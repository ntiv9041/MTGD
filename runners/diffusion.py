import os
import csv
import logging
import time
import numpy as np
import torch
import sys
import cv2
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
torch.backends.cudnn.benchmark = True

sys.path.append('..')
from models.MTGD import Model
from models.load_public_dataset_all import load_data
from models.resample import LossSecondMomentResampler, LossAwareSampler, UniformSampler
from functions import get_optimizer
from functions.losses import loss_registry, create_brain_mask


def pad_tensor(tensor):
    # Kept for compatibility; not used in current pipeline
    target_stack_size = 224
    padded_tensor = F.pad(tensor, ((3, 3, 21, 21, 0, 0, 0, 0)), mode='constant', value=0)
    return padded_tensor


# from [-1,1] to [0,1]
def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """Standard beta schedule helper."""
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5,
                             num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1,
                                  num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)

    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]
        self.schedule_sampler = LossSecondMomentResampler(self.num_timesteps)

        # parameters
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0)

        # variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # logvar config
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        # >>> logging for hyperparams 2/11/25 <<<
        logging.info(f"[CONFIG] img_size? (handled in data), T={config.diffusion.num_diffusion_timesteps}, "
             f"batch={config.training.batch_size}, lr={config.optim.lr}, "
             f"modalities={config.model.num_input_modality}, sequences={list(config.mri.mri_sequence)}")

        # DataLoader with perf knobs from config.data
        train_loader = load_data(
            config.mri.mri_sequence,
            config.pet.pet_modalities,
            config.folder_path.path,
            num_workers=getattr(config.data, "num_workers", 0),
            pin_memory=getattr(config.data, "pin_memory", False),
            persistent_workers=getattr(config.data, "persistent_workers", False),
        )

        model = Model(self.config).to(self.device)

        # >>> REPLACE optimizer 2/11/25 <<<
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.optim.lr, betas=(self.config.optim.beta1, 0.999), weight_decay=1e-4)
        total_steps = getattr(self.config.training, "n_iters", 20000)
        warmup = getattr(self.config.training, "warmup_steps", 2000)
        def lr_lambda(step):
            if step < warmup:
                return float(step + 1) / float(max(1, warmup))
            # Cosine to 0.1× final
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(1.0, max(0.0, progress))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


        # >>> ADD (EMA) 2/11/25<<<
        ema_decay = getattr(self.config.training, "ema_decay", 0.999)
        ema_model = Model(self.config).to(self.device)
        ema_model.load_state_dict(model.state_dict(), strict=True)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)


        use_amp = getattr(self.config.training, "mixed_precision", False)
        scaler = GradScaler(device="cuda", enabled=use_amp)

        start_epoch, step = 0, 0

        # resume_training
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])
            # keep optimizer eps from config
            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]

        model.train()

        # stop conditions & logging cadence
        wall_start = time.time()
        max_minutes = getattr(self.config.training, "max_minutes", None)
        max_steps = getattr(self.config.training, "n_iters", None)
        log_every = getattr(self.config.training, "log_every", 100)

        # >>> PER-TRACER METRICS 2/11/25 <<<
        tracer_running = {}   # idx -> [sum, count]
        tracer_best = {}      # idx -> best_wloss

        # optional central-slice training filter
        center_cfg = getattr(self.config.data, "train_central_slices", None)
        center_enabled = bool(center_cfg and getattr(center_cfg, "enabled", False))
        center_count = int(center_cfg.count) if center_enabled else None

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0.0
            minibatch = self.config.training.batch_size

            for i, (input_img, labels, pet_type) in enumerate(train_loader):
                # Input shape wrangling
                input_img = input_img.squeeze(0).permute(3, 0, 1, 2)
                input_img = F.interpolate(input_img, size=(224, 224),
                                          mode='bilinear', align_corners=False).unsqueeze(2)

                labels = labels.permute(3, 0, 1, 2)
                labels = F.interpolate(labels, size=(224, 224),
                                       mode='bilinear', align_corners=False)

                # optional central-slice filter per subject
                if center_enabled and isinstance(center_count, int) and center_count > 0:
                    total = labels.shape[0]
                    if center_count < total:
                        start_c = max(0, (total - center_count) // 2)
                        end_c = start_c + center_count
                        labels = labels[start_c:end_c]
                        input_img = input_img[start_c:end_c]

                index = 0
                while index < len(labels):
                    start_idx = index
                    stop_idx = index + minibatch
                    index += minibatch
                    if stop_idx >= len(labels):
                        stop_idx = len(labels)

                    mini_labels = labels[start_idx:stop_idx]
                    mini_inputs = input_img[start_idx:stop_idx]

                    # drop all-black target slices
                    batch_sums = mini_labels.sum(dim=(1, 2, 3))
                    non_zero_indices = batch_sums != 0
                    if torch.sum(non_zero_indices == False).item() == (stop_idx - start_idx):
                        continue
                    else:
                        mini_labels = mini_labels[non_zero_indices]
                        mini_inputs = mini_inputs[non_zero_indices]

                    # edge-case: single slice -> add batch dim
                    if len(mini_inputs.shape) == 3:
                        mini_labels = mini_labels.unsqueeze(0)
                        mini_inputs = mini_inputs.unsqueeze(0)

                    n = mini_labels.size(0)
                    data_time += time.time() - data_start
                    step += 1

                    # >> Changes for Colab A100 2/11/25<<<
                    mini_labels = mini_labels.to(self.device).float()
                    mini_inputs = mini_inputs.to(self.device).float()
                    
                    # Apply channels_last only if 4D (N, C, H, W)
                    if mini_labels.dim() == 4:
                        mini_labels = mini_labels.to(memory_format=torch.channels_last)
                    if mini_inputs.dim() == 4:
                        mini_inputs = mini_inputs.to(memory_format=torch.channels_last)

                    pet_type_batch = pet_type.to(self.device).float().repeat_interleave(n)
                    if step == 1:
                        logging.info(f"[Batch stats] inputs shape={mini_inputs.shape}, labels shape={mini_labels.shape}, "
                                     f"pet_type (first 4)={pet_type_batch[:4].tolist()}")
                        logging.info(f"[Norm check] mini_inputs mean={mini_inputs.mean().item():.3f}, "
                                     f"std={mini_inputs.std().item():.3f}; "
                                     f"mini_labels min={mini_labels.min().item():.3f}, "
                                     f"max={mini_labels.max().item():.3f}")
                                    
                    # Optional: drop tracer condition with small prob
                    p_uncond = float(getattr(self.config, "p_uncond", 0.1))
                    if p_uncond > 0.0:
                        mask = (torch.rand_like(pet_type_batch) < p_uncond)
                        pet_type_batch = pet_type_batch * (~mask)  # zero => "uncond id"

                    e = torch.randn_like(mini_labels)
                    b = self.betas
                    t, weights = self.schedule_sampler.sample(mini_labels.shape[0], self.device)
                    
                    # --- Create brain mask for masked loss ---
                    with torch.no_grad():
                        brain_mask = create_brain_mask(mini_labels)

                    # ------ Better logs for marking ---------
                    if step == 1 or step % 1000 == 0:
                        valid_ratio = brain_mask.mean().item()
                        logging.info(f"[Mask] Avg nonzero ratio={valid_ratio:.3f}")


                    # ----- forward loss with/without AMP -----
                    if use_amp:
                        with autocast(device_type="cuda"):
                            loss = loss_registry[config.model.type](
                                model, mini_labels, t, e, b, mini_inputs, pet_type_batch, mask=brain_mask
                            )
                        self.schedule_sampler.update_with_local_losses(t, loss.detach())

                        w_loss = (loss * weights).mean(dim=0)
                        # pet_type_batch is shape (n,), but you repeated earlier; keep per-sample mapping 2/11/25
                        for lt, lval in zip(pet_type_batch.detach().long().tolist(), (loss * weights).detach().cpu().tolist()):
                            s, c = tracer_running.get(lt, [0.0, 0])
                            tracer_running[lt] = [s + float(lval), c + 1]

                        optimizer.zero_grad(set_to_none=True)

                        # >>> Grad Clip AMP_sage 2/11/25 <<<
                        scaler.scale(w_loss).backward()
                        # Unscale, then clip
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(config.optim, "grad_clip", 1.0))
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        # >>> UPDATE EMA 2/11/25 <<<
                        with torch.no_grad():
                            m, m_ema = model.state_dict(), ema_model.state_dict()
                            for k in m_ema.keys():
                                m_ema[k].mul_(ema_decay).add_(m[k], alpha=1.0 - ema_decay)

                    else:
                        loss = loss_registry[config.model.type](
                            model, mini_labels, t, e, b, mini_inputs, pet_type_batch, mask=brain_mask
                        )                               
                        self.schedule_sampler.update_with_local_losses(t, loss.detach())

                        w_loss = (loss * weights).mean(dim=0)
                        # pet_type_batch is shape (n,), but you repeated earlier; keep per-sample mapping 2/11/25
                        for lt, lval in zip(pet_type_batch.detach().long().tolist(), (loss * weights).detach().cpu().tolist()):
                            s, c = tracer_running.get(lt, [0.0, 0])
                            tracer_running[lt] = [s + float(lval), c + 1]

                        optimizer.zero_grad(set_to_none=True)
                        w_loss.backward()
                        try:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
                        except Exception:
                            pass
                        optimizer.step()
                        # >>> UPDATE EMA 2/11/25 <<<
                        scheduler.step()
                        with torch.no_grad():
                            m, m_ema = model.state_dict(), ema_model.state_dict()
                            for k in m_ema.keys():
                                m_ema[k].mul_(ema_decay).add_(m[k], alpha=1.0 - ema_decay)

                    # logging
                    if step % log_every == 0:
                        logging.info(
                            f"step={step} loss={loss.mean(dim=0):.6f} "
                            f"w_loss={w_loss:.6f} data_time={data_time/(i+1):.4f}s"
                        )

                    # checkpointing
                    if step % self.config.training.snapshot_freq == 0 or step == 1:
                        # >>> 2/11/25 <<<
                        states = [ema_model.state_dict(), optimizer.state_dict(), epoch, step]
                        # >>> SAVE 'BEST' EMA ckpt per tracer 2/11/25<<<
                        for tid, (s, c) in tracer_running.items():
                            avg = s / max(1, c)
                            prev = tracer_best.get(tid, float("inf"))
                            if avg < prev:
                                tracer_best[tid] = avg
                                torch.save([ema_model.state_dict(), optimizer.state_dict(), epoch, step],
                                           os.path.join(self.args.log_path, f"ckpt_best_tracer{tid}.pth"))
                        tracer_running.clear()

                        torch.save(states, os.path.join(self.args.log_path, f"ckpt_{step}.pth"))
                        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                    # stop conditions
                    elapsed_min = (time.time() - wall_start) / 60.0
                    if (max_steps is not None and step >= max_steps) or \
                       (max_minutes is not None and elapsed_min >= max_minutes):
                        # >>> 2/11/25 <<<
                        states = [ema_model.state_dict(), optimizer.state_dict(), epoch, step]
                        # >>> SAVE 'BEST' EMA ckpt per tracer 2/11/25<<<
                        for tid, (s, c) in tracer_running.items():
                            avg = s / max(1, c)
                            prev = tracer_best.get(tid, float("inf"))
                            if avg < prev:
                                tracer_best[tid] = avg
                                torch.save([ema_model.state_dict(), optimizer.state_dict(), epoch, step],
                                           os.path.join(self.args.log_path, f"ckpt_best_tracer{tid}.pth"))
                        tracer_running.clear()
                        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                        torch.save(states, os.path.join(self.args.log_path, f"ckpt_{step}.pth"))
                        logging.info(f"Stopping: step={step}, elapsed_min={elapsed_min:.2f}. Saved ckpt.")
                        return  # clean exit

                data_start = time.time()

    def sample(self):
        args, config = self.args, self.config
        model = Model(self.config).to(self.device)
        # >>> logging for hyperparams 2/11/25 <<<
        logging.info(f"[CONFIG] img_size? (handled in data), T={config.diffusion.num_diffusion_timesteps}, "
             f"batch={config.training.batch_size}, lr={config.optim.lr}, "
             f"modalities={config.model.num_input_modality}, sequences={list(config.mri.mri_sequence)}")

        # # load parameters
        # try:
        #     states = torch.load(
        #         os.path.join(self.args.log_path, "ckpt.pth"),
        #         map_location=self.config.device,
        #     )
        #     model.load_state_dict(states[0], strict=True)
        # except Exception:
        #     logging.info('could not load parameters')
        #     exit()
        
        # --- Debug checkpoint loading 2/11/25 ---
        ckpt_path = os.path.join(self.args.log_path, "ckpt.pth")
        logging.info(f"[DEBUG] Trying to load checkpoint from: {ckpt_path}")
        
        if not os.path.exists(ckpt_path):
            logging.error(f"[ERROR] Checkpoint not found at: {ckpt_path}")
            exit()
        
        try:
            states = torch.load(ckpt_path, map_location=self.config.device, weights_only=False)
            logging.info(f"[DEBUG] Checkpoint loaded successfully. Keys: {list(states.keys()) if isinstance(states, dict) else type(states)}")
        
            # handle both dict and list formats
            if isinstance(states, dict) and "model" in states:
                model.load_state_dict(states["model"], strict=False)
            elif isinstance(states, (list, tuple)):
                model.load_state_dict(states[0], strict=False)
            else:
                model.load_state_dict(states, strict=False)
        
            logging.info("[DEBUG] Model weights loaded successfully!")
        
        except Exception as e:
            logging.error(f"[ERROR] Failed to load checkpoint: {e}")
            import traceback
            logging.error(traceback.format_exc())
            exit()



        model = model.to(self.device)
        logging.info('loading parameters successfully')
        model.eval()

        # test loader with perf knobs
        test_loader = load_data(
            config.mri.mri_sequence, config.pet.pet_modalities, config.folder_path.path,
            train=False,
            num_workers=getattr(config.data, "num_workers", 0),
            pin_memory=getattr(config.data, "pin_memory", False),
            persistent_workers=getattr(config.data, "persistent_workers", False),
        )
        self.sample_sequence(model, test_loader)

    def sample_sequence(self, model, test_loader):
        """
        Generate images for all items in test_loader, save PNGs, and write a manifest CSV.

        Manifest columns:
        png_path, subject_id, tracer_name, pet_index, batch_index, sequence_index, pet_path
        """
        minibatch = self.config.sampling.batch_size
        logging.info('start generating images')

        # Prepare manifest CSV under image_samples/
        manifest_path = os.path.join(self.args.image_folder, "samples_manifest.csv")
        os.makedirs(self.args.image_folder, exist_ok=True)
        write_header = not os.path.exists(manifest_path)
        manifest_f = open(manifest_path, 'a', newline='')
        manifest_w = csv.writer(manifest_f)
        if write_header:
            manifest_w.writerow([
                "png_path", "subject_id", "tracer_name", "pet_index",
                "batch_index", "sequence_index", "pet_path"
            ])

        max_slices_cfg = getattr(self.config.sampling, 'max_slices', None)

        try:
            for i, batch in enumerate(test_loader):
                # The sampling dataset may return:
                # (condition, x0, pet_type, pet_path, mri_list) -> len==5
                # (condition, x0, pet_type) -> len==3
                if isinstance(batch, (list, tuple)) and len(batch) == 5:
                    condition, x0, pet_type, pet_path_list, mri_list = batch
                    pet_path = pet_path_list[0] if isinstance(pet_path_list, (list, tuple)) else pet_path_list
                elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                    condition, x0, pet_type = batch
                    pet_path = "UNKNOWN"
                    mri_list = []
                else:
                    raise RuntimeError(
                        f"Unexpected batch structure: type={type(batch)} "
                        f"len={len(batch) if hasattr(batch,'__len__') else 'NA'}"
                    )

                tracer_name = os.path.basename(pet_path) if pet_path != "UNKNOWN" else "UNKNOWN"
                subject_id = os.path.basename(os.path.dirname(pet_path)) if pet_path != "UNKNOWN" else "UNKNOWN"
                logging.info(f"[Sampling] Subject={subject_id} Tracer={tracer_name} (batch {i})")

                # Shape wrangling
                condition = condition.squeeze(0).permute(3, 0, 1, 2)
                condition = F.interpolate(condition, size=(224, 224),
                                          mode='bilinear', align_corners=False).unsqueeze(2)
                x0 = x0.permute(3, 0, 1, 2)
                x0 = F.interpolate(x0, size=(224, 224), mode='bilinear', align_corners=False)

                # optional: limit central slices for sampling
                if isinstance(max_slices_cfg, int) and max_slices_cfg > 0:
                    total = len(x0)
                    if max_slices_cfg < total:
                        start_c = max(0, (total - max_slices_cfg) // 2)
                        end_c = start_c + max_slices_cfg
                        x0 = x0[start_c:end_c]
                        condition = condition[start_c:end_c]
                        logging.info(f"[Slices] Limiting to {max_slices_cfg} central slices (from total={total})")

                seq_counter = 0
                index = 0
                while index < len(x0):
                    start_idx = index
                    stop_idx = index + minibatch
                    index = stop_idx
                    if stop_idx >= len(x0):
                        stop_idx = len(x0)

                    x0_mini = x0[start_idx:stop_idx]
                    condition_mini = condition[start_idx:stop_idx]

                    # drop all-black target slices
                    batch_sums = x0_mini.sum(dim=(1, 2, 3))
                    non_zero_indices = batch_sums != 0
                    if torch.sum(non_zero_indices == False).item() == (stop_idx - start_idx):
                        continue
                    else:
                        condition_mini = condition_mini[non_zero_indices]
                        x0_mini = x0_mini[non_zero_indices]

                    # Edge-case: single slice -> add batch dim
                    if len(condition_mini.shape) == 3:
                        x0_mini = x0_mini.unsqueeze(0)
                        condition_mini = condition_mini.unsqueeze(0)

                    x0_mini = x0_mini.to(self.device).float()
                    condition_mini = condition_mini.to(self.device).float()

                    # Random noise for sampling
                    x = torch.randn(
                        (len(x0_mini), 1, x0_mini.shape[2], x0_mini.shape[3]),
                        device=self.device,
                    )
                    with torch.no_grad():
                        pet_type_batch = pet_type.to(self.device).float().repeat_interleave(x.shape[0])
                        x = self.sample_image(x, model, condition_mini, pet_type_batch, last=True)

                    # Convert to uint8 for PNG saving
                    x_gen_u8 = (x * 255.0).clamp(0, 255).to(torch.uint8)
                    x0_u8 = (x0_mini * 255.0).clamp(0, 255).to(torch.uint8)

                    # HWC for cv2.imwrite
                    x_gen_u8 = x_gen_u8.permute(0, 2, 3, 1).contiguous().cpu().numpy()   # (N,H,W,1)
                    x0_u8 = x0_u8.permute(0, 2, 3, 1).contiguous().cpu().numpy()         # (N,H,W,1)

                    # Concatenate horizontally: [generated | ground-truth]
                    images = np.concatenate([x_gen_u8, x0_u8], axis=1)  # width doubled

                    # Save and write manifest rows
                    batch_folder = os.path.join(self.args.image_folder, str(i))
                    os.makedirs(batch_folder, exist_ok=True)
                    for img_np in images:
                        png_path = os.path.join(batch_folder, str(seq_counter)) + '.png'
                        cv2.imwrite(png_path, img_np)
                        manifest_w.writerow([
                            png_path,
                            subject_id,
                            tracer_name,
                            int(pet_type.item()),  # pet_index as scalar
                            i,                      # batch_index
                            seq_counter,            # sequence_index
                            pet_path
                        ])
                        seq_counter += 1

                logging.info(f"[Sampling] Saved {seq_counter} slices to {os.path.join(self.args.image_folder, str(i))}")
                logging.info(f"Manifest updated: {manifest_path}")
        finally:
            manifest_f.close()

    def sample_image(self, x, model, condition, pet_type_batch, last=True):
        # Use a stride to cut the number of denoising steps (e.g., 10 -> ~100 steps if T=1000).
        skip = getattr(self.config.sampling, 'timestep_stride', 1)
        seq = range(0, self.num_timesteps, skip)
    
        # Get classifier-free guidance strength (default 1.0 = no guidance)
        guidance_scale = float(getattr(self.config.sampling, "guidance_scale", 1.0))
    
        # >>> Choose sampler based on config (updated 2/11/25) <<<
        stride = int(getattr(self.config.sampling, "timestep_stride", 10))
        if stride <= 1:
            from functions.denoising import ddpm_steps
            _, x = ddpm_steps(
                x, seq, model, self.betas, condition, pet_type_batch,
                guidance_scale=guidance_scale
            )
        else:
            from functions.denoising import ddim_steps
            _, x = ddim_steps(
                x, seq, model, self.betas, condition, pet_type_batch,
                guidance_scale=guidance_scale
            )
    
        return x[-1]


    def save_img(self, imgs, p_idx, idx):
        folder = os.path.join(self.args.image_folder, str(p_idx))
        if not os.path.exists(folder):
            os.makedirs(folder)
        for mini_index in range(len(imgs)):
            cv2.imwrite(os.path.join(folder, str(idx)) + '.png', imgs[mini_index])
            idx += 1
        return idx













