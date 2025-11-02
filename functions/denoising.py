import torch

# return alpha
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def ddpm_steps(x, seq, model, b, condition, pet_type_batch, guidance_scale=2.0, **kwargs):
    """
    DDPM sampling with optional classifier-free guidance (CFG)
    guidance_scale = 1.0 => normal DDPM (no guidance)
    guidance_scale > 1.0 => stronger conditioning
    """

    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n, device=x.device) * i)
            next_t = (torch.ones(n, device=x.device) * j)

            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1

            x = xs[-1].to(x.device)

            # --- Classifier-free guidance ---
            if guidance_scale > 1.0:
                # unconditional tracer id = 0 (already trained for this)
                pet_type_uncond = torch.zeros_like(pet_type_batch)
                # unconditional forward
                eps_uncond = model(x, t.float(), condition, pet_type_uncond)
                # conditional forward
                eps_cond = model(x, t.float(), condition, pet_type_batch)
                # combine
                e = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                # plain conditional generation
                e = model(x, t.float(), condition, pet_type_batch)

            # reconstruct x0 from predicted noise
            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to("cpu"))

            # compute mean and variance of next sample
            mean_eps = ((atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x) / (1.0 - at)
            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to("cpu"))

    return xs, x0_preds
