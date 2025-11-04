# evaluate_generated.py
import argparse, os, glob, re
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import nibabel as nib
from skimage.metrics import structural_similarity as ssim_sk
from skimage.metrics import peak_signal_noise_ratio as psnr_sk

def robust_minmax(x, low_q=1.0, high_q=99.0, eps=1e-8):
    lo = np.percentile(x, low_q)
    hi = np.percentile(x, high_q)
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + eps)

def load_png(path):
    im = Image.open(path).convert("L")
    arr = np.asarray(im).astype(np.float32) / 255.0  # normalize 0..1
    return arr

def resize_slice(arr, target_hw):
    im = Image.fromarray(np.asarray(arr))
    im = im.resize(target_hw[::-1], resample=Image.BILINEAR)
    return np.asarray(im).astype(np.float32)

def compute_metrics(pred, gt):
    mae = float(np.mean(np.abs(pred - gt)))
    mse = float(np.mean((pred - gt) ** 2))
    psnr = float(psnr_sk(gt, pred, data_range=1.0)) if np.std(gt) > 0 else np.inf
    try:
        ssim = float(ssim_sk(gt, pred, data_range=1.0))
    except Exception:
        ssim = np.nan
    return mae, mse, psnr, ssim

def find_gt_pet(subject_id, tracer, data_root):
    """
    Try to locate a PET .nii.gz matching the subject and tracer name.
    """
    subj_glob = sorted(glob.glob(os.path.join(data_root, "**", f"*{subject_id}*{tracer}"), recursive=True))
    if not subj_glob:
        subj_glob = sorted(glob.glob(os.path.join(data_root, "**", f"*{tracer}"), recursive=True))
    return subj_glob[0] if subj_glob else None

def parse_slice_idx(png_name):
    m = re.search(r"(\d+)\.png$", png_name)
    return int(m.group(1)) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="./exp/image_samples/images/samples_manifest.csv")
    ap.add_argument("--images_root", default="./exp/image_samples/images")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--axis", default="axial", choices=["axial","sagittal","coronal"])
    ap.add_argument("--lowq", type=float, default=1.0)
    ap.add_argument("--highq", type=float, default=99.0)
    ap.add_argument("--out_csv", default="./exp/image_samples/metrics_report.csv")
    args = ap.parse_args()

    Path(os.path.dirname(args.out_csv)).mkdir(parents=True, exist_ok=True)

    # --- Read manifest ---
    df = pd.read_csv(args.manifest)
    cols = {c.lower(): c for c in df.columns}
    folder_col = cols.get("folder_id", cols.get("folder", list(df.columns)[0]))
    subj_col   = cols.get("subject_id", cols.get("subject", None))
    tracer_col = cols.get("tracer_name", cols.get("tracer", None))

    if subj_col is None or tracer_col is None:
        raise RuntimeError(
            f"Manifest must contain either subject_id/subject and tracer_name/tracer columns. Found columns: {df.columns}"
        )

    rows = []
    for _, r in df.iterrows():
        folder_id = str(r[folder_col])
        subject_id = str(r[subj_col])
        tracer = str(r[tracer_col])
        
        # --- New robust logic: use png_path if provided ---
        if "png_path" in df.columns:
            png_path = str(r["png_path"])
            if os.path.isfile(png_path):
                pngs = [png_path]
                gen_dir = os.path.dirname(png_path)
            else:
                gen_dir = os.path.join(args.images_root, folder_id)
                pngs = sorted(glob.glob(os.path.join(gen_dir, "*.png")), key=lambda p: parse_slice_idx(os.path.basename(p)))
        else:
            gen_dir = os.path.join(args.images_root, folder_id)
            pngs = sorted(glob.glob(os.path.join(gen_dir, "*.png")), key=lambda p: parse_slice_idx(os.path.basename(p)))
        
        if not pngs:
            print(f"[WARN] No PNGs found for entry={folder_id}")
            continue


        print(f"[DEBUG] Looking for PET: subject_id={subject_id}, tracer={tracer}")
        gt_path = find_gt_pet(subject_id, tracer, args.data_root)
        if gt_path is None:
            print(f"[WARN] GT PET not found for subject={subject_id}, tracer={tracer}")
            continue
        else:
            print(f"[INFO] Found PET for subject={subject_id}: {gt_path}")

        pet_img = nib.load(gt_path)
        pet = pet_img.get_fdata().astype(np.float32)
        if args.axis == "axial":
            vol_slices = [pet[:,:,i] for i in range(pet.shape[2])]
        elif args.axis == "sagittal":
            vol_slices = [pet[i,:,:] for i in range(pet.shape[0])]
        else:
            vol_slices = [pet[:,i,:] for i in range(pet.shape[1])]

        slice_metrics = []
        for png_path in pngs:
            idx = parse_slice_idx(os.path.basename(png_path))
            if idx is None or idx >= len(vol_slices):
                break
            pred = load_png(png_path)
            gt = vol_slices[idx]
            gt = robust_minmax(gt, args.lowq, args.highq)
            gt = resize_slice(gt, pred.shape)
            mae, mse, psnr, ssim = compute_metrics(pred, gt)
            slice_metrics.append((idx, mae, mse, psnr, ssim))

        if len(slice_metrics) == 0:
            print(f"[WARN] No valid slice metrics for subject={subject_id}. Using rank-paired fallback.")
            k = min(len(pngs), len(vol_slices))
            take_idxs = np.linspace(0, len(vol_slices)-1, k).round().astype(int)
            for rank, (png_path, vi) in enumerate(zip(pngs[:k], take_idxs)):
                pred = load_png(png_path)
                gt = vol_slices[vi]
                gt = robust_minmax(gt, args.lowq, args.highq)
                gt = resize_slice(gt, pred.shape)
                mae, mse, psnr, ssim = compute_metrics(pred, gt)
                slice_metrics.append((vi, mae, mse, psnr, ssim))

        if slice_metrics:
            sm = np.array(slice_metrics, dtype=np.float32)
            mean_mae, mean_mse, mean_psnr, mean_ssim = sm[:,1].mean(), sm[:,2].mean(), sm[:,3].mean(), sm[:,4].mean()
            rows.append({
                "folder_id": folder_id,
                "subject_id": subject_id,
                "tracer": tracer,
                "n_slices_eval": len(slice_metrics),
                "MAE": mean_mae,
                "MSE": mean_mse,
                "PSNR": mean_psnr,
                "SSIM": mean_ssim
            })

    print(f"[DEBUG] Processed {len(rows)} subjects with valid metrics.")
    if len(rows) == 0:
        print("[DEBUG] No metrics recorded. Possible reasons:")
        print("  - PET ground-truth file not found (path mismatch)")
        print("  - Slice index mismatch between generated PNGs and PET volume")
        print("  - PNG folders empty or named differently")

    # even if no rows, create a blank CSV
    out = pd.DataFrame(rows)
    if not out.empty and "SSIM" in out.columns:
        out = out.sort_values(by="SSIM", ascending=False)
    out.to_csv(args.out_csv, index=False)
    print(f"[DONE] Wrote: {args.out_csv}")

    if not out.empty:
        print("\nTop 5 subjects by SSIM:")
        print(out.head(5).to_string(index=False))
        print("\nBottom 5 subjects by SSIM:")
        print(out.tail(5).to_string(index=False))

if __name__ == "__main__":
    main()
