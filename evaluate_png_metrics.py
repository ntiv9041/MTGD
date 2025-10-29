#!/usr/bin/env python3
"""
Evaluate PNG samples saved by MTGD sampler.

Each PNG contains two halves (width-wise):
  Left  = generated PET slice
  Right = ground-truth PET slice

This script:
  - Loads samples_manifest.csv
  - Splits PNGs into (gen, gt)
  - Normalizes to [0,1]
  - Creates a foreground mask on gt (Otsu or percentile) [optional]
  - Computes PSNR, SSIM (ROI), NMAE, NCC
  - Saves per-image and aggregated metrics to CSV

Author: Nico + Copilot
"""

import argparse
import os
import math
import csv
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import cv2
from skimage.metrics import structural_similarity as ssim


# -----------------------
# Metric helpers
# -----------------------
def compute_psnr(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return float('inf')
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def compute_nmae(gt: np.ndarray, pred: np.ndarray) -> float:
    mae = np.mean(np.abs(gt - pred))
    denom = gt.max() - gt.min()
    if denom < 1e-12:
        denom = 1.0
    return float(mae / denom)


def compute_ncc(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_centered = gt - gt.mean()
    pred_centered = pred - pred.mean()
    denom = np.linalg.norm(gt_centered) * np.linalg.norm(pred_centered)
    if denom < 1e-12:
        return 0.0
    return float(np.sum(gt_centered * pred_centered) / denom)


# -----------------------
# Mask/ROI helpers
# -----------------------
def make_brain_mask(gt: np.ndarray, method: str = "otsu", pct: int = 70) -> np.ndarray:
    """
    Build a foreground mask from ground-truth intensities.
    - method="otsu": automatic threshold on 8-bit version of gt.
    - method="percentile": keep pixels >= pct-th percentile (0..100).
    Returns boolean (H,W) mask.
    """
    if method == "otsu":
        g8 = np.clip(gt * 255.0, 0, 255).astype(np.uint8)
        thr, _ = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = gt >= (thr / 255.0)
    elif method == "percentile":
        t = np.percentile(gt, pct)
        mask = gt >= t
    else:
        # default: everything
        mask = np.ones_like(gt, dtype=bool)

    # Optional clean (small noise removal)
    mask = mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(bool)


def mask_or_roi(gt: np.ndarray, gen: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Tuple[int,int,int,int]]:
    """
    Prepare masked arrays for PSNR/NMAE/NCC and an ROI (ymin,ymax,xmin,xmax) for SSIM.
    If mask is None or empty, use full image.
    """
    if mask is None or mask.sum() == 0:
        H, W = gt.shape
        return gt, gen, (0, H, 0, W)

    ys, xs = np.where(mask)
    ymin, ymax = ys.min(), ys.max() + 1
    xmin, xmax = xs.min(), xs.max() + 1

    gt_m = gt[mask]
    gen_m = gen[mask]
    return gt_m, gen_m, (ymin, ymax, xmin, xmax)


# -----------------------
# PNG loading & splitting
# -----------------------
def load_halves_from_png(png_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read PNG and split into (generated, ground-truth).
    Auto-detect orientation:
      - Horizontal (W >= H): [gen | gt]
      - Vertical   (H >  W): [gen
                              gt]
    Normalize to [0,1] float32.
    """
    import cv2
    import numpy as np

    img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {png_path}")

    # Convert to grayscale if needed
    if img.ndim == 3:
        if img.shape[2] == 1:
            gray = img[:, :, 0]
        elif img.shape[2] in (3, 4):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if img.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
        else:
            raise RuntimeError(f"Unexpected image shape: {img.shape}")
    else:
        gray = img

    H, W = gray.shape

    # Orientation check
    if W >= H:
        # Horizontal: [gen | gt]
        mid = W // 2
        gen = gray[:, :mid]
        gt  = gray[:, mid:]
    else:
        # Vertical: [gen
        #           gt]
        mid = H // 2
        gen = gray[:mid, :]
        gt  = gray[mid:, :]

    # Normalize to [0,1]
    gen = gen.astype(np.float32) / 255.0
    gt  = gt.astype(np.float32) / 255.0
    return gen, gt


# -----------------------
# Main evaluation
# -----------------------
def evaluate_manifest(manifest_path: str,
                      output_dir: str,
                      data_range: float = 1.0,
                      mask_zero_gt: bool = False,
                      mask_method: str = "none",
                      mask_pct: int = 70,
                      limit_folders: Optional[List[int]] = None,
                      limit_images: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(manifest_path)

    # Optional filter by batch_index (folder ids)
    if limit_folders is not None and len(limit_folders) > 0:
        df = df[df["batch_index"].isin(limit_folders)].copy()

    # Optional cap on total rows
    if limit_images is not None and limit_images > 0:
        df = df.head(limit_images).copy()

    records = []
    total = len(df)
    print(f"Evaluating {total} images...")

    for _, row in df.iterrows():
        png_path    = row["png_path"]
        subject_id  = row.get("subject_id", "UNKNOWN")
        tracer_name = row.get("tracer_name", "UNKNOWN")
        pet_index   = int(row.get("pet_index", 0))
        batch_index = int(row.get("batch_index", -1))
        seq_index   = int(row.get("sequence_index", -1))
        pet_path    = row.get("pet_path", "UNKNOWN")

        if not os.path.exists(png_path):
            print(f"[WARN] Missing PNG: {png_path}")
            continue

        try:
            gen, gt = load_halves_from_png(png_path)

            # Build mask if requested
            mask = None
            if mask_zero_gt:
                mask = gt != 0
            if mask_method != "none":
                # Combine zero mask with method mask if both are requested
                method_mask = make_brain_mask(gt, method=mask_method, pct=mask_pct)
                mask = method_mask if mask is None else (mask & method_mask)

            gt_m, gen_m, (ymin, ymax, xmin, xmax) = mask_or_roi(gt, gen, mask)

            # Metrics (masked arrays for PSNR/NMAE/NCC)
            m_psnr = compute_psnr(gt_m, gen_m, data_range=data_range)
            m_nmae = compute_nmae(gt_m, gen_m)
            m_ncc  = compute_ncc(gt_m, gen_m)

            # SSIM (compute on ROI to avoid background)
            roi_gt  = gt[ymin:ymax, xmin:xmax]
            roi_gen = gen[ymin:ymax, xmin:xmax]
            # small ROI guard: SSIM needs at least a few pixels
            if roi_gt.size < 16 or roi_gen.size < 16:
                m_ssim = ssim(gt, gen, data_range=data_range, gaussian_weights=True)
            else:
                m_ssim = ssim(roi_gt.astype(np.float64),
                              roi_gen.astype(np.float64),
                              data_range=data_range,
                              gaussian_weights=True)

            records.append({
                "png_path": png_path,
                "subject_id": subject_id,
                "tracer_name": tracer_name,
                "pet_index": pet_index,
                "batch_index": batch_index,
                "sequence_index": seq_index,
                "psnr": m_psnr,
                "ssim": m_ssim,
                "nmae": m_nmae,
                "ncc": m_ncc
            })
        except Exception as e:
            print(f"[ERROR] {png_path}: {e}")

    per_image_df = pd.DataFrame.from_records(records)
    per_image_out = os.path.join(output_dir, "metrics_per_image.csv")
    per_image_df.to_csv(per_image_out, index=False)
    print(f"[OK] Saved per-image metrics -> {per_image_out}")

    by_sub_tracer = (per_image_df
                     .groupby(["subject_id", "tracer_name"])
                     .agg({"psnr":["mean","std","count"],
                           "ssim":["mean","std","count"],
                           "nmae":["mean","std","count"],
                           "ncc":["mean","std","count"]})
                     .reset_index())
    by_sub_tracer_out = os.path.join(output_dir, "metrics_by_subject_tracer.csv")
    by_sub_tracer.to_csv(by_sub_tracer_out, index=False)
    print(f"[OK] Saved aggregated metrics (subject,tracer) -> {by_sub_tracer_out}")

    by_tracer = (per_image_df
                 .groupby(["tracer_name"])
                 .agg({"psnr":["mean","std","count"],
                       "ssim":["mean","std","count"],
                       "nmae":["mean","std","count"],
                       "ncc":["mean","std","count"]})
                 .reset_index())
    by_tracer_out = os.path.join(output_dir, "metrics_by_tracer.csv")
    by_tracer.to_csv(by_tracer_out, index=False)
    print(f"[OK] Saved aggregated metrics (tracer) -> {by_tracer_out}")

    overall = per_image_df[["psnr","ssim","nmae","ncc"]].mean().to_dict()
    print("\n=== Overall Averages ===")
    for k, v in overall.items():
        print(f"{k}: {v:.4f}")

    return per_image_df, by_sub_tracer, by_tracer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=str,
                   help="Path to exp/image_samples/samples_manifest.csv")
    p.add_argument("--output_dir", default="exp/metrics", type=str)
    p.add_argument("--data_range", default=1.0, type=float,
                   help="Dynamic range for PSNR/SSIM (1.0 for [0,1], 255 for [0,255])")
    p.add_argument("--mask_zero_gt", default=False, action="store_true",
                   help="Mask out pixels where GT==0")
    p.add_argument("--mask_method", default="none", type=str,
                   choices=["none", "otsu", "percentile"],
                   help="Foreground mask method")
    p.add_argument("--mask_pct", default=70, type=int,
                   help="Percentile threshold for mask_method=percentile")
    p.add_argument("--limit_folders", nargs="*", type=int, default=None,
                   help="Optional subset of batch_index folders to evaluate (e.g., 0 1 2 3)")
    p.add_argument("--limit_images", type=int, default=None,
                   help="Optional cap on number of images to process")
    return p.parse_args()


def main():
    args = parse_args()
    evaluate_manifest(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        data_range=args.data_range,
        mask_zero_gt=args.mask_zero_gt,
        mask_method=args.mask_method,
        mask_pct=args.mask_pct,
        limit_folders=args.limit_folders,
        limit_images=args.limit_images,
    )


if __name__ == "__main__":
    main()