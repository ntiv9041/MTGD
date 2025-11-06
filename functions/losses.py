import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim


# --------------------------------------------------------
# Helper: Correct brain mask creation
# --------------------------------------------------------
def create_brain_mask(x, threshold=0.01):
    """
    Create a binary brain mask from PET input:
      - 1 = brain region (voxels > threshold * local max)
      - 0 = background
    Works slice-wise to avoid global over-thresholding.
    """
    with torch.no_grad():
        # Compute per-slice local maxima to adapt to intensity range
        local_max = x.amax(dim=(-1, -2), keepdim=True)
        mask = (x > threshold * local_max).float()
    return mask


# --------------------------------------------------------
# Main masked diffusion loss
# --------------------------------------------------------
def noise_estimation_loss(
    model,
    x0: torch.Tensor,
    t: torch.LongTensor,
    e: torch.Tensor,
    b: torch.Tensor,
    condition,
    pet_type,
    keepdim=False,
    loss_moment=False,
    mask=None,
):
    """
    Diffusion noise-prediction loss with masked supervision:
      - Focuses on brain voxels
      - Suppresses background
      - Combines masked MSE + SSIM structure term
    """
    # time-dependent weighting
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float(), condition, pet_type)

    err = (e - output)

    # --------------------------------------------------------
    # Create or validate mask
    # --------------------------------------------------------
    if mask is None:
        mask = create_brain_mask(x0, threshold=0.01)

    # Ensure proper shape & normalization
    mask = F.interpolate(mask, size=x0.shape[-2:], mode='nearest')
    inv_mask = 1.0 - mask

    # --------------------------------------------------------
    # Masked region loss (inside brain only)
    # --------------------------------------------------------
    mse_in = ((err ** 2) * mask).sum() / (mask.sum() + 1e-6)

    # --------------------------------------------------------
    # Weak background suppression (encourages dark background)
    # --------------------------------------------------------
    mse_out = ((output ** 2) * inv_mask).mean()

    # --------------------------------------------------------
    # Structural Similarity (inside mask)
    # --------------------------------------------------------
    a_sqrt = a.sqrt()
    x0_pred = (x - (1.0 - a).sqrt() * output) / a_sqrt
    x0_pred = torch.clamp(x0_pred, 0.0, 1.0)
    x0_true = torch.clamp(x0, 0.0, 1.0)

    try:
        ssim_val = ssim(x0_pred * mask, x0_true * mask, data_range=1.0)
    except Exception:
        ssim_val = torch.tensor(0.0, device=x0.device)

    # --------------------------------------------------------
    # Final weighted combination
    # --------------------------------------------------------
    loss = (
        0.80 * mse_in    # main brain loss
        + 0.15 * (1.0 - ssim_val)  # structure similarity
        + 0.05 * mse_out  # weak background penalty
    )

    return loss


loss_registry = {
    "simple": noise_estimation_loss,
}
