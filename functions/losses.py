import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim


# --------------------------------------------------------
# Helper: Create simple PET brain mask
# --------------------------------------------------------
def create_brain_mask(x, threshold=0.05):
    """
    Create a binary brain mask from PET (or MRI-like) input.
    Ignores background and focuses loss on tissue region.
    """
    with torch.no_grad():
        # mask = (x > threshold * x.max(dim=(-1, -2), keepdim=True).values).float()
        mask = (x > threshold * x.max()).float()
    return mask


# --------------------------------------------------------
# Main diffusion noise estimation loss (with masking)
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
    Combined masked diffusion loss:
      - Predicts added noise (eps)
      - Uses MSE + SSIM hybrid objective
      - Optionally applies a PET brain mask to focus on relevant voxels
    """
    # alpha cumprod
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

    # generate noisy sample x_t
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()

    # model predicts noise
    output = model(x, t.float(), condition, pet_type)

    # raw noise estimation error
    err = (e - output)

    # --------------------------------------------------------
    # optional shape handling for debugging
    # --------------------------------------------------------
    if keepdim:
        return err.square().sum(dim=(1, 2, 3))
    if loss_moment:
        return err.square().mean(dim=list(range(1, len(output.shape))))

    # --------------------------------------------------------
    # Masked MSE region loss
    # --------------------------------------------------------
    mse_map = err.square()
    if mask is not None:
        mse_map = mse_map * mask
        denom = mask.sum() + 1e-6
        mse = mse_map.sum() / denom
    else:
        mse = mse_map.mean()

    # --------------------------------------------------------
    # Structural Similarity on predicted x0
    # --------------------------------------------------------
    a_sqrt = a.sqrt()
    x0_pred = (x - (1.0 - a).sqrt() * output) / a_sqrt
    x0_pred = torch.clamp(x0_pred, 0.0, 1.0)
    x0_true = torch.clamp(x0, 0.0, 1.0)

    try:
        ssim_val = ssim(x0_pred, x0_true, data_range=1.0)
    except Exception:
        ssim_val = torch.tensor(0.0, device=x0.device)

    # --------------------------------------------------------
    # Weighted blend
    # --------------------------------------------------------
    loss = 0.84 * mse + 0.16 * (1.0 - ssim_val)

    return loss


# --------------------------------------------------------
# Loss registry (keeps consistency with runner)
# --------------------------------------------------------
loss_registry = {
    "simple": noise_estimation_loss,
}
