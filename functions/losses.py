import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim


# model,x0,t,noise,beta
def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, condition,pet_type,keepdim=False,loss_moment=False):
    # alpha cumprod
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # get xt
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    # predict noise from xt and t
    output = model(x, t.float(),condition,pet_type)
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    if loss_moment:
        return (e - output).square().mean(dim=list(range(1, len(output.shape))))
        # return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=1)
    else:
        # ---- combined L1 + SSIM loss ----
        # Compute predicted noise and target noise difference
        mse = (e - output).square().mean(dim=(1, 2, 3))   # per-sample MSE

        # Reconstruct predicted x0 for SSIM: x0_pred = (xt - sqrt(1-a)*eps_pred) / sqrt(a)
        a_sqrt = a.sqrt()
        x0_pred = (x - (1.0 - a).sqrt() * output) / a_sqrt

        # Clamp to [0,1] before SSIM to avoid NaNs
        x0_pred = torch.clamp(x0_pred, 0.0, 1.0)
        x0_true = torch.clamp(x0, 0.0, 1.0)

        ssim_val = ssim(x0_pred, x0_true, data_range=1.0)

        # Weighted blend: 84% L1/MSE term + 16% SSIM structure term
        loss = 0.84 * mse.mean() + 0.16 * (1.0 - ssim_val)

        return loss

loss_registry = {
    'simple': noise_estimation_loss,
}
