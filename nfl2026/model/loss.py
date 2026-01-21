import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class GaussianMixtureLoss(nn.Module):
    def __init__(
        self,
        time_decay: float = 0.03,
        log_std_range: tuple[float, float] = (-1.609, 3.912),
        rho_limit: float = 0.5,
    ) -> None:
        super().__init__()
        self.time_decay = max(0.0, time_decay)
        self.log_std_range = log_std_range
        self.rho_limit = rho_limit

    def forward(
        self,
        pred_traj: Float[Tensor, "B T 5"],
        target: Float[Tensor, "B T 2"],
    ) -> Float[Tensor, "B"]:
        B, T = pred_traj.shape[:2]
        nearest_xy = torch.cumsum(pred_traj[..., 0:2], dim=1)
        res_trajs = target - nearest_xy  # (B, T, 2)
        dx = res_trajs[:, :, 0]
        dy = res_trajs[:, :, 1]

        log_std1 = torch.clip(
            pred_traj[:, :, 2], min=self.log_std_range[0], max=self.log_std_range[1]
        )
        log_std2 = torch.clip(
            pred_traj[:, :, 3], min=self.log_std_range[0], max=self.log_std_range[1]
        )
        std1 = torch.exp(log_std1)  # (0.2 yds to 50 yds)
        std2 = torch.exp(log_std2)  # (0.2 yds to 50 yds)
        rho = torch.clip(pred_traj[:, :, 4], min=-self.rho_limit, max=self.rho_limit)

        t = torch.arange(T, device=pred_traj.device).float()
        weight = torch.exp(-self.time_decay * t).view(1, T).repeat(B, 1)

        # -log(a^-1 * e^b) = log(a) - b
        reg_gmm_log_coefficient = (
            log_std1 + log_std2 + 0.5 * torch.log(1 - rho**2)
        )  # (B, T)
        reg_gmm_exp = (0.5 * 1 / (1 - rho**2)) * (
            (dx**2) / (std1**2)
            + (dy**2) / (std2**2)
            - 2 * rho * dx * dy / (std1 * std2)
        )  # (B, T)

        reg_loss: Tensor = ((reg_gmm_log_coefficient + reg_gmm_exp) * weight).sum(
            dim=-1
        )

        # velocity smooth
        d1 = nearest_xy[:, 1:] - nearest_xy[:, :-1]  # (A, T-1, 2)
        d2 = d1[:, 1:] - d1[:, :-1]  # (A, T-2, 2)
        smooth = torch.norm(d2, dim=-1)
        smooth_loss = smooth.sum(dim=-1)

        return reg_loss + smooth_loss * 0.1
