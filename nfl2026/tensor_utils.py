from jaxtyping import Float
from torch import Tensor


def repeat_target_for_num_timesteps(
    target: Float[Tensor, "P1+P2+~+PN D"], num_timesteps: int
) -> Float[Tensor, "P1+P2+~+PN T D"]:
    if target.dim() == 1:
        repeated_tensor = target.unsqueeze(0).repeat(num_timesteps, 1)
    elif target.dim() == 2:
        repeated_tensor = target.unsqueeze(0).repeat(num_timesteps, 1, 1)
    else:
        raise ValueError("Tensor must be either 1D or 2D")
    p = repeated_tensor.transpose(0, 1)
    return p
