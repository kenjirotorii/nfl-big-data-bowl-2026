import math

import torch
from jaxtyping import Float


def wrap_angle(
    angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi
) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)


def global_to_source(
    source_in_global: Float[torch.Tensor, "... 3"],
    target_in_global: Float[torch.Tensor, "... 3"],
) -> Float[torch.Tensor, "... 3"]:
    if source_in_global.shape != target_in_global.shape:
        raise ValueError(
            f"Size mismatch: {source_in_global.shape} != {target_in_global.shape}"
        )

    cos_yaw = torch.cos(source_in_global[..., 2])
    sin_yaw = torch.sin(source_in_global[..., 2])

    dx = target_in_global[..., 0] - source_in_global[..., 0]
    dy = target_in_global[..., 1] - source_in_global[..., 1]
    d_yaw = wrap_angle(target_in_global[..., 2] - source_in_global[..., 2])

    x = dx * cos_yaw + dy * sin_yaw
    y = -dx * sin_yaw + dy * cos_yaw
    return torch.stack([x, y, d_yaw], dim=-1)


def source_to_global(
    source_in_global: Float[torch.Tensor, "... 3"],
    target_in_source: Float[torch.Tensor, "... 3"],
) -> Float[torch.Tensor, "... 3"]:
    if source_in_global.shape != target_in_source.shape:
        raise ValueError(
            f"Size mismatch: {source_in_global.shape} != {target_in_source.shape}"
        )
    cos_yaw = torch.cos(source_in_global[..., 2])
    sin_yaw = torch.sin(source_in_global[..., 2])

    dx = target_in_source[..., 0]
    dy = target_in_source[..., 1]

    x = dx * cos_yaw - dy * sin_yaw + source_in_global[..., 0]
    y = dx * sin_yaw + dy * cos_yaw + source_in_global[..., 1]
    yaw = wrap_angle(target_in_source[..., 2] + source_in_global[..., 2])
    return torch.stack([x, y, yaw], dim=-1)


def angle_between_2d_vectors(
    reference_vector: Float[torch.Tensor, "... 2"],
    vector_from_reference: Float[torch.Tensor, "... 2"],
    eps: float = 1e-7,
) -> Float[torch.Tensor, "..."]:
    # Ensure the input tensors have the correct shape
    if reference_vector.shape[-1] != 2 or vector_from_reference.shape[-1] != 2:
        raise ValueError("Input tensors must have a shape of (..., 2)")

    det = (
        reference_vector[..., 0] * vector_from_reference[..., 1]
        - reference_vector[..., 1] * vector_from_reference[..., 0]
    )
    dot = (reference_vector * vector_from_reference).sum(dim=-1)

    angle = torch.atan2(det, dot)

    # Handle zero vectors by setting the angle to zero where either vector is near zero
    ref_norm = reference_vector.norm(dim=-1, keepdim=True)
    vec_norm = vector_from_reference.norm(dim=-1, keepdim=True)
    zero_mask = (ref_norm < eps) | (vec_norm < eps)
    angle = torch.where(
        zero_mask.squeeze(-1), torch.tensor(0.0, device=angle.device), angle
    )

    return angle
