import torch
from jaxtyping import Float, Int
from torch import Tensor


class Flattener:
    def __init__(self, num_timesteps: int) -> None:
        self.num_timesteps = num_timesteps

    def flatten(
        self,
        tensor: Float[Tensor, "A1+A2+~+AN T"] | Float[Tensor, "A1+A2+~+AN T D"],
    ) -> Float[Tensor, "T*(A1+A2+~+AN)"] | Float[Tensor, "T*(A1+A2+~+AN) D"]:
        swapped = tensor.transpose(0, 1)
        return swapped.flatten(start_dim=0, end_dim=1).contiguous()

    def unflatten(
        self,
        tensor: Float[Tensor, "T*(A1+A2+~+AN)"] | Float[Tensor, "T*(A1+A2+~+AN) D"],
    ) -> Float[Tensor, "A1+A2+~+AN T"] | Float[Tensor, "A1+A2+~+AN T D"]:
        timefirst = tensor.unflatten(dim=0, sizes=(self.num_timesteps, -1))
        return timefirst.transpose(1, 0).contiguous()

    def partition(
        self, num_instances: list[int], device: torch.device
    ) -> Int[Tensor, "T*(A1+A2+~+AN)"]:
        batch_size = len(num_instances)
        partition_one_timestep = torch.cat(
            [
                torch.full((count,), batch_idx, dtype=torch.long)
                for batch_idx, count in enumerate(num_instances)
            ]
        )
        partition_indices = torch.cat(
            [
                partition_one_timestep + timestep * batch_size
                for timestep in range(self.num_timesteps)
            ]
        )
        return partition_indices.to(device)
