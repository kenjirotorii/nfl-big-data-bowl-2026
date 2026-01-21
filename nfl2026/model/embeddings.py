import math
from collections.abc import Callable

import torch
from jaxtyping import Float
from torch import Tensor, nn

from ..data.dataclass import EdgeArray
from ..geometry import wrap_angle
from .weight_init import weight_init


class FourierEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_freq_bands: int,
        aggregate: Callable[[list[Tensor]], Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.freqs = nn.Embedding(input_dim, num_freq_bands)
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(input_dim)
            ]
        )
        if aggregate is None:
            self.aggregate = lambda features: torch.stack(features, dim=-1).sum(-1)
        else:
            self.aggregate = aggregate

        self.apply(weight_init)

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi
        # Warning: if your data are noisy, don't use learnable sinusoidal embedding
        x = torch.cat([x.cos(), x.sin(), inputs.unsqueeze(-1)], dim=-1)
        embeddings = [self.mlps[i](x[:, i]) for i in range(self.input_dim)]
        out = self.aggregate(embeddings)
        return out


class MLPEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.apply(weight_init)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x)
        return x


class _RelativePositionEmbeddingBase(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_freq_bands: int) -> None:
        super().__init__()
        self.fourier_embedding = FourierEmbedding(input_dim, hidden_dim, num_freq_bands)
        self.apply(weight_init)

    def forward(
        self, *args: Tensor, **kwargs: Tensor
    ) -> Float[Tensor, "num_edges self.hidden_dim"]:
        with torch.no_grad():
            feature = self._compute_feature(*args, **kwargs)
        return self.fourier_embedding(feature)


class RelativePositionEmbedding(_RelativePositionEmbeddingBase):
    def __init__(self, hidden_dim: int, num_freq_bands: int) -> None:
        super().__init__(3, hidden_dim, num_freq_bands)

    @staticmethod
    def _compute_feature(
        source: Float[Tensor, "num_sources XYH"],
        target: Float[Tensor, "num_targets XYH"],
        edge: EdgeArray,
    ) -> Float[Tensor, "num_edges 3"]:
        source_position, source_heading = source[:, :2], source[:, 2]
        target_position, target_heading = target[:, :2], target[:, 2]

        position_delta = source_position[edge[0]] - target_position[edge[1]]
        dx = position_delta[:, 0]
        dy = position_delta[:, 1]
        dh = wrap_angle(source_heading[edge[0]] - target_heading[edge[1]])
        return torch.stack([dx, dy, dh], dim=-1)


class RelativeAgentPositionEmbedding(_RelativePositionEmbeddingBase):
    def __init__(self, hidden_dim: int, num_freq_bands: int) -> None:
        super().__init__(6, hidden_dim, num_freq_bands)

    @staticmethod
    def _compute_feature(
        source: Float[Tensor, "num_sources XYHSO"],
        target: Float[Tensor, "num_targets XYHSO"],
        edge: EdgeArray,
    ) -> Float[Tensor, "num_edges 6"]:
        source_position, source_heading, source_speed = (
            source[:, :2],
            source[:, 2],
            source[:, 3],
        )
        target_position, target_heading, target_speed = (
            target[:, :2],
            target[:, 2],
            target[:, 3],
        )

        position_delta = source_position[edge[0]] - target_position[edge[1]]
        dx = position_delta[:, 0]
        dy = position_delta[:, 1]
        distance = torch.norm(position_delta, p=2, dim=-1)
        is_close = (distance < 3.0).float()
        angle_to_source = torch.atan2(position_delta[:, 1], position_delta[:, 0])

        dh = wrap_angle(source_heading[edge[0]] - target_heading[edge[1]])

        target_velocity_x = target_speed * target_heading.cos()
        target_velocity_y = target_speed * target_heading.sin()

        target_closing_speed = (
            target_velocity_x[edge[1]] * angle_to_source.cos()
            + target_velocity_y[edge[1]] * angle_to_source.sin()
        )

        source_velocity_x = source_speed * source_heading.cos()
        source_velocity_y = source_speed * source_heading.sin()

        source_closing_speed = (
            source_velocity_x[edge[0]] * angle_to_source.cos()
            + source_velocity_y[edge[0]] * angle_to_source.sin()
        )

        return torch.stack(
            [dx, dy, dh, is_close, target_closing_speed, source_closing_speed], dim=-1
        )


class RelativeContextPositionEmbedding(_RelativePositionEmbeddingBase):
    def __init__(self, hidden_dim: int, num_freq_bands: int) -> None:
        super().__init__(4, hidden_dim, num_freq_bands)

    @staticmethod
    def _compute_feature(
        source: Float[Tensor, "num_sources XYH"],
        target: Float[Tensor, "num_targets XYHSO"],
        edge: EdgeArray,
    ) -> Float[Tensor, "num_edges 4"]:
        source_position, source_heading = source[:, :2], source[:, 2]
        target_position, target_heading, target_speed = (
            target[:, :2],
            target[:, 2],
            target[:, 3],
        )

        position_delta = source_position[edge[0]] - target_position[edge[1]]
        dx = position_delta[:, 0]
        dy = position_delta[:, 1]
        dh = wrap_angle(source_heading[edge[0]] - target_heading[edge[1]])

        angle_to_source = torch.atan2(position_delta[:, 1], position_delta[:, 0])

        target_velocity_x = target_speed * target_heading.cos()
        target_velocity_y = target_speed * target_heading.sin()

        closing_speed = (
            target_velocity_x[edge[1]] * angle_to_source.cos()
            + target_velocity_y[edge[1]] * angle_to_source.sin()
        )

        return torch.stack([dx, dy, dh, closing_speed], dim=-1)


class RelativeQBPositionEmbedding(_RelativePositionEmbeddingBase):
    def __init__(self, hidden_dim: int, num_freq_bands: int) -> None:
        super().__init__(4, hidden_dim, num_freq_bands)

    @staticmethod
    def _compute_feature(
        source: Float[Tensor, "num_sources XYH"],
        target: Float[Tensor, "num_targets XYHSO"],
        edge: EdgeArray,
    ) -> Float[Tensor, "num_edges 4"]:
        source_position, source_heading = source[:, :2], source[:, 2]
        target_position, target_heading = target[:, :2], target[:, 2]
        target_orient = target[:, 4]

        position_delta = source_position[edge[0]] - target_position[edge[1]]
        dx = position_delta[:, 0]
        dy = position_delta[:, 1]
        dh = wrap_angle(source_heading[edge[0]] - target_heading[edge[1]])

        angle_to_source = torch.atan2(position_delta[:, 1], position_delta[:, 0])

        diff_angle = wrap_angle(target_orient[edge[1]] - angle_to_source).abs()

        return torch.stack([dx, dy, dh, diff_angle], dim=-1)


class RelativeSpatiotemporalEmbedding(_RelativePositionEmbeddingBase):
    def __init__(self, hidden_dim: int, num_freq_bands: int) -> None:
        super().__init__(4, hidden_dim, num_freq_bands)

    @staticmethod
    def _compute_feature(
        source: Float[Tensor, "num_sources XYH"],
        target: Float[Tensor, "num_targets XYH"],
        edge: EdgeArray,
        num_agents: list[int],
    ) -> Float[Tensor, "num_edges self.input_dim"]:
        spatial = RelativePositionEmbedding._compute_feature(source, target, edge)

        # Compute the temporal differences
        # TODO Create a time-index tensor corresponding to the feature, and then slice it with the edge(General Approach)
        temporal_index = edge // sum(num_agents)
        temporal = (temporal_index[0] - temporal_index[1]).float().unsqueeze(-1)
        return torch.cat((spatial, temporal), dim=-1)
