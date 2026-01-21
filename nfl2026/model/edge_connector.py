import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch_geometric.utils import dense_to_sparse

from ..data.dataclass import EdgeArray
from .edge_selector import EdgeSelector
from .flattener import Flattener


class InterAgentTemporalEdgeConnector(nn.Module):
    def __init__(
        self,
        max_historical_aggregate_steps: int = 10,
        hist_drop_prob: float = 0.0,
        is_casual: bool = False,
    ):
        super().__init__()
        self.max_historical_aggregate_steps = max_historical_aggregate_steps
        self.hist_drop_prob = hist_drop_prob
        self.is_casual = is_casual

    def forward(
        self,
        source_valid: Bool[Tensor, "T*(A1+A2+~+AN)"],  # noqa: F821
        target_valid: Bool[Tensor, "T*(A1+A2+~+AN)"] | None,  # noqa: F821
        flattener: Flattener,
    ) -> EdgeArray:
        source_AT = flattener.unflatten(source_valid)

        if target_valid is not None:
            target_AT = flattener.unflatten(target_valid)
        else:
            target_AT = source_AT.clone()

        if self.training:
            source_AT = self._probabilistic_drop(source_AT)
        indices_AT = self._forward(source_AT, target_AT)

        A, T = source_AT.shape
        indices_TA = self._convert_to_TA(indices_AT, A, T)

        agent_indices = indices_TA % A
        if (agent_indices[0] != agent_indices[1]).any():
            raise ValueError(
                "Edge tensor contains edges connecting nodes from different agents."
            )

        return indices_TA

    def _forward(
        self,
        source_valid: Bool[Tensor, "A1+A2+~+AN T"],  # noqa: F821
        target_valid: Bool[Tensor, "A1+A2+~+AN T"],  # noqa: F821
    ) -> EdgeArray:
        # Compute pair validity for each column.
        # pair_valid[i, j, k] is True if and only if agent k at time i and j are valid
        source: Bool[Tensor, "A1+A2+~+AN T 1"] = source_valid.unsqueeze(2)
        target: Bool[Tensor, "A1+A2+~+AN 1 T"] = target_valid.unsqueeze(1)
        pair_valid: Bool[Tensor, "A1+A2+~+AN T T"] = source & target

        index = dense_to_sparse(pair_valid)[0]
        # Select only causal pairs (where the second time index is after the first)
        if self.is_casual:
            index = index[:, index[1] > index[0]]
        # Limit by max historical aggregation steps
        index = index[:, index[1] - index[0] <= self.max_historical_aggregate_steps]
        return index

    def _probabilistic_drop(
        self, source_valid: Bool[Tensor, "A1+A2+~+AN T"]
    ) -> Bool[Tensor, "A1+A2+~+AN T"]:
        ones = torch.ones_like(
            source_valid, dtype=torch.long, device=source_valid.device
        )
        mask_prob = (1 - self.hist_drop_prob) * ones
        mask_keep = torch.bernoulli(mask_prob).bool()
        return source_valid & mask_keep

    @staticmethod
    def _convert_to_TA(
        edge_index: EdgeArray, num_agents: int, num_timesteps: int
    ) -> EdgeArray:
        a_indices = edge_index // num_timesteps  # Get agent indices
        t_indices = edge_index % num_timesteps  # Get timestep indices
        return t_indices * num_agents + a_indices  # Convert to T*A ordering


class InterAgentSpatialEdgeConnector(nn.Module):
    def __init__(self, edge_selector: EdgeSelector) -> None:
        super().__init__()
        self.edge_selector = edge_selector

    def forward(
        self,
        agent_position: Float[Tensor, "T*(A1+A2+~+AN) XY"],
        agent_valid: Bool[Tensor, "T*(A1+A2+~+AN)"],
        agent_partition: Int[Tensor, "T*(A1+A2+~+AN)"],
    ) -> EdgeArray:
        return self.edge_selector(
            target_position=agent_position,
            target_partition=agent_partition,
            target_valid=agent_valid,
            source_position=agent_position,
            source_partition=agent_partition,
            source_valid=agent_valid,
        )


class AgentToPolylineSpatialEdgeConnector(nn.Module):
    def __init__(self, edge_selector: EdgeSelector) -> None:
        super().__init__()
        self.edge_selector = edge_selector

    def forward(
        self,
        agent_position: Float[Tensor, "T*(A1+A2+~+AN) XY"],
        agent_valid: Bool[Tensor, "T*(A1+A2+~+AN)"],
        agent_partition: Int[Tensor, "T*(A1+A2+~+AN)"],
        polyline_start_position: Float[Tensor, "T*(P1+P2+~+PN) XY"],
        polyline_partition: Int[Tensor, "T*(P1+P2+~+PN)"],
    ) -> EdgeArray:
        edge_index_a2pl = self.edge_selector(
            target_position=agent_position,
            target_partition=agent_partition,
            target_valid=agent_valid,
            source_position=polyline_start_position,
            source_partition=polyline_partition,
            source_valid=None,
        )
        return edge_index_a2pl.flip(0)


class AgentToBallSpatialEdgeConnector(nn.Module):
    def __init__(self, edge_selector: EdgeSelector) -> None:
        super().__init__()
        self.edge_selector = edge_selector

    def forward(
        self,
        agent_position: Float[Tensor, "T*(A1+A2+~+AN) XY"],
        agent_valid: Bool[Tensor, "T*(A1+A2+~+AN)"],
        agent_partition: Int[Tensor, "T*(A1+A2+~+AN)"],
        ball_land_position: Float[Tensor, "T*B XY"],
        ball_land_partition: Int[Tensor, "T*B"],
    ) -> EdgeArray:
        edge_index_a2pl = self.edge_selector(
            target_position=agent_position,
            target_partition=agent_partition,
            target_valid=agent_valid,
            source_position=ball_land_position,
            source_partition=ball_land_partition,
            source_valid=None,
        )
        return edge_index_a2pl.flip(0)


class ModeToFeatureEdgeConnector(nn.Module):
    def __init__(
        self,
        max_historical_aggregate_steps: int = 10,
        hist_drop_prob: float = 0.0,
    ):
        super().__init__()
        self.max_historical_aggregate_steps = max_historical_aggregate_steps
        self.hist_drop_prob = hist_drop_prob

    def forward(
        self,
        source_valid: Bool[Tensor, "T*(A1+A2+~+AN)"],  # noqa: F821
        target_valid: Bool[Tensor, "(A1+A2+~+AN)"],  # noqa: F821
        temporal_flattener: Flattener,
    ) -> EdgeArray:
        source_AT = temporal_flattener.unflatten(source_valid)
        target_AM = target_valid.unsqueeze(-1)

        if self.training:
            source_AT = self._probabilistic_drop(source_AT)
        indices_TMA = self._forward(source_AT, target_AM)

        A, T = source_AT.shape
        _, M = target_AM.shape
        indices_TMA = self._convert_to_TMA(indices_TMA, A, T, M)

        agent_indices = indices_TMA % A
        if (agent_indices[0] != agent_indices[1]).any():
            raise ValueError(
                "Edge tensor contains edges connecting nodes from different agents."
            )

        return indices_TMA

    def _forward(
        self,
        source_valid: Bool[Tensor, "A1+A2+~+AN T"],  # noqa: F821
        target_valid: Bool[Tensor, "A1+A2+~+AN M"],  # noqa: F821
    ) -> EdgeArray:
        # Compute pair validity for each column.
        # pair_valid[i, j, k] is True if and only if agent k at time i and j are valid
        source: Bool[Tensor, "A1+A2+~+AN T 1"] = source_valid.unsqueeze(2)
        target: Bool[Tensor, "A1+A2+~+AN 1 M"] = target_valid.unsqueeze(1)
        pair_valid: Bool[Tensor, "A1+A2+~+AN T M"] = source & target

        index = dense_to_sparse(pair_valid)[0]
        # Limit by max historical aggregation steps
        index = index[:, index[1] - index[0] <= self.max_historical_aggregate_steps]
        return index

    def _probabilistic_drop(
        self, source_valid: Bool[Tensor, "A1+A2+~+AN T"]
    ) -> Bool[Tensor, "A1+A2+~+AN T"]:
        ones = torch.ones_like(
            source_valid, dtype=torch.long, device=source_valid.device
        )
        mask_prob = (1 - self.hist_drop_prob) * ones
        mask_keep = torch.bernoulli(mask_prob).bool()
        return source_valid & mask_keep

    @staticmethod
    def _convert_to_TMA(
        edge_index: EdgeArray, num_agents: int, num_timesteps: int, num_modes: int
    ) -> EdgeArray:
        a_indices = edge_index[0] // num_timesteps  # Get agent indices
        t_indices = edge_index[0] % num_timesteps  # Get timestep indices
        edge_index_0 = t_indices * num_agents + a_indices  # Convert to T*A ordering

        a_indices = edge_index[1] // num_modes  # Get agent indices
        m_indices = edge_index[1] % num_modes  # Get timestep indices
        edge_index_1 = m_indices * num_agents + a_indices  # Convert to M*A ordering
        return torch.stack([edge_index_0, edge_index_1], dim=0)
