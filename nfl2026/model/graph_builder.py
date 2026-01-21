import torch
from torch import nn
from torch_geometric.data import HeteroData

from ..data.dataclass import PlayerRole, Prompt
from ..tensor_utils import repeat_target_for_num_timesteps
from .edge_connector import (
    AgentToBallSpatialEdgeConnector,
    AgentToPolylineSpatialEdgeConnector,
    InterAgentSpatialEdgeConnector,
    InterAgentTemporalEdgeConnector,
    ModeToFeatureEdgeConnector,
)
from .flattener import Flattener


class Graph(HeteroData):
    @property
    def num_plays(self) -> list[int]:
        return self["num_plays_tensor"].tolist()

    @property
    def num_players(self) -> list[int]:
        return self["num_players_tensor"].tolist()

    @property
    def num_polylines(self) -> list[int]:
        return self["num_polylines_tensor"].tolist()

    @property
    def num_timesteps(self) -> int:
        return self["num_timesteps"][0].item()


class GraphBuilder(nn.Module):
    def __init__(
        self,
        temporal_edge_connector: InterAgentTemporalEdgeConnector,
        interaction_edge_connector: InterAgentSpatialEdgeConnector,
        map_edge_connector: AgentToPolylineSpatialEdgeConnector,
        ball_edge_connector: AgentToBallSpatialEdgeConnector,
    ) -> None:
        super().__init__()
        self._connect_edge_aTa = temporal_edge_connector
        self._connect_edge_aSa = interaction_edge_connector
        self._connect_edge_pSa = map_edge_connector
        self._connect_edge_bSa = ball_edge_connector

    def forward(self, prompt: Prompt) -> Graph:
        flattener = Flattener(prompt.num_timesteps)

        graph = Graph()
        player = graph["player"]
        player.num_nodes = sum(prompt.num_players) * prompt.num_timesteps
        player_attributes = (
            prompt.time_dependent_attributes + prompt.player_time_dependent_attributes
        )
        for attr, value in prompt["player"].items():
            if attr in player_attributes:
                player[attr] = flattener.flatten(value)

        qb = graph["qb"]
        qb.num_nodes = sum(prompt.num_plays) * prompt.num_timesteps
        for attr, value in prompt["qb"].items():
            if attr in prompt.time_dependent_attributes:
                qb[attr] = flattener.flatten(value)

        meta = graph["meta"]
        meta.num_nodes = sum(prompt.num_players)
        meta["num_frames_output"] = prompt["player"]["num_frames_output"]

        polyline = graph["polyline"]
        polyline.num_nodes = sum(prompt.num_polylines)
        for attr, value in prompt["polyline"].items():
            if attr not in ("num_nodes", "batch", "ptr"):
                polyline[attr] = value

        ball = graph["ball"]
        ball.num_nodes = sum(prompt.num_plays)
        ball["xyh"] = prompt["ball"]["xyh"]
        ball["feature"] = prompt["ball"]["feature"]

        graph["num_plays_tensor"] = prompt["num_plays_tensor"]
        graph["num_players_tensor"] = prompt["num_players_tensor"]
        graph["num_polylines_tensor"] = prompt["num_polylines_tensor"]
        graph["num_timesteps"] = torch.tensor(
            [prompt.num_timesteps],
            dtype=torch.int32,
            device=prompt["num_players_tensor"].device,
        )
        self._connect_edges(graph)
        return graph

    def _connect_edges(self, graph: Graph) -> None:
        flattener = Flattener(graph.num_timesteps)

        # temporal edge
        player_valid = graph["player"]["valid"]
        graph["player", "temporal", "player"].edge_index = self._connect_edge_aTa(
            player_valid, None, flattener
        )

        # interaction edge
        player_xy = graph["player"]["xyh"][:, :2]
        player_partition = flattener.partition(graph.num_players, player_xy.device)
        graph["player", "spatial", "player"].edge_index = self._connect_edge_aSa(
            player_xy, player_valid, player_partition
        )

        # polyline edge
        polyline_xy = repeat_target_for_num_timesteps(
            graph["polyline"]["xyh"][:, :2], graph.num_timesteps
        )
        polyline_xy = flattener.flatten(polyline_xy)
        polyline_partition = flattener.partition(
            graph.num_polylines, polyline_xy.device
        )
        edge_pSa = self._connect_edge_pSa(
            player_xy,
            player_valid,
            player_partition,
            polyline_xy,
            polyline_partition,
        )
        edge_pSa[0] = edge_pSa[0] % sum(graph.num_polylines)
        graph["polyline", "spatial", "player"].edge_index = edge_pSa

        # ball edge
        ball_land_xy = repeat_target_for_num_timesteps(
            graph["ball"]["xyh"][..., :2], graph.num_timesteps
        )
        ball_land_xy = flattener.flatten(ball_land_xy)
        ball_land_partition = flattener.partition(graph.num_plays, ball_land_xy.device)
        edge_bSa = self._connect_edge_bSa(
            player_xy,
            player_valid,
            player_partition,
            ball_land_xy,
            ball_land_partition,
        )
        # Reindex the edge indices because the polylines remain identical across all timesteps.
        edge_bSa[0] = edge_bSa[0] % sum(graph.num_plays)
        graph["ball", "spatial", "player"].edge_index = edge_bSa

        # qb edge
        qb_xy = graph["qb"]["xyh"][..., :2]
        qb_partition = flattener.partition(graph.num_plays, qb_xy.device)
        edge_qSa = self._connect_edge_bSa(
            player_xy,
            player_valid,
            player_partition,
            qb_xy,
            qb_partition,
        )
        # Reindex the edge indices because the polylines remain identical across all timesteps.
        edge_qSa[0] = edge_qSa[0] % sum(graph.num_plays)
        graph["qb", "spatial", "player"].edge_index = edge_qSa


class ModeGraph(HeteroData):
    @property
    def num_plays(self) -> list[int]:
        return self["num_plays_tensor"].tolist()

    @property
    def num_players(self) -> list[int]:
        return self["num_players_tensor"].tolist()

    @property
    def num_timesteps(self) -> int:
        return self["num_timesteps"][0].item()


class ModeGraphBuilder(nn.Module):
    def __init__(
        self,
        feature_edge_connector: ModeToFeatureEdgeConnector,
    ) -> None:
        super().__init__()
        self._connect_edge_aTm = feature_edge_connector

    def forward(self, prompt: Prompt) -> ModeGraph:
        temporal_flattener = Flattener(prompt.num_timesteps)

        graph = ModeGraph()
        player = graph["player"]
        player.num_nodes = sum(prompt.num_players) * prompt.num_timesteps
        player["valid"] = temporal_flattener.flatten(prompt["player"]["valid"])
        player["is_predicted"] = prompt["player"]["is_predicted"]
        player["is_receiver"] = (
            prompt["player"]["role"][:, 0] == PlayerRole.TargetedReceiver.value
        ).long()
        player["num_frames_output"] = prompt["player"]["num_frames_output"].long()

        meta = graph["meta"]
        meta.num_nodes = sum(prompt.num_players)
        meta["num_frames_output"] = prompt["player"]["num_frames_output"].long()

        graph["num_plays_tensor"] = prompt["num_plays_tensor"]
        graph["num_players_tensor"] = prompt["num_players_tensor"]
        graph["num_timesteps"] = torch.tensor(
            [prompt.num_timesteps],
            dtype=torch.int32,
            device=prompt["num_players_tensor"].device,
        )
        self._connect_edges(graph)
        return graph

    def _connect_edges(self, graph: ModeGraph) -> None:
        temporal_flattener = Flattener(graph.num_timesteps)

        player_valid = graph["player"]["valid"]
        mode_valid = graph["player"]["is_predicted"]
        graph["player", "to", "mode"].edge_index = self._connect_edge_aTm(
            player_valid, mode_valid, temporal_flattener
        )
