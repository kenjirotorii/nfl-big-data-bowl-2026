from copy import deepcopy

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from torch_geometric.data.storage import EdgeStorage

from ..data.dataclass import EdgeArray, PlayerPosition, PlayerRole
from .attentions import RPECrossAttention, RPESelfAttention
from .embeddings import (
    MLPEmbedding,
    RelativeAgentPositionEmbedding,
    RelativeContextPositionEmbedding,
    RelativeQBPositionEmbedding,
    RelativeSpatiotemporalEmbedding,
)
from .flattener import Flattener
from .graph_builder import Graph
from .weight_init import weight_init


class MLPLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TrajectoryEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # players
        self.xyh_embedding = MLPEmbedding(input_dim=3, hidden_dim=hidden_dim)
        self.speed_embedding = MLPEmbedding(input_dim=4, hidden_dim=hidden_dim)
        self.angle_embedding = MLPEmbedding(input_dim=6, hidden_dim=hidden_dim)
        self.kinematic_embedding = MLPEmbedding(input_dim=4, hidden_dim=hidden_dim)

        num_roles = len(PlayerRole)
        self.role_embedding = nn.Embedding(num_roles, hidden_dim)

        num_positions = len(PlayerPosition)
        self.position_embedding = nn.Embedding(num_positions, hidden_dim)

        self.predict_embedding = nn.Embedding(2, hidden_dim)

        num_fusions = 7
        self.fusion_embedding = MLPEmbedding(
            input_dim=hidden_dim * num_fusions, hidden_dim=hidden_dim
        )

        # qb
        self.qb_xyh_embedding = MLPEmbedding(input_dim=3, hidden_dim=hidden_dim)
        self.qb_speed_embedding = MLPEmbedding(input_dim=4, hidden_dim=hidden_dim)
        self.qb_angle_embedding = MLPEmbedding(input_dim=6, hidden_dim=hidden_dim)
        self.qb_kinematic_embedding = MLPEmbedding(input_dim=4, hidden_dim=hidden_dim)

        num_fusions = 4
        self.qb_fusion_embedding = MLPEmbedding(
            input_dim=hidden_dim * num_fusions, hidden_dim=hidden_dim
        )

        # polylines
        num_polylines = 7
        self.polyline_embedding = nn.Embedding(num_polylines, hidden_dim)

        # ball
        self.ball_embedding = MLPEmbedding(input_dim=3, hidden_dim=hidden_dim)

        # relative positional embedding
        self.embedding_aTa = RelativeSpatiotemporalEmbedding(
            hidden_dim=hidden_dim, num_freq_bands=num_freq_bands
        )
        self.embedding_pSa = RelativeContextPositionEmbedding(
            hidden_dim=hidden_dim, num_freq_bands=num_freq_bands
        )
        self.embedding_aSa = RelativeAgentPositionEmbedding(
            hidden_dim=hidden_dim, num_freq_bands=num_freq_bands
        )
        self.embedding_bSa = RelativeContextPositionEmbedding(
            hidden_dim=hidden_dim, num_freq_bands=num_freq_bands
        )
        self.embedding_qSa = RelativeQBPositionEmbedding(
            hidden_dim=hidden_dim, num_freq_bands=num_freq_bands
        )

        # attentions
        self_attn = RPESelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        cross_attn = RPECrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.attn_aTa = nn.ModuleList([deepcopy(self_attn) for _ in range(num_layers)])
        self.attn_pSa = nn.ModuleList([deepcopy(cross_attn) for _ in range(num_layers)])
        self.attn_bSa = nn.ModuleList([deepcopy(cross_attn) for _ in range(num_layers)])
        self.attn_qSa = nn.ModuleList([deepcopy(cross_attn) for _ in range(num_layers)])
        self.attn_aSa = nn.ModuleList([deepcopy(self_attn) for _ in range(num_layers)])

        self.apply(weight_init)

    def forward(self, graph: Graph) -> Float[Tensor, "A1+A2+~+AN T hidden_dim"]:
        player, polyline, ball, qb = (
            graph["player"],
            graph["polyline"],
            graph["ball"],
            graph["qb"],
        )
        a = self.player_embedding(
            player["dxdydh"],
            player["speeds"],
            player["angles"],
            player["kinematics"],
            player["role"],
            player["position"],
            player["predict"],
        )
        q = self.qb_embedding(
            qb["dxdydh"], qb["speeds"], qb["angles"], qb["kinematics"]
        )
        p = self.polyline_embedding(polyline["token_id"])
        b = self.ball_embedding(ball["feature"])
        aTa, pSa, aSa, bSa, qSa = self.positional_embedding(
            player["xyh"],
            polyline["xyh"],
            ball["xyh"],
            qb["xyh"],
            graph["player", "temporal", "player"].edge_index,
            graph["polyline", "spatial", "player"].edge_index,
            graph["player", "spatial", "player"].edge_index,
            graph["ball", "spatial", "player"].edge_index,
            graph["qb", "spatial", "player"].edge_index,
            graph.num_players,
        )
        a = self.attention(a, p, b, q, aTa, pSa, aSa, bSa, qSa)
        a = Flattener(graph.num_timesteps).unflatten(a)
        return a

    def player_embedding(
        self,
        xyh: Float[Tensor, "T*(A1+A2+~+AN) XYH"],
        speeds: Float[Tensor, "T*(A1+A2+~+AN) 3"],
        angles: Float[Tensor, "T*(A1+A2+~+AN) 6"],
        kinematics: Float[Tensor, "T*(A1+A2+~+AN) 4"],
        role: Int[Tensor, "T*(A1+A2+~+AN)"],
        position: Int[Tensor, "T*(A1+A2+~+AN)"],
        predict: Int[Tensor, "T*(A1+A2+~+AN)"],
    ) -> Float[Tensor, "T*(A1+A2+~+AN) hidden_dim"]:
        xyh_embedding = self.xyh_embedding(xyh)
        speed_embedding = self.speed_embedding(speeds)
        angle_embedding = self.angle_embedding(angles)
        kinematic_embedding = self.kinematic_embedding(kinematics)
        role_embedding = self.role_embedding(role)
        position_embedding = self.position_embedding(position)
        predict_embedding = self.predict_embedding(predict)

        a = torch.cat(
            (
                xyh_embedding,
                speed_embedding,
                angle_embedding,
                kinematic_embedding,
                role_embedding,
                position_embedding,
                predict_embedding,
            ),
            dim=-1,
        )
        a = self.fusion_embedding(a)
        return a

    def qb_embedding(
        self,
        xyh: Float[Tensor, "T*(A1+A2+~+AN) XYH"],
        speeds: Float[Tensor, "T*(A1+A2+~+AN) 3"],
        angles: Float[Tensor, "T*(A1+A2+~+AN) 6"],
        kinematics: Float[Tensor, "T*(A1+A2+~+AN) 4"],
    ) -> Float[Tensor, "T*(A1+A2+~+AN) hidden_dim"]:
        xyh_embedding = self.qb_xyh_embedding(xyh)
        speed_embedding = self.qb_speed_embedding(speeds)
        angle_embedding = self.qb_angle_embedding(angles)
        kinematic_embedding = self.qb_kinematic_embedding(kinematics)

        a = torch.cat(
            (xyh_embedding, speed_embedding, angle_embedding, kinematic_embedding),
            dim=-1,
        )
        a = self.qb_fusion_embedding(a)
        return a

    def positional_embedding(
        self,
        player_pose: Float[Tensor, "T*(A1+A2+~+AN) XYH"],
        polyline_pose: Float[Tensor, "P1+P2+~+PN XYH"],
        ball_pose: Float[Tensor, "B XYH"],
        qb_pose: Float[Tensor, "B XYH"],
        index_aTa: EdgeArray,
        index_pSa: EdgeArray,
        index_aSa: EdgeArray,
        index_bSa: EdgeArray,
        index_qSa: EdgeArray,
        num_players: list[int],
    ) -> tuple[EdgeStorage, EdgeStorage, EdgeStorage, EdgeStorage, EdgeStorage]:
        aTa = EdgeStorage(
            edge_index=index_aTa,
            edge_attr=self.embedding_aTa(
                player_pose, player_pose, index_aTa, num_players
            ),
        )
        pSa = EdgeStorage(
            edge_index=index_pSa,
            edge_attr=self.embedding_pSa(polyline_pose, player_pose, index_pSa),
        )
        aSa = EdgeStorage(
            edge_index=index_aSa,
            edge_attr=self.embedding_aSa(player_pose, player_pose, index_aSa),
        )
        bSa = EdgeStorage(
            edge_index=index_bSa,
            edge_attr=self.embedding_bSa(ball_pose, player_pose, index_bSa),
        )
        qSa = EdgeStorage(
            edge_index=index_qSa,
            edge_attr=self.embedding_qSa(qb_pose, player_pose, index_qSa),
        )
        return aTa, pSa, aSa, bSa, qSa

    def attention(
        self,
        a: Float[Tensor, "T*(A1+A2+~+AN) hidden_dim"],
        p: Float[Tensor, "P1+P2+~+PN hidden_dim"],
        b: Float[Tensor, "B hidden_dim"],
        q: Float[Tensor, "B hidden_dim"],
        aTa: EdgeStorage,
        pSa: EdgeStorage,
        aSa: EdgeStorage,
        bSa: EdgeStorage,
        qSa: EdgeStorage,
    ) -> Float[Tensor, "T*(A1+A2+~+AN) hidden_dim"]:
        for i in range(self.num_layers):
            a = self.attn_aTa[i](a, aTa.edge_attr, aTa.edge_index)
            a = self.attn_pSa[i](p, a, pSa.edge_attr, pSa.edge_index)
            a = self.attn_qSa[i](q, a, qSa.edge_attr, qSa.edge_index)
            a = self.attn_aSa[i](a, aSa.edge_attr, aSa.edge_index)
            a = self.attn_bSa[i](b, a, bSa.edge_attr, bSa.edge_index)
        return a
