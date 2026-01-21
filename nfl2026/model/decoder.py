from copy import deepcopy

import torch
from jaxtyping import Float
from torch import Tensor, nn

from .attention import CrossAttention
from .embeddings import MLPEmbedding
from .encoder import MLPLayer
from .flattener import Flattener
from .graph_builder import ModeGraph
from .weight_init import weight_init


class TrajectoryDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        num_output_frames: int,
        step_size: int = 10,
    ) -> None:
        super().__init__()
        self.num_output_frames = num_output_frames
        self.step_size = step_size
        self.num_head_iter = num_output_frames // self.step_size

        self.time_compressed_ratio = 2
        self.time_mode_embedding = nn.Embedding(
            num_output_frames // self.time_compressed_ratio, hidden_dim
        )
        self.role_mode_embedding = nn.Embedding(2, hidden_dim)
        self.mode_embedding = MLPEmbedding(hidden_dim * 2, hidden_dim)

        cross_attn = CrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.attn_aTm = nn.ModuleList(
            [deepcopy(cross_attn) for _ in range(self.num_head_iter)]
        )

        self.num_output_dim = 5
        emb_to_traj = MLPLayer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=self.step_size * self.num_output_dim,
        )
        traj_to_emb = MLPLayer(
            input_dim=self.step_size * self.num_output_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
        )
        self.emb_to_traj = nn.ModuleList(
            [deepcopy(emb_to_traj) for _ in range(self.num_head_iter)]
        )
        self.traj_to_emb = nn.ModuleList(
            [deepcopy(traj_to_emb) for _ in range(self.num_head_iter)]
        )
        self.apply(weight_init)

    def forward(
        self, embedding: Float[Tensor, "A1+A2+~+AN T hidden_dim"], graph: ModeGraph
    ) -> Float[Tensor, "A1+A2+~+AN num_output_frames D=5"]:
        A, T = embedding.shape[:2]
        embedding = Flattener(T).flatten(embedding)
        tm = self.time_mode_embedding(
            graph["player"]["num_frames_output"] // self.time_compressed_ratio
        )
        rm = self.role_mode_embedding(graph["player"]["is_receiver"])
        m = torch.cat([tm, rm], dim=-1)
        m = self.mode_embedding(m)

        trajs = []
        for i in range(self.num_head_iter):
            # attention
            m = self.attn_aTm[i](embedding, m, graph["player", "to", "mode"].edge_index)
            m = self.emb_to_traj[i](m)  # (A, s*5)
            trajs.append(m.view(A, self.step_size, self.num_output_dim))
            m = self.traj_to_emb[i](m)  # (A, D)

        pred_traj = torch.cat(trajs, dim=-2)

        return pred_traj
