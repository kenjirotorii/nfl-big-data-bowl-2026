from typing import Any

import torch
import torch.nn as nn
from jaxtyping import Int
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

from ..model.weight_init import weight_init


class _Attention(MessagePassing):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        bipartite: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(aggr="add", node_dim=0, **kwargs)

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.to_q = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_k = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
        self.to_v = nn.Linear(hidden_dim, head_dim * num_heads)

        self.to_s = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_g = nn.Linear(head_dim * num_heads + hidden_dim, head_dim * num_heads)
        self.to_out = nn.Linear(head_dim * num_heads, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.ff_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        if bipartite:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = nn.LayerNorm(hidden_dim)
        else:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = self.attn_prenorm_x_src

        self.attn_postnorm = nn.LayerNorm(hidden_dim)
        self.ff_prenorm = nn.LayerNorm(hidden_dim)
        self.ff_postnorm = nn.LayerNorm(hidden_dim)
        self.apply(weight_init)

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        edge_index: Int[Tensor, "2(src,dst) num_edges"],
    ) -> Tensor:
        x_src = self.attn_prenorm_x_src(source)
        x_tgt = self.attn_prenorm_x_dst(target)

        x = target
        x = x + self.attn_postnorm(self._attn_block(x_src, x_tgt, edge_index))
        x = x + self.ff_postnorm(self._ff_block(self.ff_prenorm(target)))
        return x

    def message(
        self, q_i: Tensor, k_j: Tensor, v_j: Tensor, index: Tensor, ptr: Any
    ) -> Tensor:
        sim = (q_i * k_j).sum(dim=-1) * self.scale
        attn = softmax(sim, index, ptr)
        attn = self.attn_drop(attn)
        return v_j * attn.unsqueeze(-1)

    def update(self, inputs: Tensor, x_dst: Tensor) -> Tensor:
        inputs = inputs.view(-1, self.num_heads * self.head_dim)
        g = self.to_g(torch.cat([inputs, x_dst], dim=-1)).sigmoid_()
        return inputs + g * (self.to_s(x_dst) - inputs)

    def _attn_block(self, x_src: Tensor, x_dst: Tensor, edge_index: Tensor) -> Tensor:
        q = self.to_q(x_dst).view(-1, self.num_heads, self.head_dim)
        k = self.to_k(x_src).view(-1, self.num_heads, self.head_dim)
        v = self.to_v(x_src).view(-1, self.num_heads, self.head_dim)
        agg = self.propagate(edge_index=edge_index, x_dst=x_dst, q=q, k=k, v=v)
        return self.to_out(agg)

    def _ff_block(self, x: Tensor) -> Tensor:
        return self.ff_mlp(x)


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._attn = _Attention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            bipartite=False,
            **kwargs,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self._attn(x, x, edge_index)


class CrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._attn = _Attention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            bipartite=True,
            **kwargs,
        )

    def forward(self, source: Tensor, target: Tensor, edge_index: Tensor) -> Tensor:
        return self._attn(source, target, edge_index)
