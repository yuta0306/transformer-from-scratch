import copy
from typing import Optional

import torch
import torch.nn as nn

from transformer.attention import MultiHeadSelfAttention
from transformer.ffn import FeedForward


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadSelfAttention(n_heads=n_heads, dim=d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=eps)
        self.dropout1 = nn.Dropout(p=dropout)

        self.ffn = FeedForward(d_model, dim_ff, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=eps)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = x + self.dropout1(self.self_attn(x, mask)["outputs"])
        out = self.norm1(out)

        out = out + self.dropout2(self.ffn(out))
        out = self.norm2(out)

        return out


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
    ) -> None:
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = x

        for layer in self.layers:
            out = layer(out, mask)

        return out
