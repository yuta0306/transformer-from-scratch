import copy
from typing import Optional

import torch
import torch.nn as nn

from transformer.attention import MultiHeadCrossAttention, MultiHeadSelfAttention
from transformer.ffn import FeedForward


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadSelfAttention(n_heads=n_heads, dim=d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=eps)

        self.cross_attn = MultiHeadCrossAttention(n_heads=n_heads, dim=d_model)
        self.dropout2 = nn.Dropout(p=dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, dim_ff, dropout=dropout)
        self.dropout3 = nn.Dropout(p=dropout)
        self.norm3 = nn.LayerNorm(d_model, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        out = x + self.dropout1(self.self_attn(x, target_mask)["outputs"])
        out = self.norm1(out)

        out = out + self.dropout2(
            self.cross_attn(out, context, context_mask)["outputs"]
        )
        out = self.norm2(out)

        out = out + self.dropout3(self.ffn(out))
        out = self.norm3(out)

        return out


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
    ) -> None:
        super(TransformerDecoder, self).__init__()

        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = x

        for layer in self.layers:
            out = layer(out, context, target_mask, context_mask)

        return out
