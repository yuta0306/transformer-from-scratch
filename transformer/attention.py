from typing import Optional

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    Self Attention
    """

    def __init__(
        self,
        dim: int,
        hidden_size: Optional[int] = None,
        scale: bool = True,
    ) -> None:
        super(SelfAttention, self).__init__()

        self.scale = 1.0
        if scale:
            self.scale = dim ** (-0.5)

        if hidden_size is None:
            hidden_size = dim

        self.q_proj = nn.Linear(dim, hidden_size, bias=False)
        self.k_proj = nn.Linear(dim, hidden_size, bias=False)
        self.v_proj = nn.Linear(dim, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        attention_score = torch.einsum("b i d, b j d -> b i j", query, key)
        attention_weights = torch.softmax(attention_score, dim=-1)

        out = (
            torch.einsum("b i j, b j d -> b i d", attention_weights, value) * self.scale
        )
        out = self.o_proj(out)

        return out


class MultiHeadSelfAttention(nn.Module):
    """
    Multi Head Self Attention
    """

    def __init__(self) -> None:
        super(MultiHeadSelfAttention, self).__init__()


class CrossAttention(nn.Module):
    pass


class MultiHeadCrossAttention(nn.Module):
    pass
