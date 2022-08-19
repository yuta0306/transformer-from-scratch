from typing import Dict, Optional

import einops
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

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        attention_score = torch.einsum("b i d, b j d -> b i j", query, key)

        if mask is not None:
            if mask.ndim == 2:
                mask = einops.repeat(mask, "b i -> b 1 i")
            big_neg = -torch.finfo(attention_score.dtype).max
            attention_score.masked_fill_(~mask.bool(), big_neg)

        attention_weights = torch.softmax(attention_score, dim=-1)

        out = (
            torch.einsum("b i j, b j d -> b i d", attention_weights, value) * self.scale
        )
        out = self.o_proj(out)

        return {"outputs": out, "attention_weights": attention_weights.detach()}


class MultiHeadSelfAttention(nn.Module):
    """
    Multi Head Self Attention
    """

    def __init__(
        self,
        n_heads: int,
        dim: int,
        hidden_size: Optional[int] = None,
        scale: bool = True,
    ) -> None:
        super(MultiHeadSelfAttention, self).__init__()

        if hidden_size is None:
            hidden_size = dim

        assert hidden_size % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.scale = 1.0
        if scale:
            self.scale = hidden_size ** (-0.5)

        self.q_proj = nn.Linear(dim, hidden_size, bias=False)
        self.k_proj = nn.Linear(dim, hidden_size, bias=False)
        self.v_proj = nn.Linear(dim, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        query = einops.rearrange(query, "b i (h d) -> b h i d", h=self.n_heads)
        key = einops.rearrange(key, "b i (h d) -> b h i d", h=self.n_heads)
        value = einops.rearrange(value, "b i (h d) -> b h i d", h=self.n_heads)

        attention_score = torch.einsum("b h i d, b h j d -> b h i j", query, key)

        if mask is not None:
            if mask.ndim == 2:
                mask = einops.repeat(mask, "b i -> b h 1 i", h=self.n_heads)
            elif mask.ndim == 3:
                mask = einops.repeat(mask, "b h i -> b h 1 i")
            big_neg = -torch.finfo(attention_score.dtype).max
            attention_score.masked_fill_(~mask.bool(), big_neg)

        attention_weights = torch.softmax(attention_score, dim=-1)

        out = torch.einsum("b h i j, b h j d -> b h i d", attention_weights, value)
        out = einops.rearrange(out, "b h i d -> b i (h d)") * self.scale

        out = self.o_proj(out)

        return {"outputs": out, "attention_weights": attention_weights.detach()}


class CrossAttention(nn.Module):
    """
    Cross Attention
    """

    def __init__(
        self,
        dim: int,
        context_dim: Optional[int] = None,
        hidden_size: Optional[int] = None,
        scale: bool = True,
    ) -> None:
        super(CrossAttention, self).__init__()

        self.scale = 1.0
        if scale:
            self.scale = dim ** (-0.5)

        if hidden_size is None:
            hidden_size = dim
        if context_dim is None:
            context_dim = dim

        self.q_proj = nn.Linear(dim, hidden_size, bias=False)
        self.k_proj = nn.Linear(context_dim, hidden_size, bias=False)
        self.v_proj = nn.Linear(context_dim, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        query = self.q_proj(x)
        key = self.k_proj(context)
        value = self.v_proj(context)

        attention_score = torch.einsum("b i d, b j d -> b i j", query, key)

        if mask is not None:
            if mask.ndim == 2:
                mask = einops.repeat(mask, "b i -> b i 1")
            big_neg = -torch.finfo(attention_score.dtype).max
            attention_score.masked_fill_(~mask.bool(), big_neg)

        attention_weights = torch.softmax(attention_score, dim=-1)

        out = (
            torch.einsum("b i j, b j d -> b i d", attention_weights, value) * self.scale
        )
        out = self.o_proj(out)

        return {"outputs": out, "attention_weights": attention_weights.detach()}


class MultiHeadCrossAttention(nn.Module):
    """
    Multi Head Cross Attention
    """

    def __init__(
        self,
        n_heads: int,
        dim: int,
        context_dim: Optional[int] = None,
        hidden_size: Optional[int] = None,
        scale: bool = True,
    ) -> None:
        super(MultiHeadCrossAttention, self).__init__()

        self.scale = 1.0
        if scale:
            self.scale = dim ** (-0.5)

        if hidden_size is None:
            hidden_size = dim
        if context_dim is None:
            context_dim = dim

        assert hidden_size % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.q_proj = nn.Linear(dim, hidden_size, bias=False)
        self.k_proj = nn.Linear(context_dim, hidden_size, bias=False)
        self.v_proj = nn.Linear(context_dim, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        query = self.q_proj(x)
        key = self.k_proj(context)
        value = self.v_proj(context)

        query = einops.rearrange(query, "b i (h d) -> b h i d", h=self.n_heads)
        key = einops.rearrange(key, "b i (h d) -> b h i d", h=self.n_heads)
        value = einops.rearrange(value, "b i (h d) -> b h i d", h=self.n_heads)

        attention_score = torch.einsum("b h i d, b h j d -> b h i j", query, key)

        if mask is not None:
            if mask.ndim == 2:
                mask = einops.repeat(mask, "b i -> b h i 1", h=self.n_heads)
            elif mask.ndim == 3:
                mask = einops.repeat(mask, "b h i -> b h i 1")
            big_neg = -torch.finfo(attention_score.dtype).max
            attention_score.masked_fill_(~mask.bool(), big_neg)

        attention_weights = torch.softmax(attention_score, dim=-1)

        out = torch.einsum("b h i j, b h j d -> b h i d", attention_weights, value)
        out = einops.rearrange(out, "b h i d -> b i (h d)") * self.scale

        out = self.o_proj(out)

        return {"outputs": out, "attention_weights": attention_weights.detach()}
