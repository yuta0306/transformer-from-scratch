from typing import Optional

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_size: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super(FeedForward, self).__init__()

        if hidden_size is None:
            hidden_size = dim

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)

        return out
