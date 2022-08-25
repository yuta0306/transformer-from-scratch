import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    pe: torch.Tensor

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_length: int = 5000,
    ) -> None:
        super(PositionalEncoding, self).__init__()

        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.pe[: x.size(0)]
        out = self.dropout(out)

        return out
