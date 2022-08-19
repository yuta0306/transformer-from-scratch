import torch
import torch.nn as nn


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super(GEGLU, self).__init__()

        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, gate = torch.chunk(self.proj(x), chunks=2, dim=-1)
        return out * nn.GELU()(gate)
