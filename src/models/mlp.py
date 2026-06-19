"""Small MLP models for toy diffusion and flow experiments."""

from __future__ import annotations

import torch
from torch import nn


class TimeMLP(nn.Module):
    """A small MLP that takes [x, t] and outputs a vector field/score."""

    def __init__(self, dim: int = 2, hidden: int = 128, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = dim + 1
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat([x, t], dim=1))
