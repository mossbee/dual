"""
Uncertainty-based loss weighting (Kendall et al. 2018).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import torch
from torch import nn


class UncertaintyWeighting(nn.Module):
    """
    Maintains learnable scalars w_i to balance multiple losses:
        L = 0.5 * sum(exp(-w_i) * L_i + w_i)
    """

    def __init__(self, terms: Sequence[str]):
        super().__init__()
        self.terms = list(terms)
        self.log_vars = nn.Parameter(torch.zeros(len(self.terms)))

    def forward(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if set(loss_dict.keys()) != set(self.terms):
            missing = set(self.terms) - set(loss_dict.keys())
            extra = set(loss_dict.keys()) - set(self.terms)
            raise KeyError(f"Loss terms mismatch. Missing: {missing}, extra: {extra}")
        total = 0.0
        for idx, name in enumerate(self.terms):
            coeff = torch.exp(-self.log_vars[idx])
            total = total + 0.5 * (coeff * loss_dict[name] + self.log_vars[idx])
        return total


