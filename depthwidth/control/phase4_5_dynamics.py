"""
Phase 4 & 5 asymptotic analysis and control bound computations.
"""

import torch

def compute_asymptotic_bounds(depth: int, width: int, epsilon: float = 1e-3) -> torch.Tensor:
    """
    Computes theoretical control capacity bounds as a function of network depth and width.
    """
    bound = float(depth) / (float(width) + epsilon)
    return torch.tensor(bound)
