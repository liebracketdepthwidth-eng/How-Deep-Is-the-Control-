"""
Evaluation metrics and spectral analysis helpers.
"""

import torch

def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)

def compute_spectral_radius(matrix: torch.Tensor) -> torch.Tensor:
    eigenvalues = torch.linalg.eigvals(matrix)
    return torch.max(torch.abs(eigenvalues))
