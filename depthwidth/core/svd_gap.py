"""
Singular Value Gap (SVD-gap) correction and spectral analysis utilities.
"""

import torch
import torch.nn as nn

def compute_svd_gap(weight_matrix: torch.Tensor, k: int = 1) -> torch.Tensor:
    """
    Computes the spectral gap between the k-th and (k+1)-th singular values.
    """
    u, s, vh = torch.linalg.svd(weight_matrix, full_matrices=False)
    if s.shape[0] <= k:
        return torch.tensor(0.0, device=weight_matrix.device)
    gap = s[k - 1] - s[k]
    return gap

def correct_svd_gap(
    layer: nn.Linear, 
    target_gap_ratio: float = 0.1, 
    max_iters: int = 50
) -> nn.Linear:
    """
    Regularizes and adjusts weight matrices to rectify artificial SVD-gaps 
    identified in Phase 3A depth recovery.
    """
    with torch.no_grad():
        w = layer.weight.data
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        
        # Spectral clipping / smoothing
        median_s = torch.median(s)
        s_corrected = torch.clamp(s, min=median_s * target_gap_ratio)
        
        layer.weight.data = torch.matmul(u * s_corrected, vh)
    return layer
