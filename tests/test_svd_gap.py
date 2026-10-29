import torch
import torch.nn as nn
from depthwidth.core.svd_gap import compute_svd_gap, correct_svd_gap

def test_compute_svd_gap():
    # Construct a matrix with a known singular value gap
    u = torch.eye(10)
    s = torch.tensor([10.0, 8.0, 2.0, 1.5, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1])
    vh = torch.eye(10)
    w = torch.matmul(u * s, vh)
    
    gap_1 = compute_svd_gap(w, k=1) # s[0] - s[1] = 10.0 - 8.0 = 2.0
    assert torch.allclose(gap_1, torch.tensor(2.0), atol=1e-4)

def test_correct_svd_gap():
    layer = nn.Linear(10, 10)
    corrected_layer = correct_svd_gap(layer, target_gap_ratio=0.05)
    
    assert isinstance(corrected_layer, nn.Linear)
    assert corrected_layer.weight.shape == (10, 10)
