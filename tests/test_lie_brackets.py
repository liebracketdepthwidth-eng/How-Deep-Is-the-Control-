import torch
import pytest
from depthwidth.core.lie_algebra import lie_bracket, compute_vector_field_jacobian

def test_vector_field_jacobian():
    # Simple linear vector field f(x) = A x where A = [[2, 1], [0, 3]]
    A = torch.tensor([[2.0, 1.0], [0.0, 3.0]])
    f = lambda x: torch.matmul(x, A.T)
    
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    jac = compute_vector_field_jacobian(f, x)
    
    # For linear vector field, Jacobian should equal A for all batch elements
    expected_jac = A.unsqueeze(0).expand(2, 2, 2)
    assert torch.allclose(jac, expected_jac, atol=1e-5)

def test_lie_bracket_computation():
    # Test Lie bracket properties on smooth vector fields
    f = lambda x: torch.stack([x[:, 0] ** 2, x[:, 1]], dim=1)
    g = lambda x: torch.stack([x[:, 1], x[:, 0]], dim=1)
    
    x = torch.tensor([[1.0, 1.0], [2.0, 3.0]], requires_grad=True)
    bracket_val = lie_bracket(f, g, x)
    
    assert bracket_val.shape == x.shape
    assert not torch.isnan(bracket_val).any()
