"""
Lie-bracket computations and vector field bracket operators for analyzing 
depth-width expressivity trade-offs.
"""

import torch
import torch.nn as nn
from typing import Callable, Tuple

def compute_vector_field_jacobian(
    f: Callable[[torch.Tensor], torch.Tensor], 
    x: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Jacobian matrix J_f(x) for vector field f at point x.
    """
    x = x.detach().requires_grad_(True)
    y = f(x)
    batch_size = y.shape[0]
    dim = y.shape[1]
    
    jac = torch.zeros(batch_size, dim, dim, device=x.device)
    for i in range(dim):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[:, i] = 1.0
        grads = torch.autograd.grad(
            outputs=y, inputs=x, grad_outputs=grad_outputs,
            create_graph=True, retain_graph=True
        )[0]
        jac[:, i, :] = grads
    return jac

def lie_bracket(
    f: Callable[[torch.Tensor], torch.Tensor],
    g: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Lie bracket [f, g](x) = J_g(x)f(x) - J_f(x)g(x).
    """
    fx = f(x)
    gx = g(x)
    
    jac_g = compute_vector_field_jacobian(g, x)
    jac_f = compute_vector_field_jacobian(f, x)
    
    term1 = torch.bmm(jac_g, fx.unsqueeze(-1)).squeeze(-1)
    term2 = torch.bmm(jac_f, gx.unsqueeze(-1)).squeeze(-1)
    
    return term1 - term2
