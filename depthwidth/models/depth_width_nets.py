"""
Configurable multi-layer perceptrons with variable depth and width scaling.
"""

import torch
import torch.nn as nn

class ScalableDepthWidthNet(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        depth: int, 
        activation: str = "relu"
    ):
        super().__init__()
        act_fn = nn.ReLU() if activation == "relu" else nn.Tanh()
        
        layers = [nn.Linear(input_dim, hidden_dim), act_fn]
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
