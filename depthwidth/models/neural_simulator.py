"""
Phase 3B Neural-Style Simulator for modeling deep control trajectories.
"""

import torch
import torch.nn as nn

class NeuralStyleSimulator(nn.Module):
    def __init__(
        self, 
        state_dim: int = 2, 
        latent_dim: int = 32, 
        hidden_dim: int = 128, 
        depth: int = 4
    ):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
