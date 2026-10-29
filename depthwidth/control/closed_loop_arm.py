"""
Phase 2 Close-Loop Arm Simulator for dynamical control verification.
"""

import torch
import numpy as np

class ClosedLoopArmSimulator:
    def __init__(self, arm_length: float = 1.0, dt: float = 0.01):
        self.L = arm_length
        self.dt = dt

    def step(self, state: torch.Tensor, control_torque: torch.Tensor) -> torch.Tensor:
        """
        Simulates one step of robotic arm dynamics: theta, omega.
        state shape: [batch_size, 2] -> [theta, omega]
        """
        theta = state[:, 0]
        omega = state[:, 1]
        
        g = 9.81
        alpha = (control_torque - (g / self.L) * torch.sin(theta)) / (self.L ** 2)
        
        next_theta = theta + omega * self.dt
        next_omega = omega + alpha * self.dt
        
        return torch.stack([next_theta, next_omega], dim=1)
