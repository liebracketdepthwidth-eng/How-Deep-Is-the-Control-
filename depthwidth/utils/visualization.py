"""
Visualization utilities for phase portraits and trajectory plots.
"""

import matplotlib.pyplot as plt
import torch

def plot_phase_trajectory(states: torch.Tensor, save_path: str = "trajectory.png"):
    states_np = states.detach().cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.plot(states_np[:, 0], states_np[:, 1], label="Trajectory", color="b", lw=2)
    plt.xlabel(r"$\theta$ (Position)")
    plt.ylabel(r"$\omega$ (Velocity)")
    plt.title("Closed-Loop Arm Phase Portrait")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
