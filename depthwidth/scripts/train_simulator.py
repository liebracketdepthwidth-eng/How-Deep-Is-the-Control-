#!/usr/bin/env python3
"""
Training script for Phase 3B Neural-Style Simulator.
"""

import torch
import torch.nn as nn
import argparse
from depthwidth.models.neural_simulator import NeuralStyleSimulator

def main():
    parser = argparse.ArgumentParser(description="Train Neural-Style Simulator")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    args = parser.parse_args()

    print("=== Training Phase 3B Neural-Style Simulator ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    simulator = NeuralStyleSimulator(state_dim=2, latent_dim=32, hidden_dim=128, depth=4).to(device)
    optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        states = torch.randn(64, 2, device=device)
        targets = torch.randn(64, 2, device=device)
        
        preds = simulator(states)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}")

    print("Training completed successfully.")

if __name__ == "__main__":
    main()
