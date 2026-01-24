"""
Plotting module to visualize the solution found by Physics-Informed Neural Network

Author: elphaim
Date: January 23, 2026
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional

from models.heat_pinn import analytical_solution

def plot_solution(model, data, alpha_true=0.01, save_path: Optional[str] = None):
    # Generate mesh
    x_eval = torch.linspace(0, 1, 100).reshape(-1, 1)
    t_eval = torch.linspace(0, 1, 100).reshape(-1, 1)
    X, T = torch.meshgrid(x_eval.squeeze(), t_eval.squeeze(), indexing='ij')
    x_flat = X.flatten().reshape(-1, 1)
    t_flat = T.flatten().reshape(-1, 1)

    if model.inverse:
        # Predict solution and parameter
        u_pred_inv = model.predict(x_flat, t_flat).reshape(100, 100)
        alpha_pred = model.get_alpha()
        error_pct = abs(alpha_pred - alpha_true) / alpha_true * 100
        # Get measurement values
        x_m = data['x_m'].cpu().numpy()
        t_m = data['t_m'].cpu().numpy()
        u_m = data['u_m'].cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Reconstructed solution with measurements
        im1 = axes[0].contourf(X.numpy(), T.numpy(), u_pred_inv, levels=20, cmap='hot')
        axes[0].scatter(x_m, t_m, c='cyan', s=50, marker='x', linewidths=2, label='Measurements')
        axes[0].set_title(f'PINN Reconstruction (α={alpha_pred:.6f})')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('t')
        axes[0].legend()
        plt.colorbar(im1, ax=axes[0])

        # Plot 2: Measurement fit
        u_m_pred = model.predict(data['x_m'], data['t_m'])
        axes[1].scatter(u_m, u_m_pred, alpha=0.6, s=50)
        axes[1].plot([u_m.min(), u_m.max()], [u_m.min(), u_m.max()], 'r--', linewidth=2)
        axes[1].set_xlabel('Measured u')
        axes[1].set_ylabel('Predicted u')
        axes[1].set_title('Measurement Fit')
        axes[1].grid(True, alpha=0.3)

        r2 = 1 - np.sum((u_m - u_m_pred)**2) / np.sum((u_m - u_m.mean())**2)
        axes[1].text(0.05, 0.95, f'R² = {r2:.4f}\nα error = {error_pct:.3f}%', 
            transform=axes[1].transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    else:
        # Predict solution
        u_pred = model.predict(x_flat, t_flat).reshape(100, 100)
        # Compute ground truth
        u_true = analytical_solution(x_flat.numpy(), t_flat.numpy(), alpha=alpha_true).reshape(100, 100)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Predicted solution
        im1 = axes[0].contourf(X.numpy(), T.numpy(), u_pred, levels=20, cmap='hot')
        axes[0].set_title('PINN Solution')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('t')
        plt.colorbar(im1, ax=axes[0])

        # Plot 2: Exact solution
        im2 = axes[1].contourf(X.numpy(), T.numpy(), u_true, levels=20, cmap='hot')
        axes[1].set_title('Analytical Solution')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('t')
        plt.colorbar(im2, ax=axes[1])

        # Plot 3: error
        error = np.abs(u_pred - u_true)
        im3 = axes[2].contourf(X.numpy(), T.numpy(), error, levels=20, cmap='viridis')
        axes[2].set_title(f'Error (L2={np.sqrt(np.sum((u_pred - u_true)**2)) / np.sqrt(np.sum(u_true**2)) * 100:.4f}%)')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('t')
        plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()