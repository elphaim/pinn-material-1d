"""
Multi-Fidelity PINN Training Script

Compares three approaches:
1. High-fidelity only (few expensive measurements)
2. Low-fidelity only (many cheap measurements)
3. Multi-fidelity (combines both)

Author: elphaim
Date: February 3rd, 2026
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Optional

from data.heat_data import HeatEquationData, prepare_multi_fidelity_data
from models.heat_pinn import analytical_solution
from models.heat_pinn_strategy import StrategicPINN, StrongFormLoss, MultiFidelityLoss
from training.trainer_strategy import StrategicPINNTrainer


# ============================================================================
# Data Preparation
# ============================================================================


def prepare_hf_only_data(mf_data: dict) -> dict:
    """
    Extract high-fidelity only dataset from multi-fidelity data.
    Uses standard measurement keys expected by StrongFormLoss.
    """
    return {
        'x_f': mf_data['x_f'], 't_f': mf_data['t_f'],
        'x_bc': mf_data['x_bc'], 't_bc': mf_data['t_bc'], 'u_bc': mf_data['u_bc'],
        'x_ic': mf_data['x_ic'], 't_ic': mf_data['t_ic'], 'u_ic': mf_data['u_ic'],
        'x_m': mf_data['x_hf'], 't_m': mf_data['t_hf'], 'u_m': mf_data['u_hf']
    }


def prepare_lf_only_data(mf_data: dict) -> dict:
    """
    Extract low-fidelity only dataset from multi-fidelity data.
    """
    return {
        'x_f': mf_data['x_f'], 't_f': mf_data['t_f'],
        'x_bc': mf_data['x_bc'], 't_bc': mf_data['t_bc'], 'u_bc': mf_data['u_bc'],
        'x_ic': mf_data['x_ic'], 't_ic': mf_data['t_ic'], 'u_ic': mf_data['u_ic'],
        'x_m': mf_data['x_lf'], 't_m': mf_data['t_lf'], 'u_m': mf_data['u_lf']
    }


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_model(
    model: StrategicPINN,
    alpha_true: float = 0.01,
    n_points: int = 100
) -> dict:
    """
    Evaluate trained model against analytical solution.
    
    Returns:
        metrics: Dictionary with L2 error, relative error, alpha error
    """
    model.eval()
    
    # Create evaluation grid
    x_eval = torch.linspace(0, 1, n_points).reshape(-1, 1)
    t_eval = torch.linspace(0, 1, n_points).reshape(-1, 1)
    X, T = torch.meshgrid(x_eval.squeeze(), t_eval.squeeze(), indexing='ij')
    x_flat = X.flatten().reshape(-1, 1)
    t_flat = T.flatten().reshape(-1, 1)
    
    # Predict
    u_pred = model.predict(x_flat, t_flat)
    
    # Exact solution
    u_exact = analytical_solution(
        x_flat.numpy(), t_flat.numpy(), alpha=alpha_true
    )
    
    # Compute errors
    l2_error = np.sqrt(np.mean((u_pred - u_exact)**2))
    rel_error = l2_error / np.sqrt(np.mean(u_exact**2))
    max_error = np.max(np.abs(u_pred - u_exact))
    
    # Alpha error (for inverse problems)
    if model.inverse:
        alpha_pred = model.get_alpha()
        alpha_error = abs(alpha_pred - alpha_true) / alpha_true
    else:
        alpha_pred = None
        alpha_error = None
    
    return {
        'l2_error': l2_error,
        'rel_error': rel_error,
        'max_error': max_error,
        'alpha_pred': alpha_pred,
        'alpha_error': alpha_error,
        'u_pred': u_pred.reshape(n_points, n_points),
        'u_exact': u_exact.reshape(n_points, n_points),
        'X': X.numpy(),
        'T': T.numpy()
    }


# ============================================================================
# Visualization
# ============================================================================


def plot_comparison(
    results: dict,
    mf_data: dict,
    save_path: Optional[str] = None
):
    """
    Plot comparison of all three approaches.
    """
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    
    approaches = ['hf_only', 'lf_only', 'multi_fidelity']
    titles = ['High-Fidelity Only', 'Low-Fidelity Only', 'Multi-Fidelity']
    
    for row, (approach, title) in enumerate(zip(approaches, titles)):
        res = results[approach]
        X, T = res['X'], res['T']
        
        # Column 1: Predicted solution
        im1 = axes[row, 0].contourf(X, T, res['u_pred'], levels=20, cmap='hot')
        axes[row, 0].set_xlabel('x')
        axes[row, 0].set_ylabel('t')
        axes[row, 0].set_title(f'{title}: Prediction')
        plt.colorbar(im1, ax=axes[row, 0])
        
        # Column 2: Exact solution (same for all)
        im2 = axes[row, 1].contourf(X, T, res['u_exact'], levels=20, cmap='hot')
        axes[row, 1].set_xlabel('x')
        axes[row, 1].set_ylabel('t')
        axes[row, 1].set_title('Exact Solution')
        plt.colorbar(im2, ax=axes[row, 1])
        
        # Column 3: Error field
        error_field = np.abs(res['u_pred'] - res['u_exact'])
        im3 = axes[row, 2].contourf(X, T, error_field, levels=20, cmap='viridis')
        axes[row, 2].set_xlabel('x')
        axes[row, 2].set_ylabel('t')
        axes[row, 2].set_title(f'Error (Rel. L2: {res["rel_error"]:.2%})')
        plt.colorbar(im3, ax=axes[row, 2])
        
        # Column 4: Data locations
        x_m = []
        x_hf = []
        x_lf = []
        if approach == 'hf_only':
            x_m = mf_data['x_hf'].cpu().numpy()
            t_m = mf_data['t_hf'].cpu().numpy()
            axes[row, 3].scatter(x_m, t_m, c='blue', s=30, label='HF data')
        elif approach == 'lf_only':
            x_m = mf_data['x_lf'].cpu().numpy()
            t_m = mf_data['t_lf'].cpu().numpy()
            axes[row, 3].scatter(x_m, t_m, c='red', s=10, alpha=0.5, label='LF data')
        else:
            x_hf = mf_data['x_hf'].cpu().numpy()
            t_hf = mf_data['t_hf'].cpu().numpy()
            x_lf = mf_data['x_lf'].cpu().numpy()
            t_lf = mf_data['t_lf'].cpu().numpy()
            axes[row, 3].scatter(x_lf, t_lf, c='red', s=10, alpha=0.3, label='LF data')
            axes[row, 3].scatter(x_hf, t_hf, c='blue', s=30, label='HF data')
        
        axes[row, 3].set_xlabel('x')
        axes[row, 3].set_ylabel('t')
        axes[row, 3].set_title(f'Data Locations (n={len(x_m) if approach != "multi_fidelity" else len(x_hf)+len(x_lf)})')
        axes[row, 3].legend()
        axes[row, 3].set_xlim(0, 1)
        axes[row, 3].set_ylim(0, 1)
        axes[row, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_training_curves(
    trainers: dict,
    save_path: Optional[str] = None
):
    """
    Plot training loss curves for all approaches.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'hf_only': 'blue', 'lf_only': 'red', 'multi_fidelity': 'green'}
    labels = {'hf_only': 'HF Only', 'lf_only': 'LF Only', 'multi_fidelity': 'Multi-Fidelity'}
    
    # Total loss
    for approach, trainer in trainers.items():
        epochs = trainer.history['epoch']
        total_loss = trainer.history['total_loss']
        axes[0].semilogy(epochs, total_loss, color=colors[approach], 
                         label=labels[approach], linewidth=2)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual loss (physics)
    for approach, trainer in trainers.items():
        epochs = trainer.history['epoch']
        res_loss = trainer.history['residual_loss']
        axes[1].semilogy(epochs, res_loss, color=colors[approach],
                         label=labels[approach], linewidth=2)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Residual Loss')
    axes[1].set_title('Physics Residual Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def print_summary(results: dict, mf_data: dict):
    """
    Print summary table of results.
    """
    print("\n" + "=" * 70)
    print("MULTI-FIDELITY COMPARISON RESULTS")
    print("=" * 70)
    
    print(f"\nData Configuration:")
    print(f"  High-fidelity points: {mf_data['x_hf'].shape[0]}")
    print(f"  Low-fidelity points: {mf_data['x_lf'].shape[0]}")
    print(f"  σ_hf: {mf_data['sigma_hf']:.4f}")
    print(f"  σ_lf: {mf_data['sigma_lf']:.4f}")
    
    print(f"\n{'Approach':<20} {'Rel. L2 Error':<15} {'Max Error':<15} {'Improvement':<15}")
    print("-" * 65)
    
    baseline_error = results['hf_only']['rel_error']
    
    for approach, label in [('hf_only', 'HF Only'), 
                            ('lf_only', 'LF Only'), 
                            ('multi_fidelity', 'Multi-Fidelity')]:
        res = results[approach]
        improvement = (baseline_error - res['rel_error']) / baseline_error * 100
        sign = '+' if improvement > 0 else ''
        
        print(f"{label:<20} {res['rel_error']:<15.4%} {res['max_error']:<15.4e} {sign}{improvement:<14.1f}%")
    
    print("-" * 65)
    
    # Highlight winner
    errors = {k: v['rel_error'] for k, v in results.items()}
    winner = min(errors, key=errors.get) # pyright: ignore[reportCallIssue, reportArgumentType]
    winner_label = {'hf_only': 'HF Only', 'lf_only': 'LF Only', 'multi_fidelity': 'Multi-Fidelity'}[winner]
    
    print(f"\nBest approach: {winner_label} (Rel. L2 = {errors[winner]:.4%})")
    
    if winner == 'multi_fidelity':
        hf_improvement = (errors['hf_only'] - errors['multi_fidelity']) / errors['hf_only'] * 100
        print(f"Multi-fidelity improvement over HF-only: {hf_improvement:.1f}%")


# ============================================================================
# Main Training Script
# ============================================================================


def run_multi_fidelity_comparison(
    # Data parameters
    alpha_true: float = 0.01,
    hf_sensors: int = 10,
    hf_times: int = 10,
    hf_noise: float = 0.02,
    lf_nx: int = 11,
    lf_nt: int = 31,
    lf_alpha_ratio: float = 1.2,
    lf_noise: float = 0.05,
    # Training parameters
    epochs: int = 3000,
    learning_rate: float = 1e-3,
    print_every: int = 500,
    # Model parameters
    layers: list = [2, 50, 50, 50, 50, 1],
    # Output
    save_dir: Optional[str] = None,
    random_seed: int = 42
):
    """
    Run complete multi-fidelity comparison experiment.
    """
    print("=" * 70)
    print("MULTI-FIDELITY PINN COMPARISON")
    print("=" * 70)
    
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Create output directory
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Data Generation
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("GENERATING DATA")
    print("-" * 70)
    
    data_gen = HeatEquationData(
        alpha=alpha_true,
        N_f=2000,
        N_bc=150,
        N_ic=100,
        noise_level=hf_noise,
        random_seed=random_seed
    )
    
    mf_data = prepare_multi_fidelity_data(
        data_gen,
        hf_sensors=hf_sensors,
        hf_times=hf_times,
        lf_nx=lf_nx,
        lf_nt=lf_nt,
        lf_alpha_ratio=lf_alpha_ratio,
        lf_noise=lf_noise
    )
    
    hf_data = prepare_hf_only_data(mf_data)
    lf_data = prepare_lf_only_data(mf_data)
    
    # ========================================================================
    # Train HF-Only Model
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("TRAINING: HIGH-FIDELITY ONLY")
    print("-" * 70)
    
    model_hf = StrategicPINN(
        layers=layers,
        alpha_true=alpha_true,
        inverse=False
    )
    model_hf.set_loss_strategy(StrongFormLoss())
    
    trainer_hf = StrategicPINNTrainer(
        model=model_hf,
        data=hf_data,
        learning_rate=learning_rate,
        min_adam_epochs=epochs + 1  # Disable L-BFGS switch for fair comparison
    )
    
    trainer_hf.train(
        epochs=epochs,
        print_every=print_every,
        plot_every=epochs + 1  # Disable intermediate plots
    )
    
    # ========================================================================
    # Train LF-Only Model
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("TRAINING: LOW-FIDELITY ONLY")
    print("-" * 70)
    
    model_lf = StrategicPINN(
        layers=layers,
        alpha_true=alpha_true,
        inverse=False
    )
    model_lf.set_loss_strategy(StrongFormLoss())
    
    trainer_lf = StrategicPINNTrainer(
        model=model_lf,
        data=lf_data,
        learning_rate=learning_rate,
        min_adam_epochs=epochs + 1
    )
    
    trainer_lf.train(
        epochs=epochs,
        print_every=print_every,
        plot_every=epochs + 1
    )
    
    # ========================================================================
    # Train Multi-Fidelity Model
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("TRAINING: MULTI-FIDELITY")
    print("-" * 70)
    
    model_mf = StrategicPINN(
        layers=layers,
        alpha_true=alpha_true,
        inverse=False
    )
    model_mf.set_loss_strategy(
        MultiFidelityLoss(weighting='uncertainty')
    )
    
    trainer_mf = StrategicPINNTrainer(
        model=model_mf,
        data=mf_data,
        learning_rate=learning_rate,
        min_adam_epochs=epochs + 1
    )
    
    trainer_mf.train(
        epochs=epochs,
        print_every=print_every,
        plot_every=epochs + 1
    )
    
    # ========================================================================
    # Evaluation
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("EVALUATING MODELS")
    print("-" * 70)
    
    results = {
        'hf_only': evaluate_model(model_hf, alpha_true),
        'lf_only': evaluate_model(model_lf, alpha_true),
        'multi_fidelity': evaluate_model(model_mf, alpha_true)
    }
    
    trainers = {
        'hf_only': trainer_hf,
        'lf_only': trainer_lf,
        'multi_fidelity': trainer_mf
    }
    
    # ========================================================================
    # Results
    # ========================================================================
    
    print_summary(results, mf_data)
    
    # Plots
    plot_comparison(
        results, mf_data,
        save_path=f"{save_dir}/solution_comparison.png" if save_dir else None
    )
    
    plot_training_curves(
        trainers,
        save_path=f"{save_dir}/training_curves.png" if save_dir else None
    )
    
    # Save results
    if save_dir:
        summary = {
            'config': {
                'alpha_true': alpha_true,
                'hf_sensors': hf_sensors,
                'hf_times': hf_times,
                'hf_noise': hf_noise,
                'lf_nx': lf_nx,
                'lf_nt': lf_nt,
                'lf_alpha_ratio': lf_alpha_ratio,
                'epochs': epochs
            },
            'results': {
                approach: {
                    'rel_error': float(res['rel_error']),
                    'l2_error': float(res['l2_error']),
                    'max_error': float(res['max_error'])
                }
                for approach, res in results.items()
            }
        }
        
        with open(f"{save_dir}/summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to {save_dir}/")
    
    return results, trainers, mf_data


# ============================================================================
# Usage Example and Testing
# ============================================================================


if __name__ == "__main__":
    results, trainers, data = run_multi_fidelity_comparison(
        alpha_true=0.01,
        # HF data
        hf_sensors=5,
        hf_times=5,
        hf_noise=0.01,
        # LF data
        lf_nx=6,
        lf_nt=11,
        lf_alpha_ratio=1.5,
        lf_noise=0.1,
        # Training
        epochs=3000,
        learning_rate=1e-3,
        print_every=500,
        # Output
        #save_dir='results/multi_fidelity_comparison',
        random_seed=42
    )