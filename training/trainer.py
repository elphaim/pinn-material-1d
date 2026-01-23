"""
Training module for Physics-Informed Neural Networks

Implements training loop with:
- Adaptive loss weighting
- Progress monitoring and visualization
- Checkpointing
- Early stopping

Author: elphaim
Date: January 19, 2026
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import time
from pathlib import Path
import json

from models.heat_pinn import analytical_solution

class PINNTrainer:
    """
    Trainer class for Physics-Informed Neural Networks.
    
    Args:
        model: PINN model to train
        data: Dictionary containing all training data
        device: Device for computation ('cpu' or 'cuda')
        learning_rate: Initial learning rate (default: 1e-3)
        switch_threshold: Loss threshold after which optimizer is switched from Adam to L-BFGS (default: 1e-5)
        max_iter: Max iterations for L-BFGS (default: 500)
        track_gradient_norms: Compute gradients of loss functions at every epoch (default: False)
        adaptive_weights: Use adaptive loss weighting (default: False)
        weight_update_freq: How often to update weights (default: 100 epochs)
        weight_ema: EMA smoothing for gradient statistics (default: 0.9)
    """
    
    def __init__(
        self,
        model,
        data: Dict[str, torch.Tensor],
        device: str = 'cpu',
        learning_rate: float = 1e-3,
        switch_threshold: float = 1e-5,
        max_iter: int = 500,
        track_gradient_norms: bool = False,
        adaptive_weights: bool = False,
        weight_update_freq: int = 100,
        weight_ema: float = 0.9,
    ):
        self.model = model.to(device)
        self.data = data
        self.device = device
        self.switch_threshold = switch_threshold
        self.track_gradient_norms = track_gradient_norms
        self.adaptive_weights = adaptive_weights
        self.weight_update_freq = weight_update_freq
        self.weight_ema = weight_ema
        
        # Initialize optimizers
        self.adam = optim.Adam(model.parameters(), lr=learning_rate)
        self.lbfgs = optim.LBFGS(model.parameters(), max_iter=max_iter, line_search_fn='strong_wolfe')
        
        # Initialize loss weights
        self.lambda_f = 1.0
        self.lambda_bc = 1.0
        self.lambda_ic = 1.0
        self.lambda_m = 1.0 if model.inverse else 0.0

        # Running mean of gradient magnitudes for EMA
        self.grad_mean_f = None
        self.grad_mean_bc = None
        self.grad_mean_ic = None
        self.grad_mean_m = None
        
        # History tracking
        self.history = {
            'epoch': [],
            'total_loss': [],
            'residual_loss': [],
            'boundary_loss': [],
            'initial_loss': [],
            'measurement_loss': [],
            'lambda_f': [],
            'lambda_bc': [],
            'lambda_ic': [],
            'lambda_m': [],
        }
        # alpha for inverse problem
        if model.inverse:
            self.history['alpha'] = []
        # L2 norms of gradients
        if track_gradient_norms:
            self.history['grad_norm_f'] = []
            self.history['grad_norm_bc'] = []
            self.history['grad_norm_ic'] = []
            self.history['grad_norm_m'] = []

        # Print status
        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Adam Learning rate: {learning_rate}")
        print(f"  Switch threshold for L-BFGS: {switch_threshold}")
        print(f"  Tracking gradient L2 norms: {track_gradient_norms}")
        print(f"  Adaptive weights: {adaptive_weights}")
        if adaptive_weights:
            print(f"  EMA smoothing: {weight_ema}")
            print(f"  Update frequency: every {weight_update_freq} epochs")
        print(f"  Problem type: {'Inverse' if model.inverse else 'Forward'}")


    def compute_loss_gradients(self) -> Dict[str, float]:
        self.model.eval()

        # Get model parameters for gradient computation
        params = list(self.model.parameters())

        # Enable gradients on collocation points for residual
        # This is needed because .residual() uses autograd
        x_f = self.data['x_f'].requires_grad_(True)
        t_f = self.data['t_f'].requires_grad_(True)
        # Residual loss
        residual = self.model.residual(x_f, t_f)
        loss_f = torch.mean(residual ** 2)
        # Compute gradients w.r.t. parameters
        grad_f = torch.autograd.grad(
            outputs=loss_f,
            inputs=params,
            create_graph=False,
            allow_unused=True
        )
        # Discard inexisting gradients
        valid_grad_f = [g for g in grad_f if g is not None]
        # Take L2 norm
        total_f = torch.zeros((), device=valid_grad_f[0].device)
        for g in valid_grad_f:
            total_f += g.norm()**2
        grad_norm_f = torch.sqrt(total_f)

        # Boundary condition loss
        u_bc_pred = self.model.forward(self.data['x_bc'], self.data['t_bc'])
        loss_bc = torch.mean((u_bc_pred - self.data['u_bc']) ** 2)
        grad_bc = torch.autograd.grad(
            outputs=loss_bc,
            inputs=params,
            create_graph=False,
            allow_unused=True
        )
        valid_grad_bc = [g for g in grad_bc if g is not None]
        total_bc = torch.zeros((), device=valid_grad_bc[0].device)
        for g in valid_grad_bc:
            total_bc += g.norm()**2
        grad_norm_bc = torch.sqrt(total_bc)

        # Initial condition loss
        u_ic_pred = self.model.forward(self.data['x_ic'], self.data['t_ic'])
        loss_ic = torch.mean((u_ic_pred - self.data['u_ic']) ** 2)
        grad_ic = torch.autograd.grad(
            outputs=loss_ic,
            inputs=params,
            create_graph=False,
            allow_unused=True
        )
        valid_grad_ic = [g for g in grad_ic if g is not None]
        total_ic = torch.zeros((), device=valid_grad_ic[0].device)
        for g in valid_grad_ic:
            total_ic += g.norm()**2
        grad_norm_ic = torch.sqrt(total_ic)

         # Measurement loss (inverse problem)
        if self.model.inverse and 'x_m' in self.data:
            u_m_pred = self.model.forward(self.data['x_m'], self.data['t_m'])
            loss_m = torch.mean((u_m_pred - self.data['u_m']) ** 2)
            grad_m = torch.autograd.grad(
                outputs=loss_m,
                inputs=params,
                create_graph=False,
                allow_unused=True
            )
            valid_grad_m = [g for g in grad_m if g is not None]
            total_m = torch.zeros((), device=valid_grad_m[0].device)
            for g in valid_grad_m:
                total_m += g.norm()**2
            grad_norm_m = torch.sqrt(total_m)
        else:
            grad_norm_m = torch.tensor(0.0)

        return {
            'grad_norm_f': grad_norm_f.item(),
            'grad_norm_bc': grad_norm_bc.item(),
            'grad_norm_ic': grad_norm_ic.item(),
            'grad_norm_m': grad_norm_m.item() if self.model.inverse else 0.0
        }
        

    def compute_adaptive_weights(self) -> Dict[str, float]:
        """
        Compute adaptive loss weights using loss gradients
        
        Following Wang et al. (2021): λ = Tr[K] / Tr[K_component]
        but NTK is computationally intensive so we use a proxy:
            λ_i = mean(||∇_θ L_i||_2) / ||∇_θ L_i||_2
        
        Returns:
            weights: Dictionary with updated lambda values
        """
        grads = self.compute_loss_gradients()
 
        # Apply EMA on gradients
        if self.grad_mean_f is None or self.grad_mean_bc is None or self.grad_mean_ic is None or self.grad_mean_m is None:
            # Initialize on first call
            self.grad_mean_f = grads['grad_norm_f']
            self.grad_mean_bc = grads['grad_norm_bc']
            self.grad_mean_ic = grads['grad_norm_ic']
            self.grad_mean_m = grads['grad_norm_m'] if self.model.inverse else 0.0
        else:
            # Update with EMA: new = α·old + (1-α)·current
            self.grad_mean_f = self.weight_ema * self.grad_mean_f + (1 - self.weight_ema) * grads['grad_norm_f']
            self.grad_mean_bc = self.weight_ema * self.grad_mean_bc + (1 - self.weight_ema) * grads['grad_norm_bc']
            self.grad_mean_ic = self.weight_ema * self.grad_mean_ic + (1 - self.weight_ema) * grads['grad_norm_ic']
            if self.model.inverse:
                self.grad_mean_m = self.weight_ema * self.grad_mean_m + (1 - self.weight_ema) * grads['grad_norm_m']

        # Update weights
        grad_means = [self.grad_mean_f, self.grad_mean_bc, self.grad_mean_ic]
        if self.model.inverse:
            grad_means.append(self.grad_mean_m)

        mean_grad = np.mean(grad_means)
        lambda_f_new = mean_grad / (self.grad_mean_f + 1e-8)
        lambda_bc_new = mean_grad / (self.grad_mean_bc + 1e-8)
        lambda_ic_new = mean_grad / (self.grad_mean_ic + 1e-8)
        lambda_m_new = mean_grad / (self.grad_mean_m + 1e-8) if self.model.inverse else 0.0
        
        return {
            'lambda_f': float(lambda_f_new),
            'lambda_bc': float(lambda_bc_new),
            'lambda_ic': float(lambda_ic_new),
            'lambda_m': float(lambda_m_new),
        }
    
    
    def train_epoch(self, use_lbfgs: bool = False) -> Dict[str, float]:
        """
        Execute one training epoch.

        Args:
            use_lbfgs: whether to use the L-BFGS optimizer
        
        Returns:
            losses: Dictionary with loss values
        """
        self.model.train()

        # Prepare data for measurement loss
        x_m = self.data.get('x_m', None)
        t_m = self.data.get('t_m', None)
        u_m = self.data.get('u_m', None)
        # Enable gradients on collocation points for residual
        # This is needed because .residual() uses autograd
        x_f = self.data['x_f'].requires_grad_(True)
        t_f = self.data['t_f'].requires_grad_(True)

        if not use_lbfgs:
            # Use Adam optimizer
            self.adam.zero_grad()
            # Compute loss
            total_loss, losses = self.model.loss_function(
                x_f=x_f,
                t_f=t_f,
                x_bc=self.data['x_bc'],
                t_bc=self.data['t_bc'],
                u_bc=self.data['u_bc'],
                x_ic=self.data['x_ic'],
                t_ic=self.data['t_ic'],
                u_ic=self.data['u_ic'],
                x_m=x_m,
                t_m=t_m,
                u_m=u_m,
                lambda_f=self.lambda_f,
                lambda_bc=self.lambda_bc,
                lambda_ic=self.lambda_ic,
                lambda_m=self.lambda_m
            )
            # Backward pass
            total_loss.backward()
            self.adam.step()

            return losses

        if use_lbfgs:
            # Switch to L-BFGS
            def closure():
                self.lbfgs.zero_grad()
                # Compute loss
                total_loss, _ = self.model.loss_function(
                    x_f=x_f,
                    t_f=t_f,
                    x_bc=self.data['x_bc'],
                    t_bc=self.data['t_bc'],
                    u_bc=self.data['u_bc'],
                    x_ic=self.data['x_ic'],
                    t_ic=self.data['t_ic'],
                    u_ic=self.data['u_ic'],
                    x_m=x_m,
                    t_m=t_m,
                    u_m=u_m,
                    lambda_f=self.lambda_f,
                    lambda_bc=self.lambda_bc,
                    lambda_ic=self.lambda_ic,
                    lambda_m=self.lambda_m
                )
                total_loss.backward()
                return total_loss
            # Final L-BFGS
            self.lbfgs.step(closure)
            # Recompute final losses after closure
            total_loss, losses = self.model.loss_function(
                    x_f=x_f,
                    t_f=t_f,
                    x_bc=self.data['x_bc'],
                    t_bc=self.data['t_bc'],
                    u_bc=self.data['u_bc'],
                    x_ic=self.data['x_ic'],
                    t_ic=self.data['t_ic'],
                    u_ic=self.data['u_ic'],
                    x_m=x_m,
                    t_m=t_m,
                    u_m=u_m,
                    lambda_f=self.lambda_f,
                    lambda_bc=self.lambda_bc,
                    lambda_ic=self.lambda_ic,
                    lambda_m=self.lambda_m
                )
            return losses
                
        
    def train(
        self,
        epochs: int = 10000,
        print_every: int = 1000,
        plot_every: int = 2000,
        save_path: Optional[str] = None
    ):
        """
        Main training loop.
        
        Args:
            epochs: Number of training epochs
            print_every: Print progress every N epochs
            plot_every: Plot progress every N epochs
            save_path: Path to save model checkpoints
        """
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)
        
        start_time = time.time()

        # Begin training with Adam optimizer
        # Use L-BFGS as final shot
        use_lbfgs = False
        lbfgs_epoch = 0
        
        for epoch in range(epochs):
            # Compute loss gradient norms
            if self.track_gradient_norms:
                grads = self.compute_loss_gradients()
                self.grad_norm_f = grads['grad_norm_f']
                self.grad_norm_bc = grads['grad_norm_bc']
                self.grad_norm_ic = grads['grad_norm_ic']
                self.grad_norm_m = grads['grad_norm_m']

            # Update adaptive weights periodically
            if self.adaptive_weights and epoch > 0 and epoch % self.weight_update_freq == 0:
                weights = self.compute_adaptive_weights()
                self.lambda_f = weights['lambda_f']
                self.lambda_bc = weights['lambda_bc']
                self.lambda_ic = weights['lambda_ic']
                self.lambda_m = weights['lambda_m']
            
            # Train one epoch
            losses = self.train_epoch(use_lbfgs)
            if losses['total'] < self.switch_threshold:
                if not use_lbfgs:
                    # Inform of the switch
                    print("\n" + "=" * 30)
                    print(f"Switching to L-BFGS at epoch {epoch}")
                    print("=" * 30)
                    lbfgs_epoch = epoch
                use_lbfgs = True
            
            # Break 10 epochs after switch to L-BFGS for visualization
            if use_lbfgs and epoch >= lbfgs_epoch + 10:
                break

            # Record history
            self.history['epoch'].append(epoch)
            self.history['total_loss'].append(losses['total'])
            self.history['residual_loss'].append(losses['residual'])
            self.history['boundary_loss'].append(losses['boundary'])
            self.history['initial_loss'].append(losses['initial'])
            self.history['measurement_loss'].append(losses['measurement'])
            self.history['lambda_f'].append(self.lambda_f)
            self.history['lambda_bc'].append(self.lambda_bc)
            self.history['lambda_ic'].append(self.lambda_ic)
            self.history['lambda_m'].append(self.lambda_m)
            # Record alpha
            if self.model.inverse:
                self.history['alpha'].append(self.model.get_alpha())
            # Record gradients
            if self.track_gradient_norms:
                self.history['grad_norm_f'].append(self.grad_norm_f)
                self.history['grad_norm_bc'].append(self.grad_norm_bc)
                self.history['grad_norm_ic'].append(self.grad_norm_ic)
                self.history['grad_norm_m'].append(self.grad_norm_m)
            
            # Print progress
            if epoch % print_every == 0:
                elapsed = time.time() - start_time
                print(f"\nEpoch {epoch}/{epochs} ({elapsed:.1f}s)")
                print(f"  Total Loss: {losses['total']:.6e}")
                print(f"  Residual: {losses['residual']:.6e} (λ={self.lambda_f:.2f})")
                print(f"  Boundary: {losses['boundary']:.6e} (λ={self.lambda_bc:.2f})")
                print(f"  Initial: {losses['initial']:.6e} (λ={self.lambda_ic:.2f})")
                if self.model.inverse:
                    print(f"  Measurement: {losses['measurement']:.6e} (λ={self.lambda_m:.2f})")
                    print(f"  Alpha: {self.model.get_alpha():.6f} (true: 0.01)")
            
            # Plot progress
            if epoch % plot_every == 0 and epoch > 0:
                self.plot_progress()
        
        # Last plot
        print(f"\nFinal training plot:")
        self.plot_progress()
        total_time = time.time() - start_time
        print(f"\nTraining complete. Total time: {total_time:.1f}s")
        
        # Save final model
        if save_path:
            self.save_checkpoint(save_path)

    
    def plot_progress(self, save_path: Optional[str] = None):
        """
        Plot training progress.
        
        Args:
            save_path: Optional path to save figure
        """        
        epochs = self.history['epoch']

        if self.adaptive_weights:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
            # Plot 1: Loss components
            ax = axes[0, 0]
            ax.semilogy(epochs, self.history['total_loss'], 'k-', label='Total', linewidth=2)
            ax.semilogy(epochs, self.history['residual_loss'], 'b-', label='Residual', alpha=0.7)
            ax.semilogy(epochs, self.history['boundary_loss'], 'r-', label='Boundary', alpha=0.7)
            ax.semilogy(epochs, self.history['initial_loss'], 'g-', label='Initial', alpha=0.7)
            if self.model.inverse:
                ax.semilogy(epochs, self.history['measurement_loss'], 'm-', label='Measurement', alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss Components')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 2: Adaptive weights
            ax = axes[0, 1]
            ax.plot(epochs, self.history['lambda_f'], 'b-', label='λ_f (residual)')
            ax.plot(epochs, self.history['lambda_bc'], 'r-', label='λ_bc (boundary)')
            ax.plot(epochs, self.history['lambda_ic'], 'g-', label='λ_ic (initial)')
            if self.model.inverse:
                ax.plot(epochs, self.history['lambda_m'], 'm-', label='λ_m (measurement)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Weight')
            ax.set_title('Gradient-based Adaptive Weights')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 3: Alpha convergence (inverse)
            if self.model.inverse:
                ax = axes[1, 0]
                ax.plot(epochs, self.history['alpha'], 'b-', linewidth=2)
                ax.axhline(y=0.01, color='r', linestyle='--', linewidth=2, label='True α')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('α')
                ax.set_title('Parameter Recovery')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                axes[1, 0].set_axis_off()
        
            # Plot 4: Gradient norms (when tracked)
            if self.track_gradient_norms:
                ax = axes[1, 1]
                ax.plot(epochs, self.history['grad_norm_f'], 'b-', label='||∇L_f||_2', alpha=0.7)
                ax.plot(epochs, self.history['grad_norm_bc'], 'r-', label='||∇L_bc|_2', alpha=0.7)
                ax.plot(epochs, self.history['grad_norm_ic'], 'g-', label='||∇L_ic||_2', alpha=0.7)
                if self.model.inverse and len(self.history['grad_norm_m']) > 0:
                    ax.plt(epochs, self.history['grad_norm_m'], 'm-', label='||∇L_m||_2', alpha=0.7)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Gradient L2 Norm')
                ax.set_title('Loss landscape')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                axes[1, 1].set_axis_off()

        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Plot 1: Loss components
            ax = axes[0]
            ax.semilogy(epochs, self.history['total_loss'], 'k-', label='Total', linewidth=2)
            ax.semilogy(epochs, self.history['residual_loss'], 'b-', label='Residual', alpha=0.7)
            ax.semilogy(epochs, self.history['boundary_loss'], 'r-', label='Boundary', alpha=0.7)
            ax.semilogy(epochs, self.history['initial_loss'], 'g-', label='Initial', alpha=0.7)
            if self.model.inverse:
                ax.semilogy(epochs, self.history['measurement_loss'], 'm-', label='Measurement', alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss Components')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 2: Alpha convergence (inverse) or gradient norms (when tracked)
            if self.model.inverse:
                ax = axes[1]
                ax.plot(epochs, self.history['alpha'], 'b-', linewidth=2)
                ax.axhline(y=0.01, color='r', linestyle='--', linewidth=2, label='True α')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('α')
                ax.set_title('Parameter Recovery')
                ax.legend()
                ax.grid(True, alpha=0.3)
            elif self.track_gradient_norms:
                ax = axes[1]
                ax.plot(epochs, self.history['grad_norm_f'], 'b-', label='||∇L_f||_2', alpha=0.7)
                ax.plot(epochs, self.history['grad_norm_bc'], 'r-', label='||∇L_bc||_2', alpha=0.7)
                ax.plot(epochs, self.history['grad_norm_ic'], 'g-', label='||∇L_ic||_2', alpha=0.7)
                if self.model.inverse and len(self.history['grad_norm_m']) > 0:
                    ax.plt(epochs, self.history['grad_norm_m'], 'm-', label='||∇L_m||_2', alpha=0.7)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Gradient L2 Norm')
                ax.set_title('Loss landscape')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                axes[1].set_axis_off()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint and training history."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.lbfgs.state_dict(),
            'history': self.history,
            'final_alpha': self.model.get_alpha() if self.model.inverse else None
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
        
        # Also save history as JSON
        history_path = path.replace('.pt', '_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)