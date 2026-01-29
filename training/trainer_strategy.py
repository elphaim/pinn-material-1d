"""
Trainer compatible with Strategy Pattern PINNs

Works with both strong-form and weak-form loss strategies.

Author: elphaim
Date: January 29, 2026
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List
import time
from pathlib import Path
import json

from models.heat_pinn_strategy import StrategicPINN


class StrategicPINNTrainer:
    """
    Trainer for StrategicPINN that works with any loss strategy.

    Args:
        model: StrategicPINN instance
        data: Data dictionary (format depends on loss strategy)
        learning_rate: Initial Adam learning rate (default: 1e-3)
        reduce_lr_patience: Epochs before Adam LR is divided by 2 when loss plateaus (default: 100)
        switch_var: Condition on loss spread to switch from Adam to L-BFGS (default: 1e-12, disabled)
        switch_slope: Condition on loss slope to switch from Adam L-BFGS (default: 1e-12, disabled)
        switch_window: Window for computing switch criteria (default: 200)
        min_adam_epochs: Minimum epochs before allowing switch (default: 1000)
        lbfgs_max_iter: Max iterations per L-BFGS step (default: 20)
        lbfgs_tolerance: L-BFGS convergence tolerance (default: 1e-9)
        lbfgs_max_eval: Max function evaluations per L-BFGS step (default: 25)
        track_gradient_norms: Compute all gradients of loss functions (default: False)
        adaptive_weights: Use adaptive loss weighting (default: False)
        weight_update_freq: How often to update weights (default: 100 epochs)
        weight_ema: EMA smoothing for gradient statistics (default: 0.9)
        device: 'cpu' or 'cuda'
    """
    
    def __init__(
        self,
        model: StrategicPINN,
        data: Dict,
        learning_rate: float = 1e-3,
        reduce_lr_patience: int = 100,
        switch_var: float = 1e-12,
        switch_slope: float = 1e-12,
        switch_window: int = 200,
        min_adam_epochs: int = 1000,
        lbfgs_max_iter: int = 20,
        lbfgs_tolerance: float = 1e-9,
        lbfgs_max_eval: int = 25,
        track_gradient_norms: bool = False,
        adaptive_weights: bool = False,
        weight_update_freq: int = 100,
        weight_ema: float = 0.9,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.data = data
        self.device = device
        self.track_gradient_norms = track_gradient_norms
        self.adaptive_weights = adaptive_weights
        self.weight_update_freq = weight_update_freq
        self.weight_ema = weight_ema

        # Set default dtype for torch to float64 on CPU
        if device == 'cpu':
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)
        
        # Adam optimizer
        self.adam = optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=0.0
        )
        
        # LR scheduler for Adam
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.adam, 
            factor=0.5, 
            patience=reduce_lr_patience,
            min_lr=1e-6
        )

        # L-BFGS optimizer
        self.lbfgs = optim.LBFGS(
            model.parameters(),
            lr=1.0, 
            max_iter=lbfgs_max_iter,
            max_eval=lbfgs_max_eval,
            tolerance_grad=lbfgs_tolerance,
            tolerance_change=lbfgs_tolerance,
            history_size=100, 
            line_search_fn='strong_wolfe'
        )
        
        # Switch parameters
        self.switch_var = switch_var
        self.switch_slope = switch_slope
        self.switch_window = switch_window
        self.min_adam_epochs = min_adam_epochs
        
        # Loss weights
        self.lambda_f = 1.0
        self.lambda_bc = 1.0
        self.lambda_ic = 1.0
        self.lambda_m = 1.0 if model.inverse else 0.0

        # Running mean of gradient L2 norms
        self.grad_mean_f = None
        self.grad_mean_bc = None
        self.grad_mean_ic = None
        self.grad_mean_m = None
        
        # History
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
            'optimizer': [],
            'lbfgs_iter': []
        }
        
        if model.inverse:
            self.history['alpha'] = []

        if track_gradient_norms:
            self.history['grad_norm_f'] = []
            self.history['grad_norm_bc'] = []
            self.history['grad_norm_ic'] = []
            self.history['grad_norm_m'] = []

        # L-BFGS tracking
        self.lbfgs_iteration = 0
        self.lbfgs_losses = []

        # Print status
        print(f"Trainer initialized:")
        print(f"  Loss strategy: {model.loss_strategy.__class__.__name__}")
        print(f"  Adam LR: {learning_rate}")
        print(f"  Switch criteria: var < {switch_var}, |slope| < {switch_slope}")
        print(f"  L-BFGS max iter: {lbfgs_max_iter}")
        print(f"  Tracking gradient L2 norms: {track_gradient_norms}")
        print(f"  Adaptive weights: {adaptive_weights}")
        if adaptive_weights:
            print(f"  EMA smoothing: {weight_ema}")
            print(f"  Update frequency: every {weight_update_freq} epochs")
        print(f"  Problem type: {'Inverse' if model.inverse else 'Forward'}")


    def should_switch_to_lbfgs(self, epoch: int) -> Tuple[bool, str]:
        """
        Determine if should switch from Adam to L-BFGS.
        
        Returns:
            should_switch: Boolean
            reason: String explaining decision
        """
        # Too early
        if epoch < self.min_adam_epochs:
            return False, f"Too early (< {self.min_adam_epochs})"
        
        # Not enough history
        if len(self.history['total_loss']) < self.switch_window:
            return False, "Insufficient history"
        
        # Compute switching criteria
        past_losses = self.history['total_loss'][-self.switch_window:]
        
        # Variance ratio (plateau detection)
        p95 = np.percentile(past_losses, 95)
        p5 = np.percentile(past_losses, 5)
        var_ratio = (p95 - p5) / p95
        
        # Slope (stagnation detection)
        t = np.arange(self.switch_window)
        slope = np.polyfit(t, np.log(past_losses), 1)[0]
        
        # Check criteria
        plateau = var_ratio < self.switch_var
        stagnant = abs(slope) < self.switch_slope
        
        if plateau and stagnant:
            return True, f"Plateau detected (var={var_ratio:.6f}, slope={abs(slope):.6f})"
        
        return False, f"Not ready (var={var_ratio:.6f}, slope={abs(slope):.6f})"

    
    def compute_loss_gradients(self) -> Dict[str, float]:
        """
        Compute L2 norms of loss gradients ||∇_θ L_i||_2
        """
        # Get model parameters for gradient computation
        params = list(self.model.parameters())

        # Compute loss using strategy
        self.model.eval()
        _, losses = self.model.compute_loss(
            data=self.data,
            lambda_f=self.lambda_f,
            lambda_bc=self.lambda_bc,
            lambda_ic=self.lambda_ic,
            lambda_m=self.lambda_m
        )

        # Residual loss gradients
        grad_f = torch.autograd.grad(
            outputs=losses['residual_t'],
            inputs=params,
            create_graph=False,
            allow_unused=True
        )
        # Discard inexisting gradients
        valid_grad_f = [g for g in grad_f if g is not None]
        # Compute L2 norm
        total_f = torch.zeros((), device=valid_grad_f[0].device)
        for g in valid_grad_f:
            total_f += g.norm()**2
        grad_norm_f = torch.sqrt(total_f)

        # Boundary loss gradients
        grad_bc = torch.autograd.grad(
            outputs=losses['boundary_t'],
            inputs=params,
            create_graph=False,
            allow_unused=True
        )
        valid_grad_bc = [g for g in grad_bc if g is not None]
        total_bc = torch.zeros((), device=valid_grad_bc[0].device)
        for g in valid_grad_bc:
            total_bc += g.norm()**2
        grad_norm_bc = torch.sqrt(total_bc)

        # Initial loss gradients
        grad_ic = torch.autograd.grad(
            outputs=losses['initial_t'],
            inputs=params,
            create_graph=False,
            allow_unused=True
        )
        valid_grad_ic = [g for g in grad_ic if g is not None]
        total_ic = torch.zeros((), device=valid_grad_ic[0].device)
        for g in valid_grad_ic:
            total_ic += g.norm()**2
        grad_norm_ic = torch.sqrt(total_ic)

        # Measurement loss gradients (inverse problem)
        if self.model.inverse:
            grad_m = torch.autograd.grad(
                outputs=losses['measurement_t'],
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
        Compute adaptive loss weights using gradient proxy
        
        Following Wang et al. (2021): λ = Tr[K] / Tr[K_component]
        
        NTK is computationally intensive so we use a proxy:
            λ_i = mean(||∇_θ L_i||_2) / ||∇_θ L_i||_2
        along with EMA smoothing
        
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
            # Update with EMA: new = s * old + (1-s) * current
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
    

    def train_epoch_adam(self) -> Tuple[Dict[str, float], Optional[float]]:
        """
        Execute one training epoch with Adam optimizer
        
        Returns:
            losses: Dictionary with loss values
            lr: Current Adam LR
        """
        self.model.train()
        self.adam.zero_grad()

        # Compute loss using strategy
        total_loss, losses = self.model.compute_loss(
            data=self.data,
            lambda_f=self.lambda_f,
            lambda_bc=self.lambda_bc,
            lambda_ic=self.lambda_ic,
            lambda_m=self.lambda_m
        )

        # Backward
        total_loss.backward()
        self.adam.step()

        # LR scheduler
        old_lr = self.adam.param_groups[0]["lr"]
        self.scheduler.step(total_loss.item())
        new_lr = self.adam.param_groups[0]["lr"]

        if new_lr < old_lr:
            print("\n" + "=" * 40)
            print(f"Adam LR reduced: {old_lr:.2e} -> {new_lr:.2e}")
            print("=" * 40)

        return losses, new_lr
        

    def train_lbfgs_step(self) -> List[Dict[str, float]]:
        """
        Single L-BFGS step (which includes multiple line searches).
        
        Returns:
            losses_list: List of losses from each internal iteration
        """
        self.model.train()
        
        # Reset L-BFGS iteration counter for this step
        self.lbfgs_iteration = 0
        self.lbfgs_losses = []
        
        def closure():
            """
            L-BFGS closure function.
            
            Called multiple times per .step() during line search.
            We track each call to visualize L-BFGS convergence.
            """
            self.lbfgs.zero_grad()
            
            total_loss, losses = self.model.compute_loss(
                data=self.data,
                lambda_f=self.lambda_f,
                lambda_bc=self.lambda_bc,
                lambda_ic=self.lambda_ic,
                lambda_m=self.lambda_m
            )
            
            total_loss.backward()
            
            # Store this iteration's losses
            self.lbfgs_losses.append(losses)
            self.lbfgs_iteration += 1
            
            return total_loss
        
        # Single L-BFGS step (internally does multiple line searches)
        self.lbfgs.step(closure)
        
        # Return all intermediate losses from this step
        return self.lbfgs_losses
            
    
    def train(
        self,
        epochs: int = 5000,
        print_every: int = 500,
        plot_every: int = 1000,
        lbfgs_max_steps: int = 100,
        lbfgs_convergence_tol: float = 1e-7,
        save_path: Optional[str] = None
    ):
        """
        Main training loop.
        
        Args:
            epochs: Max number of Adam training epochs (default: 5000)
            print_every: Print progress every N Adam epochs (default: 500)
            plot_every: Plot progress every N Adam epochs (default: 1000)
            lbfgs_max_steps: Maximum L-BFGS steps after switch
            lbfgs_convergence_tol: Stop L-BFGS if loss change < this
            save_path: Path to save model checkpoints (default: None)
        """
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)
        
        start_time = time.time()
        use_lbfgs = False
        lbfgs_step_count = 0
        lbfgs_switch_epoch = None
        
        # Begin training with Adam optimizer
        for epoch in range(epochs):

            # Exit loop if switch
            if use_lbfgs:
                break

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
            losses, lr = self.train_epoch_adam()

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
            self.history['optimizer'].append('adam')
            self.history['lbfgs_iter'].append(0)
            
            if self.model.inverse:
                self.history['alpha'].append(self.model.get_alpha())

            if self.track_gradient_norms:
                self.history['grad_norm_f'].append(self.grad_norm_f)
                self.history['grad_norm_bc'].append(self.grad_norm_bc)
                self.history['grad_norm_ic'].append(self.grad_norm_ic)
                self.history['grad_norm_m'].append(self.grad_norm_m)
            
            # Print progress
            if epoch % print_every == 0:
                elapsed = time.time() - start_time
                print(f"\nEpoch {epoch}/{epochs} ({elapsed:.1f}s)")
                print(f"  Adam learning rate: {lr:.2e}")
                print(f"  Total: {losses['total']:.6e}")
                print(f"  Residual: {losses['residual']:.6e} (λ={self.lambda_f:.2f})")
                print(f"  Boundary: {losses['boundary']:.6e} (λ={self.lambda_bc:.2f})")
                print(f"  Initial: {losses['initial']:.6e} (λ={self.lambda_ic:.2f})")
                
                if self.model.inverse:
                    print(f"  Measurement: {losses['measurement']:.6e} (λ={self.lambda_m:.2f})")
                    print(f"  Alpha: {self.model.get_alpha():.6f} (true: 0.01)")
                
                # Strategy-specific info
                if 'weak_res_nonzero' in losses:
                    print(f"  Non-zero weak residuals: {losses['weak_res_nonzero']}")

            # Plot progress
            if epoch % plot_every == 0 and epoch > 0:
                self.plot_progress()

            # Check if should switch to L-BFGS
            should_switch, reason = self.should_switch_to_lbfgs(epoch)
            
            if should_switch:
                use_lbfgs = True
                lbfgs_switch_epoch = epoch
                
                print("\n" + "=" * 60)
                print(f"Switching to L-BFGS at Epoch {epoch}")
                print(f"   Reason: {reason}")
                print(f"   Loss before switch: {losses['total']:.6e}")
                print("=" * 60)

        # L-BFGS training
        if use_lbfgs and lbfgs_switch_epoch is not None:

            loss_before_lbfgs = self.history['total_loss'][-1]
            prev_step_losses = []

            for lbfgs_step in range(lbfgs_max_steps):
                # Single L-BFGS step (multiple internal iterations)
                step_losses = self.train_lbfgs_step()
                
                # Record ALL intermediate iterations
                for iter_losses in step_losses:
                    # Use pseudo-epoch number for continuity in plots
                    pseudo_epoch = lbfgs_switch_epoch + lbfgs_step_count + 1
                    
                    self.history['epoch'].append(pseudo_epoch)
                    self.history['total_loss'].append(iter_losses['total'])
                    self.history['residual_loss'].append(iter_losses['residual'])
                    self.history['boundary_loss'].append(iter_losses['boundary'])
                    self.history['initial_loss'].append(iter_losses['initial'])
                    self.history['measurement_loss'].append(iter_losses['measurement'])
                    self.history['lambda_f'].append(self.lambda_f)
                    self.history['lambda_bc'].append(self.lambda_bc)
                    self.history['lambda_ic'].append(self.lambda_ic)
                    self.history['lambda_m'].append(self.lambda_m)
                    self.history['optimizer'].append('lbfgs')
                    self.history['lbfgs_iter'].append(self.lbfgs_iteration)
                    
                    if self.model.inverse:
                        self.history['alpha'].append(self.model.get_alpha())

                    if self.track_gradient_norms:
                        self.history['grad_norm_f'].append(self.grad_norm_f)
                        self.history['grad_norm_bc'].append(self.grad_norm_bc)
                        self.history['grad_norm_ic'].append(self.grad_norm_ic)
                        self.history['grad_norm_m'].append(self.grad_norm_m)
                    
                    lbfgs_step_count += 1
                
                # Print progress
                final_loss = step_losses[-1]['total']
                print(f"\nL-BFGS Step {lbfgs_step+1}/{lbfgs_max_steps}")
                print(f"  Internal iterations: {len(step_losses)}")
                print(f"  Loss: {final_loss:.6e}")
                print(f"  Improvement: {abs(loss_before_lbfgs - final_loss) / loss_before_lbfgs:.2%}")
                
                # Check convergence
                if lbfgs_step > 0:
                    prev_loss = self.history['total_loss'][
                        -(len(step_losses) + len(prev_step_losses))
                    ]
                    loss_change = abs(final_loss - prev_loss)
                    
                    if loss_change < lbfgs_convergence_tol:
                        print(f"\nL-BFGS converged (loss change: {loss_change:.6e})")
                        break
                
                prev_step_losses = step_losses

        # Last plot
        print(f"\nFinal training plot:")
        self.plot_progress()

        total_time = time.time() - start_time
        print(f"\nTraining complete. Time: {total_time:.1f}s")

        # Save final model
        if save_path:
            self.save_checkpoint(save_path)


    def plot_progress(self, save_path: Optional[str] = None):
        """
        Plot training progress
        
        Args:
            save_path: Optional path to save figure
        """        
        epochs = self.history['epoch']
        plots = []

        # Find where L-BFGS starts
        lbfgs_start = None
        for i, opt in enumerate(self.history['optimizer']):
            if opt == 'lbfgs':
                lbfgs_start = i
                break

        def plot_losses(ax):
            ax.semilogy(epochs, self.history['total_loss'], 'k-', label='Total', linewidth=2)
            ax.semilogy(epochs, self.history['residual_loss'], 'b-', label='Residual', alpha=0.7)
            ax.semilogy(epochs, self.history['boundary_loss'], 'r-', label='Boundary', alpha=0.7)
            ax.semilogy(epochs, self.history['initial_loss'], 'g-', label='Initial', alpha=0.7)
            if self.model.inverse:
                ax.semilogy(epochs, self.history['measurement_loss'], 'm-', label='Measurement', alpha=0.7)
            # Mark start of L-BFGS
            if lbfgs_start:
                ax.axvline(x=epochs[lbfgs_start], color='green', linestyle='--', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss Components')
            ax.legend()
            ax.grid(True, alpha=0.3)

        def plot_weights(ax):
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

        def plot_alpha(ax):
            ax.plot(epochs, self.history['alpha'], 'b-', linewidth=2)
            ax.axhline(y=0.01, color='r', linestyle='--', linewidth=2, label='True α')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('α')
            ax.set_title('Parameter Recovery')
            ax.legend()
            ax.grid(True, alpha=0.3)

        def plot_grad_norms(ax):
            ax.plot(epochs, self.history['grad_norm_f'], 'b-', label='||∇L_f||_2', alpha=0.7)
            ax.plot(epochs, self.history['grad_norm_bc'], 'r-', label='||∇L_bc|_2', alpha=0.7)
            ax.plot(epochs, self.history['grad_norm_ic'], 'g-', label='||∇L_ic||_2', alpha=0.7)
            if self.model.inverse:
                ax.plt(epochs, self.history['grad_norm_m'], 'm-', label='||∇L_m||_2', alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Gradient L2 Norm')
            ax.set_title('Loss Landscape')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Select plots
        plots.append(plot_losses)

        if self.adaptive_weights:
            plots.append(plot_weights)

        if self.model.inverse:
            plots.append(plot_alpha)

        if self.track_gradient_norms:
            plots.append(plot_grad_norms)

        # Build grid
        n_plots = len(plots)

        if n_plots == 1:
            nrows, ncols = 1, 1
        elif n_plots == 2:
            nrows, ncols = 1, 2
        else:
            ncols = 2
            nrows = int(np.ceil(n_plots / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows))
        axes = np.atleast_1d(axes).flatten()

        # Draw plots
        for ax, plot_fn in zip(axes, plots):
            plot_fn(ax)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


    def save_checkpoint(self, path: str):
        """
        Save model checkpoint and training history
        """
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


# ============================================================================
# Complete Training Example
# ============================================================================

import sys
sys.path.append('..')

def train_comparison():
    """
    Train with both strong and weak forms for comparison.
    """
    from data.heat_data import HeatEquationData
    from models.heat_pinn_strategy import StrongFormLoss, WeakFormLoss
    from utils.test_functions import generate_compact_gaussians
    
    print("="*70)
    print("STRONG VS WEAK FORM COMPARISON")
    print("="*70)
    
    # Generate base data
    data_gen = HeatEquationData(alpha=0.01)
    base_data = data_gen.generate_full_dataset()
    
    # ========================================================================
    # Train Strong Form
    # ========================================================================
    
    print("\n" + "="*70)
    print("Training Strong Form")
    print("="*70)
    
    model_strong = StrategicPINN(alpha_true=0.01, inverse=False)
    model_strong.set_loss_strategy(StrongFormLoss())
    
    strong_data = {
        'x_f': base_data['x_f'],
        't_f': base_data['t_f'],
        'x_bc': base_data['x_bc'],
        't_bc': base_data['t_bc'],
        'u_bc': base_data['u_bc'],
        'x_ic': base_data['x_ic'],
        't_ic': base_data['t_ic'],
        'u_ic': base_data['u_ic']
    }
    
    trainer_strong = StrategicPINNTrainer(model_strong, strong_data)
    trainer_strong.train(epochs=1000, print_every=500, plot_every=1000)
    
    # ========================================================================
    # Train Weak Form
    # ========================================================================
    
    print("\n" + "="*70)
    print("Training Weak Form")
    print("="*70)
    
    model_weak = StrategicPINN(alpha_true=0.01, inverse=False)
    model_weak.set_loss_strategy(
        WeakFormLoss(integration_method='gauss_legendre', n_integration_points=15)
    )
    
    test_funcs, test_doms = generate_compact_gaussians(
        n_funcs=10,
        support_radius=0.3
    )
    
    weak_data = {
        'test_funcs': test_funcs,
        'test_doms': test_doms,
        'x_bc': base_data['x_bc'],
        't_bc': base_data['t_bc'],
        'u_bc': base_data['u_bc'],
        'x_ic': base_data['x_ic'],
        't_ic': base_data['t_ic'],
        'u_ic': base_data['u_ic']
    }
    
    trainer_weak = StrategicPINNTrainer(model_weak, weak_data)
    trainer_weak.train(epochs=1000, print_every=500, plot_every=1000)
    
    # ========================================================================
    # Compare Results
    # ========================================================================
    
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    from models.heat_pinn import analytical_solution
    
    # Evaluate both models
    x_eval = torch.linspace(0, 1, 100).reshape(-1, 1)
    t_eval = torch.linspace(0, 1, 100).reshape(-1, 1)
    X, T = torch.meshgrid(x_eval.squeeze(), t_eval.squeeze(), indexing='ij')
    
    x_flat = X.flatten().reshape(-1, 1)
    t_flat = T.flatten().reshape(-1, 1)
    
    # Strong form
    u_strong = model_strong.predict(x_flat, t_flat)
    u_exact = analytical_solution(x_flat.numpy(), t_flat.numpy(), alpha=0.01)
    error_strong = np.sqrt(np.mean((u_strong - u_exact)**2))
    rel_error_strong = error_strong / np.sqrt(np.mean(u_exact**2))
    
    # Weak form
    u_weak = model_weak.predict(x_flat, t_flat)
    error_weak = np.sqrt(np.mean((u_weak - u_exact)**2))
    rel_error_weak = error_weak / np.sqrt(np.mean(u_exact**2))
    
    print(f"\nStrong Form:")
    print(f"  Final loss: {trainer_strong.history['total_loss'][-1]:.6e}")
    print(f"  L2 error: {error_strong:.6e}")
    print(f"  Relative L2: {rel_error_strong*100:.4f}%")
    
    print(f"\nWeak Form:")
    print(f"  Final loss: {trainer_weak.history['total_loss'][-1]:.6e}")
    print(f"  L2 error: {error_weak:.6e}")
    print(f"  Relative L2: {rel_error_weak*100:.4f}%")
    
    print(f"\nConclusion:")
    if rel_error_weak < rel_error_strong:
        print(" Weak form achieved lower error.")
    elif abs(rel_error_weak - rel_error_strong) / rel_error_strong < 0.1:
        print(" Both methods achieved similar accuracy.")
    else:
        print("Strong form performed better (expected for smooth problem).")


if __name__ == "__main__":
    train_comparison()