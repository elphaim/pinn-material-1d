"""
Strategy Pattern for PINN Loss Computation

Separates loss computation strategy from model architecture.
Useful since e.g. Strong/Weak PINNs use different losses.

Author: elphaim
Date: January 29, 2026
"""

import torch
from abc import ABC, abstractmethod
from typing import Callable

from models.heat_pinn import HeatPINN
from utils.integrator import IntegratorFactory


# ============================================================================
# Abstract Strategy
# ============================================================================

class LossStrategy(ABC):
    """
    Abstract strategy for computing PINN loss.
    
    Different strategies for different formulations:
    - Strong form (point-wise PDE residual)
    - Weak form (integrated residual)
    - Hybrid approaches
    """
    
    @abstractmethod
    def compute_loss(
        self,
        model: HeatPINN,
        data: dict,
        lambdas: dict[str, float]
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss given model and data.
        
        Args:
            model: The PINN model
            data: Dictionary with all required data
            lambdas: Loss weights
            
        Returns:
            total_loss: Scalar loss tensor
            losses: Dictionary with loss components
        """
        pass


# ============================================================================
# Concrete Strategies
# ============================================================================

class StrongFormLoss(LossStrategy):
    """
    Strong-form loss: point-wise PDE residual evaluation.
    
    Required data keys:
    - x_f, t_f: Collocation points
    - x_bc, t_bc, u_bc: Boundary conditions
    - x_ic, t_ic, u_ic: Initial conditions
    - x_m, t_m, u_m: Measurements (for inverse problem)
    """

    def __init__(
        self,
        device: str = 'cpu'
    ):
        
        # Set default dtype for torch to float64 on CPU
        if device == 'cpu':
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)
    
    def compute_loss(
        self,
        model: HeatPINN,
        data: dict,
        lambdas: dict[str, float]
    ) -> tuple[torch.Tensor, dict]:
        """
        Strong-form loss computation for HeatPINN
        """
        
        # Extract data
        x_f = data['x_f'].requires_grad_(True)
        t_f = data['t_f'].requires_grad_(True)
        
        # 1. PDE Residual loss
        residual = model.residual(x_f, t_f)
        loss_f = torch.mean(residual ** 2)
        
        # 2. Boundary loss
        u_bc_pred = model.forward(data['x_bc'], data['t_bc'])
        loss_bc = torch.mean((u_bc_pred - data['u_bc']) ** 2)
        
        # 3. Initial loss
        u_ic_pred = model.forward(data['x_ic'], data['t_ic'])
        loss_ic = torch.mean((u_ic_pred - data['u_ic']) ** 2)
        
        # 4. Measurement loss (for inverse problem)
        if 'x_m' in data and data['x_m'] is not None:
            u_m_pred = model.forward(data['x_m'], data['t_m'])
            loss_m = torch.mean((u_m_pred - data['u_m']) ** 2)
        else:
            loss_m = torch.tensor(0.0)
        
        # Total loss
        total_loss = (
            lambdas.get('f', 1.0) * loss_f +
            lambdas.get('bc', 1.0) * loss_bc +
            lambdas.get('ic', 1.0) * loss_ic +
            lambdas.get('m', 1.0) * loss_m
        )
        
        losses = {
            'total': total_loss.item(),
            'residual': loss_f.item(),
            'boundary': loss_bc.item(),
            'initial': loss_ic.item(),
            'measurement': loss_m.item() if torch.is_tensor(loss_m) else 0.0,
            # also record full tensors for loss gradient tracking
            'total_t': total_loss,
            'residual_t': loss_f,
            'boundary_t': loss_bc,
            'initial_t': loss_ic,
            'measurement_t': loss_m
        }
        
        return total_loss, losses


class WeakFormLoss(LossStrategy):
    """
    Weak-form loss: integrated PDE residual.
    
    Required data keys:
    - test_funcs: List of test functions
    - test_doms: List of integration domains
    - x_bc, t_bc, u_bc: Boundary conditions
    - x_ic, t_ic, u_ic: Initial conditions
    - x_m, t_m, u_m: Measurements (optional)
    """
    
    def __init__(
        self,
        integration_method: str = 'gauss_legendre',
        n_integration_points: int = 15,
        device: str = 'cpu'
    ):
        """
        Args:
            integration_method: 'gauss_legendre', 'simpson', 'monte_carlo', 'adaptive'
            n_integration_points: Number of quadrature points
            device: 'cpu' or 'cuda'
        """

        # Set default dtype for torch to float64 on CPU
        if device == 'cpu':
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

        self.integrator = IntegratorFactory.create(
            method=integration_method,
            n_points=n_integration_points,
            device=device
        )
        self.integration_method = integration_method
        self.n_points = n_integration_points
        
        print(f"Weak-form loss strategy:")
        print(f"  Method: {integration_method}")
        print(f"  Points: {n_integration_points}")
    
    def _compute_weak_residual(
        self,
        model: HeatPINN,
        test_func: Callable,
        domain: list[list[float]]
    ) -> torch.Tensor:
        """
        Compute weak residual for one test function
        """

        def integrand(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """
            Weak form integrand: u·φ_t - α·u_x·φ_x
            """
            if not x.requires_grad:
                x = x.requires_grad_(True)
            if not t.requires_grad:
                t = t.requires_grad_(True)
            
            # Test function
            phi = test_func(x, t)
            
            phi_x = torch.autograd.grad(
                phi, x, grad_outputs=torch.ones_like(phi),
                create_graph=True, retain_graph=True
            )[0]
            
            phi_t = torch.autograd.grad(
                phi, t, grad_outputs=torch.ones_like(phi),
                create_graph=True, retain_graph=True
            )[0]
            
            # Solution
            u = model.forward(x, t)
            
            u_x = torch.autograd.grad(
                u, x, grad_outputs=torch.ones_like(u),
                create_graph=True, retain_graph=True
            )[0]
            
            # Weak form
            return u * phi_t - model.alpha * u_x * phi_x
        
        return self.integrator.integrate(integrand, domain)
    
    def compute_loss(
        self,
        model: HeatPINN,
        data: dict,
        lambdas: dict[str, float]
    ) -> tuple[torch.Tensor, dict]:
        """
        Weak-form loss computation
        """
        
        # Extract test functions and domains
        test_funcs = data['test_funcs']
        test_doms = data['test_doms']
        
        # 1. Weak-form residuals
        weak_residuals = []
        for phi_func, domain in zip(test_funcs, test_doms):
            try:
                weak_res = self._compute_weak_residual(model, phi_func, domain)
                weak_residuals.append(weak_res)
            except Exception as e:
                print(f"Warning: Integration failed: {e}")
                weak_residuals.append(torch.tensor(0.0))
        
        weak_residuals_tensor = torch.stack(weak_residuals)
        loss_f = torch.mean(weak_residuals_tensor ** 2)
        
        # 2-4. Boundary, initial, measurement (same as strong form)
        u_bc_pred = model.forward(data['x_bc'], data['t_bc'])
        loss_bc = torch.mean((u_bc_pred - data['u_bc']) ** 2)
        
        u_ic_pred = model.forward(data['x_ic'], data['t_ic'])
        loss_ic = torch.mean((u_ic_pred - data['u_ic']) ** 2)
        
        if 'x_m' in data and data['x_m'] is not None:
            u_m_pred = model.forward(data['x_m'], data['t_m'])
            loss_m = torch.mean((u_m_pred - data['u_m']) ** 2)
        else:
            loss_m = torch.tensor(0.0)
        
        # Total loss
        total_loss = (
            lambdas.get('f', 1.0) * loss_f +
            lambdas.get('bc', 1.0) * loss_bc +
            lambdas.get('ic', 1.0) * loss_ic +
            lambdas.get('m', 1.0) * loss_m
        )
        
        # Diagnostics
        nonzero = (weak_residuals_tensor.abs() > 1e-8).sum().item()
        
        losses = {
            'total': total_loss.item(),
            'residual': loss_f.item(),
            'boundary': loss_bc.item(),
            'initial': loss_ic.item(),
            'measurement': loss_m.item() if torch.is_tensor(loss_m) else 0.0,
            'weak_res_nonzero': nonzero,
            'weak_res_mean': weak_residuals_tensor.mean().item(),
            'weak_res_std': weak_residuals_tensor.std().item(),
            # also record full tensors for loss gradient tracking
            'total_t': total_loss,
            'residual_t': loss_f,
            'boundary_t': loss_bc,
            'initial_t': loss_ic,
            'measurement_t': loss_m
        }
        
        return total_loss, losses


# ============================================================================
# Enhanced PINN with Strategy Pattern
# ============================================================================

class StrategicPINN(HeatPINN):
    """
    PINN with pluggable loss computation strategy.
    
    Usage:
        # Strong form
        model = StrategicPINN(alpha_true=0.01)
        model.set_loss_strategy(StrongFormLoss())
        
        # Weak form
        model = StrategicPINN(alpha_true=0.01)
        model.set_loss_strategy(WeakFormLoss('gauss_legendre', 15))
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Default to strong form
        self.loss_strategy = StrongFormLoss()
        print("StrategicPINN initialized with StrongFormLoss (default)")
    
    def set_loss_strategy(self, strategy: LossStrategy):
        """
        Set the loss computation strategy.
        
        Args:
            strategy: Instance of LossStrategy subclass
        """
        self.loss_strategy = strategy
        print(f"Loss strategy changed to {strategy.__class__.__name__}")
    
    def compute_loss(
        self,
        data: dict,
        lambda_f: float = 1.0,
        lambda_bc: float = 1.0,
        lambda_ic: float = 1.0,
        lambda_m: float = 1.0
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss using current strategy.
        
        Args:
            data: Dictionary with all required data
                  (contents depend on strategy)
            lambda_*: Loss weights
            
        Returns:
            total_loss: Scalar loss
            losses: Dictionary with components
        """
        lambdas = {
            'f': lambda_f,
            'bc': lambda_bc,
            'ic': lambda_ic,
            'm': lambda_m
        }
        
        return self.loss_strategy.compute_loss(self, data, lambdas)


# ============================================================================
# Usage Example
# ============================================================================

import sys
sys.path.append('..')

def example_usage():
    """Demonstrate strategy pattern."""
    print("="*70)
    print("Strategy Pattern Example")
    print("="*70)
    
    # Create model
    model = StrategicPINN(
        layers=[2, 50, 50, 50, 50, 1],
        alpha_true=0.01,
        inverse=False
    )
    
    # Generate data
    from data.heat_data import HeatEquationData
    
    data_gen = HeatEquationData()
    base_data = data_gen.generate_full_dataset()
    
    print("\n" + "="*70)
    print("Test 1: Strong Form")
    print("="*70)
    
    # Use strong form
    model.set_loss_strategy(StrongFormLoss())
    
    # Prepare data for strong form
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
    
    loss_strong, losses_strong = model.compute_loss(strong_data)
    print(f"\nStrong form loss: {loss_strong.item():.6e}")
    print(f"  Residual: {losses_strong['residual']:.6e}")
    print(f"  Boundary: {losses_strong['boundary']:.6e}")
    
    print("\n" + "="*70)
    print("Test 2: Weak Form")
    print("="*70)
    
    # Switch to weak form
    model.set_loss_strategy(
        WeakFormLoss(
            integration_method='gauss_legendre',
            n_integration_points=15,
            device='cpu'
        )
    )
    
    # Prepare data for weak form
    from utils.test_functions import generate_compact_gaussians
    
    test_funcs, test_doms = generate_compact_gaussians(
        n_funcs=5,
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
    
    loss_weak, losses_weak = model.compute_loss(weak_data)
    print(f"\nWeak form loss: {loss_weak.item():.6e}")
    print(f"  Residual: {losses_weak['residual']:.6e}")
    print(f"  Non-zero: {losses_weak['weak_res_nonzero']}/5")
    
    # Test backprop
    print("\n" + "="*70)
    print("Test 3: Gradient Flow")
    print("="*70)
    
    loss_weak.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"Gradient norms: min={min(grad_norms):.2e}, max={max(grad_norms):.2e}")
    print("Gradients flow correctly.")
    
    print("\n" + "="*70)
    print("Strategy Pattern Working.")
    print("="*70)


if __name__ == "__main__":
    example_usage()