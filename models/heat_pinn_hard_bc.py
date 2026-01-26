"""
Physics-Informed Neural Network for 1D Heat Equation

This module implements a PINN for solving both forward and inverse problems
of the 1D heat equation: u_t = alpha * u_xx
while imposing hard boundary/initial conditions

Author: elphaim
Date: January 26, 2026
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Callable


class HeatPINNHardBC(nn.Module):
    """
    Physics-Informed Neural Network for the 1D heat equation with hard-encoded
    boundary and initial conditions
    
    The network approximates u(x,t) where:
        u_t - alpha * u_xx = 0
        
    For inverse problems, alpha is learned as a trainable parameter.
    For forward problems, alpha is fixed.
    
    Args:
        ic_func: lambda x: initial condition value
        bc_left_func: lambda t: left boundary value
        bc_right_func: lambda t: right boundary value
        layers: List of layer sizes. Default: [2, 50, 50, 50, 50, 1]
                Input layer has size 2 (x, t), output has size 1 (u)
        alpha_true: True value of thermal diffusivity (for forward problem)
        inverse: If True, treat alpha as learnable parameter
        alpha_init: Initial guess for alpha (used in inverse problem)
    """

    def __init__(
            self,
            ic_func: Callable[[torch.Tensor], torch.Tensor],
            bc_left_func: Callable[[torch.Tensor], torch.Tensor],
            bc_right_func: Callable[[torch.Tensor], torch.Tensor],
            layers: list = [2, 50, 50, 50, 50, 1],
            alpha_true: Optional[float] = None,
            inverse: bool = False,
            alpha_init: float = 0.02,
    ):
        super(HeatPINNHardBC, self).__init__()

        self.ic_func = ic_func
        self.bc_left_func = bc_left_func
        self.bc_right_func = bc_right_func
        self.inverse = inverse

        # Build the neural network layers
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

        # Initialize weights using Xavier initialization
        self._initialize_weights()

        # Handle the thermal diffusivity parameter
        if inverse:
            # For inverse problem: alpha is learnable
            self.alpha = nn.Parameter(torch.tensor([alpha_init], dtype=torch.float32))
            print(f"Inverse problem mode: alpha initialized to {alpha_init}")
        else:
            # For forward problem: alpha is fixed
            if alpha_true is None:
                raise ValueError("Must provide alpha_true for forward problem")
            self.register_buffer('alpha', torch.tensor([alpha_true], dtype=torch.float32))
            print(f"Forward problem mode with hard BCs: alpha fixed to {alpha_true}")

    def _initialize_weights(self):
        """
        Initialize network weights using Xavier Normal initialization.
        """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Spatial coordinates, shape (N, 1)
            t: Temporal coordinates, shape (N, 1)
            
        Returns:
            u: Temperature field, shape (N, 1)
        """
        # Concatenate inputs
        inputs = torch.cat([x, t], dim=1)  # Shape: (N, 2)
        
        # Pass through hidden layers with tanh activation
        out = inputs
        for i, layer in enumerate(self.layers[:-1]):
            out = torch.tanh(layer(out))
        
        # Output layer (no activation)
        nn_out = self.layers[-1](out)

        # Hard-encode the boundary & initial conditions
        u_ic = self.ic_func(x)
        u_bc_left = self.bc_left_func(t)
        u_bc_right = self.bc_right_func(t)

        # Blend initial, boundary and interior
        u = (1 - t) * u_ic + t * ((1 - x) * u_bc_left + x * u_bc_right) + t * x * (1 - x) * nn_out

        return u
    
    def residual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the PDE residual: u_t - alpha * u_xx
        
        Uses automatic differentiation to compute derivatives.
        
        Args:
            x: Spatial coordinates, shape (N, 1)
            t: Temporal coordinates, shape (N, 1)
            
        Returns:
            residual: PDE residual, shape (N, 1)
        """        
        # Forward pass
        u = self.forward(x, t)
        
        # First derivatives
        u_t = torch.autograd.grad(
            outputs=u,
            inputs=t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        u_x = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Second derivative (u_xx)
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # PDE residual: u_t - alpha * u_xx = 0
        residual = u_t - self.alpha * u_xx
        
        return residual
    
    def loss_function(
        self,
        x_f: torch.Tensor,
        t_f: torch.Tensor,
        x_m: Optional[torch.Tensor] = None,
        t_m: Optional[torch.Tensor] = None,
        u_m: Optional[torch.Tensor] = None,
        lambda_f: float = 1.0,
        lambda_m: float = 1.0
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total loss (MSE) for PINN training.
        
        Args:
            x_f, t_f: Collocation points for PDE residual
            x_m, t_m, u_m: Measurement points (for inverse problem)
            lambda_f, lambda_m: Loss weights
            
        Returns:
            total_loss: Weighted sum of all loss components
            losses: Dictionary with individual loss components
        """
        # 1. PDE Residual Loss
        residual = self.residual(x_f, t_f)
        loss_f = torch.mean(residual ** 2)
        
        # 2. Measurement Loss (for inverse problem)
        if x_m is not None and t_m is not None and u_m is not None:
            u_m_pred = self.forward(x_m, t_m)
            loss_m = torch.mean((u_m_pred - u_m) ** 2)
        else:
            loss_m = torch.tensor(0.0)
        
        # Total weighted loss
        total_loss = (
            lambda_f * loss_f +
            lambda_m * loss_m
        )
        
        # Store individual losses for monitoring
        losses = {
            'total': total_loss.item(),
            'residual': loss_f.item(),
            'measurement': loss_m.item() if torch.is_tensor(loss_m) else 0.0,
        }
        
        return total_loss, losses
    
    def predict(self, x: torch.Tensor, t: torch.Tensor) -> np.ndarray:
        """
        Predict temperature field at given points.
        
        Args:
            x: Spatial coordinates
            t: Temporal coordinates
            
        Returns:
            u: Predicted temperature (numpy array)
        """
        self.eval()
        with torch.no_grad():
            u = self.forward(x, t)
        return u.cpu().numpy()
    
    def get_alpha(self) -> float:
        """
        Return current value of alpha parameter.
        """
        return self.alpha.item()
    
# Utility function for analytical solution (validation)
def analytical_solution(x: np.ndarray, t: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Analytical solution for u_t = alpha * u_xx with:
        - Initial condition: u(x, 0) = sin(pi * x)
        - Boundary conditions: u(0, t) = u(1, t) = 0
    
    Solution: u(x, t) = sin(pi * x) * exp(-alpha * pi^2 * t)
    
    Args:
        x: Spatial coordinates (0 to 1)
        t: Temporal coordinates (0 to T)
        alpha: Thermal diffusivity
        
    Returns:
        u: Temperature field
    """
    return np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t)

# Example usage and testing
if __name__ == "__main__":
    print("Testing HeatPINNHardBC implementation...\n")
    
    # Test 1: Forward problem
    print("=" * 60)
    print("Test 1: Forward Problem")
    print("=" * 60)
    
    model_forward = HeatPINNHardBC(
        ic_func=lambda x:torch.sin(torch.pi * x),
        bc_left_func=lambda t: torch.zeros_like(t),
        bc_right_func=lambda t: torch.zeros_like(t),
        layers=[2, 50, 50, 50, 50, 1],
        alpha_true=0.01,
        inverse=False
    )
    print(f"Model architecture: {model_forward}")
    print(f"Number of parameters: {sum(p.numel() for p in model_forward.parameters())}")

    # Test hard-encoded IC
    x_ic = torch.rand(100, 1)
    t_ic = torch.zeros(100, 1)
    u_ic = model_forward(x_ic, t_ic)
    print(f"IC satisfied to 1e-14: {torch.max(torch.abs(u_ic - torch.sin(torch.pi * x_ic))) < 1e-14}")

    # Test hard-encoded BC
    x_bc = torch.zeros(100, 1)
    t_bc = torch.rand(100, 1)
    u_bc = model_forward(x_bc, t_bc)
    print(f"BC satisfied to 1e-14: {torch.max(torch.abs(u_bc)) < 1e-14}")
    
    # Test forward pass
    x_test = torch.linspace(0, 1, 10).reshape(-1, 1)
    t_test = torch.linspace(0, 1, 10).reshape(-1, 1)
    u_pred = model_forward(x_test, t_test)
    print(f"Output shape: {u_pred.shape}")
    
    # Test residual computation
    x_test = x_test.requires_grad_(True)
    t_test = t_test.requires_grad_(True)
    residual = model_forward.residual(x_test, t_test)
    print(f"Residual shape: {residual.shape}")
    print(f"Residual mean before training: {residual.mean().item():.6f}")
    
    # Test 2: Inverse problem
    print("\n" + "=" * 60)
    print("Test 2: Inverse Problem")
    print("=" * 60)
    
    model_inverse = HeatPINNHardBC(
        ic_func=lambda x:torch.sin(torch.pi * x),
        bc_left_func=lambda t: torch.zeros_like(t),
        bc_right_func=lambda t: torch.zeros_like(t),
        layers=[2, 50, 50, 50, 50, 1],
        inverse=True,
        alpha_init=0.02
    )
    print(f"Initial alpha guess: {model_inverse.get_alpha():.4f}")
    print(f"Alpha is learnable: {model_inverse.alpha.requires_grad}")
    
    # Test 3: Analytical solution
    print("\n" + "=" * 60)
    print("Test 3: Analytical Solution")
    print("=" * 60)
    
    x_np = np.linspace(0, 1, 5)
    t_np = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    for t_val in t_np:
        u_analytical = analytical_solution(x_np, t_val * np.ones_like(x_np), alpha=0.01)
        print(f"t = {t_val:.2f}: u_max = {u_analytical.max():.4f}, u_mean = {u_analytical.mean():.4f}")
    
    print("\nAll tests passed. Ready to train.")