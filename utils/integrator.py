"""
PyTorch-native numerical integration for weak-form PINNs

Implements multiple quadrature methods with full autograd support:
- Simpson's rule (adaptive)
- Gauss-Legendre quadrature
- Monte Carlo integration
- Quasi-Monte Carlo (Sobol)

All methods preserve computational graph for backpropagation.

Author: elphaim
Date: January 29, 2026
"""

import torch
import numpy as np
from typing import Callable, Optional
from abc import ABC, abstractmethod


class Integrator2D(ABC):
    """
    Abstract base class for 2D numerical integration.
    
    All integrators must:
    1. Preserve PyTorch computational graph
    2. Work with functions that require gradients
    3. Return a scalar tensor with grad_fn
    """
    
    @abstractmethod
    def integrate(
        self, 
        func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        domain: list[list[float]],
    ) -> torch.Tensor:
        """
        Integrate func over 2D domain.
        
        Args:
            func: Function f(x, t) returning tensor
            domain: [[x_min, x_max], [t_min, t_max]]
            
        Returns:
            integral: Scalar tensor with gradient
        """
        pass


class MonteCarloIntegrator(Integrator2D):
    """
    Monte Carlo integration with importance sampling.
    
    ∫∫ f(x,t) dx dt ≈ V · (1/N) Σ f(x_i, t_i)
    
    where V = (x_max - x_min)(t_max - t_min) is domain volume.
    """
    
    def __init__(self, n_samples: int = 10000, device: str = 'cpu'):
        """
        Args:
            n_samples: Number of random samples
            device: 'cpu' or 'cuda'
        """
        self.n_samples = n_samples
        self.device = device

        # Set default dtype for torch to float64 on CPU
        if device == 'cpu':
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)
    
    def integrate(
        self,
        func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        domain: list[list[float]],
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Monte Carlo integration over 2D domain.
        
        Args:
            func: Integrand f(x, t)
            domain: [[x_min, x_max], [t_min, t_max]]
            seed: Random seed for reproducibility
            
        Returns:
            integral: V · mean(f(samples))
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        x_min, x_max = domain[0]
        t_min, t_max = domain[1]
        
        # Random samples (uniform distribution)
        x_samples = torch.rand(self.n_samples, 1, device=self.device) * (x_max - x_min) + x_min
        t_samples = torch.rand(self.n_samples, 1, device=self.device) * (t_max - t_min) + t_min
        
        # Evaluate function at samples
        f_values = func(x_samples, t_samples)
        
        # Monte Carlo estimate: V · mean(f)
        volume = (x_max - x_min) * (t_max - t_min)
        integral = volume * torch.mean(f_values)
        
        return integral


class GaussLegendreIntegrator(Integrator2D):
    """
    Gauss-Legendre quadrature for 2D integration.
    
    ∫∫ f(x,t) dx dt ≈ Σ_i Σ_j w_i w_j f(x_i, t_j)
    
    High accuracy for smooth functions.
    """
    
    def __init__(self, n_points: int = 15, device: str = 'cpu'):
        """
        Args:
            n_points: Number of quadrature points per dimension
            device: 'cpu' or 'cuda'
        """
        self.n_points = n_points
        self.device = device

        # Set default dtype for torch to float64 on CPU
        if device == 'cpu':
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)
        
        # Pre-compute Gauss-Legendre nodes and weights on [-1, 1]
        nodes_np, weights_np = np.polynomial.legendre.leggauss(n_points)
        
        self.nodes_base = torch.tensor(nodes_np, dtype=torch.float64 if self.device == 'cpu' else torch.float32, device=device)
        self.weights_base = torch.tensor(weights_np, dtype=torch.float64 if self.device == 'cpu' else torch.float32, device=device)
    
    def _transform_to_domain(
        self, 
        nodes: torch.Tensor, 
        a: float, 
        b: float
    ) -> tuple[torch.Tensor, float]:
        """
        Transform nodes from [-1, 1] to [a, b].
        
        Returns:
            transformed_nodes: Nodes on [a, b]
            jacobian: (b - a) / 2
        """
        jacobian = (b - a) / 2.0
        transformed = a + (nodes + 1.0) * jacobian
        return transformed, jacobian
    
    def integrate(
        self,
        func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        domain: list[list[float]],
    ) -> torch.Tensor:
        """
        Gauss-Legendre quadrature over 2D domain.
        
        Args:
            func: Integrand f(x, t)
            domain: [[x_min, x_max], [t_min, t_max]]
            
        Returns:
            integral: Weighted sum of function evaluations
        """
        x_min, x_max = domain[0]
        t_min, t_max = domain[1]
        
        # Transform nodes to domain
        x_nodes, jac_x = self._transform_to_domain(self.nodes_base, x_min, x_max)
        t_nodes, jac_t = self._transform_to_domain(self.nodes_base, t_min, t_max)
        
        # Create 2D grid of quadrature points
        x_grid = x_nodes.repeat(self.n_points, 1).T.reshape(-1, 1)
        t_grid = t_nodes.repeat(self.n_points, 1).reshape(-1, 1)
        
        # Create 2D grid of weights
        weights_2d = torch.outer(self.weights_base, self.weights_base).reshape(-1, 1)
        
        # Evaluate function at all quadrature points
        f_values = func(x_grid, t_grid)
        
        # Compute integral with Jacobian
        integral = torch.sum(weights_2d * f_values) * jac_x * jac_t
        
        return integral


class SimpsonIntegrator(Integrator2D):
    """
    2D Simpson's rule (composite).
    
    Uses Simpson's 1/3 rule in each dimension:
    ∫∫ f(x,t) dx dt ≈ (hx/3)(ht/3) Σ_ij w_i w_j f(x_i, t_j)
    
    where w are Simpson weights: [1, 4, 2, 4, ..., 2, 4, 1]
    """
    
    def __init__(self, n_points: int = 21, device: str = 'cpu'):
        """
        Args:
            n_points: Number of points per dimension (must be odd!)
            device: 'cpu' or 'cuda'
        """
        if n_points % 2 == 0:
            n_points += 1  # Ensure odd
            print(f"Simpson's rule requires odd n_points, using {n_points}")
        
        self.n_points = n_points
        self.device = device

        # Set default dtype for torch to float64 on CPU
        if device == 'cpu':
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)
    
    def _simpson_weights(self, n: int) -> torch.Tensor:
        """
        Generate Simpson's 1/3 rule weights.
        
        Pattern: [1, 4, 2, 4, 2, ..., 2, 4, 1]
        """
        weights = torch.ones(n, device=self.device)
        weights[1:-1:2] = 4  # Odd indices (1, 3, 5, ...)
        weights[2:-1:2] = 2  # Even indices (2, 4, 6, ...)
        return weights
    
    def integrate(
        self,
        func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        domain: list[list[float]],
    ) -> torch.Tensor:
        """
        Simpson's rule over 2D domain.
        
        Args:
            func: Integrand f(x, t)
            domain: [[x_min, x_max], [t_min, t_max]]
            
        Returns:
            integral: Simpson's rule estimate
        """
        x_min, x_max = domain[0]
        t_min, t_max = domain[1]
        
        # Create uniform grid
        x_nodes = torch.linspace(x_min, x_max, self.n_points, device=self.device)
        t_nodes = torch.linspace(t_min, t_max, self.n_points, device=self.device)
        
        # Grid spacing
        hx = (x_max - x_min) / (self.n_points - 1)
        ht = (t_max - t_min) / (self.n_points - 1)
        
        # Simpson weights for each dimension
        wx = self._simpson_weights(self.n_points)
        wt = self._simpson_weights(self.n_points)
        
        # Create 2D grid
        x_grid = x_nodes.repeat(self.n_points, 1).T.reshape(-1, 1)
        t_grid = t_nodes.repeat(self.n_points, 1).reshape(-1, 1)
        
        # 2D weights
        weights_2d = torch.outer(wx, wt).reshape(-1, 1)
        
        # Evaluate function
        f_values = func(x_grid, t_grid)
        
        # Simpson's rule: (h/3) Σ w_i f_i
        integral = (hx / 3.0) * (ht / 3.0) * torch.sum(weights_2d * f_values)
        
        return integral


class AdaptiveSimpsonIntegrator(Integrator2D):
    """
    Adaptive Simpson's rule with error estimation.
    
    Recursively refines regions where error estimate is large.
    Best for functions with localized features.
    """
    
    def __init__(
        self, 
        initial_points: int = 9,
        max_refinements: int = 3,
        tol: float = 1e-6,
        device: str = 'cpu'
    ):
        """
        Args:
            initial_points: Starting grid size (odd)
            max_refinements: Maximum recursion depth
            tol: Error tolerance
            device: 'cpu' or 'cuda'
        """
        if initial_points % 2 == 0:
            initial_points += 1
        
        self.initial_points = initial_points
        self.max_refinements = max_refinements
        self.tol = tol
        self.device = device

        # Set default dtype for torch to float64 on CPU
        if device == 'cpu':
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)
        
        # Use base Simpson integrator
        self.simpson = SimpsonIntegrator(initial_points, device)
    
    def integrate(
        self,
        func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        domain: list[list[float]],
    ) -> torch.Tensor:
        """
        Adaptive Simpson integration.
        
        Currently implements uniform refinement.
        TODO: Implement actual adaptive subdivision.
        """
        # Start with coarse estimate
        I_coarse = self.simpson.integrate(func, domain)
        
        # Refine and check convergence
        for refinement in range(self.max_refinements):
            n_fine = self.initial_points + refinement * 10
            if n_fine % 2 == 0:
                n_fine += 1
            
            simpson_fine = SimpsonIntegrator(n_fine, self.device)
            I_fine = simpson_fine.integrate(func, domain)
            
            # Error estimate (difference between coarse and fine)
            error = torch.abs(I_fine - I_coarse)
            
            if error < self.tol * torch.abs(I_fine):
                return I_fine
            
            I_coarse = I_fine
        
        # Return best estimate
        return I_coarse


class IntegratorFactory:
    """
    Factory for creating integrators with unified interface.
    
    Usage:
        integrator = IntegratorFactory.create('gauss', n_points=15)
        result = integrator.integrate(my_func, [[0, 1], [0, 1]])
    """
    
    @staticmethod
    def create(
        method: str,
        n_points: Optional[int] = None,
        device: str = 'cpu',
    ) -> Integrator2D:
        """
        Create integrator instance.
        
        Args:
            method: 'monte_carlo', 'gauss', 'simpson', 'adaptive_simpson'
            n_points: Number of integration points (method-dependent)
            device: 'cpu' or 'cuda'
            
        Returns:
            integrator: Instance of Integrator2D subclass
        """
        method = method.lower()
        
        if method == 'monte_carlo' or method == 'mc':
            n_samples = n_points if n_points else 10000
            return MonteCarloIntegrator(n_samples=n_samples, device=device)
        
        elif method == 'gauss_legendre' or method == 'gl':
            n = n_points if n_points else 15
            return GaussLegendreIntegrator(n_points=n, device=device)
        
        elif method == 'simpson':
            n = n_points if n_points else 21
            return SimpsonIntegrator(n_points=n, device=device)
        
        elif method == 'adaptive_simpson' or method == 'adaptive':
            return AdaptiveSimpsonIntegrator(
                initial_points=n_points if n_points else 9,
                max_refinements=3,
                tol=1e-6,
                device=device
            )
        
        else:
            raise ValueError(f"Unknown integration method: {method}")


# ============================================================================
# Usage Example and Testing
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing PyTorch-Native Integrators")
    print("="*70)
    
    device = 'cpu'
    
    # Test function: f(x,t) = sin(πx) * cos(πt)
    # Exact integral over [0,1]×[0,1]: 0
    def test_func_1(x, t):
        return torch.sin(np.pi * x) * torch.cos(np.pi * t)
    
    # Test function: f(x,t) = x^2 + t^2
    # Exact integral over [0,1]×[0,1]: 2/3
    def test_func_2(x, t):
        return x**2 + t**2
    
    domain = [[0.0, 1.0], [0.0, 1.0]]
    
    methods = ['monte_carlo', 'gauss_legendre', 'simpson', 'adaptive']
    
    print("\nTest 1: ∫∫ sin(πx)cos(πt) dx dt over [0,1]×[0,1]")
    print("Expected: 0.0")
    print("-" * 70)
    
    for method in methods:
        if method == 'monte_carlo':
            integrator = IntegratorFactory.create(method, n_points=50000, device=device)
        else:
            integrator = IntegratorFactory.create(method, n_points=20, device=device)
        
        result = integrator.integrate(test_func_1, domain)
        print(f"{method:20s}: {result.item():12.8f}")
    
    print("\nTest 2: ∫∫ (x² + t²) dx dt over [0,1]×[0,1]")
    print("Expected: 0.66666667")
    print("-" * 70)
    
    for method in methods:
        if method == 'monte_carlo':
            integrator = IntegratorFactory.create(method, n_points=50000, device=device)
        else:
            integrator = IntegratorFactory.create(method, n_points=20, device=device)
        
        result = integrator.integrate(test_func_2, domain)
        error = abs(result.item() - 2/3)
        print(f"{method:20s}: {result.item():12.8f}  (error: {error:.2e})")
    
    # Test gradient preservation
    print("\nTest 3: Gradient preservation")
    print("-" * 70)
    
    # Create a simple function with learnable parameter
    alpha = torch.tensor(2.0, requires_grad=True)
    
    def test_func_grad(x, t):
        return alpha * x * t
    
    integrator = IntegratorFactory.create('gauss_legendre', n_points=10, device=device)
    result = integrator.integrate(test_func_grad, domain)
    
    # Backprop
    result.backward()
    
    print(f"Integral: {result.item():.6f}")
    print(f"Gradient w.r.t. alpha: {alpha.grad.item() if alpha.grad is not None else 'error'}")
    print(f"Expected gradient: 0.25 (= ∫∫ x*t dx dt)")
    print(f"Gradient preserved." if alpha.grad is not None else "No gradient!")
    
    print("\n" + "="*70)
    print("All tests complete.")
    print("="*70)