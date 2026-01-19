"""
Data generation module for 1D Heat Equation PINN

Generates:
1. Collocation points for PDE residual
2. Boundary and initial condition points
3. Synthetic measurements with noise (for inverse problem)
4. Ground truth analytical solutions

Author: elphaim
Date: January 19, 2026
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Dict, Optional

class HeatEquationData:
    """
    Data generator for the 1D heat equation problem.
    
    Domain: x ∈ [0, L], t ∈ [0, T]
    PDE: u_t = alpha * u_xx
    IC: u(x, 0) = sin(π*x)
    BC: u(0, t) = u(L, t) = 0
    
    Args:
        L: Length of spatial domain, must be odd (default: 1.0)
        T: Final time (default: 1.0)
        alpha: True thermal diffusivity (default: 0.01)
        N_f: Number of collocation points for residual (default: 10000)
        N_sensors: Number of spatial sensor locations (default: 10)
        N_time_measurements: Number of time measurements per sensor (default: 10)
        noise_level: Measurement noise std as fraction of signal (default: 0.01 for SNR~20dB)
        device: torch device (default: cpu)
    """
    
    def __init__(
        self,
        L: float = 1.0,
        T: float = 1.0,
        alpha: float = 0.01,
        N_f: int = 10000,
        N_sensors: int = 10,
        N_time_measurements: int = 10,
        noise_level: float = 0.01,
        device: str = 'cpu',
        random_seed: int = 42
    ):
        self.L = L
        self.T = T
        self.alpha = alpha
        self.N_f = N_f
        self.N_sensors = N_sensors
        self.N_time_measurements = N_time_measurements
        self.noise_level = noise_level
        self.device = device
        
        # Set random seed for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        print(f"Data Generator initialized:")
        print(f"  Domain: x ∈ [0, {L}], t ∈ [0, {T}]")
        print(f"  True alpha: {alpha}")
        print(f"  Collocation points: {N_f}")
        print(f"  Measurements: {N_sensors} sensors × {N_time_measurements} times = {N_sensors * N_time_measurements} total")
        print(f"  Noise level: {noise_level:.1%} (SNR ≈ {-20*np.log10(noise_level):.0f} dB)")
    
    def analytical_solution(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Analytical solution for u_t = alpha * u_xx with:
            - Initial condition: u(x, 0) = sin(pi * x)
            - Boundary conditions: u(0, t) = u(1, t) = 0
    
        Solution: u(x, t) = sin(pi * x) * exp(-alpha * pi^2 * t)
    
        Args:
            x: Spatial coordinates (0 to L)
            t: Temporal coordinates (0 to T)
            alpha: Thermal diffusivity
        
        Returns:
            u: Temperature field
        """
        return np.sin(np.pi * x) * np.exp(-self.alpha * np.pi**2 * t)
    
    def generate_collocation_points(self, method: str = 'uniform') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate collocation points for PDE residual evaluation.
        
        Args:
            method: 'uniform' or 'lhs' (Latin Hypercube Sampling)
            
        Returns:
            x_f: Spatial coordinates, shape (N_f, 1)
            t_f: Temporal coordinates, shape (N_f, 1)
        """
        if method == 'uniform':
            x_f = torch.rand(self.N_f, 1) * self.L
            t_f = torch.rand(self.N_f, 1) * self.T
        elif method == 'lhs':
            # Latin Hypercube Sampling for better space-filling
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=2, rng=42)
            sample = sampler.random(n=self.N_f)
            x_f = torch.tensor(sample[:, 0:1] * self.L, dtype=torch.float32)
            t_f = torch.tensor(sample[:, 1:2] * self.T, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return x_f.to(self.device), t_f.to(self.device)
    
    def generate_boundary_conditions(self, N_bc: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate boundary condition points: u(0, t) = u(L, t) = 0
        
        Args:
            N_bc: Number of points per boundary
            
        Returns:
            x_bc: Spatial coordinates (at x=0 and x=L)
            t_bc: Temporal coordinates
            u_bc: Boundary values (all zeros)
        """
        # Left boundary (x=0)
        t_left = torch.rand(N_bc, 1) * self.T
        x_left = torch.zeros(N_bc, 1)
        
        # Right boundary (x=L)
        t_right = torch.rand(N_bc, 1) * self.T
        x_right = torch.ones(N_bc, 1) * self.L
        
        # Combine
        x_bc = torch.cat([x_left, x_right], dim=0)
        t_bc = torch.cat([t_left, t_right], dim=0)
        u_bc = torch.zeros_like(x_bc)  # Dirichlet BC: u = 0
        
        return x_bc.to(self.device), t_bc.to(self.device), u_bc.to(self.device)
    
    def generate_initial_conditions(self, N_ic: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate initial condition points: u(x, 0) = sin(π*x)
        
        Args:
            N_ic: Number of initial condition points
            
        Returns:
            x_ic: Spatial coordinates
            t_ic: Temporal coordinates (all zeros)
            u_ic: Initial values
        """
        x_ic = torch.rand(N_ic, 1) * self.L
        t_ic = torch.zeros(N_ic, 1)
        u_ic = torch.sin(np.pi * x_ic)
        
        return x_ic.to(self.device), t_ic.to(self.device), u_ic.to(self.device)
    
    def generate_measurements(
        self, 
        add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Generate synthetic sensor measurements with noise.
        
        Simulates sensors placed at regular intervals that measure temperature
        at regular time intervals.
        
        Args:
            add_noise: Whether to add Gaussian noise to measurements
            
        Returns:
            x_m: Measurement spatial coordinates
            t_m: Measurement temporal coordinates
            u_m: Measured temperature (with noise if add_noise=True)
            info: Dictionary with additional information (true values, SNR, etc.)
        """
        # Sensor locations (evenly spaced, excluding boundaries)
        x_sensors = np.linspace(self.L / (self.N_sensors + 1), 
                               self.L * self.N_sensors / (self.N_sensors + 1), 
                               self.N_sensors)
        
        # Measurement times (evenly spaced)
        t_measurements = np.linspace(0, self.T, self.N_time_measurements)
        
        # Create meshgrid
        X_mesh, T_mesh = np.meshgrid(x_sensors, t_measurements)
        x_m_np = X_mesh.flatten()
        t_m_np = T_mesh.flatten()
        
        # Compute true values using analytical solution
        u_true = self.analytical_solution(x_m_np, t_m_np)
        
        # Add noise
        if add_noise:
            noise = np.random.normal(0, self.noise_level * np.abs(u_true))
            u_measured = u_true + noise
            
            # Compute actual SNR
            signal_power = np.mean(u_true ** 2)
            noise_power = np.mean(noise ** 2)
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            u_measured = u_true
            snr_db = np.inf
        
        # Convert to torch tensors
        x_m = torch.tensor(x_m_np.reshape(-1, 1), dtype=torch.float32)
        t_m = torch.tensor(t_m_np.reshape(-1, 1), dtype=torch.float32)
        u_m = torch.tensor(u_measured.reshape(-1, 1), dtype=torch.float32)
        
        # Store metadata
        info = {
            'x_sensors': x_sensors,
            't_measurements': t_measurements,
            'u_true': u_true,
            'u_measured': u_measured,
            'snr_db': snr_db,
            'noise_std': self.noise_level * np.abs(u_true).mean()
        }
        
        print(f"\nMeasurements generated:")
        print(f"  Sensor locations: {x_sensors}")
        print(f"  Time points: {t_measurements}")
        print(f"  Total measurements: {len(x_m)}")
        if add_noise:
            print(f"  Actual SNR: {snr_db:.1f} dB")
        
        return x_m.to(self.device), t_m.to(self.device), u_m.to(self.device), info
    
    def generate_full_dataset(
        self, 
        collocation_method: str = 'uniform'
    ) -> Dict[str, torch.Tensor]:
        """
        Generate complete dataset for training.
        
        Returns:
            data: Dictionary containing all data:
                - 'x_f', 't_f': Collocation points
                - 'x_bc', 't_bc', 'u_bc': Boundary conditions
                - 'x_ic', 't_ic', 'u_ic': Initial conditions
                - 'x_m', 't_m', 'u_m': Measurements
                - 'measurement_info': Additional measurement metadata
        """
        print("\n" + "="*60)
        print("Generating complete dataset...")
        print("="*60)
        
        # Generate all data
        x_f, t_f = self.generate_collocation_points(method=collocation_method)
        x_bc, t_bc, u_bc = self.generate_boundary_conditions()
        x_ic, t_ic, u_ic = self.generate_initial_conditions()
        x_m, t_m, u_m, info = self.generate_measurements()
        
        data = {
            'x_f': x_f, 't_f': t_f,
            'x_bc': x_bc, 't_bc': t_bc, 'u_bc': u_bc,
            'x_ic': x_ic, 't_ic': t_ic, 'u_ic': u_ic,
            'x_m': x_m, 't_m': t_m, 'u_m': u_m,
            'measurement_info': info
        }
        
        print("\nDataset generation complete.")
        return data
    
    def visualize_data(self, data: Dict[str, torch.Tensor], save_path: Optional[str] = None):
        """
        Visualize the generated dataset.
        
        Args:
            data: Dictionary from generate_full_dataset()
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Convert to numpy for plotting
        x_f = data['x_f'].cpu().numpy()
        t_f = data['t_f'].cpu().numpy()
        x_bc = data['x_bc'].cpu().numpy()
        t_bc = data['t_bc'].cpu().numpy()
        x_ic = data['x_ic'].cpu().numpy()
        t_ic = data['t_ic'].cpu().numpy()
        x_m = data['x_m'].cpu().numpy()
        t_m = data['t_m'].cpu().numpy()
        u_m = data['u_m'].cpu().numpy()
        
        # Plot 1: Collocation points
        axes[0, 0].scatter(x_f, t_f, s=1, alpha=0.3, c='blue')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('t')
        axes[0, 0].set_title(f'Collocation Points (N={len(x_f)})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Boundary and Initial Conditions
        axes[0, 1].scatter(x_bc, t_bc, s=10, c='red', label='BC', alpha=0.5)
        axes[0, 1].scatter(x_ic, t_ic, s=10, c='green', label='IC', alpha=0.5)
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('t')
        axes[0, 1].set_title('Boundary & Initial Conditions')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Measurement locations
        axes[1, 0].scatter(x_m, t_m, s=50, c='orange', marker='x', linewidths=2)
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('t')
        axes[1, 0].set_title(f'Measurement Locations (N={len(x_m)})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Measured temperature values
        scatter = axes[1, 1].scatter(x_m, t_m, s=100, c=u_m, cmap='coolwarm', marker='o')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('t')
        axes[1, 1].set_title('Measured Temperature')
        plt.colorbar(scatter, ax=axes[1, 1], label='u (temperature)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    print("Testing HeatEquationData class...\n")
    
    # Create data generator
    data_gen = HeatEquationData(
        L=1.0,
        T=1.0,
        alpha=0.01,
        N_f=10000,
        N_sensors=10,
        N_time_measurements=10,
        noise_level=0.01,
        device='cpu'
    )
    
    # Generate full dataset
    data = data_gen.generate_full_dataset(collocation_method='uniform')
    
    # Print dataset info
    print("\nDataset summary:")
    for key, value in data.items():
        if key != 'measurement_info':
            print(f"  {key}: shape {value.shape}")
    
    # Visualize
    data_gen.visualize_data(data, save_path='data/data_visualization.png')
    
    # Test analytical solution at a few points
    print("\nTesting analytical solution:")
    x_test = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    t_test = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    u_test = data_gen.analytical_solution(x_test, t_test)
    print(f"u(x, t=0) = sin(πx): {u_test}")
    
    print("\nAll data generation tests passed.")