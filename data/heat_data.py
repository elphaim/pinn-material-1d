"""
Data generation module for 1D Heat Equation PINN

Generates:
1. Collocation points for PDE residual
2. Boundary and initial condition points
3. Synthetic measurements with noise (for inverse problem)
4. Ground truth analytical solutions
5. Low-fidelity data based on finite-difference solution

Author: elphaim
Date: January 19, 2026
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional


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
        N_bc: Number of collocation points for boundary conditions (default: 100)
        N_ic: Number of collocation points for initial conditions (default: 100)
        N_sensors: Number of spatial sensor locations (default: 10)
        N_time_measurements: Number of time measurements per sensor (default: 10)
        noise_level: Measurement noise std as fraction of signal (default: 0.01 for SNR~40dB)
        device: torch device (default: cpu)
    """
    
    def __init__(
        self,
        L: float = 1.0,
        T: float = 1.0,
        alpha: float = 0.01,
        N_f: int = 10000,
        N_bc: int = 100,
        N_ic: int = 100,
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
        self.N_bc = N_bc
        self.N_ic = N_ic
        self.N_sensors = N_sensors
        self.N_time_measurements = N_time_measurements
        self.noise_level = noise_level
        self.device = device
        
        # Set random seed for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # Set default dtype for torch to float64 on CPU
        if device == 'cpu':
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)
        
        print(f"Data Generator initialized:")
        print(f"  Domain: x ∈ [0, {L}], t ∈ [0, {T}]")
        print(f"  True alpha: {alpha}")
        print(f"  Collocation points: {N_f}")
        print(f"  Boundary points: {N_bc}")
        print(f"  Initial condition points: {N_ic}")
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
    

    def generate_collocation_points(self, method: str = 'uniform') -> tuple[torch.Tensor, torch.Tensor]:
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
            dtype = torch.float64 if self.device == 'cpu' else torch.float32
            x_f = torch.tensor(sample[:, 0:1] * self.L, dtype=dtype)
            t_f = torch.tensor(sample[:, 1:2] * self.T, dtype=dtype)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return x_f.to(self.device), t_f.to(self.device)
    
    
    def generate_boundary_conditions(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate boundary condition points: u(0, t) = u(L, t) = 0
            
        Returns:
            x_bc: Spatial coordinates (at x=0 and x=L)
            t_bc: Temporal coordinates
            u_bc: Boundary values (all zeros)
        """
        # Left boundary (x=0)
        t_left = torch.rand(self.N_bc, 1) * self.T
        x_left = torch.zeros(self.N_bc, 1)
        
        # Right boundary (x=L)
        t_right = torch.rand(self.N_bc, 1) * self.T
        x_right = torch.ones(self.N_bc, 1) * self.L
        
        # Combine
        x_bc = torch.cat([x_left, x_right], dim=0)
        t_bc = torch.cat([t_left, t_right], dim=0)
        u_bc = torch.zeros_like(x_bc)  # Dirichlet BC: u = 0
        
        return x_bc.to(self.device), t_bc.to(self.device), u_bc.to(self.device)
    
    
    def generate_initial_conditions(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate initial condition points: u(x, 0) = sin(π*x)
            
        Returns:
            x_ic: Spatial coordinates
            t_ic: Temporal coordinates (all zeros)
            u_ic: Initial values
        """
        x_ic = torch.rand(self.N_ic, 1) * self.L
        t_ic = torch.zeros(self.N_ic, 1)
        u_ic = torch.sin(np.pi * x_ic)
        
        return x_ic.to(self.device), t_ic.to(self.device), u_ic.to(self.device)
    
    
    def generate_measurements(
        self, 
        add_noise: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
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
        dtype = torch.float64 if self.device == 'cpu' else torch.float32
        x_m = torch.tensor(x_m_np.reshape(-1, 1), dtype=dtype)
        t_m = torch.tensor(t_m_np.reshape(-1, 1), dtype=dtype)
        u_m = torch.tensor(u_measured.reshape(-1, 1), dtype=dtype)
        
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
    

    def finite_difference_solution(
        self,
        nx: int = 21,
        nt: int = 101,
        alpha_sim: Optional[float] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve heat equation using explicit finite difference (FTCS scheme).
        
        u_t = alpha * u_xx
        IC: u(x, 0) = sin(π·x)
        BC: u(0, t) = u(L, t) = 0
        
        Args:
            nx: Number of spatial grid points (including boundaries)
            nt: Number of time steps
            alpha_sim: Thermal diffusivity for simulation (default: self.alpha)
                    Can differ from true alpha to simulate model error.
        
        Returns:
            x_grid: Spatial coordinates, shape (nx,)
            t_grid: Temporal coordinates, shape (nt,)
            u_grid: Solution field, shape (nx, nt)
        """
        if alpha_sim is None:
            alpha_sim = self.alpha
        
        # Grid setup
        x_grid = np.linspace(0, self.L, nx)
        t_grid = np.linspace(0, self.T, nt)
        dx = x_grid[1] - x_grid[0]
        dt = t_grid[1] - t_grid[0]
        
        # Stability check (CFL condition)
        r = alpha_sim * dt / dx**2
        if r > 0.5:
            print(f"Warning: CFL condition violated (r={r:.3f} > 0.5). "
                f"Solution may be unstable. Consider increasing nx or nt.")
        
        # Initialize solution array
        u_grid = np.zeros((nx, nt))
        
        # Initial condition: u(x, 0) = sin(π·x)
        u_grid[:, 0] = np.sin(np.pi * x_grid)
        
        # Boundary conditions are already zero (array initialized to zero)
        
        # Time stepping (FTCS scheme)
        for n in range(nt - 1):
            for i in range(1, nx - 1):
                u_grid[i, n+1] = u_grid[i, n] + r * (
                    u_grid[i+1, n] - 2*u_grid[i, n] + u_grid[i-1, n]
                )
            # BCs: u[0, n+1] = u[nx-1, n+1] = 0 (already satisfied)
        
        return x_grid, t_grid, u_grid


    def generate_low_fidelity_measurements(
        self,
        nx: int = 11,
        nt: int = 51,
        alpha_sim: Optional[float] = None,
        noise_level: Optional[float] = None,
        N_samples: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Generate low-fidelity measurements from finite difference solution.
        
        Args:
            nx: FD spatial resolution (coarser = lower fidelity)
            nt: FD temporal resolution
            alpha_sim: Possibly incorrect alpha for model error
            noise_level: Measurement noise (default: 2x self.noise_level)
            N_samples: Number of measurement points to sample from FD grid
                    (default: sample all interior grid points)
        
        Returns:
            x_lf: Spatial coordinates of measurements
            t_lf: Temporal coordinates of measurements
            u_lf: Low-fidelity temperature values
            info: Metadata dictionary
        """
        if noise_level is None:
            noise_level = 2.0 * self.noise_level
        
        if alpha_sim is None:
            alpha_sim = self.alpha * 1.2  # 20% model error by default
        
        # Solve with finite difference
        x_grid, t_grid, u_grid = self.finite_difference_solution(
            nx=nx, nt=nt, alpha_sim=alpha_sim
        )
        
        # Create measurement points from interior grid
        # Exclude boundaries (x=0, x=L) and initial condition (t=0)
        x_interior = x_grid[1:-1]
        t_interior = t_grid[1:]
        
        X_mesh, T_mesh = np.meshgrid(x_interior, t_interior, indexing='ij')
        x_all = X_mesh.flatten()
        t_all = T_mesh.flatten()
        u_all = u_grid[1:-1, 1:].flatten()
        
        # Subsample if requested
        if N_samples is not None and N_samples < len(x_all):
            indices = np.random.choice(len(x_all), N_samples, replace=False)
            x_all = x_all[indices]
            t_all = t_all[indices]
            u_all = u_all[indices]
        
        # Add noise
        noise = np.random.normal(0, noise_level * np.abs(u_all).mean(), size=u_all.shape)
        u_noisy = u_all + noise
        
        # Compute error vs analytical solution
        u_exact = self.analytical_solution(x_all, t_all)
        model_error = np.sqrt(np.mean((u_all - u_exact)**2))
        
        # Convert to tensors
        dtype = torch.float64 if self.device == 'cpu' else torch.float32
        x_lf = torch.tensor(x_all.reshape(-1, 1), dtype=dtype).to(self.device)
        t_lf = torch.tensor(t_all.reshape(-1, 1), dtype=dtype).to(self.device)
        u_lf = torch.tensor(u_noisy.reshape(-1, 1), dtype=dtype).to(self.device)
        
        info = {
            'nx': nx,
            'nt': nt,
            'alpha_sim': alpha_sim,
            'alpha_true': self.alpha,
            'noise_level': noise_level,
            'model_error_rmse': model_error,
            'n_measurements': len(x_all),
            'u_fd_clean': u_all,
            'u_exact': u_exact
        }
        
        print(f"\nLow-fidelity data generated:")
        print(f"  FD grid: {nx}×{nt}")
        print(f"  α_sim: {alpha_sim:.4f} (true: {self.alpha:.4f})")
        print(f"  Model error (RMSE): {model_error:.4e}")
        print(f"  Noise level: {noise_level:.1%}")
        print(f"  Measurements: {len(x_all)}")
        
        return x_lf, t_lf, u_lf, info
    

    def generate_full_dataset(
        self, 
        collocation_method: str = 'uniform'
    ) -> dict[str, torch.Tensor]:
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
        print("\n" + "=" * 60)
        print("Generating complete dataset...")
        print("=" * 60)
        
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
    
    
    def visualize_data(self, data: dict[str, torch.Tensor], save_path: Optional[str] = None):
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
        axes[0, 1].set_title(f'Boundary (N={len(x_bc)}) & Initial Conditions (N={len(x_ic)})')
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


    def visualize_fidelity_comparison(
        self,
        nx: int = 11,
        nt: int = 51,
        alpha_sim: Optional[float] = None,
        save_path: Optional[str] = None
    ):
        """
        Compare finite difference solution against analytical solution.
        
        Visualizes:
        1. Analytical solution (ground truth)
        2. Finite difference solution (low-fidelity)
        3. Absolute error field
        4. Error profiles at selected time slices
        
        Args:
            nx: FD spatial resolution
            nt: FD temporal resolution
            alpha_sim: FD thermal diffusivity (default: 1.2 * self.alpha)
            save_path: Optional path to save figure
        """
        if alpha_sim is None:
            alpha_sim = self.alpha * 1.2
        
        # Compute FD solution
        x_fd, t_fd, u_fd = self.finite_difference_solution(
            nx=nx, nt=nt, alpha_sim=alpha_sim
        )
        
        # Compute analytical solution on same grid
        X_fd, T_fd = np.meshgrid(x_fd, t_fd, indexing='ij')
        u_exact = self.analytical_solution(X_fd, T_fd)
        
        # Compute analytical on fine grid for smooth reference
        x_fine = np.linspace(0, self.L, 100)
        t_fine = np.linspace(0, self.T, 100)
        X_fine, T_fine = np.meshgrid(x_fine, t_fine, indexing='ij')
        u_exact_fine = self.analytical_solution(X_fine, T_fine)
        
        # Error metrics
        error = np.abs(u_fd - u_exact)
        rmse = np.sqrt(np.mean((u_fd - u_exact)**2))
        max_error = np.max(error)
        rel_error = rmse / np.sqrt(np.mean(u_exact**2))
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Analytical solution (fine grid)
        im1 = axes[0, 0].contourf(X_fine, T_fine, u_exact_fine, levels=20, cmap='hot')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('t')
        axes[0, 0].set_title(f'Analytical Solution (α={self.alpha})')
        plt.colorbar(im1, ax=axes[0, 0], label='u')
        
        # Plot 2: FD solution (coarse grid)
        im2 = axes[0, 1].contourf(X_fd, T_fd, u_fd, levels=20, cmap='hot')
        # Overlay grid points
        axes[0, 1].scatter(X_fd.flatten(), T_fd.flatten(), c='white', s=3, alpha=0.3)
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('t')
        axes[0, 1].set_title(f'Finite Difference (α={alpha_sim}, {nx}×{nt} grid)')
        plt.colorbar(im2, ax=axes[0, 1], label='u')
        
        # Plot 3: Error field
        im3 = axes[1, 0].contourf(X_fd, T_fd, error, levels=20, cmap='viridis')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('t')
        axes[1, 0].set_title(f'Absolute Error (RMSE={rmse:.2e}, Rel={rel_error:.2%})')
        plt.colorbar(im3, ax=axes[1, 0], label='|u_fd - u_exact|')
        
        # Plot 4: Error profiles at selected times
        time_slices = [0.0, 0.25, 0.5, 0.75]
        colors = plt.colormaps['viridis'](np.linspace(0, 0.8, len(time_slices)))
        
        for t_val, color in zip(time_slices, colors):
            # Find closest time index in FD grid
            t_idx = np.argmin(np.abs(t_fd - t_val * self.T))
            actual_t = t_fd[t_idx]
            
            # FD solution at this time
            u_fd_slice = u_fd[:, t_idx]
            
            # Analytical solution at this time (fine grid)
            u_exact_slice = self.analytical_solution(x_fine, actual_t * np.ones_like(x_fine))
            
            # Plot both
            axes[1, 1].plot(x_fine, u_exact_slice, '-', color=color, 
                            label=f't={actual_t:.2f} (exact)', linewidth=1.5)
            axes[1, 1].plot(x_fd, u_fd_slice, 'o', color=color, 
                            markersize=5, alpha=0.7)
        
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('u')
        axes[1, 1].set_title('Solution Profiles (lines: exact, dots: FD)')
        axes[1, 1].legend(loc='upper right', fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add text box with error summary
        textstr = f'Model error: α_sim/α_true = {alpha_sim/self.alpha:.2f}\n'
        textstr += f'Grid: {nx}×{nt}\n'
        textstr += f'Max error: {max_error:.2e}'
        axes[1, 1].text(0.02, 0.98, textstr, transform=axes[1, 1].transAxes,
                        verticalalignment='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
        # Print summary
        print(f"\nFidelity Comparison Summary:")
        print(f"  FD grid: {nx} × {nt}")
        print(f"  α_true: {self.alpha}, α_sim: {alpha_sim} (ratio: {alpha_sim/self.alpha:.2f})")
        print(f"  RMSE: {rmse:.4e}")
        print(f"  Relative L2 error: {rel_error:.2%}")
        print(f"  Max absolute error: {max_error:.4e}")


def prepare_multi_fidelity_data(
    data_gen: HeatEquationData,
    hf_sensors: int = 15,
    hf_times: int = 15,
    lf_nx: int = 11,
    lf_nt: int = 31,
    lf_alpha_ratio: float = 1.2,
    lf_noise: float = 0.05,
    collocation_method: str = 'uniform'
) -> dict:
    """
    Prepare complete multi-fidelity dataset for training.
    
    Args:
        data_gen: HeatEquationData instance
        hf_sensors: Number of high-fidelity sensor locations
        hf_times: Number of high-fidelity time measurements
        lf_nx: Low-fidelity FD spatial resolution
        lf_nt: Low-fidelity FD temporal resolution
        lf_alpha_ratio: Ratio of LF alpha to true alpha (model error)
        collocation_method: 'uniform' or 'lhs'
    
    Returns:
        data: Dictionary with all training data
    """
    # Store original settings
    orig_sensors = data_gen.N_sensors
    orig_times = data_gen.N_time_measurements
    
    # Generate collocation points
    x_f, t_f = data_gen.generate_collocation_points(method=collocation_method)
    
    # Generate BC and IC
    x_bc, t_bc, u_bc = data_gen.generate_boundary_conditions()
    x_ic, t_ic, u_ic = data_gen.generate_initial_conditions()
    
    # High-fidelity measurements
    data_gen.N_sensors = hf_sensors
    data_gen.N_time_measurements = hf_times
    x_hf, t_hf, u_hf, hf_info = data_gen.generate_measurements(add_noise=True)
    
    # Low-fidelity measurements
    alpha_lf = data_gen.alpha * lf_alpha_ratio
    x_lf, t_lf, u_lf, lf_info = data_gen.generate_low_fidelity_measurements(
        nx=lf_nx, nt=lf_nt, alpha_sim=alpha_lf, noise_level=lf_noise
    )
    
    # Restore original settings
    data_gen.N_sensors = orig_sensors
    data_gen.N_time_measurements = orig_times
    
    # Estimate noise standard deviations
    sigma_hf = data_gen.noise_level * np.abs(hf_info['u_true']).mean()
    sigma_lf = lf_info['noise_level'] * np.abs(lf_info['u_fd_clean']).mean()
    # Add model error contribution to LF uncertainty
    sigma_lf = np.sqrt(sigma_lf**2 + lf_info['model_error_rmse']**2)
    
    data = {
        'x_f': x_f, 't_f': t_f,
        'x_bc': x_bc, 't_bc': t_bc, 'u_bc': u_bc,
        'x_ic': x_ic, 't_ic': t_ic, 'u_ic': u_ic,
        'x_hf': x_hf, 't_hf': t_hf, 'u_hf': u_hf,
        'x_lf': x_lf, 't_lf': t_lf, 'u_lf': u_lf,
        'sigma_hf': sigma_hf,
        'sigma_lf': sigma_lf,
        'hf_info': hf_info,
        'lf_info': lf_info
    }
    
    print(f"\nMulti-fidelity dataset prepared:")
    print(f"  High-fidelity: {x_hf.shape[0]} points, σ={sigma_hf:.4f}")
    print(f"  Low-fidelity: {x_lf.shape[0]} points, σ={sigma_lf:.4f}")
    print(f"  Effective weight ratio (HF/LF): {(sigma_lf/sigma_hf)**2:.1f}x")
    
    return data


# ============================================================================
# Usage Example and Testing
# ============================================================================


if __name__ == "__main__":
    print("Testing HeatEquationData class...\n")
    
    # Create data generator
    data_gen = HeatEquationData(
        L=1.0,
        T=1.0,
        alpha=0.01,
        N_f=10000,
        N_bc = 150,
        N_ic = 100,
        N_sensors=10,
        N_time_measurements=10,
        noise_level=0.05,
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
    data_gen.visualize_data(data)
    
    # Test analytical solution at a few points
    print("\nTesting analytical solution:")
    x_test = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    t_test = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    u_test = data_gen.analytical_solution(x_test, t_test)
    print(f"u(x, t=0) = sin(πx): {u_test}")

    print("\n" + "=" * 60)
    print("Testing finite difference solver and fidelity comparison...")
    print("=" * 60)

    # Test FD solver
    x_fd, t_fd, u_fd = data_gen.finite_difference_solution(nx=21, nt=101)
    print(f"\nFD solution shape: {u_fd.shape}")

    # Test low-fidelity measurement generation
    x_lf, t_lf, u_lf, lf_info = data_gen.generate_low_fidelity_measurements(
        nx=11, nt=31, alpha_sim=0.012
    )
    print(f"Low-fidelity measurements: {x_lf.shape[0]} points")

    # Visualize fidelity comparison
    data_gen.visualize_fidelity_comparison(
        nx=11, nt=31, alpha_sim=0.012)
    
    print("\n" + "=" * 60)
    print("Testing multi-fidelity data generation...")
    print("=" * 60)

    prepare_multi_fidelity_data(data_gen)
    
    print("\nAll data generation tests passed.")