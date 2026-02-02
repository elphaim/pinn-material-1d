"""
Module to generate smooth compact 2D Gaussian test functions for weak-form PINNs

Author: elphaim
Date: January 29, 2026
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from typing import Optional


def compact_gaussian(
    x: torch.Tensor, 
    t: torch.Tensor, 
    xc: float, 
    tc: float, 
    R: float, 
    k: float = 1e8):
    """
    Compact support Gaussian: 
    φ(x,t) = Heaviside(R - r) * exp(-r²/(R² - r²))
    where r² = (x-xc)² + (t-tc)²

    Args:
        (x, t): spatio-temporal coordinates, tensors (N, 1)
        (xc, tc): center coordinates, tensors (N, 1)
        R: support radius
        k: smoothing factor to replace Heaviside by steep sigmoid (default: 1e8)
    """
    r_squared = (x - xc)**2 + (t - tc)**2
    R_squared = R**2

    if k > 0:
        # k is the smoothing factor: higher = steeper step
        phi = torch.sigmoid(2*k*(R_squared - r_squared)) * torch.exp(-r_squared / (R_squared - r_squared + 1e-8))
    else:
        # Heaviside version
        inside = r_squared < R_squared
        phi = torch.where(inside, torch.exp(-r_squared / (R_squared - r_squared + 1e-8)), torch.zeros_like(x))

    return phi


def generate_compact_gaussians(
    L: float = 1.0,
    T: float = 1.0,
    n_funcs: int = 10,
    support_radius: Optional[float] = None, 
    shrink: Optional[float] = None,
    placement: str = 'uniform',
    min_separation: Optional[float] = None,
    smooth: float = 0.0,
) -> tuple[list, list]:
    """
    Generates a list of 2D compactly supported Gaussian test functions
    
    Args:
        L: Length of spatial domain, must be odd (default: 1.0)
        T: Final time (default: 1.0)
        n_funcs: Number of test functions to be generated (default: 10)
        support_radius: Radius outside of which the function vanishes (default: min(L/5, T/5))
        shrink: offset for the x and t domains (default: support_radius so functions vanish on domain boundary for valid IBP)
        placement: strategy for placing Gaussians, one of 'uniform', 'random', or 'boundary'
        min_separation: Minimum distance between centers to prevent clustering (default: min(L/10, T/10))
        smooth: Smoothen Heaviside(R - r) with steep sigmoid (default: 0, no smoothing)
    
    Returns:
        test_funcs: List of callables, each accepting (x, t) tensors
        test_doms: List of support domain [[x_center - support_radius, x_center + support_radius], [t_center - support_radius, t_center + support_radius]]
    """
    # Global PDE domain
    domain = [[0, L], [0, T]]
    
    # Parameters       
    if support_radius is None:
        support_radius = min(L/5, T/5)
    
    if shrink is None:
        shrink = support_radius
    # Validate the shrink
    x_range = domain[0][1] - domain[0][0]
    t_range = domain[1][1] - domain[1][0]
    if 2 * shrink >= min(x_range, t_range):
        raise ValueError(f"shrink ({shrink}) too large for domain")
    # Shrunk domain for center placement
    shrunk_domain = [[domain[0][0] + shrink, domain[0][1] - shrink], [domain[1][0] + shrink, domain[1][1] - shrink]]

    # Initialize results
    test_funcs = []
    test_doms = []

    # Uniform grid placement
    if placement == 'uniform':
        aspect_ratio = (shrunk_domain[0][1] - shrunk_domain[0][0]) / (shrunk_domain[1][1] - shrunk_domain[1][0])
        # Determine grid dimensions that give approximately n_funcs test functions
        nx = int(np.sqrt(n_funcs * aspect_ratio))
        nt = int(np.sqrt(n_funcs / aspect_ratio))
        # Ensure at least 2 points in each direction
        nx = max(2, nx)
        nt = max(2, nt)
        # Create uniform grid
        x_centers = np.linspace(shrunk_domain[0][0], shrunk_domain[0][1], nx)
        t_centers = np.linspace(shrunk_domain[1][0], shrunk_domain[1][1], nt)
        # Create all combinations
        xx, tt = np.meshgrid(x_centers, t_centers)
        centers = np.column_stack([xx.ravel(), tt.ravel()])
        print(f"Uniform Placement: Successfully placed {nx*nt}/{n_funcs} test functions on {nx}×{nt} grid")

    # Random placement
    elif placement == 'random':
        # Fix minimum separation to avoid clustering
        if min_separation is None:
            min_separation = min(L/10, T/10)
        max_attempts = n_funcs * 100
        centers = []
        
        for i in range(n_funcs):
            placed = False
            
            for attempt in range(max_attempts):
                # Random candidate position
                x_cand = np.random.uniform(shrunk_domain[0][0], shrunk_domain[0][1])
                t_cand = np.random.uniform(shrunk_domain[1][0], shrunk_domain[1][1])
                candidate = np.array([x_cand, t_cand])
                
                # Check minimum separation from existing centers
                if len(centers) == 0:
                    centers.append(candidate)
                    placed = True
                    break
                
                distances = np.linalg.norm(np.array(centers) - candidate, axis=1)
                if np.all(distances >= min_separation):
                    centers.append(candidate)
                    placed = True
                    break
            
            if not placed:
                print(f"Warning: Could not place test function {i+1}/{n_funcs} "
                      f"after {max_attempts} attempts")
        
        centers = np.array(centers)
        print(f"Random Placement: Successfully placed {len(centers)}/{n_funcs} test functions")

    # Boundary enriched placement
    elif placement == 'boundary':
        boundary_fraction = 0.4
        # Split into boundary and interior regions
        n_boundary = int(n_funcs * boundary_fraction)
        n_interior = n_funcs - n_boundary
        # Define boundary layer thickness
        boundary_thickness = 2.5 * support_radius
        # Interior region (away from boundaries)
        x_interior_min = shrunk_domain[0][0] + boundary_thickness
        x_interior_max = shrunk_domain[0][1] - boundary_thickness
        t_interior_min = shrunk_domain[1][0] + boundary_thickness
        t_interior_max = shrunk_domain[1][1] - boundary_thickness
        
        centers_list = []
        
        # Place interior test functions (coarser grid)
        if n_interior > 0 and x_interior_min < x_interior_max and t_interior_min < t_interior_max:
            aspect_ratio = (x_interior_max - x_interior_min) / (t_interior_max - t_interior_min)
            nx_int = max(2, int(np.sqrt(n_interior * aspect_ratio)))
            nt_int = max(2, int(np.sqrt(n_interior / aspect_ratio)))
            
            x_int = np.linspace(x_interior_min, x_interior_max, nx_int)
            t_int = np.linspace(t_interior_min, t_interior_max, nt_int)
            xx_int, tt_int = np.meshgrid(x_int, t_int)
            centers_list.append(np.column_stack([xx_int.ravel(), tt_int.ravel()]))
        
        # Place boundary test functions (finer spacing)
        if n_boundary > 0:
            # Distribute boundary points among 4 boundary regions
            n_per_boundary = n_boundary // 4
            
            # Left boundary (x ≈ x_min)
            x_left = np.linspace(shrunk_domain[0][0], shrunk_domain[0][0] + boundary_thickness, 2)
            t_left = np.linspace(shrunk_domain[1][0], shrunk_domain[1][1], n_per_boundary)
            for x in x_left:
                centers_list.append(np.column_stack([np.full_like(t_left, x), t_left]))
            
            # Right boundary (x ≈ x_max)
            x_right = np.linspace(shrunk_domain[0][1] - boundary_thickness, shrunk_domain[0][1], 2)
            t_right = np.linspace(shrunk_domain[1][0], shrunk_domain[1][1], n_per_boundary)
            for x in x_right:
                centers_list.append(np.column_stack([np.full_like(t_right, x), t_right]))
            
            # Bottom boundary (t ≈ t_min)
            x_bottom = np.linspace(shrunk_domain[0][0], shrunk_domain[0][1], n_per_boundary)
            t_bottom = np.linspace(shrunk_domain[1][0], shrunk_domain[1][0] + boundary_thickness, 2)
            for t in t_bottom:
                centers_list.append(np.column_stack([x_bottom, np.full_like(x_bottom, t)]))
            
            # Top boundary (t ≈ t_max)
            x_top = np.linspace(shrunk_domain[0][0], shrunk_domain[0][1], n_per_boundary)
            t_top = np.linspace(shrunk_domain[1][1] - boundary_thickness, shrunk_domain[1][1], 2)
            for t in t_top:
                centers_list.append(np.column_stack([x_top, np.full_like(x_top, t)]))
        
        # Combine all centers
        centers = np.vstack(centers_list)
        # Remove duplicates (points that might appear in multiple regions)
        centers = np.unique(centers, axis=0)
        print(f"Boundary Placement: Placed {len(centers)} test functions "
              f"({boundary_fraction*100:.0f}% near boundaries)")
        
    else:
        print(f"Placement strategy {placement} not recognized, returning empty lists.")
        return [], []

    for x_center, t_center in centers:
        # Callable wrapper with captured parameters
        def placed_gaussian(x, t, xc=x_center, tc=t_center, R=support_radius, k=smooth):
            return compact_gaussian(x, t, xc=xc, tc=tc, R=R, k=k)
        # Compute support domain clipped to global domain
        dom_xmin = max(x_center - support_radius, domain[0][0])
        dom_xmax = min(x_center + support_radius, domain[0][1])
        dom_tmin = max(t_center - support_radius, domain[1][0])
        dom_tmax = min(t_center + support_radius, domain[1][1])
    
        test_funcs.append(placed_gaussian)
        test_doms.append([[dom_xmin, dom_xmax], [dom_tmin, dom_tmax]])

    return test_funcs, test_doms


def plot_compact_gaussians(
    test_funcs,
    test_doms,
    show_support: bool = True,
    domain: list = [[0, 1], [0, 1]],
    resolution: int = 200,
    alpha: float = 0.8,
    save_path: Optional[str] = None
):
    """
    Overlay all compact Gaussians and their support domains on one plot.

    Args:
        test_funcs: List of callables f(x, t)
        test_doms: List of compact support domains
        show_support: Display compact supports
        domain: Global plotting domain
        resolution: Grid resolution
        alpha: Transparency for Gaussian fields
        save_path: Optional path to save figure
    """
    assert len(test_funcs) == len(test_doms)

    # Grid
    x = torch.linspace(domain[0][0], domain[0][1], resolution)
    t = torch.linspace(domain[1][0], domain[1][1], resolution)
    X, T = torch.meshgrid(x, t, indexing="ij")

    # Accumulate all Gaussians
    Z_total = torch.zeros_like(X)

    with torch.no_grad():
        for func in test_funcs:
            Z_total += func(X, T)

    # Plot combined field
    plt.figure(figsize=(6, 5))
    plt.imshow(
        Z_total.numpy(),
        extent=(domain[0][0], domain[0][1],
                domain[1][0], domain[1][1]),
        origin="lower",
        cmap="viridis",
        alpha=alpha,
        aspect="auto"
    )

    if show_support:
        # Draw support domains
        ax = plt.gca()
        for dom in test_doms:
            (x_min, x_max), (t_min, t_max) = dom
            radius = (x_max - x_min) / 2
            x_center = x_min + radius
            t_center = t_min + radius
            circle = Circle(
                (x_center, t_center),
                radius,
                fill=False,
                linewidth=0.8,
                edgecolor="red",
                linestyle="--",
                alpha=0.5
            )
            ax.add_patch(circle)

    plt.colorbar(label="Σ φ(x,t)")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(f"All compact Gaussians {"and support domains" if show_support else ""}")
    plt.tight_layout()

    if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
            
    plt.show()