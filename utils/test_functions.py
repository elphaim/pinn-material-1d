"""
Module to generate smooth compact 2D Gaussian test functions for weak-form PINNs

Author: elphaim
Date: January 29, 2026
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from typing import Optional


def generate_compact_gaussians(
    L: float = 1.0,
    T: float = 1.0,
    n_funcs: int = 10,
    support_radius: Optional[float] = None, 
    shrink: Optional[float] = None,
    min_separation: Optional[float] = None,
    smooth: bool = True,
) -> tuple[list, list]:
    """
    Generates a list of 2D compactly supported Gaussian test functions
    
    Args:
        L: Length of spatial domain, must be odd (default: 1.0)
        T: Final time (default: 1.0)
        n_funcs: Number of test functions to be generated (default: 10)
        support_radius: Radius outside of which the function vanishes (default: min(L/5, T/5))
        shrink: offset for the x and t domains (default: support_radius so functions vanish on domain boundary for valid IBP)
        min_separation: Minimum distance between centers to prevent clustering (default: min(L/10, T/10))
        smooth: Smoothen Heaviside(R - r) with steep sigmoid (default: True)
    
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
    
    if min_separation is None:
        min_separation = min(L/10, T/10)

    # Initialize results
    test_funcs = []
    test_doms = []
    centers = []

    # Attempt to spread centers according to min_separation
    attempts = 0
    max_attempts = n_funcs * 100

    while len(test_funcs) < n_funcs and attempts < max_attempts:
        attempts += 1
        # Random center within shrunk domain
        x_center = np.random.uniform(domain[0][0] + shrink, domain[0][1] - shrink)
        t_center = np.random.uniform(domain[1][0] + shrink, domain[1][1] - shrink)
        # Check separation from existing centers
        if min_separation > 0:
            too_close = False
            for prev_center in centers:
                dist = np.sqrt((x_center - prev_center[0])**2 + 
                            (t_center - prev_center[1])**2)
                if dist < min_separation:
                    too_close = True
                    break
            if too_close:
                continue

        # Create function with captured parameters
        def compact_gaussian(x, t, xc=x_center, tc=t_center, R=support_radius, s=smooth):
            """
            Compact support Gaussian: 
            φ(x,t) = Heaviside(R - r) * exp(-r²/(R² - r²))
            where r² = (x-xc)² + (t-tc)²

            Args:
                smooth: Heaviside smoothed out using steep sigmoid
            """
            r_squared = (x - xc)**2 + (t - tc)**2
            R_squared = R**2

            if s:
                # k is the smoothing factor: higher = steeper step
                k = 1e8
                # Smooth version
                phi = torch.sigmoid(2*k*(R_squared - r_squared)) * torch.exp(-r_squared / (R_squared - r_squared + 1e-8))
            else:
                # Non-smooth version with Heaviside function
                inside = r_squared < R_squared
                phi = torch.where(inside, torch.exp(-r_squared / (R_squared - r_squared + 1e-8)), torch.zeros_like(x))

            return phi
    
        # Compute support domain clipped to global domain
        dom_xmin = max(x_center - support_radius, domain[0][0])
        dom_xmax = min(x_center + support_radius, domain[0][1])
        dom_tmin = max(t_center - support_radius, domain[1][0])
        dom_tmax = min(t_center + support_radius, domain[1][1])
    
        test_funcs.append(compact_gaussian)
        test_doms.append([[dom_xmin, dom_xmax], [dom_tmin, dom_tmax]])
        centers.append((x_center, t_center))

    if len(test_funcs) < n_funcs:
        print(f"Warning: Only generated {len(test_funcs)}/{n_funcs} test functions")

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
        test_funcs: list of callables f(x, t)
        test_doms: list of [[x_min, x_max], [t_min, t_max]]
        show_support: display compact supports
        domain: global plotting domain [[x_min, x_max], [t_min, t_max]]
        resolution: grid resolution
        alpha: transparency for Gaussian fields
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
            rect = Rectangle(
                (t_min, x_min),
                t_max - t_min,
                x_max - x_min,
                linewidth=1.5,
                edgecolor="red",
                facecolor="none",
                linestyle="--",
                alpha=0.9
            )
            ax.add_patch(rect)

    plt.colorbar(label="Σ φ(x,t)")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(f"All compact Gaussians {"and support domains" if show_support else ""}")
    plt.tight_layout()

    if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
            
    plt.show()