"""
Quantum Lattice — PhiLattice, FibonacciLattice, HaltonLattice
============================================================
Implements optimal time-point sampling using irrational bases and low-discrepancy sequences.
This leverages Weyl's equidistribution theorem to avoid arithmetic resonances in chaotic attractors.

Author: Paul E. Harris IV — SynechismCore v20.1
"""

import torch
import numpy as np

class PhiLattice:
    """Weyl equidistribution using the golden ratio (phi)."""
    def __init__(self, phi_base: float = 1.618033988749895):
        self.phi = phi_base

    def generate(self, n_points: int, device='cpu'):
        """Generate n_points using (k * phi) mod 1."""
        indices = torch.arange(1, n_points + 1, device=device).float()
        return (indices * self.phi) % 1.0

class FibonacciLattice:
    """Sampling based on Fibonacci ratios for optimal 2D/3D coverage."""
    def __init__(self):
        self.phi = (1 + 5**0.5) / 2

    def generate(self, n_points: int, device='cpu'):
        indices = torch.arange(n_points, device=device).float()
        theta = 2 * np.pi * indices / self.phi**2
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

class HaltonLattice:
    """Low-discrepancy Halton sequence for high-dimensional sampling."""
    def __init__(self, base: int = 2):
        self.base = base

    def generate(self, n_points: int, device='cpu'):
        points = []
        for i in range(1, n_points + 1):
            f = 1
            r = 0
            while i > 0:
                f = f / self.base
                r = r + f * (i % self.base)
                i = i // self.base
            points.append(r)
        return torch.tensor(points, device=device).float()
