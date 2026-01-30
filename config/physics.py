import numpy as np
import torch

class PoissonPhysics:
    """
    Centralized physics definition for -Δu = f(x, y)
    Supports both Numpy (FEM) and Torch (PINN).
    """
    def __init__(self, source_type='sine', scale=1.0):
        self.source_type = source_type
        self.scale = scale

    def source_term(self, x, y, library='numpy'):
        """Term f(x, y)"""
        if library == 'numpy':
            if self.source_type == 'sine':
                return self.scale * 2 * (np.pi**2) * np.sin(np.pi * x) * np.sin(np.pi * y)
            return np.ones_like(x) * self.scale
        else:
            if self.source_type == 'sine':
                return self.scale * 2 * (torch.pi**2) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
            return torch.ones_like(x) * self.scale

    def exact_solution(self, x, y):
        """Analytical solution for validation (Numpy only)"""
        if self.source_type == 'sine':
            return self.scale * np.sin(np.pi * x) * np.sin(np.pi * y)
        return None # No analytical solution for other types

    def boundary_condition(self, x, y, library='numpy'):
        """u(x, y) = 0 on boundary"""
        if library == 'numpy':
            return np.zeros_like(x)
        return torch.zeros_like(x)