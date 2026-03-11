# this is especific for 2D
import numpy as np
import torch
import sys

from torch_geometric.utils import scatter

from config.physics import GraphMathOps
def scatter_add(src, index, dim=0, dim_size=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='sum')
def scatter_mean(src, index, dim=0, dim_size=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='mean')

class PoissonPhysics:
    """
    Versión optimizada 2D. 
    Interfaz compatible con PoissonGeneral.
    """
    def __init__(self, source_type='sine', scale=1.0):
        self.source_type = source_type
        self.scale = scale
        self.ops = GraphMathOps()

    def source_term(self, x):
        """Calcula f(x, y) detectando la librería automáticamente"""
        if torch.is_tensor(x):
            # Extraemos x e y del tensor unificado [batch, 2]
            ix, iy = x[:, 0:1], x[:, 1:2]
            if self.source_type == 'sine':
                return self.scale * 2 * (torch.pi**2) * torch.sin(torch.pi * ix) * torch.sin(torch.pi * iy)
            return torch.ones_like(ix) * self.scale
        else:
            # Versión Numpy para validación
            ix, iy = x[:, 0:1], x[:, 1:2]
            if self.source_type == 'sine':
                return self.scale * 2 * (np.pi**2) * np.sin(np.pi * ix) * np.sin(np.pi * iy)
            return np.ones_like(ix) * self.scale

    def exact_solution(self, x):
        """u(x) analítica (Numpy)"""
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
            
        if self.source_type == 'sine':
            ix, iy = x[:, 0:1], x[:, 1:2]
            return self.scale * np.sin(np.pi * ix) * np.sin(np.pi * iy)
        return None

    def boundary_condition(self, x, library='torch'):
        """u(boundary) = 0"""
        if library == 'torch':
            return torch.zeros((x.shape[0], 1), device=x.device)
        return np.zeros((x.shape[0], 1))

    def compute_laplacian(self, x, grads):
        """Δu = u_xx + u_yy"""
        u_xx = torch.autograd.grad(
            grads[:, 0], x, 
            grad_outputs=torch.ones_like(grads[:, 0]),
            create_graph=True
        )[0][:, 0:1]
        
        u_yy = torch.autograd.grad(
            grads[:, 1], x, 
            grad_outputs=torch.ones_like(grads[:, 1]),
            create_graph=True
        )[0][:, 1:2]
        
        return u_xx + u_yy

    def compute_pde_residual(self, x, u, grads):
        """R = Δu + f"""
        laplacian = self.ops.laplacian_autograd(x, grads)
        f = self.source_term(x) # Llamada unificada
        return laplacian + f