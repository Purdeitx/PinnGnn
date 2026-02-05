import torch
import numpy as np

class SquareGeometry:
    def __init__(self, x_range=[0, 1], y_range=[0, 1], device='cpu'):
        self.x_range = x_range
        self.y_range = y_range
        self.device = device

    def sample_interior(self, n_points):
        """Genera puntos de colocación (física)"""
        x = torch.rand(n_points, 1, device=self.device) * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        y = torch.rand(n_points, 1, device=self.device) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        return torch.cat([x, y], dim=1).requires_grad_(True)

    def sample_boundary(self, n_points):
        """Genera puntos para las condiciones de contorno"""
        n_per_side = n_points // 4
        # Lados del cuadrado
        bottom = torch.stack([torch.rand(n_per_side, device=self.device), torch.zeros(n_per_side, device=self.device)], dim=1)
        top    = torch.stack([torch.rand(n_per_side, device=self.device), torch.ones(n_per_side, device=self.device)], dim=1)
        left   = torch.stack([torch.zeros(n_per_side, device=self.device), torch.rand(n_per_side, device=self.device)], dim=1)
        right  = torch.stack([torch.ones(n_per_side, device=self.device), torch.rand(n_per_side, device=self.device)], dim=1)
        
        return torch.cat([bottom, top, left, right], dim=0)
    
    def get_skfem_mesh(self, nx, ny, mesh_type='tri'):
        """Genera una malla compatible con skfem usando sus rangos"""
        from skfem import MeshTri, MeshQuad
        x = np.linspace(self.x_range[0], self.x_range[1], nx + 1)
        y = np.linspace(self.y_range[0], self.y_range[1], ny + 1)
        if mesh_type == 'tri':
            return MeshTri.init_tensor(x, y)
        return MeshQuad.init_tensor(x, y)
    
'''
# TODO: implementar carga de malla externa
class MeshGeometry:
    def __init__(self, mesh_file):
        # Aquí cargaría los nodos de un .vtu o .obj
        self.nodes = load_mesh(mesh_file) 
    
    def sample_interior(self, n_points):
        # Retorna puntos aleatorios de dentro de la malla
        pass
'''
