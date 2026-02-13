import torch
import numpy as np

class PINNSampler:
    def __init__(self, geometry, device='cpu'):
        self.geo = geometry
        self.device = device

    def sample_interior(self, n_points):
        if self.geo.name == "square":
            x = torch.rand(n_points, 1) * (self.geo.x_range[1] - self.geo.x_range[0]) + self.geo.x_range[0]
            y = torch.rand(n_points, 1) * (self.geo.y_range[1] - self.geo.y_range[0]) + self.geo.y_range[0]
            pts = torch.cat([x, y], dim=1)
        
        elif self.geo.name == "circle":
            r = self.geo.radius * torch.sqrt(torch.rand(n_points, 1))
            theta = 2 * np.pi * torch.rand(n_points, 1)
            x = self.geo.center[0] + r * torch.cos(theta)
            y = self.geo.center[1] + r * torch.sin(theta)
            pts = torch.cat([x, y], dim=1)

        else:
            raise NotImplementedError(
                f"Interior sampling not implemented for geometry '{self.geo.name}'."
            )
            
        return pts.to(self.device).requires_grad_(True)

    def sample_boundary(self, n_points):
        ## TODO:
        # Aquí implementamos la lógica según el tipo para que los puntos 
        # caigan exactamente en el borde (perímetro o lados)
        if self.geo.name == "square":
            """Genera puntos para las condiciones de contorno"""
            n_per_side = n_points // 4
            # tomo los limites geometricos del contorno: 
            x0, x1 = self.geo.x_range
            y0, y1 = self.geo.y_range
            r = torch.rand(n_per_side)

            # Lados del cuadrado
            bottom = torch.stack([r * (x1 - x0) + x0, torch.full_like(r, y0)], dim=1)
            top    = torch.stack([r * (x1 - x0) + x0, torch.full_like(r, y1)], dim=1)
            left   = torch.stack([torch.full_like(r, x0), r * (y1 - y0) + y0], dim=1)
            right  = torch.stack([torch.full_like(r, x1), r * (y1 - y0) + y0], dim=1)
            
            return torch.cat([bottom, top, left, right], dim=0).to(self.device)
        elif self.geo.name == "circle":
            theta = 2 * np.pi * torch.rand(n_points, 1)
            x = self.geo.center[0] + self.geo.radius * torch.cos(theta)
            y = self.geo.center[1] + self.geo.radius * torch.sin(theta)
            return torch.cat([x, y], dim=1).to(self.device)
        else:
            raise NotImplementedError(
                f"Interior sampling not implemented for geometry '{self.geo.name}'."
            )
