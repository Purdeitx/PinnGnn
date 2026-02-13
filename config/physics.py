import numpy as np
import torch

# this is especific for 2D
class PoissonPhysics:
    """
    Versión optimizada 2D. 
    Interfaz compatible con PoissonGeneral.
    """
    def __init__(self, source_type='sine', scale=1.0):
        self.source_type = source_type
        self.scale = scale

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
        laplacian = self.compute_laplacian(x, grads)
        f = self.source_term(x) # Llamada unificada
        return laplacian + f

class PoissonGeneral:
    def __init__(self, source_type='sine', scale=1.0):
        self.source_type = source_type
        self.scale = scale

    def source_term(self, x):
        """Detecta automáticamente si x es Torch o Numpy y calcula f(x)"""
        if torch.is_tensor(x):
            if self.source_type == 'sine':
                dims = x.shape[1]
                # torch.prod da fallos en GPU con tensores 2D, así que hacemos el producto manualmente
                # source = torch.prod(torch.sin(torch.pi * x), dim=1, keepdim=True)

                # 2D solution: manual implementation 
                # source = torch.sin(torch.pi * x[:, 0:1]) * torch.sin(torch.pi * x[:, 1:2])

                # ND solution: general implementation (descomentar si se quiere probar en 3D o más)
                source = torch.sin(torch.pi * x[:, 0:1])
                for d in range(1, dims):
                    source = source * torch.sin(torch.pi * x[:, d:d+1])
                
                return self.scale * dims * (torch.pi**2) * source
            return torch.ones((x.shape[0], 1), device=x.device) * self.scale
        else:
            if self.source_type == 'sine':
                dims = x.shape[1]
                source = np.prod(np.sin(np.pi * x), axis=1, keepdims=True)
                return self.scale * dims * (np.pi**2) * source
            return np.ones((x.shape[0], 1)) * self.scale
        
    def exact_solution(self, x):
        """u(x) analítica (Numpy)"""
        if self.source_type == 'sine':
            return self.scale * np.prod(np.sin(np.pi * x), axis=1, keepdims=True)
        return None
    
    def boundary_condition(self, x, library='torch'):  
        """u(boundary) = 0"""
        if library == 'torch':
            return torch.zeros((x.shape[0], 1), device=x.device)
        return np.zeros((x.shape[0], 1)) 

    def compute_laplacian(self, x, grads):
        """Δu generalista"""
        laplacian = 0
        dims = x.shape[1]
        for i in range(dims):
            grad_ii = torch.autograd.grad(
                grads[:, i], x, 
                grad_outputs=torch.ones_like(grads[:, i]),
                create_graph=True,
                retain_graph=True
            )[0][:, i:i+1]
            laplacian += grad_ii
        return laplacian

    def compute_pde_residual(self, x, u, grads):
        """R = Δu + f"""
        laplacian = self.compute_laplacian(x, grads)
        f = self.source_term(x) # Llamada limpia sin especificar librería
        return laplacian + f

