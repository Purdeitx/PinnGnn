import numpy as np
import torch
import sys
from utils.mathOps import GraphMathOps

from torch_geometric.utils import scatter
def scatter_add(src, index, dim=0, dim_size=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='sum')
def scatter_mean(src, index, dim=0, dim_size=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='mean')

class PoissonGeneral:
    def __init__(self, source_type='sine', scale=1.0, bc_type='zero'):
        self.source_type = source_type
        self.scale = scale
        self.bc_type = bc_type
        self.ops = GraphMathOps()

    def exact_solution(self, x):
        """Calcula la solución analítica unificando todo el cálculo en PyTorch."""
        is_numpy = isinstance(x, np.ndarray)
        # Convertimos a tensor si es numpy (asegurando float32)
        x_t = torch.tensor(x, dtype=torch.float32) if is_numpy else x
        
        scale = self.scale
        
        if self.source_type == 'sine':
            out = scale * torch.prod(torch.sin(torch.pi * x_t), dim=1, keepdim=True)
            
        elif self.source_type == 'linear':
            out = scale * torch.sum(x_t, dim=1, keepdim=True)
            
        elif self.source_type == 'poly2':
            out = scale * torch.sum(x_t**2, dim=1, keepdim=True)
            
        else:
            raise NotImplementedError(f"Solución exacta no implementada para: {self.source_type}")

        # Devolvemos numpy o tensor según lo que entró
        return out.detach().cpu().numpy() if is_numpy else out

    def source_term(self, x):
        """Calcula el término fuente unificando todo el cálculo en PyTorch."""
        is_numpy = isinstance(x, np.ndarray)
        x_t = torch.tensor(x, dtype=torch.float32) if is_numpy else x
        
        dims = x_t.shape[1]
        scale = self.scale
        
        if self.source_type == 'sine':
            u = torch.prod(torch.sin(torch.pi * x_t), dim=1, keepdim=True)
            out = scale * dims * (torch.pi**2) * u
            
        elif self.source_type == 'linear':
            out = torch.zeros((x_t.shape[0], 1), device=x_t.device)
            
        elif self.source_type == 'poly2':
            f_val = - 2.0 * dims * scale
            out = torch.full((x_t.shape[0], 1), f_val, device=x_t.device)
            
        else:
            raise NotImplementedError(f"Término fuente no implementado para: {self.source_type}")

        return out.detach().cpu().numpy() if is_numpy else out
    
    def analytical_derivatives(self, x):
        """
        Calcula de forma exacta (analítica) el gradiente, la divergencia del flujo
        (div(-grad u)) y el laplaciano para el source_type actual.
        Devuelve: (gradiente, divergencia, laplaciano)
        """
        is_numpy = isinstance(x, np.ndarray)
        x_t = torch.tensor(x, dtype=torch.float32) if is_numpy else x
        
        A = self.scale
        dims = x_t.shape[1]
        
        if self.source_type == 'sine':
            # u = A * prod(sin(pi * x_i))
            u = A * torch.prod(torch.sin(torch.pi * x_t), dim=1, keepdim=True)
            
            # Gradiente: Regla de la cadena adaptada para N dimensiones
            grad = torch.zeros_like(x_t)
            for j in range(dims):
                grad_j = A * torch.pi * torch.cos(torch.pi * x_t[:, j:j+1])
                for i in range(dims):
                    if i != j:
                        grad_j = grad_j * torch.sin(torch.pi * x_t[:, i:i+1])
                grad[:, j:j+1] = grad_j
                
            laplacian = - (torch.pi**2) * dims * u
            divergence = -laplacian # div(-grad u) = -Laplacian(u) en Poisson
            
        elif self.source_type == 'linear':
            # u = A * (x + y + ...)
            grad = torch.full_like(x_t, A)
            laplacian = torch.zeros((x_t.shape[0], 1), device=x_t.device)
            divergence = torch.zeros((x_t.shape[0], 1), device=x_t.device)
            
        elif self.source_type == 'poly2':
            # u = A * (x^2 + y^2 + ...)
            grad = 2.0 * A * x_t
            laplacian = torch.full((x_t.shape[0], 1), 2.0 * A * dims, device=x_t.device)
            divergence = -laplacian
            
        else:
            raise NotImplementedError(f"Derivadas analíticas no implementadas para: {self.source_type}")
            
        if is_numpy:
            return grad.detach().cpu().numpy(), divergence.detach().cpu().numpy(), laplacian.detach().cpu().numpy()
        return grad, divergence, laplacian

    
    def boundary_condition(self, x):  
        """
        Aplica la condición de contorno según el bc_type configurado.
        Todo el cálculo interno se hace en PyTorch.
        """
        # TODO: condiciones de contorno más complejas (ej. u=0 en x=0, solución exacta en el resto)
        # is_bc actual 0-1, 0 interior, 1 frontera con Dirichlet u=0
        # is_bc pro:
        # 0 - nodo interior
        # 1 - nodo de frontera, condicion Dirichlet u = u0
        # 2 - nodo de frontera, condicion Neumann du/dn = g0                (no implementada aún)
        # 3 - nodo de frontera, condicion Neumann flujo libre, du/dn = 0    (no implementada aún)
        is_numpy = isinstance(x, np.ndarray)
        x_t = torch.tensor(x, dtype=torch.float32) if is_numpy else x
        
        if self.bc_type == 'exact':
            out = self.exact_solution(x_t)
            
        elif self.bc_type == 'zero':
            out = torch.zeros((x_t.shape[0], 1), device=x_t.device)
            
        elif self.bc_type == 'zero_x0':
            # 0 en la pared izquierda (x=0). El resto asume la solución exacta 
            # TODO: usar solucion exacta es hacer trampa, pero vamos paso a paso
            out = self.exact_solution(x_t)
            x_min = torch.min(x_t[:, 0])
            mask_x0 = torch.abs(x_t[:, 0:1] - x_min) < 1e-7
            out[mask_x0] = 0.0
            
        else:
            raise NotImplementedError(f"Tipo de condición de contorno no soportada: {self.bc_type}")
            
        # Devolvemos numpy o tensor según la entrada original
        return out.detach().cpu().numpy() if is_numpy else out

    def compute_pde_residual(self, x, u, grads):
        """R = Δu + f"""
        laplacian = self.ops.laplacian_autograd(u=u, x=x, grads=grads)
        f = self.source_term(x) # Llamada limpia sin especificar librería
        return laplacian + f
    
    def compute_graph_residual(self, u_pred, graph, strong_form=True, autograd=False):
        """Residual específico para PiGNN.
        """
        num_nodes = graph.pos.size(0)
        dims = graph.pos.shape[1]

        if autograd:
            grad_u = self.ops.gradient_autograd(u=u_pred, x=graph.pos) 
            laplacian = self.ops.divergence_discrete(q=grad_u, graph=graph)
            # laplacian = self.ops.laplacian_autograd(u=u_pred, x=graph.pos)
        else:
            laplacian = self.ops.laplacian_graph(u_pred, graph)

        if strong_form:
            f_pointwise = self.source_term(graph.pos)
            residual = laplacian + f_pointwise
        else:
            node_volumes = getattr(graph, 'node_volumes', torch.ones((num_nodes, 1), device=u_pred.device))
            f_integrated = graph.F       
            residual = laplacian * node_volumes + f_integrated

        # TODO: si hay mas de un tipo de contorno puede no ser 0-1 
        bc_mask = graph.is_bc

        return residual * (1.0 - bc_mask)

    def compute_graph_residual_with_flux(self, u_pred, flux_pred, graph, strong_form=True, autograd=False):
        """
        Residual para PiGNN usando flujo.        
        u_pred: (N, 1) - Potencial en los nodos
        flux_pred: (E, 1) - Flujo escalar predicho en las aristas
        autograd: Si True, usa derivadas automáticas; si False, usa operadores de grafo.
        """
        num_nodes = graph.pos.size(0)
        u_pred = u_pred.view(-1, 1)
        flux_pred = flux_pred.view(-1, 1)
        # senders, receivers
        s, r = graph.edge_index
        edge_vec = graph.pos[r] - graph.pos[s]
        dist = torch.norm(edge_vec, dim=-1, keepdim=True) + 1e-8
        unit_vec = edge_vec / dist

        # 1. LEY DE FICK/FOURIER (q = -∇u) -> Edges
        if autograd:
            grad_u_node = self.ops.gradient_autograd(u_pred, graph.pos) # (N, D)
            grad_u_edge = torch.sum(grad_u_node[s] * unit_vec, dim=1, keepdim=True) # (E, 1)
        else:
            # grad_u_node = self.ops.gradient_discrete(u_pred, graph) # (N, D)
            # grad_u_edge = torch.sum(grad_u_node[s] * unit_vec, dim=1, keepdim=True) # (E, 1)
            grad_u_edge = self.ops.gradient_edge_discrete(u_pred, graph)

        loss_Fick = flux_pred + grad_u_edge # q = -∇u -> residual Fick = q + ∇u

        # 2. LEY DE CONSERVACIÓN (∇·q = f) -> NODOS
        flux_vec_edges = flux_pred * unit_vec
        q_nodal = scatter_mean(flux_vec_edges, s, dim=0, dim_size=num_nodes)
        if autograd:
            divergence = self.ops.divergence_autograd(q_nodal, graph.pos) # (N, 1)
        else: 
            # divergence = self.ops.divergence_discrete(q_nodal, graph) # (N, 1)
            divergence = self.ops.divergence_edge_aggregated(flux_pred, graph) # (N, 1)

        # 3. CÁLCULO DEL RESIDUO FINAL
        f_target = self.source_term(graph.pos) if strong_form else graph.F

        bc_mask = graph.is_bc
        # Relacion q/u:     div(q) = div(-grad u) = -Laplacian(u)
        # La PDE es:        Laplacian(u) + f = 0
        # Loss es:          -div(q) + f = 0  =>  div(q) - f = 0
        loss_pde = (divergence - f_target) * (1.0 - bc_mask) # div(q) = f -> residual PDE = div(q) - f

        return loss_pde, loss_Fick

    # Desde aqui todo es opcional
    def residuo_FEM(self, u_pred, graph):
        K_coo = graph.K  
        F = graph.F
        residual = torch.sparse.mm(K_coo, u_pred) - F
        return residual


