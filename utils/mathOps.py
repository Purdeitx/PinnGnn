import numpy as np
import torch
import sys

from torch.mtia import graph
from torch_geometric.utils import scatter
def scatter_add(src, index, dim=0, dim_size=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='sum')
def scatter_mean(src, index, dim=0, dim_size=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='mean')


class GraphMathOps:
    # --- AUTOGRAD ---
    @staticmethod
    def laplacian_autograd(u, x, grads=None):
        if grads is None:
            # 1. Primera derivada: grad(u, x)
            grads = torch.autograd.grad(
                u, x, 
                grad_outputs=torch.ones_like(u),
                create_graph=True, 
                retain_graph=True, 
                allow_unused=True
            )[0]

        if grads is None:
            return torch.zeros_like(u)

        laplacian = torch.zeros_like(u)
        
        # 2. Segunda derivada por componente
        for i in range(x.shape[1]):
            grad_i = grads[:, i:i+1]
            
            # OJO: grad_outputs debe ser del tamaño de grad_i
            # Usamos un tensor de unos limpio
            target_grad = torch.ones_like(grad_i)
            
            grad_ii = torch.autograd.grad(
                grad_i, x, 
                grad_outputs=target_grad,
                create_graph=True, 
                retain_graph=True, 
                allow_unused=True
            )[0]

            if grad_ii is not None:
                comp_ii = grad_ii[:, i:i+1]
                # Si una componente da NaN, la neutralizamos para que no mate a toda la loss
                comp_ii = torch.where(torch.isnan(comp_ii), torch.zeros_like(comp_ii), comp_ii)
                laplacian = laplacian + comp_ii

        return laplacian
    
    @staticmethod
    def divergence_autograd(flux, x):
        """
        Calcula la divergencia ∇·q = dqx/dx + dqy/dy usando autogrado.
        flux: Tensor (N, dim) - El vector de flujo en cada nodo.
        x: Tensor (N, dim) - Posiciones.
        """
        spatial_dims = x.shape[1]
        div = torch.zeros((x.shape[0], 1), device=x.device)

        for i in range(spatial_dims):
            flux_i = flux[:, i:i+1]
            grad_flux_i = torch.autograd.grad(
                flux_i, x, grad_outputs=torch.ones_like(flux_i),
                create_graph=True, retain_graph=True, allow_unused=True
                )[0]
            
            if grad_flux_i is not None:
                div += grad_flux_i[:, i:i+1]

        return div 
    
    @staticmethod
    def gradient_autograd(u, x):
        """
        Calcula el gradiente ∇u = (du/dx, du/dy) usando autogrado.
        u: Tensor (N, 1)
        x: Tensor (N, 2) con posiciones que requieren gradiente.
        """
        grads = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        if grads is None:
            return torch.zeros_like(x)

        return grads
    
    # --- OPERADORES DE GRAFO (Formulación Fuerte Puntual) ---
    @staticmethod
    def laplacian_graph(u, graph):
        """ Δu puntual (GFD): 2D * sum( (u_j - u_i) / d_ij^2 ) / degree """
        if u.dim() == 1: u = u.unsqueeze(1)
        
        s, r = graph.edge_index
        # 1. Filtro mágico: ignorar self-loops
        mask = s != r
        senders, receivers = s[mask], r[mask]
        
        # 2. Extraer distancia sin miedo al 0
        diff = graph.pos[receivers] - graph.pos[senders]
        dist_sq = torch.sum(diff**2, dim=-1, keepdim=True) # [E_mask, 1]
        dist_sq = torch.clamp(dist_sq, min=1e-8) # Seguridad numérica

        D = graph.pos.shape[1]
        
        # 3. Suma ponderada de diferencias: (u_j - u_i) / d^2
        lap_edges = (u[receivers] - u[senders]) / dist_sq # [E_mask, 1]
        sum_lap = scatter_add(lap_edges, senders, dim=0, dim_size=graph.num_nodes)
        
        # 4. Conteo de vecinos reales
        ones = torch.ones_like(lap_edges)
        degree = scatter_add(ones, senders, dim=0, dim_size=graph.num_nodes)
        
        # 5. Normalización GFD (Añadimos clamp por si algún nodo queda aislado sin vecinos)
        laplacian_nodal = (2.0 * D) * sum_lap / torch.clamp(degree, min=1)

        return laplacian_nodal

    @staticmethod
    def gradient_discrete(u, graph):
        """ 
        ∇u puntual en los Nodos. 
        Retorna (N, D), directamente comparable con gradient_autograd.
        """
        if u.dim() == 1: u = u.unsqueeze(1)
        
        s, r = graph.edge_index
        mask = s != r
        senders, receivers = s[mask], r[mask]
        
        vec_ij = graph.pos[receivers] - graph.pos[senders] # Vector arista [E, D]
        dist_sq = torch.sum(vec_ij**2, dim=-1, keepdim=True).clamp(min=1e-8)
                
        # 1. Contribución de la arista
        diff = (u[receivers] - u[senders]) / dist_sq
        grad_edges = diff * vec_ij 
        
        # 2. Suma vectorial en los nodos
        sum_grad = scatter_add(grad_edges, senders, dim=0, dim_size=graph.num_nodes)
        
        # 3. Normalización GFD
        ones = torch.ones((len(senders), 1), device=u.device)
        degree = scatter_add(ones, senders, dim=0, dim_size=graph.num_nodes)
        
        D = graph.pos.shape[1]
        grad_nodal = float(D) * sum_grad / torch.clamp(degree, min=1.0)
        return grad_nodal

    @staticmethod
    def divergence_discrete(q, graph):
        """ 
        ∇·q puntual en los Nodos.
        q: Tensor (N, D) - El campo vectorial (ej. el flujo o -∇u).
        """
        s, r = graph.edge_index
        mask = s != r
        senders, receivers = s[mask], r[mask]
        
        vec_ij = graph.pos[receivers] - graph.pos[senders] # [E, D]
        dist_sq = torch.sum(vec_ij**2, dim=-1, keepdim=True).clamp(min=1e-8)
        D = graph.pos.shape[1]
        
        # 1. Diferencia del campo vectorial
        dq = q[receivers] - q[senders]
        
        # 2. Producto punto
        dot_prod = torch.sum(dq * vec_ij, dim=1, keepdim=True) # [E, 1]
        
        # 3. Suma ponderada
        div_edges = dot_prod / dist_sq
        sum_div = scatter_add(div_edges, senders, dim=0, dim_size=graph.num_nodes)
        
        # 4. Normalización GFD
        ones = torch.ones_like(dist_sq)
        degree = scatter_add(ones, senders, dim=0, dim_size=graph.num_nodes)
        
        div_nodal = float(D) * sum_div / torch.clamp(degree, min=1.0)
        return div_nodal

    @staticmethod
    def gradient_edge_discrete(u, graph):
        """
        Derivada direccional escalar evaluada en las aristas.
        Calcula (u_r - u_s) / d_ij para cada arista del grafo.
        
        u: (N, 1) - Potencial en los nodos
        Retorna: (E, 1) - Gradiente escalar proyectado en la dirección de la arista
        """

        if u.dim() == 1: u = u.unsqueeze(1)
            
        s, r = graph.edge_index
        vec_ij = graph.pos[r] - graph.pos[s] # Vector arista [E, D]
        dist = torch.norm(vec_ij, dim=-1, keepdim=True).clamp(min=1e-8)
        grad_u_edge = (u[r] - u[s]) / dist 
        
        return grad_u_edge

    @staticmethod
    def divergence_edge_aggregated(q_edge, graph):
        """
        Divergencia Topológica (Volúmenes Finitos en Grafo).
        Calcula ∇·q en los nodos a partir del flujo escalar predicho en las aristas.
        
        q_edge: (E, 1) - Flujo que va del nodo 's' (sender) al nodo 'r' (receiver).
        """
        s, r = graph.edge_index
        vec_ij = graph.pos[r] - graph.pos[s]
        dist = torch.norm(vec_ij, dim=-1, keepdim=True).clamp(min=1e-8)
        D = float(graph.pos.shape[1])
        
        # 1. Ponderamos el flujo escalar de la arista por la distancia geométrica
        div_edges = q_edge / dist
        
        # 2. Sumamos todo el flujo que "SALE" del nodo (agrupamos por 's')
        net_flux = scatter_add(div_edges, s, dim=0, dim_size=graph.num_nodes) - \
               scatter_add(div_edges, r, dim=0, dim_size=graph.num_nodes)

        sum_flux = net_flux / 2.0
        # sum_flux = scatter_add(div_edges, s, dim=0, dim_size=graph.num_nodes)
        
        # 3. Normalización topológica (GFD)
        ones = torch.ones_like(div_edges)
        degree = scatter_add(ones, s, dim=0, dim_size=graph.num_nodes)
        
        div_nodal = (2.0 * D) * sum_flux / torch.clamp(degree, min=1.0)
        return div_nodal
    

class PhysValidation:
    def __init__(self, physics, printed=25):
        self.physics = physics
        self.math = physics.ops
        self.min_print = printed 

    def _get_base_data(self, graph):
        """Prepara las coordenadas, la u exacta y las máscaras."""
        pos = graph.pos.detach().clone().requires_grad_(True)
        u_exact = self.physics.exact_solution(pos)
        if u_exact.dim() == 1: u_exact = u_exact.unsqueeze(1)
        is_bc_tensor = graph.x[:, 0].detach() == 1
        return pos, u_exact, is_bc_tensor.cpu().numpy(), ~is_bc_tensor
    
    def _print_table(self, title, headers, data_cols, coords, is_bc_cpu, print_bc, is_edge=False):
        """Genera una tabla ASCII dinámica adaptada al número de columnas pasadas."""
        import numpy as np # Asegúrate de tenerlo importado arriba en tu script
        
        # Convertimos las columnas a CPU/numpy para impresión segura
        cols = [c.flatten().detach().cpu().numpy() if torch.is_tensor(c) else c for c in data_cols]
        total_items = len(cols[0])
        
        # 1. Filtrar los índices que son VÁLIDOS para imprimir
        if is_edge:
            valid_indices = np.arange(total_items)
        else:
            if print_bc:
                valid_indices = np.arange(total_items)
            else:
                # Nos quedamos solo con los índices donde is_bc_cpu es Falso
                valid_indices = np.where(~np.array(is_bc_cpu).astype(bool))[0]
                
        # 2. Control de seguridad por si no hay nada que imprimir
        if len(valid_indices) == 0:
            sys.stdout.write(f"\n{title}\n[Malla sin datos válidos para los filtros actuales]\n")
            return
            
        # 3. Muestreo espacial homogéneo si hay más puntos que el límite
        if len(valid_indices) <= self.min_print:
            selected_indices = valid_indices
        else:
            # linspace saca 'min_print' valores equiespaciados entre el primer y último índice válido
            step_indices = np.linspace(0, len(valid_indices) - 1, self.min_print, dtype=int)
            selected_indices = valid_indices[step_indices]

        # 4. Cabeceras
        sys.stdout.write(f"\n{title}\n")
        header_str = f"║ {'ID':<5} │ {'Nodos/Coords':<15} │ " + " │ ".join([f"{h:<18}" for h in headers]) + " ║"
        border = "═" * (len(header_str) - 2)
        
        sys.stdout.write(f"╔{border}╗\n{header_str}\n")
        sys.stdout.write("╠" + "═"*7 + "╪" + "═"*17 + "╪" + "╪".join(["═"*20]*len(headers)) + "╣\n")
        
        # 5. Filas (ahora iteramos solo sobre la muestra perfecta)
        for i in selected_indices:
            if is_edge:
                coord_str = f"{coords[0][i]:>4} -> {coords[1][i]:<4}"
            else:
                is_bc = is_bc_cpu[i]
                coord_str = f"{coords[i][0]:.2f},{coords[i][1]:.2f}{' (B)' if is_bc else '':<5}"
            
            row = f"║ {i:<5} │ {coord_str:<15} │ " + " │ ".join([f"{col[i]:>18.6e}" for col in cols]) + " ║\n"
            sys.stdout.write(row)
            
        sys.stdout.write(f"╚{border}╝\n")

    def _calc_ratio(self, exact, graph_val, mask, name, is_magnitude=False):
        """Calcula e imprime el ratio de escala para corregir operadores discretos."""
        if is_magnitude:
            ex, gr = torch.norm(exact[mask], dim=1), torch.norm(graph_val[mask], dim=1)
        else:
            ex, gr = exact.flatten()[mask], graph_val.flatten()[mask]
            
        valid = torch.abs(ex) > 1e-5
        if valid.any():
            ratio = (gr[valid] / ex[valid]).mean().item()
            sys.stdout.write(f" RATIO {name:<25}: {ratio:.4f} ")
            if abs(ratio - 1.0) > 0.02:
                sys.stdout.write(f"--> Sugerencia: Multiplicar operador por {1.0/ratio:.4f}\n")
            else:
                sys.stdout.write("--> OK\n")
        else:
            sys.stdout.write(f" RATIO {name:<25}: No calculable (analítico nulo en interior)\n")
        sys.stdout.flush()

    # Funciones de verificación 
    def verifica_laplacian(self, graph, print_bc=False):
        pos, u, bc_cpu, mask = self._get_base_data(graph)
        _, _, lap_hand = self.physics.analytical_derivatives(pos)
        lap_auto = self.math.laplacian_autograd(u, pos) 
        lap_graph = self.math.laplacian_graph(u.detach(), graph)

        self._print_table("TEST: LAPLACIANO (Analítico vs Autograd vs Grafo)", 
                          ["Analítico", "Autograd", "Grafo (FD)"], [lap_hand, lap_auto, lap_graph], 
                          pos.detach().cpu().numpy(), bc_cpu, print_bc) # <-- AÑADIDO .detach()
        self._calc_ratio(lap_hand, lap_graph, mask, "Laplaciano")

    def verifica_gradiente(self, graph, print_bc=False):
        pos, u, bc_cpu, mask = self._get_base_data(graph)
        g_hand, _, _ = self.physics.analytical_derivatives(pos)
        g_auto = self.math.gradient_autograd(u, pos)
        g_graph = self.math.gradient_discrete(u.detach(), graph)

        coords = pos.detach().cpu().numpy() # <-- AÑADIDO .detach()
        self._print_table("TEST: GRADIENTE X", ["Handmade X", "Autograd X", "Grafo X"], 
                          [g_hand[:,0], g_auto[:,0], g_graph[:,0]], coords, bc_cpu, print_bc)
        self._print_table("TEST: GRADIENTE Y", ["Handmade Y", "Autograd Y", "Grafo Y"], 
                          [g_hand[:,1], g_auto[:,1], g_graph[:,1]], coords, bc_cpu, print_bc)
        self._calc_ratio(g_hand, g_graph, mask, "Gradiente (Magnitud)", is_magnitude=True)

    def verifica_divergencia(self, graph, print_bc=False):
        pos, u, bc_cpu, mask = self._get_base_data(graph)
        grad_hand, div_hand, _ = self.physics.analytical_derivatives(pos)
        flux_hand = -grad_hand
        
        flux_auto = -self.math.gradient_autograd(u, pos)
        div_auto = self.math.divergence_autograd(flux_auto, pos)

        with torch.no_grad():
            # flux_graph = -self.math.gradient_discrete(u.detach(), graph)
            div_graph = self.math.divergence_discrete(flux_hand.detach(), graph)

        self._print_table("TEST: DIVERGENCIA", ["Handmade", "Autograd", "Grafo"], 
                          [div_hand, div_auto, div_graph], pos.detach().cpu().numpy(), bc_cpu, print_bc) # <-- AÑADIDO .detach()
        self._calc_ratio(div_hand, div_graph, mask, "Divergencia")

    def verifica_operadores_gfd_grafo(self, graph, print_bc=False):
        pos, u, bc_cpu, mask = self._get_base_data(graph)
        g_hand, d_hand, l_hand = self.physics.analytical_derivatives(pos)
        
        with torch.no_grad():
            l_graph = self.math.laplacian_graph(u, graph)
            g_graph = self.math.gradient_discrete(u, graph)
            d_graph = self.math.divergence_discrete(-g_hand, graph)

        coords = pos.detach().cpu().numpy() # <-- AÑADIDO .detach()
        self._print_table("GFD: LAPLACIANO", ["Analítico", "Grafo (GFD)"], [l_hand, l_graph], coords, bc_cpu, print_bc)
        self._print_table("GFD: DIVERGENCIA", ["Analítico", "Grafo (GFD)"], [d_hand, d_graph], coords, bc_cpu, print_bc)
        self._print_table("GFD: GRADIENTE X", ["Analítico X", "Grafo X"], [g_hand[:,0], g_graph[:,0]], coords, bc_cpu, print_bc)
        
        sys.stdout.write("\n" + "="*45 + "\n RATIOS DE ESCALA GFD\n" + "="*45 + "\n")
        self._calc_ratio(l_hand, l_graph, mask, "Laplaciano")
        self._calc_ratio(d_hand, d_graph, mask, "Divergencia")
        self._calc_ratio(g_hand, g_graph, mask, "Gradiente", is_magnitude=True)
        sys.stdout.write("\n")

    def verifica_operadores_mixtos_grafo(self, graph, print_bc=False):
        pos, u, bc_cpu, mask = self._get_base_data(graph)
        s, r = graph.edge_index
        
        # 1. EVALUACIÓN EN EL PUNTO MEDIO (Para consistencia con diferencias finitas)
        pos_mid = (pos[s] + pos[r]) / 2.0
        # Obtenemos el gradiente analítico en los puntos medios
        g_hand_mid, d_hand_node, _ = self.physics.analytical_derivatives(pos_mid)
        
        edge_vec = pos[r] - pos[s]
        dist = torch.norm(edge_vec, dim=-1, keepdim=True) + 1e-8
        unit_vec = edge_vec / dist
        
        # Gradiente analítico proyectado en la arista (en el centro)
        g_exact_edge = torch.sum(g_hand_mid * unit_vec, dim=1, keepdim=True)
        
        # 2. OPERADOR DISCRETO
        g_disc_edge = self.math.gradient_edge_discrete(u, graph)
        
        # 3. DIVERGENCIA (Sin dividir por volumen, ya que el operador es GFD)
        # d_hand_node debe evaluarse en los nodos para comparar con la salida de div_agg
        _, d_hand_nodal_exact, _ = self.physics.analytical_derivatives(pos)
        d_disc_node = self.math.divergence_edge_aggregated(-g_exact_edge, graph)

        self._print_table("TEST MIXTO: FICK EN ARISTAS", 
                        ["Grad Analítico (Mid)", "Grad Discreto (Arista)"], 
                        [g_exact_edge, g_disc_edge], [s, r], bc_cpu, print_bc, is_edge=True)
        
        self._print_table("TEST MIXTO: DIVERGENCIA NODAL", 
                        ["Div Analítica (f)", "Div Agregada GFD"], 
                        [d_hand_nodal_exact, d_disc_node], pos.detach().cpu().numpy(), bc_cpu, print_bc)
        
        # Ratios (Deberían ser muy cercanos a 1.0 ahora)
        self._calc_ratio(g_exact_edge, g_disc_edge, torch.ones_like(g_disc_edge).bool(), "Fick (Aristas)")
        self._calc_ratio(d_hand_nodal_exact, d_disc_node, mask, "Divergencia (Nodos)")
        
    def verifica_predicciones_red(self, system, graph, print_bc=False):
        """
        Evalúa los operadores aplicados sobre las PREDICCIONES de la red entrenada.
        """
        # 1. Extraer datos base y APLANAR LA MÁSCARA para evitar IndexErrors
        pos, _, bc_cpu, mask_interior = self._get_base_data(graph)
        mask_interior = mask_interior.flatten()  
        
        # 2. Poner el sistema en modo evaluación y predecir
        system.eval()
        with torch.no_grad():
            u_pred_tensor, flux_pred_tensor = system(graph.to(system.device))
            u_pred = u_pred_tensor.cpu()
            if u_pred.dim() == 1: u_pred = u_pred.unsqueeze(1)
            graph_cpu = graph.cpu()

        # 3. Obtener el Ground Truth Analítico
        g_hand, d_hand, l_hand = self.physics.analytical_derivatives(pos)
        
        # 4. Calcular Operadores sobre la variable de estado predicha (EL FLUJO IMPLÍCITO)
        l_pred = self.math.laplacian_graph(u_pred, graph_cpu)
        g_pred = self.math.gradient_discrete(u_pred, graph_cpu)  
        d_pred = self.math.divergence_discrete(-g_pred, graph_cpu)

        # 5. Impresión de Tablas Comparativas
        coords = pos.detach().cpu().numpy()
        self._print_table("EVALUACIÓN RED: LAPLACIANO", 
                          ["Analítico Exacto", "Lap(u_pred)"], 
                          [l_hand, l_pred], coords, bc_cpu, print_bc)
        
        self._print_table("EVALUACIÓN RED: GRADIENTE X (Flujo Implícito)", 
                          ["Analítico X", "Grad_X(u_pred)"], 
                          [g_hand[:,0], g_pred[:,0]], coords, bc_cpu, print_bc)
        
        self._print_table("EVALUACIÓN RED: DIVERGENCIA", 
                          ["Analítico Exacto", "Div(-Grad(u_pred))"], 
                          [d_hand, d_pred], coords, bc_cpu, print_bc)

        # 6. Cálculo de Ratios y Métricas
        sys.stdout.write("\n" + "═"*55 + "\n RATIOS DE PREDICCIÓN (RED vs ANALÍTICO)\n" + "═"*55 + "\n")
        
        # Ahora pasamos la máscara aplanada a _calc_ratio
        self._calc_ratio(l_hand, l_pred, mask_interior, "Laplaciano Predicho")
        self._calc_ratio(d_hand, d_pred, mask_interior, "Divergencia Predicha")
        self._calc_ratio(g_hand, g_pred, mask_interior, "Gradiente Predicho", is_magnitude=True)
        sys.stdout.write("\n")
