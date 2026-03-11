"""pignn_utils.py"""

import torch
import numpy as np
import torch
import enum
from torch_geometric.data import Data
from config.physics import PoissonGeneral

def decompose_graph(graph):
    # Devolvemos x, edge_index y edge_attr
    return graph.x, graph.edge_index, graph.edge_attr

def copy_geometric_data(graph):
    ret = Data()
    for key, value in graph:   # Data es iterable: yield (key, tensor)
        setattr(ret, key, value)
    return ret

def update_graph_geometry(g):
    """
    Actualiza las distancias y diferencias relativas en el grafo 
    para asegurar que el grafo de computación de Autograd esté conectado.
    """
    s_idx, r_idx = g.edge_index
    # Diferencia viva: vinculada a g.pos para autograd
    diff = g.pos[r_idx] - g.pos[s_idx]
    dist = torch.norm(diff, dim=-1, keepdim=True)
    
    dims = g.pos.shape[1]
    
    # Actualizamos el bloque de geometría en edge_attr 
    # (Asumiendo que empieza en el índice 0: [diff, dist, ...])
    g.edge_attr = torch.cat([
        diff, 
        dist, 
        g.edge_attr[:, (dims + 1):] # Mantiene el resto intacto (flags, props, state)
    ], dim=-1)
    
    g.x = torch.cat([
        g.pos, 
        g.x[:, dims:] 
    ], dim=-1)

    return g

def set_node_props(graph, node_props=None):
    if node_props is None:
        node_props = torch.tensor([[1.0]], device=graph.x.device, dtype=torch.float32)

    dims = graph.pos.shape[1]
    start_idx = dims + 1

    material_ids = graph.x[:, start_idx].long()
    actual_props = node_props[material_ids]
    n_props = node_props.shape[1]
    graph.x[:, start_idx : start_idx + n_props] = actual_props

    return graph
    
def set_edge_props(graph, edge_props=None):
    if edge_props is None:
        edge_props = torch.tensor([[1.0]], device=graph.edge_attr.device, dtype=torch.float32)

    dims = graph.pos.shape[1]
    start_idx = dims + 2

    material_ids = graph.edge_attr[:, start_idx].long()
    actual_props = edge_props[material_ids]

    n_props = edge_props.shape[1]
    graph.edge_attr[:, start_idx : start_idx + n_props] = actual_props

    return graph

def FEM_to_PiGnnData(prob, node_out=1, edge_out=1, node_props=1, edge_props=1):
    phys = prob['physics']
    pos = torch.tensor(prob['doflocs'], dtype=torch.float32)
    num_nodes = pos.shape[0]
    dims = pos.shape[1]

    # almaceno fuerza evaluada puntualmente:
    f_pointwise=phys.source_term(pos)    

    # --- ATRIBUTOS DE NODO (x) ---
    node_type = torch.zeros((num_nodes, 1), dtype=torch.float32)
    node_type[prob['boundary_indices']] = 1.0           # Dirichlet por defecto
    is_bc_binary = (node_type > 0).float()

    node_material = torch.zeros((num_nodes, node_props))
    node_material[:, 0] = 0.0                           # Material ID 0 by default
    node_state = torch.zeros((num_nodes, node_out))     # Inicialización de u

    # Initiañlize state variables in BC nodes
    mask_bc = (node_type == 1.0).view(-1)
    if hasattr(phys, 'boundary_condition'):
        node_state[mask_bc] = phys.boundary_condition(pos[mask_bc]).view(-1, node_out)

    node_attr = torch.cat([
        pos,                    # dims
        node_type,              # 1
        node_material,          # node_props
        node_state              # node_out
        ], dim=-1)

    # --- ATRIBUTOS DE ARISTA (edge_attr) ---
    K_coo = prob['K'].tocoo()
    edge_index = torch.tensor(np.vstack((K_coo.row, K_coo.col)), dtype=torch.long)
    num_edges = edge_index.shape[1]
    senders, receivers = edge_index[0], edge_index[1]

    edge_pos_rel = pos[receivers] - pos[senders]
    edge_dist_norm = torch.norm(edge_pos_rel, p=2, dim=-1, keepdim=True)  

    # BCs  
    edge_type = (is_bc_binary[senders] * is_bc_binary[receivers]).view(-1, 1)

    edge_material = torch.zeros((num_edges, edge_props))
    edge_material[:, 0] = 0.0
    edge_state = torch.zeros((num_edges, edge_out)) # Inicialización de flujo

    edge_attr = torch.cat([
        edge_pos_rel,       # dims
        edge_dist_norm,     # 1
        edge_type,          # 1
        edge_material,      # edge_props
        edge_state          # edge_out
        ], dim=-1)

    # --- CONDICIONES DE CONTORNO ---
    u_bc = torch.zeros((num_nodes, 1), dtype=torch.float32)
    if hasattr(phys, 'boundary_condition'):
        mask_bc = (node_type == 1.0).view(-1)
        u_bc[mask_bc] = phys.boundary_condition(pos[mask_bc])

    # Store u_exact if available for validation
    y = None
    if prob.get('u_exact') is not None:
        y = torch.tensor(prob['u_exact'], dtype=torch.float32).view(-1, 1)

    data = Data(
        x=node_attr,            # Atributos estáticos de nodo
        edge_attr=edge_attr,    # Atributos estáticos de arista
        pos=pos,                # Geometría viva (para autograd/operadores)
        edge_index=edge_index, 
        is_bc=is_bc_binary,        # Clasificación bc-interior
        u_bc=u_bc,
        F=torch.tensor(prob['F'], dtype=torch.float32).view(-1, 1),
        node_volumes=torch.tensor(prob['M'].diagonal(), dtype=torch.float32).view(-1, 1),
        f_pointwise=f_pointwise,
        y=y,
    )

    # actualizamos propiedades
    data = set_node_props(data, None)
    data = set_edge_props(data, None)

    return data

def FEM_to_PiGnnData_BK(prob):
    """
    Convierte el output de FEM en un objeto Data compatible con MeshGraphNet.
    """
    basis = prob['basis']
    phys = prob['physics']
    
    # Node positions [x, y]
    pos = torch.tensor(prob['doflocs'], dtype=torch.float32)
    num_nodes = pos.shape[0]

    # Conectivity from K
    K_coo = prob['K'].tocoo()
    edge_index = torch.tensor(np.vstack((K_coo.row, K_coo.col)), dtype=torch.long)
    
    # Edge attributes: vectors and distances
    senders, receivers = edge_index
    diff = pos[receivers] - pos[senders]
    dist = torch.norm(diff, dim=-1, keepdim=True)
    edge_geom = torch.cat([diff, dist], dim=-1)
    flux_init = torch.zeros((edge_geom.size(0), 1))
    edge_attr = torch.cat([edge_geom, flux_init], dim=-1)

    # BC mask and bc values 
    is_bc = torch.zeros((num_nodes, 1), dtype=torch.float32)
    is_bc[prob['boundary_indices']] = 1.0
    
    u_bc = torch.zeros((num_nodes, 1), dtype=torch.float32)
    if hasattr(phys, 'boundary_condition'):
        mask_bool = is_bc.bool().view(-1)
        u_bc[mask_bool] = phys.boundary_condition(pos[mask_bool])

    # PDE terms (Integrated Source term & control volumes)
    F_tensor = torch.tensor(prob['F'], dtype=torch.float32).view(-1, 1)
    node_volumes = torch.tensor(prob['M'].diagonal(), dtype=torch.float32).view(-1, 1)

    # Node features: [pos_x, pos_y, is_boundary_mask, u_current]
    # Initialized u_current a 0.0
    u_init = torch.zeros((num_nodes, 1), dtype=torch.float32)
    node_attr = torch.cat([pos, is_bc, u_init], dim=-1) 

    # Store u_exact if available for validation
    y = None
    if prob.get('u_exact') is not None:
        y = torch.tensor(prob['u_exact'], dtype=torch.float32).view(-1, 1)

    data = Data(
        x=node_attr,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        y=y,           # Ground truth para validación
        is_bc=is_bc,   # Máscara 0-1
        u_bc=u_bc,     # Valores objetivos en la frontera
        F=F_tensor,    # Término f de la PDE (integrado)
        node_volumes=node_volumes
    )

    return data