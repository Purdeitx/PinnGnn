"""utils.py"""

import torch
import numpy as np
import torch
import enum
from torch_geometric.data import Data

def decompose_graph(graph):
    return (graph.x, graph.edge_index, graph.edge_attr)

def copy_geometric_data(graph):
    node_attr, edge_index, edge_attr = decompose_graph(graph)
    ret = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
    return ret

def FEM_to_GraphData(prob):
    """
    Convierte el output de FEM en un objeto Data compatible con MeshGraphNet.
    """
    basis = prob['basis']
    
    # Node positions [x, y]
    pos = torch.tensor(basis.doflocs.T, dtype=torch.float32)
    
    # Connectivity from Stiffness matrix
    K_coo = prob['K'].tocoo()
    edge_index = torch.tensor(np.vstack((K_coo.row, K_coo.col)), dtype=torch.long)
    
    # Node features
    # [x, y, is_boundary]
    dofs_container = basis.get_dofs(dict(markers=[1, 2, 3, 4]))
    all_indices = []
    if isinstance(dofs_container, dict):
        for v in dofs_container.values(): all_indices.extend(v.all().flatten())
    else:
        all_indices.extend(dofs_container.all().flatten())
        
    # Añadimos el boundary_mask como una propiedad fundamental
    boundary_mask = torch.zeros((pos.shape[0], 1))
    boundary_mask[np.unique(all_indices)] = 1.0
    # x: [pos_x, pos_y, node_type_enc] 
    # node_attr = torch.cat([pos, boundary_mask], dim=-1)

    # Añadimos un canal para la variable física (inicializado a 0 o a la u del paso anterior)
    u_init = torch.zeros((pos.shape[0], 1)) 
    # x: [pos_x, pos_y, node_type_enc, u] 
    node_attr = torch.cat([pos, boundary_mask, u_init], dim=-1) 

    # Edge attributes: vectors and distances
    senders, receivers = edge_index
    diff = pos[senders] - pos[receivers]
    dist = torch.norm(diff, dim=-1, keepdim=True)
    edge_attr = torch.cat([diff, dist], dim=-1)
    
    # TARGET (u)
    y = torch.tensor(prob['u_exact'], dtype=torch.float32).view(-1, 1)

    data = Data(
        x=node_attr,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        pos=pos # Guardamos pos por separado por si acaso
    )

    return data