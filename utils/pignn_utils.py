"""pignn_utils.py"""

import torch
import numpy as np
import torch
import enum
import random
import pandas as pd
from torch_geometric.data import Data
from config.physics import PoissonGeneral
from utils.geometry import geometry_factory
from FEM.fem_solver import get_problem, solve_problem

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

def FEM_to_PiGnnData(prob, node_out=1, edge_out=1, node_props=1, edge_props=1, use_exact_as_target=False):
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
    u_fem = solve_problem(prob)
    u_fem_torch = torch.tensor(u_fem, dtype=torch.float32).view(-1, 1)
    u_exact = prob.get('u_exact')
    if u_exact is not None:
        u_exact_torch = torch.tensor(u_exact, dtype=torch.float32).view(-1, 1)
    else:
        u_exact_torch = u_fem_torch.clone()
        
    y_target = u_exact_torch if (use_exact_as_target and u_exact is not None) else u_fem_torch

    u_bc = torch.zeros((num_nodes, 1), dtype=torch.float32)
    if hasattr(phys, 'boundary_condition'):
        mask_bc = (node_type == 1.0).view(-1)
        u_bc[mask_bc] = phys.boundary_condition(pos[mask_bc])

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
        y=y_target,
    )

    # actualizamos propiedades
    data = set_node_props(data, None)
    data = set_edge_props(data, None)

    return data

def generate_pignn_dataset(n_samples=50, 
                           physics_class=PoissonGeneral,  # Pasamos la CLASE
                           physics_fixed_args={'bc_type': 'exact'},
                           sources=['sine', 'gaussian_peak'], 
                           mesh_types=['quad', 'tri'],
                           scale_range=(1.0, 5.0),
                           freq_range=(1.0, 2.5),
                           domain_range=(-1.0, 1.0),
                           res_range=(12, 32),
                           get_resume=False):
    dataset = []
    summary_list = []
    
    for i in range(n_samples):
        res = random.randint(*res_range)
        src = random.choice(sources)
        m_type = random.choice(mesh_types)
        scale = random.uniform(*scale_range)
        freq = random.uniform(*freq_range)
        
        x_coords = sorted([random.uniform(*domain_range) for _ in range(2)])
        y_coords = sorted([random.uniform(*domain_range) for _ in range(2)])
        # (Añadir validación de ancho mínimo aquí si se desea)

        geom = geometry_factory('square', x_range=x_coords, y_range=y_coords)
        
        # --- INSTANCIACIÓN DINÁMICA ---
        # Aquí es donde ocurre la magia: creamos la física sea cual sea la clase pasada
        phys = physics_class(
            source_type=src, 
            scale=scale, 
            freq=freq,
            **physics_fixed_args
        )
        
        prob = get_problem(geometry=geom, physics=phys, nx=res, ny=res, mesh=m_type)
        data = FEM_to_PiGnnData(prob)
        dataset.append(data)
        
        if get_resume:
            summary_list.append({
                'ID': i, 'Source': src, 'Mesh': m_type, 
                'Nodes': data.num_nodes, 'Scale': round(scale, 2), 'Freq': round(freq, 2)
            })

    if get_resume:
        print("\n DATASET AGNOSTICO GENERADO (Resumen)")
        print(pd.DataFrame(summary_list).to_string(index=False))
        
    return dataset

def generate_benchmark_datasets(res_range=(12, 24), 
                                mesh_types=['quad'], samples_per_domain=10):
    """
    Genera un set de entrenamiento fijo basado en dominios específicos
    y un set de validación OOD (0-4).
    """
    # 1. Definición de los dominios de entrenamiento "de seguridad"
    train_specs = [
        ([-1.0, 1.0], [-1.0, 1.0]),
        ([0.0, 1.0], [0.0, 1.0]),
        ([0.0, 2.0], [0.0, 2.0])
    ]
    
    train_graphs = []
    print("Generando Benchmark Training Set...")
    
    for x_rng, y_rng in train_specs:
        for i in range(samples_per_domain):
            # Variamos la resolución para que aprenda a ser agnóstico a la densidad
            res = random.randint(*res_range)
            m_type = random.choice(mesh_types)
            
            geom = geometry_factory('square', x_range=x_rng, y_range=y_rng)
            # Física fija: Seno, Amp 1, Frec 1, BC 0
            phys = PoissonGeneral(source_type='sine', scale=1.0, freq=1.0, bc_type='zero')
            prob = get_problem(geom, phys, nx=res, ny=res)
            
            # FEM_to_PiGnnData calcula u_fem automáticamente como target 'y'
            data = FEM_to_PiGnnData(prob)
            train_graphs.append(data)
            
    # 2. Generación del caso de validación OOD (Extrapolación 0-4)
    print("Generando Caso de Validación OOD [0, 4]...")
    geom_ood = geometry_factory('square', x_range=[0.0, 4.0], y_range=[0.0, 4.0])
    phys_ood = PoissonGeneral(source_type='sine', scale=1.0, freq=1.0, bc_type='zero')
    prob_ood = get_problem(geom_ood, phys_ood, nx=40, ny=40)
    val_graph_ood = FEM_to_PiGnnData(prob_ood)
    
    return train_graphs, val_graph_ood