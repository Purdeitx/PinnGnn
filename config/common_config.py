# Common Configuration for all models

COMMON_CONFIG = {
    # general configuration
    'project_name': 'out',      # User project folder
    'output_root': 'outputs',
    # mesh and geometry
    'geometry_type': 'square',
    'nelem': 2,            # Mesh elements per side
    'mesh': 'quad',        # type of mesh elements 'tri' or 'quad'          
    'porder': 2,           # Polynomial order (1: P1, 2: P2)
    'x_range': [0, 1],
    'y_range': [0, 1],
    # physics 
    'problem': 'Poisson',  # For folder: PDE_Poisson
    'source_type': 'sine', # 
    'source_value': 1.0,   # Gao's S=[1], f(x,el)=source_value
    # learning
    'epochs': 500,
    'batch': 64,
    'lr': 1e-3,
    'hidden': 64,
    'num_layers': 4,
    'activation': 'silu',
    'layer_norm': True,
    'dropout': 0.0,
}

