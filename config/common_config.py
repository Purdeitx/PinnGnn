# Common Configuration for all models
COMMON_CONFIG = {
    'pde_type': 'Poisson', # For folder: PDE_Poisson
    'project_name': 'out', # User project folder
    'nelem': 2,            # Mesh elements per side
    'porder': 2,           # Polynomial order (1: P1, 2: P2)
    'source_type': 'const', # Gao's reference: constant source f=1
    'source_value': 1.0,   # Gao's S=[1], f(x,el)=source_value
    'output_root': 'outputs',
    'lambda_bc': 100.0,    # Weight for boundary condition loss
    'epochs': 500,
    'lr': 1e-3,
}

