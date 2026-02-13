from .common_config import COMMON_CONFIG

PINN_CONFIG = {
    **COMMON_CONFIG,
    # general configuration
    'project_name': COMMON_CONFIG['project_name'],
    'output_root': COMMON_CONFIG['output_root'],
    # mesh and geometry
    'geometry_type': COMMON_CONFIG['geometry_type'],
    'x_range': COMMON_CONFIG['x_range'],
    'y_range': COMMON_CONFIG['y_range'],
    # physics
    'problem': COMMON_CONFIG['problem'],
    'source_type': COMMON_CONFIG['source_type'],
    'source_value': COMMON_CONFIG['source_value'],
    # learning
    'epochs': COMMON_CONFIG['epochs'],
    'batch': COMMON_CONFIG['batch'],
    'lr': COMMON_CONFIG['lr'],
    'hidden_dim': COMMON_CONFIG['hidden'],
    'num_layers': COMMON_CONFIG['num_layers'],
    'activation': COMMON_CONFIG['activation'],
    'layer_norm': COMMON_CONFIG['layer_norm'],
    'dropout': COMMON_CONFIG['dropout'],
    # other
    'use_fem_for_train': False,
    'use_fem_for_test': False,
    'n_train': 2000,
    'n_test': 2000,
    'n_bc': 500,
    'lambda_bc': 100,
}