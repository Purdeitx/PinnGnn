from .common_config import COMMON_CONFIG

GNN_CONFIG = {
    **COMMON_CONFIG,
    # general configuration
    'project_name': COMMON_CONFIG['project_name'],
    'output_root': COMMON_CONFIG['output_root'],
    # mesh and geometry
    'geometry_type': COMMON_CONFIG['geometry_type'],
    'nx': COMMON_CONFIG['nelem'],
    'ny': COMMON_CONFIG['nelem'],
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
    'hidden': COMMON_CONFIG['hidden'],
    'num_layers': COMMON_CONFIG['num_layers'],
    'activation': COMMON_CONFIG['activation'],
    'layer_norm': COMMON_CONFIG['layer_norm'],
    'dropout': COMMON_CONFIG['dropout'],
    # other
    'node_in': 4,
    'edge_in': 3,
    'decoder_out': 1,
    'latent_dim': 64,       # Espacio latente para encoders y procesadores
    'msg_passes': 5,
}

