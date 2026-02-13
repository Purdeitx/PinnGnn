from .common_config import COMMON_CONFIG

# Default FEM Configuration
FEM_CONFIG = {
    **COMMON_CONFIG,
    # general config:
    'project_name': COMMON_CONFIG['project_name'],      # for wandb...
    'output_root': COMMON_CONFIG['output_root'],
    # mesh and geometry
    'geometry_type': COMMON_CONFIG['geometry_type'],
    'nx': COMMON_CONFIG['nelem'],
    'ny': COMMON_CONFIG['nelem'],
    'mesh': COMMON_CONFIG['mesh'],
    'porder': COMMON_CONFIG['porder'],
    'x_range': COMMON_CONFIG['x_range'],
    'y_range': COMMON_CONFIG['y_range'],
    # physics
    'problem': COMMON_CONFIG['problem'],
    'source_type': COMMON_CONFIG['source_type'], 
    'source_value': COMMON_CONFIG['source_value'],
    # other
    }

