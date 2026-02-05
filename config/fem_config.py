from .common_config import COMMON_CONFIG

# Default FEM Configuration
FEM_CONFIG = {
    'problem': 'poisson',
    'nx': COMMON_CONFIG['nelem'],
    'ny': COMMON_CONFIG['nelem'],
    'porder': COMMON_CONFIG['porder'],
    'source_type': COMMON_CONFIG['source_type']
}

