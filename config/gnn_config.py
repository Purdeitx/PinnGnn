from .common_config import COMMON_CONFIG

GNN_CONFIG = {
    'hidden_dim': 32,
    'num_layers': 8,
    'epochs': COMMON_CONFIG['epochs'],
    'lr': COMMON_CONFIG['lr'],
    'lambda_bc': 10.0, # GNN usually needs less BC weight if using K matrix
    'supervised': True,
    'use_chebconv': False,  # Set to True to use Chebyshev convolution (like Gao's PossionNet)
}

