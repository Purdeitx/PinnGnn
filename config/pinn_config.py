from .common_config import COMMON_CONFIG

PINN_CONFIG = {
    'hidden_dim': COMMON_CONFIG['hidden_dim'],
    'num_layers': COMMON_CONFIG['num_layers'],
    'activation': COMMON_CONFIG['activation'],
    'epochs': COMMON_CONFIG['epochs'],
    'lr': COMMON_CONFIG['lr'],
    'lambda_bc': COMMON_CONFIG['lambda_bc'],
    'use_fem_for_train': False,
    'use_fem_for_test': False,
    'n_train': 2000,
    'n_test': 2000,
    'n_bc': 500
}