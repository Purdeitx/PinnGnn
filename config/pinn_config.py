from .common_config import COMMON_CONFIG

PINN_CONFIG = {
    'hidden_dim': 64,
    'num_layers': 4,
    'epochs': COMMON_CONFIG['epochs'],
    'lr': COMMON_CONFIG['lr'],
    'lambda_bc': COMMON_CONFIG['lambda_bc'],
    'use_fem_for_train': COMMON_CONFIG['use_fem_for_train'],
    'use_fem_for_test': COMMON_CONFIG['use_fem_for_test'],
    'n_collocation': 2000,
    'n_boundary': 500
}