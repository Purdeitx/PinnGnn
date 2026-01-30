import os
import pandas as pd
from IPython.display import display


def print_config_summary(model, config, model_type="PINN"):
    """
    Generates a standardized configuration table for PINN or FEM.
    """

    descriptions = {
        # PINN Specific
        'hidden_dim': 'Width of the MLP layers',
        'num_layers': 'Number of hidden layers',
        'epochs': 'Maximum training iterations',
        'lr': 'Learning rate (Adam)',
        'lambda_bc': 'Weight penalty for Boundary Conditions',
        'use_fem_for_train': 'Use FEM mesh points for training',
        'use_fem_for_test': 'Use FEM mesh points for testing',
        'n_collocation': 'Train points (Interior PDE residual)',
        'n_boundary': 'Train points (Boundary conditions)',
        
        # FEM Specific
        'nelem': 'Mesh resolution (Elements per side)',
        'porder': 'Polynomial order (1: Linear, 2: Quadratic)',
        'mesh_type': 'Element geometry (tri/quad)',
        'source_type': 'Physics source term function'
    }
    
    table_data = []
    
    # 1. Standard config items
    for key, value in config.items():
        desc = descriptions.get(key, 'Model/Solver parameter')
        table_data.append({
            "Hyperparameter": f"`{key}`", 
            "Description": desc, 
            "Value": value
        })
    
    # 2. Add dynamic info for PINN
    if model_type == "PINN":
    # Check actual points from model
        train_pts = getattr(model, 'train_physics_pts', None)
        n_actual = len(train_pts) if train_pts is not None else config.get('n_collocation')
        
        table_data.append({
            "Hyperparameter": "`Actual Train Pts`", 
            "Description": "Final count of interior points used", 
            "Value": n_actual
        })
        
    # 3. Add dynamic info for FEM (Degrees of Freedom)
    elif model_type == "FEM":
        if hasattr(model, 'basis'):
            dofs = model.basis.N
            table_data.append({
                "Hyperparameter": "`DOFs`", 
                "Description": "Total Degrees of Freedom in the system", 
                "Value": dofs
            })

    df = pd.DataFrame(table_data)
    print(f"\n### {model_type} CONFIGURATION SUMMARY")
    # display(df)
    return df