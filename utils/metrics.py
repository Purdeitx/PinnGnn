import numpy as np
import pandas as pd

def calculate_mse(u_true, u_pred):
    return np.mean((u_true - u_pred)**2)

def calculate_rrmse(u_true, u_pred):
    """
    Relative Root Mean Square Error following the localized formula:
    sqrt( (1/nt) * sum_t ( (1/np) * sum_p ( (u_true - u_pred)^2 / (||u_true||_inf)^2 ) ) )
    
    Compatible with 1D (points) or 2D (points, time) arrays.
    """
    u_true = np.atleast_2d(u_true) # [nt, np]
    u_pred = np.atleast_2d(u_pred)
    
    nt, np_dim = u_true.shape
    
    # Calculate RRMSE per time-step/snapshot
    rrmse_steps = []
    for i in range(nt):
        mse_i = np.mean((u_true[i] - u_pred[i])**2)
        norm_inf_i = np.max(np.abs(u_true[i]))
        if norm_inf_i < 1e-10:
            rrmse_steps.append(0.0)
        else:
            rrmse_steps.append(mse_i / (norm_inf_i**2))
            
    return np.sqrt(np.mean(rrmse_steps))

def calculate_absolute_error(u_true, u_pred):
    return np.abs(u_true - u_pred)

def calculate_fem_metrics(u_num, u_ex, porder=None, nelem=None, return_df=True):
    """
    FEM results metrics calculation: MSE, RRMSE, Max Absolute Error, MAE.
    Optionally returns a formatted DataFrame for reporting.
    """
    # Call the metric functions
    mse = calculate_mse(u_ex, u_num)
    rrmse = calculate_rrmse(u_ex, u_num)
    abs_err = calculate_absolute_error(u_ex, u_num)
    
    # Additional metrics
    max_err = np.max(abs_err)
    mae = np.mean(abs_err)

    metrics_dict = {
        "mse": mse,
        "rrmse": rrmse,
        "max_abs_error": max_err,
        "mae": mae,
        "porder": porder,
        "nelem": nelem
    }

    if not return_df:
        return metrics_dict

    # Formated DataFrame
    data = {
        "Métrica": [
            "Error Absoluto Máximo (L_inf)",
            "Error Absoluto Promedio (MAE)",
            "MSE (Mean Squared Error)",
            "RRMSE (Relative RMSE)"
        ],
        "Valor": [
            f"{max_err:.6e}",
            f"{mae:.6e}",
            f"{mse:.6e}",
            f"{rrmse:.6%}" # percentage format
        ]
    }
    
    return pd.DataFrame(data), metrics_dict

