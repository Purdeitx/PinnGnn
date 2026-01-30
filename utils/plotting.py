import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np
import os

def plot_loss(loss_data, save_path, model_name="Model"):
    """
    Plots the evolution of the loss components.
    
    Args:
        loss_data (dict or pd.DataFrame): Multi-column data containing 'train_loss' and 'loss_*'
        save_path (str): Path to save the plot.
        model_name (str): Name of the model for titles.
    """
    plt.figure(figsize=(10, 6))
    
    if isinstance(loss_data, dict):
        for label, values in loss_data.items():
            plt.plot(values, label=label, linewidth=1.5 if label == 'train_loss' else 1.0, 
                     alpha=1.0 if label == 'train_loss' else 0.7)
    else:
        # Assume it's a pandas DataFrame
        # Fill NaNs to ensure continuous plots (PL logs val_loss only on val epochs)
        df_plot = loss_data.ffill()
        for col in df_plot.columns:
            if col.startswith('loss_') or col == 'train_loss' or col == 'val_loss':
                plt.plot(df_plot[col].values, label=col, 
                         linewidth=2.0 if col in ['train_loss', 'val_loss'] else 1.2,
                         alpha=1.0 if col in ['train_loss', 'val_loss'] else 0.8)

    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title(f'Loss Decomposition - {model_name}')
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_comparison_with_fem(u_fem, u_model, x_coords, save_path, model_name="GNN"):
    """
    Plots FEM vs Model results with 3 subplots: FEM, Model, and MSE.
    Uses common scale for FEM/Model, adjusted scale for Error.
    """
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    # Common scale for solutions
    vmin = min(u_fem.min(), u_model.min())
    vmax = max(u_fem.max(), u_model.max())
    
    # Pointwise MSE
    error_mse = (u_fem - u_model)**2
    
    # FEM
    sc1 = ax[0].tricontourf(x_coords[:,0], x_coords[:,1], u_fem, levels=20, vmin=vmin, vmax=vmax)
    ax[0].set_title("FEM Solution")
    ax[0].set_aspect('equal')
    plt.colorbar(sc1, ax=ax[0])
    
    # Model
    sc2 = ax[1].tricontourf(x_coords[:,0], x_coords[:,1], u_model, levels=20, vmin=vmin, vmax=vmax)
    ax[1].set_title(f"{model_name} Prediction")
    ax[1].set_aspect('equal')
    plt.colorbar(sc2, ax=ax[1])

    # MSE (Pointwise) - Adjusted scale
    sc3 = ax[2].tricontourf(x_coords[:,0], x_coords[:,1], error_mse, levels=20)
    ax[2].set_title("MSE (Pointwise)")
    ax[2].set_aspect('equal')
    plt.colorbar(sc3, ax=ax[2])
    
    plt.suptitle(f"Comparison: FEM vs {model_name}", fontsize=14)
    plt.tight_layout()
    fig = plt.gcf()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_error_analysis(u_fem, u_model, save_path, model_name="GNN"):
    """
    Boxplot of Absolute Error and Percentage Error distribution at all nodes.
    Layout: 1 Row, 2 Columns
    """
    err_sq = (u_fem - u_model)**2
    norm_inf = np.max(np.abs(u_fem))
    # Avoid div by zero
    eps = 1e-10 if norm_inf > 0 else 1.0
    # For percentage, we usually stay with absolute relative error or MSE relative to max
    err_percent = (np.abs(u_fem - u_model) / (norm_inf + eps)) * 100
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].boxplot(err_sq)
    ax[0].set_title(f"Squared Error Distribution ({model_name})")
    ax[0].set_ylabel("(u_fem - u_model)^2")
    ax[0].grid(True, alpha=0.3)
    
    ax[1].boxplot(err_percent)
    ax[1].set_title(f"Absolute % Error Distribution ({model_name})")
    ax[1].set_ylabel("Error (%) relative to Max Reference")
    ax[1].grid(True, alpha=0.3)
    
    plt.suptitle(f"Error Analysis for {model_name}\n(Max Reference Solution: {norm_inf:.4f})", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig



def save_simulation_gif(u_frames, u_fem, points, save_path, title="Simulation"):
    """
    Saves a GIF of the simulation evolution with 3 subplots: FEM, Model, and Absolute Error.
    """
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    # Filter out empty or None frames
    u_frames = [u for u in u_frames if u is not None]
    if not u_frames:
        return

    vmin = min(u_fem.min(), min([u.min() for u in u_frames]))
    vmax = max(u_fem.max(), max([u.max() for u in u_frames]))
    
    def update(frame):
        for a in ax:
            a.clear()
        
        u_model = u_frames[frame]
        error_mse = (u_fem - u_model)**2
        
        # Solution Scale (FEM + Model)
        # Model scale
        sc1 = ax[0].tricontourf(points[:,0], points[:,1], u_fem, levels=20, vmin=vmin, vmax=vmax)
        ax[0].set_title("FEM Solution")
        ax[0].set_aspect('equal')
        
        # Model
        sc2 = ax[1].tricontourf(points[:,0], points[:,1], u_model, levels=20, vmin=vmin, vmax=vmax)
        ax[1].set_title(f"Model Prediction (Frame {frame})")
        ax[1].set_aspect('equal')

        # MSE - Adjusted scale
        sc3 = ax[2].tricontourf(points[:,0], points[:,1], error_mse, levels=20)
        ax[2].set_title("MSE (Pointwise)")
        ax[2].set_aspect('equal')
        
        return sc1, sc2, sc3

    ani = animation.FuncAnimation(fig, update, frames=len(u_frames), blit=False)
    
    try:
        ani.save(save_path, writer='pillow', fps=10)
    except Exception as e:
        print(f"Could not save GIF: {e}. Saving as PNG instead.")
        plt.savefig(save_path.replace('.gif', '.png'))
    plt.close()

def save_extrapolation_gif(u_frames, u_fem, points, save_path):
    """
    Placeholder for temporal extrapolation GIF.
    """
    save_simulation_gif(u_frames, u_fem, points, save_path, title="Extrapolation")

def plot_pinn_sampling(eval_points, train_points, bc_points=None, title="PINN Point Distribution"):
    """
    Visualize the split between training (PDE), boundary, and evaluation points.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 8))
    
    # 1. Evaluation Points (The "Ground Truth" Mesh)
    plt.scatter(eval_points[:, 0], eval_points[:, 1], marker='x', color='blue', 
                alpha=0.4, label=f'Evaluation (Eval points: {len(eval_points)})')
    
    # 2. Training Points (PDE Collocation)
    plt.scatter(train_points[:, 0], train_points[:, 1], marker='.', color='red', 
                s=15, alpha=0.6, label=f'Train (Train points: {len(train_points)})')
    
    # 3. Boundary Points (If provided)
    if bc_points is not None:
        plt.scatter(bc_points[:, 0], bc_points[:, 1], marker='s', color='green', 
                    s=20, label=f'Boundary (BC: {len(bc_points)})')
    
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.gca().set_aspect('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    return plt.gca()

def plot_fem_mesh(prob, title="FEM Mesh Geometry"):
    """
    FEM mesh visualization.
    Plot the mesh elements and degrees of freedom (DoFs).
    """
    mesh = prob['mesh']
    coords = prob['doflocs']            # this are the DoFs [N x 2]
    b_idx = prob['boundary_indices']    # boundary DoFs
    i_idx = prob['interior_indices']    # interior DoFs
    
    fig, ax = plt.subplots(figsize=(8, 7))

    # 1. grid lines (mesh elements)
    # mesh.p are vertices, mesh.t the connectivity of the cells
    verts = mesh.p.T[mesh.t.T] 
    
    # PolyCollection plots Tri or Quad automatically depending on the shape of verts
    poly = PolyCollection(verts, facecolors='none', edgecolors='black', 
                          linewidths=1.2, alpha=0.6, zorder=1)
    ax.add_collection(poly)

    # 2. Draw the DoFs (calculation points)
    ax.scatter(coords[i_idx, 0], coords[i_idx, 1], color='blue', s=30, 
               label=f'Interior DoF ({len(i_idx)})', zorder=2)
    ax.scatter(coords[b_idx, 0], coords[b_idx, 1], color='red', s=30, 
               label=f'Boundary DoF ({len(b_idx)})', zorder=2)

    # 3. Plot settings
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.autoscale_view()
    plt.tight_layout()
    
    return fig, ax

def plot_fem_validation(prob, title="FEM Validation"):
    """
    Compares FEM solution against exact analytical solution.
    """
    coords = prob['doflocs']
    u_fem = prob['u']
    u_exact = prob['u_exact']
    
    # obtain error at each node
    error = np.abs(u_fem - u_exact)
    max_err = np.max(error)
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    # Define common scale for FEM and Exact
    vmin, vmax = u_exact.min(), u_exact.max()
    if np.abs(vmax - vmin) < 1e-10:
        vmax += 0.1
        vmin -= 0.1

    # 1. Analitic solution (GT)
    sc0 = ax[0].tricontourf(coords[:,0], coords[:,1], u_exact, levels=20, vmin=vmin, vmax=vmax)
    ax[0].set_title("Exact Analytical Solution")
    plt.colorbar(sc0, ax=ax[0])

    # 2. FEM solution (Numerical)
    sc1 = ax[1].tricontourf(coords[:,0], coords[:,1], u_fem, levels=20, vmin=vmin, vmax=vmax)
    ax[1].set_title("FEM Numerical Solution")
    plt.colorbar(sc1, ax=ax[1])

    # 3. Absolute Error
    sc2 = ax[2].tricontourf(coords[:,0], coords[:,1], error, levels=20, cmap='inferno')
    ax[2].set_title(f"Absolute Error (Max: {max_err:.2e})")
    plt.colorbar(sc2, ax=ax[2])

    for a in ax:
        a.set_aspect('equal')
        a.set_xlabel('x')
        a.set_ylabel('y')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig, ax


