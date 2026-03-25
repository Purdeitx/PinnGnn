import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import PolyCollection, LineCollection
import numpy as np
import torch
import os

def plot_loss(loss_data, model_name="Model"):
    """Devuelve la figura de la evolución de las pérdidas."""
    fig = plt.figure(figsize=(10, 6))
    
    if isinstance(loss_data, dict):
        for label, values in loss_data.items():
            if values: # Evitar listas vacías
                plt.plot(values, label=label, linewidth=1.5 if label == 'train_loss' else 1.0, 
                         alpha=1.0 if label == 'train_loss' else 0.7)
    else:
        df_plot = loss_data.ffill()
        for col in df_plot.columns:
            if col.startswith('loss_') or col in ['train_loss', 'val_loss']:
                plt.plot(df_plot[col].values, label=col, 
                         linewidth=2.0 if col in ['train_loss', 'val_loss'] else 1.2)

    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title(f'Loss Decomposition - {model_name}')
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    return fig

def plot_fem_validation(prob, title="FEM Validation"):
    """
    Compara la solución FEM contra la analítica exacta.
    Devuelve la figura con escalas unificadas y mapa de error.
    """
    coords = prob['doflocs']
    u_fem = prob['u']
    u_exact = prob['u_exact']
    
    # Cálculo de error absoluto
    error = np.abs(u_fem - u_exact)
    max_err = np.max(error)
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    # Escala unificada para comparar peras con peras
    vmin, vmax = u_exact.min(), u_exact.max()
    levels = np.linspace(vmin, vmax, 25)
    
    # 1. Solución Analítica (Exacta)
    sc0 = ax[0].tricontourf(coords[:,0], coords[:,1], u_exact, levels=levels, cmap='viridis')
    ax[0].set_title("Exact Analytical Solution")
    plt.colorbar(sc0, ax=ax[0])

    # 2. Solución FEM (Numérica)
    sc1 = ax[1].tricontourf(coords[:,0], coords[:,1], u_fem, levels=levels, cmap='viridis')
    ax[1].set_title("FEM Numerical Solution")
    plt.colorbar(sc1, ax=ax[1])

    # 3. Error Absoluto (Mapa de calor de fallos)
    sc2 = ax[2].tricontourf(coords[:,0], coords[:,1], error, levels=20, cmap='inferno')
    ax[2].set_title(f"Abs. Error (Max: {max_err:.2e})")
    plt.colorbar(sc2, ax=ax[2])

    for a in ax:
        a.set_aspect('equal')
        a.set_xlabel('x')
        a.set_ylabel('y')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig

def plot_comparison_with_fem(u_fem, u_model, x_coords, model_name="Model"):
    """Triple plot: FEM, Prediction y MSE con escala unificada."""
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    vmin, vmax = min(u_fem.min(), u_model.min()), max(u_fem.max(), u_model.max())
    levels = np.linspace(vmin, vmax, 25)
    error_mse = (u_fem - u_model)**2
    
    sc1 = ax[0].tricontourf(x_coords[:,0], x_coords[:,1], u_fem, levels=levels, cmap='viridis')
    ax[0].set_title("FEM Solution")
    plt.colorbar(sc1, ax=ax[0])
    
    sc2 = ax[1].tricontourf(x_coords[:,0], x_coords[:,1], u_model, levels=levels, vmin=vmin, vmax=vmax, cmap='viridis')
    ax[1].set_title(f"{model_name} Prediction")
    plt.colorbar(sc2, ax=ax[1])

    sc3 = ax[2].tricontourf(x_coords[:,0], x_coords[:,1], error_mse, levels=20, cmap='magma')
    ax[2].set_title("MSE (Pointwise)")
    plt.colorbar(sc3, ax=ax[2])
    
    for a in ax: a.set_aspect('equal')
    plt.suptitle(f"Comparison: FEM vs {model_name}", fontsize=14)
    plt.tight_layout()
    return fig

def plot_error_analysis(u_fem, u_model, model_name="Model"):
    """Boxplot de distribución de error."""
    err_sq = (u_fem - u_model)**2
    norm_inf = np.max(np.abs(u_fem))
    eps = 1e-10 if norm_inf > 0 else 1.0
    err_percent = (np.abs(u_fem - u_model) / (norm_inf + eps)) * 100
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].boxplot(err_sq); ax[0].set_title(f"Squared Error ({model_name})"); ax[0].grid(True, alpha=0.3)
    ax[1].boxplot(err_percent); ax[1].set_title(f"Abs % Error ({model_name})"); ax[1].grid(True, alpha=0.3)
    
    plt.suptitle(f"Error Analysis for {model_name}\n(Max Reference: {norm_inf:.4f})", fontsize=14)
    plt.tight_layout()
    return fig

def plot_pinn_sampling(eval_points, train_points, bc_points=None, title="PINN Point Distribution"):
    """Visualiza la nube de puntos de colocación."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(eval_points[:, 0], eval_points[:, 1], marker='x', color='blue', alpha=0.2, label='Evaluation')
    ax.scatter(train_points[:, 0], train_points[:, 1], marker='.', color='red', s=15, alpha=0.5, label='PDE Train')
    if bc_points is not None:
        ax.scatter(bc_points[:, 0], bc_points[:, 1], marker='s', color='green', s=20, label='Boundary')
    
    ax.set_title(title)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def plot_fem_mesh(prob, title="FEM Mesh Geometry"):
    """Visualización de la malla y DoFs."""
    mesh = prob['mesh']
    coords = prob['doflocs']
    b_idx, i_idx = prob['boundary_indices'], prob['interior_indices']
    
    fig, ax = plt.subplots(figsize=(8, 7))
    verts = mesh.p.T[mesh.t.T] 
    poly = PolyCollection(verts, facecolors='none', edgecolors='black', linewidths=0.8, alpha=0.4)
    ax.add_collection(poly)

    ax.scatter(coords[i_idx, 0], coords[i_idx, 1], color='blue', s=20, label='Interior DoF')
    ax.scatter(coords[b_idx, 0], coords[b_idx, 1], color='red', s=20, label='Boundary DoF')

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.autoscale_view()
    plt.tight_layout()
    return fig


def plot_comparison_with_pinn(pinn, u_ref, x_coords, ref_name="FEM"):
    """Inferencia y ploteo comparativo unificado (Versión Robusta)."""
    pinn.eval()
    device = next(pinn.parameters()).device 
    
    if torch.is_tensor(x_coords):
        x_plot = x_coords.detach().cpu().numpy()
        coords_torch = x_coords.detach().clone().float().to(device)
    else:
        x_plot = x_coords
        coords_torch = torch.from_numpy(x_coords).float().to(device)

    if torch.is_tensor(u_ref):
        u_ref_plot = u_ref.detach().cpu().numpy().flatten()
    else:
        u_ref_plot = u_ref.flatten()

    with torch.no_grad():
        u_pinn_plot = pinn(coords_torch).cpu().numpy().flatten()

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    vmin = min(u_ref_plot.min(), u_pinn_plot.min())
    vmax = max(u_ref_plot.max(), u_pinn_plot.max())
    levels = np.linspace(vmin, vmax, 25)
    error_abs = np.abs(u_ref_plot - u_pinn_plot)
    
    sc1 = ax[0].tricontourf(x_plot[:,0], x_plot[:,1], u_ref_plot, levels=levels, cmap='viridis')
    ax[0].set_title(f"Reference ({ref_name})")
    plt.colorbar(sc1, ax=ax[0])
    
    sc2 = ax[1].tricontourf(x_plot[:,0], x_plot[:,1], u_pinn_plot, levels=levels, cmap='viridis')
    ax[1].set_title("PINN Prediction")
    plt.colorbar(sc2, ax=ax[1])

    sc3 = ax[2].tricontourf(x_plot[:,0], x_plot[:,1], error_abs, levels=25, cmap='magma')
    ax[2].set_title(f"Abs. Error (Max: {error_abs.max():.2e})")
    plt.colorbar(sc3, ax=ax[2])
    
    for a in ax: a.set_aspect('equal')
    plt.tight_layout()
    return fig

def plot_pinn_strategy(train, bc, test, title="Sampling Strategy"):
    dims = train.shape[1]
    fig = plt.figure(figsize=(8, 6))
    
    if dims == 2:
        ax = fig.add_subplot(111)
        ax.scatter(test[:, 0], test[:, 1], c='blue', alpha=0.3, label='Test/Eval (Blue)')
        ax.scatter(train[:, 0], train[:, 1], c='red', marker='x', s=30, label='Train/Physics (Red)')
        ax.scatter(bc[:, 0], bc[:, 1], c='green', s=30, label='BC/Boundary (Green)')
    elif dims == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(test[:, 0], test[:, 1], test[:, 2], c='blue', alpha=0.1)
        ax.scatter(train[:, 0], train[:, 1], train[:, 2], c='red', marker='x')
        ax.scatter(bc[:, 0], bc[:, 1], bc[:, 2], c='green')
    
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend()
    plt.show()

def plot_error_boxplot(u_exact, u_pred, var_names=None, model_name="PINN"):
    """
    Genera un boxplot del error absoluto detectando automáticamente el nº de variables.
    
    u_exact, u_pred: arrays de forma (N,) o (N, num_vars)
    var_names: lista de strings con los nombres (ej: ['u', 'v']). Si es None, usa Var 1, Var 2...
    """
    # Asegurar que los datos tengan forma (N, num_vars)
    if u_exact.ndim == 1:
        u_exact = u_exact.reshape(-1, 1)
        u_pred = u_pred.reshape(-1, 1)
    
    num_vars = u_exact.shape[1]
    errors = [np.abs(u_exact[:, i] - u_pred[:, i]) for i in range(num_vars)]
    
    # Configurar nombres de las variables
    if var_names is None:
        var_names = [f"Var {i+1}" for i in range(num_vars)]
    elif len(var_names) != num_vars:
        var_names = [f"Var {i+1}" for i in range(num_vars)]

    fig, ax = plt.subplots(figsize=(2 + 2*num_vars, 7)) # Escala el ancho según nº de vars
    
    # Dibujar Boxplot
    bplot = ax.boxplot(errors, 
                       patch_artist=True, 
                       labels=var_names,
                       medianprops={'color': 'black', 'linewidth': 1.5},
                       flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 3, 'alpha': 0.3})
    
    # Colores estéticos (puedes añadir más a la lista)
    colors = ['#add8e6', '#90ee90', '#ffb6c1', '#f0e68c', '#e6e6fa']
    for patch, color in zip(bplot['boxes'], colors * 10): # El *10 es por si hay muchas vars
        patch.set_facecolor(color)
    
    ax.set_title(f'Distribución del Error Absoluto por Variable ({model_name})')
    ax.set_ylabel('Error Absoluto $|u_{exact} - u_{pred}|$')
    ax.set_yscale('log') # Escala logarítmica suele ser mejor para ver errores de PINNs
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig

def plot_error_histogram(u_exact, u_pred, model_name="PINN"):
    """
    Genera un histograma de la distribución del error.
    Útil para verificar si el error es aleatorio o sistemático.
    """
    if torch.is_tensor(u_exact): u_exact = u_exact.detach().cpu().numpy()
    if torch.is_tensor(u_pred): u_pred = u_pred.detach().cpu().numpy()
    
    error = (u_exact.flatten() - u_pred.flatten())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Histograma con densidad (normed) y KDE si tienes seaborn, o solo hist
    n, bins, patches = ax.hist(error, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Añadimos una línea vertical en el cero
    ax.axvline(0, color='red', linestyle='dashed', linewidth=2, label='Error 0')
    
    ax.set_title(f'Distribución del Error Residual ({model_name})')
    ax.set_xlabel('Error ($u_{exact} - u_{pred}$)')
    ax.set_ylabel('Densidad de Probabilidad')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_flux_map(edge_index, flux, coords, title="Heat Flux Map (GNN)"):
    """
    Visualiza el flujo aprendido por la GNN en las aristas del grafo.
    
    Args:
        edge_index: Tensor [2, num_edges] con los índices de los nodos.
        flux: Array/Tensor [num_edges] con el valor del flujo en cada arista.
        coords: Array [num_nodes, 2] con las coordenadas (x, y) de los nodos.
        title: Título del gráfico.
    """
    if torch.is_tensor(flux):
        flux = flux.detach().cpu().numpy().flatten()
    if torch.is_tensor(edge_index):
        edge_index = edge_index.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Extraemos puntos de inicio (senders) y fin (receivers) de cada arista
    points = coords[edge_index.T] # Shape: [num_edges, 2, 2]
    
    # Creamos una colección de líneas para que sea eficiente de plotear
    # Normilizamos el color según la intensidad del flujo
    from matplotlib.collections import LineCollection
    lc = LineCollection(points, array=flux, cmap='jet', linewidths=2)
    
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax, label='Flux Intensity ($q$)')
    
    # Ajustar límites del plot
    ax.set_xlim(coords[:, 0].min() - 0.1, coords[:, 0].max() + 0.1)
    ax.set_ylim(coords[:, 1].min() - 0.1, coords[:, 1].max() + 0.1)
    
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    
    return fig, ax  

def plot_flux_comparison(flux_gnn, graph, physics, title="Flux Comparison: GT vs GNN"):
    
    pos = graph.pos.clone().detach().requires_grad_(True)
    u_analitica = physics.exact_solution(pos)
    grad_gt_nodes = torch.autograd.grad(
        u_analitica, pos, grad_outputs=torch.ones_like(u_analitica),
        create_graph=False
    )[0]
    
    senders, receivers = graph.edge_index
    dist_vec = graph.pos[receivers] - graph.pos[senders]
    dist_norm = torch.norm(dist_vec, dim=1, keepdim=True) + 1e-8
    unit_vec = dist_vec / dist_norm
    
    grad_avg_edge = (grad_gt_nodes[senders] + grad_gt_nodes[receivers]) / 2.0
    flux_gt = -(grad_avg_edge * unit_vec).sum(dim=1).detach().cpu().numpy()

    # --- FIX 1: quedarse solo con aristas i→j donde i < j ---
    # Elimina la arista duplicada j→i, nos quedamos con una dirección canónica
    s_np = senders.cpu().numpy()
    r_np = receivers.cpu().numpy()
    mask_canonical = s_np < r_np  # una sola dirección por par de nodos

    flux_gt_plot  = flux_gt[mask_canonical]
    flux_gnn_plot = flux_gnn[mask_canonical]

    # --- FIX 2: verificar si el signo de la red está invertido ---
    from scipy.stats import pearsonr
    corr, _ = pearsonr(flux_gt_plot, flux_gnn_plot)
    print(f"Correlación GT vs GNN: {corr:.4f}")
    if corr < -0.5:
        print("⚠️  Signo invertido detectado — corrigiendo")
        # flux_gnn_plot = -flux_gnn_plot

    flux_error = np.abs(flux_gt_plot - flux_gnn_plot)

    # --- Plotting solo con aristas canónicas ---
    coords = graph.pos.detach().cpu().numpy()
    edge_index_np = graph.edge_index.cpu().numpy()
    
    # Filtrar puntos para LineCollection
    points_all = coords[edge_index_np.T]          # [E, 2, 2]
    points_plot = points_all[mask_canonical]       # [E/2, 2, 2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Ground Truth Flux", "GNN Predicted Flux", "Absolute Error"]
    datas  = [flux_gt_plot, flux_gnn_plot, flux_error]
    cmaps  = ['jet', 'jet', 'inferno']

    for ax, data, t, cmap in zip(axes, datas, titles, cmaps):
        lc = LineCollection(points_plot, array=data, cmap=cmap, linewidths=1.5)
        line = ax.add_collection(lc)
        fig.colorbar(line, ax=ax)
        ax.set_title(t)
        ax.set_aspect('equal')
        ax.set_xlim(coords[:,0].min()-0.1, coords[:,0].max()+0.1)
        ax.set_ylim(coords[:,1].min()-0.1, coords[:,1].max()+0.1)

    fig.suptitle(f"{title}  |  corr={corr:.3f}", fontsize=12)
    plt.tight_layout()
    return fig

def plot_flux_symmetry(flux_gnn, graph, physics, title="Flux Symmetry Check"):
    """
    Visualiza las dos direcciones de arista por separado para verificar
    si flux_ij ≈ -flux_ji (antisimetría física).
    """
    senders, receivers = graph.edge_index
    s_np = senders.cpu().numpy()
    r_np = receivers.cpu().numpy()

    # Separar las dos direcciones canónicas
    mask_ij = s_np < r_np   # dirección i→j (s < r)
    mask_ji = s_np > r_np   # dirección j→i (s > r)

    flux_ij = flux_gnn[mask_ij]
    flux_ji = flux_gnn[mask_ji]

    # --- Verificar que los pares están correctamente ordenados ---
    # Para cada arista i→j debe existir j→i con el mismo par de nodos
    pairs_ij = set(zip(s_np[mask_ij], r_np[mask_ij]))
    pairs_ji = set(zip(r_np[mask_ji], s_np[mask_ji]))  # invertimos para comparar
    assert pairs_ij == pairs_ji, "⚠️  Las aristas no están perfectamente emparejadas"

    # Ordenar flux_ji para que corresponda al mismo orden que flux_ij
    # Construir lookup: (i,j) → flux_ji value
    lookup_ji = {(r_np[k], s_np[k]): flux_gnn[k] for k in np.where(mask_ji)[0]}
    flux_ji_sorted = np.array([lookup_ji[(i, j)] 
                                for i, j in zip(s_np[mask_ij], r_np[mask_ij])])

    # --- GT para referencia ---
    pos = graph.pos.clone().detach().requires_grad_(True)
    u_analitica = physics.exact_solution(pos)
    grad_gt_nodes = torch.autograd.grad(
        u_analitica, pos, grad_outputs=torch.ones_like(u_analitica),
        create_graph=False
    )[0]
    dist_vec = graph.pos[receivers] - graph.pos[senders]
    dist_norm = torch.norm(dist_vec, dim=1, keepdim=True) + 1e-8
    unit_vec  = dist_vec / dist_norm
    grad_avg  = (grad_gt_nodes[senders] + grad_gt_nodes[receivers]) / 2.0
    flux_gt_all = -(grad_avg * unit_vec).sum(dim=1).detach().cpu().numpy()
    flux_gt_ij  = flux_gt_all[mask_ij]

    # --- Métricas de antisimetría ---
    antisym_residual = flux_ij + flux_ji_sorted   # debería ser ~0 si antisimétrico
    print(f"Antisimetría  — mean|flux_ij + flux_ji|: {np.abs(antisym_residual).mean():.4f}")
    print(f"              — max|flux_ij + flux_ji|:  {np.abs(antisym_residual).max():.4f}")
    
    from scipy.stats import pearsonr
    corr_sym, _  = pearsonr(flux_ij, -flux_ji_sorted)
    corr_gt_ij,_ = pearsonr(flux_gt_ij, flux_ij)
    corr_gt_ji,_ = pearsonr(flux_gt_ij, -flux_ji_sorted)
    print(f"Correlación flux_ij vs -flux_ji:  {corr_sym:.4f}  (1.0 = perfectamente antisimétrico)")
    print(f"Correlación GT vs flux_ij:        {corr_gt_ij:.4f}")
    print(f"Correlación GT vs -flux_ji:       {corr_gt_ji:.4f}")

    # --- Plotting ---
    coords     = graph.pos.detach().cpu().numpy()
    edge_index = graph.edge_index.cpu().numpy()
    points_all = coords[edge_index.T]        # [E, 2, 2]
    points_ij  = points_all[mask_ij]         # [E/2, 2, 2]

    # Escala común para flux_ij y flux_ji para que sean comparables
    vmax = max(np.abs(flux_ij).max(), np.abs(flux_ji_sorted).max())
    vmin = -vmax

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=13)

    # Fila superior: las dos direcciones de la red + diferencia
    datasets_top = [
        (flux_ij,          "GNN flux  i→j  (s < r)",   'RdBu_r'),
        (-flux_ji_sorted,  "GNN flux  j→i  negado",     'RdBu_r'),
        (antisym_residual, "Residuo antisimetría\nflux_ij + flux_ji", 'inferno'),
    ]
    for ax, (data, t, cmap) in zip(axes[0], datasets_top):
        vn = vmax if cmap == 'RdBu_r' else None
        lc = LineCollection(points_ij, array=data, cmap=cmap, linewidths=1.5,
                            norm=plt.Normalize(-vn, vn) if vn else None)
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax)
        ax.set_title(t)
        ax.set_aspect('equal')
        ax.set_xlim(coords[:,0].min()-0.1, coords[:,0].max()+0.1)
        ax.set_ylim(coords[:,1].min()-0.1, coords[:,1].max()+0.1)

    # Fila inferior: GT + comparativa scatter
    lc_gt = LineCollection(points_ij, array=flux_gt_ij, cmap='RdBu_r', linewidths=1.5,
                           norm=plt.Normalize(-vmax, vmax))
    axes[1, 0].add_collection(lc_gt)
    fig.colorbar(lc_gt, ax=axes[1, 0])
    axes[1, 0].set_title("GT flux  i→j")
    axes[1, 0].set_aspect('equal')
    axes[1, 0].set_xlim(coords[:,0].min()-0.1, coords[:,0].max()+0.1)
    axes[1, 0].set_ylim(coords[:,1].min()-0.1, coords[:,1].max()+0.1)

    # Scatter: GT vs flux_ij
    axes[1, 1].scatter(flux_gt_ij, flux_ij, s=1, alpha=0.3, c='steelblue')
    lim = max(np.abs(flux_gt_ij).max(), np.abs(flux_ij).max())
    axes[1, 1].plot([-lim, lim], [-lim, lim], 'r--', lw=1, label='y=x')
    axes[1, 1].set_xlabel("GT flux_ij")
    axes[1, 1].set_ylabel("GNN flux_ij")
    axes[1, 1].set_title(f"Scatter GT vs i→j\ncorr={corr_gt_ij:.3f}")
    axes[1, 1].legend()
    axes[1, 1].set_aspect('equal')

    # Scatter: GT vs -flux_ji (debería coincidir con el anterior si es antisimétrico)
    axes[1, 2].scatter(flux_gt_ij, -flux_ji_sorted, s=1, alpha=0.3, c='darkorange')
    axes[1, 2].plot([-lim, lim], [-lim, lim], 'r--', lw=1, label='y=x')
    axes[1, 2].set_xlabel("GT flux_ij")
    axes[1, 2].set_ylabel("GNN -flux_ji")
    axes[1, 2].set_title(f"Scatter GT vs -j→i\ncorr={corr_gt_ji:.3f}")
    axes[1, 2].legend()
    axes[1, 2].set_aspect('equal')

    plt.tight_layout()
    return fig

def plot_train_test_comparison(err_train, err_test, model_name="PiGNN"):
    """
    Genera un boxplot comparativo de los errores relativos L2 
    entre el conjunto de entrenamiento y el de test.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = [err_train, err_test]
    labels = ['Train Set', 'Test Set (Visto)']
    
    bplot = ax.boxplot(data, patch_artist=True, labels=labels)
    
    colors = ['#add8e6', '#ffb6c1'] # Azul claro y Rosa claro
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        
    ax.set_yscale('log')
    ax.set_ylabel('Relative L2 Error (Log Scale)')
    ax.set_title(f'Generalización del Operador: {model_name}')
    ax.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.tight_layout()
    return fig