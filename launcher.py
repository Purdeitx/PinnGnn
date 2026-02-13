import os
import argparse
import torch
import json
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader as PinnDataLoader
from torch_geometric.loader import DataLoader as GNNDataLoader

# =========================
# Project-specific imports
# =========================

# Configuration modules
from config.physics import PoissonPhysics, PoissonGeneral
from config.fem_config import FEM_CONFIG
from config.pinn_config import PINN_CONFIG
from config.gnn_config import GNN_CONFIG

# Model-specific modules
from PINN.pinn_module import PINNSystem, PinnDataset, ValDataset
from FEM.fem_solver import get_problem
from GNN.gnn_module import MessagePassing, DirectSystem
from GNN.MeshGraphNet import MeshGraphNet, MGNSystem

# Utility modules
from utils.train_utils import GradientMonitor, LossPlotterCallback
from utils.geometry import geometry_factory
from utils.plotting import *
from utils.metrics import *
from utils.reporting import *
from utils.pinn_utils import *
from utils.gnn_utils import *


# ==========================================================
# Helper utilities
# ==========================================================

def get_next_out_dir(base_path):
    """
    Generates the next output directory with incremental naming:
    out_001, out_002, ...
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
        return "out_001"
    existing_dirs = [d for d in os.listdir(base_path) if d.startswith("out_") and os.path.isdir(os.path.join(base_path, d))]
    if not existing_dirs:
        return "out_001"
    indices = [int(d.split('_')[1]) for d in existing_dirs if d.split('_')[1].isdigit()]
    next_idx = max(indices) + 1 if indices else 1
    return f"out_{next_idx:03d}"

def save_json(data, folder, filename):
    """
    Safely saves a dictionary in JSON format.
    """
    path = os.path.join(folder, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# ==========================================================
# Main experimental pipeline
# ==========================================================

def main(config):

    # ------------------------------------------------------
    # Device selection and output directory preparation
    # ------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_path = os.path.join(
        config['output_root'],
        f"PDE_{config['problem']}",
        config['project_name']
    )

    out_dir = get_next_out_dir(base_path)
    run_dir = os.path.join(base_path, out_dir)
    os.makedirs(run_dir, exist_ok=True)

    # ------------------------------------------------------
    # Physics model and geometry instantiation
    # ------------------------------------------------------
    physics = PoissonGeneral(
        source_type=config['source_type'],
        scale=config['source_value']
    )

    geom = geometry_factory(
        config['geometry_type'],
        x_range=config['x_range'],
        y_range=config['y_range']
    )

    # ======================================================
    # 1) FEM SOLUTION
    # ======================================================
    print(f"\n[1/3] Solving FEM on mesh {config['nx']}x{config['ny']}...")

    fem_dir = os.path.join(run_dir, "fem")
    os.makedirs(fem_dir, exist_ok=True)

    # Update FEM configuration
    FEM_CONFIG.update({
        'geometry_type': config['geometry_type'],
        'nx': config['nx'],
        'ny': config['ny'],
        'porder': config['porder'],
        'mesh': config['mesh'],
        'source_type': config['source_type'],
        'source_value': config['source_value'],
        'x_range': config['x_range'], 
        'y_range': config['y_range'],
    })

    # Solve FEM problem
    prob = get_problem(
        geometry=geom,
        nx=FEM_CONFIG['nx'],
        ny=FEM_CONFIG['ny'],
        porder=FEM_CONFIG['porder'],
        source_type=FEM_CONFIG['source_type'],
        mesh_type=FEM_CONFIG['mesh']
    )

    save_json(FEM_CONFIG, fem_dir, "fem_config.json")

    # ---------------- FEM report ----------------
    log_file = os.path.join(fem_dir, "fem_report.txt")
    with open(log_file, "w") as f:

        summary_header = (
            f"--- Mesh Technical Summary ---\n"
            f"Structure: {FEM_CONFIG['mesh'].upper()} | Order: P{FEM_CONFIG['porder']}\n"
            f"Resolution: {FEM_CONFIG['nx']}x{FEM_CONFIG['ny']} elements\n"
            f"Total DoFs (computational nodes): {len(prob['doflocs'])}\n"
        )

        l2_error = np.sqrt(np.mean((prob['u'] - prob['u_exact'])**2))
        error_str = f"Mean Squared Error (nodes): {l2_error:.6f}\n"

        df_metrics, raw_metrics = calculate_fem_metrics(prob['u'], prob['u_exact'])
        metrics_report = f"REPORT: {FEM_CONFIG['mesh']} P{FEM_CONFIG['porder']}\n{df_metrics.to_string()}\n"

        print(summary_header + error_str + metrics_report)
        f.write(summary_header + error_str + metrics_report)

    # ---------------- FEM plots ----------------
    plot_fem_mesh(prob,
        title=f"Mesh: {FEM_CONFIG['mesh'].upper()} | Order: P{FEM_CONFIG['porder']} | Res: {FEM_CONFIG['nx']}x{FEM_CONFIG['ny']}"
    )
    plt.savefig(os.path.join(fem_dir, "fem_mesh.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plot_fem_validation(prob,
        title=f"Validation: {FEM_CONFIG['mesh']} P{FEM_CONFIG['porder']} (Res: {FEM_CONFIG['nx']})"
    )
    plt.savefig(os.path.join(fem_dir, "fem_validation.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Save FEM numerical data for later comparison or inference
    np.save(os.path.join(fem_dir, "u_fem.npy"), prob['u'])
    np.save(os.path.join(fem_dir, "u_exact.npy"), prob['u_exact'])
    np.save(os.path.join(fem_dir, "doflocs.npy"), prob['doflocs'])

    print(f">>> FEM results saved in: {fem_dir}")


    # ======================================================
    # 2) PINN TRAINING
    # ======================================================
    print(f"\n[2/3] Configuring PINN for {config['problem']}...")

    pinn_dir = os.path.join(run_dir, "pinn")
    os.makedirs(pinn_dir, exist_ok=True)

    # Update PINN configuration
    PINN_CONFIG.update({
        'geometry_type': config['geometry_type'],
        'n_train': config['n_train'],
        'n_test': config['n_test'],
        'n_bc': config['n_bc'],
        'lambda_bc': config['lambda_bc'],
        'hidden': config['hidden'],
        'num_layers': config['num_layers'],
        'lr': config['lr'],
        'epochs': config['epochs'],
        'batch': config['batch'],
        'source_type': config['source_type'],
        'source_value': config['source_value'],
        'use_fem_for_train': config['use_fem_for_train'],
        'use_fem_for_test': config['use_fem_for_test'],
        'layer_norm': False,
        'x_range': config['x_range'], 
        'y_range': config['y_range'],
    })

    save_json(PINN_CONFIG, pinn_dir, "pinn_config.json")

    # ---------------- Data sampling ----------------
    sampler = PINNSampler(geom, device=device)

    # Interior collocation points
    if PINN_CONFIG['use_fem_for_train']:
        train_pts = torch.tensor(prob['doflocs'], dtype=torch.float32, device=device)
        train_label = "FEM Nodes"
    else:
        train_pts = sampler.sample_interior(PINN_CONFIG['n_train'])
        train_label = "Random Collocation"

    # Boundary points
    bc_coords = sampler.sample_boundary(PINN_CONFIG['n_bc'])

    # Validation/test points
    if PINN_CONFIG['use_fem_for_test']:
        test_pts = torch.tensor(prob['doflocs'], dtype=torch.float32, device=device)
        test_val = torch.tensor(prob['u'], dtype=torch.float32, device=device).view(-1, 1)
        test_label = "FEM Mesh"
    else:
        test_pts = sampler.sample_interior(PINN_CONFIG['n_test'])
        test_val = None
        test_label = "Random Test"

    # ---------------- DataLoaders ----------------
    train_ds = PinnDataset(geometry=geom, pde_pts=train_pts, bc_pts=bc_coords)
    train_loader = PinnDataLoader(train_ds, batch_size=PINN_CONFIG['batch'], shuffle=True, num_workers=0)

    val_ds = ValDataset(geometry=geom, pts=test_pts, vals=test_val)
    val_loader = PinnDataLoader(val_ds, batch_size=len(test_pts))

    # ---------------- Visualization of sampling strategy ----------------
    plot_pinn_strategy(
        train_pts.detach().cpu().numpy(),
        bc_coords.detach().cpu().numpy(),
        test_pts.detach().cpu().numpy(),
        title=f"Strategy: {train_label} vs {test_label}"
    )
    plt.savefig(os.path.join(pinn_dir, "pinn_collocation_points.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------- Model creation ----------------
    model_system = PINNSystem(physics=physics, config=PINN_CONFIG, geometry=geom)

    total_params = sum(p.numel() for p in model_system.model.parameters())
    pinn_report_file = os.path.join(pinn_dir, "pinn_report.txt")

    # ---------------- PINN report ----------------
    with open(pinn_report_file, "w") as f:
        report = (
            f"--- PINN Technical Summary ---\n"
            f"Architecture: MLP\n"
            f"Hidden neurons: {config['hidden']}\n"
            f"Depth (hidden layers): {config['num_layers']}\n"
            f"Total trainable parameters: {total_params:,}\n"
            f"Capacity (Params / FEM DoFs): {total_params / len(prob['doflocs']):.2f}x\n\n"
            f"--- Training Strategy ---\n"
            f"Interior collocation points: {len(train_pts)} ({train_label})\n"
            f"Boundary points: {len(bc_coords)}\n"
            f"Boundary loss weight (lambda): {config['lambda_bc']}\n"
            f"Optimizer: Adam | LR: {config['lr']} | Epochs: {config['epochs']}\n"
        )
        print(report)
        f.write(report)

    # ---------------- Training ----------------
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=300, mode='min', min_delta=1e-6, verbose=True),
        ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', filename='best_pinn'),
        GradientMonitor(verbose=False),
        LossPlotterCallback(model_name="Poisson PINN")
    ]

    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        log_every_n_steps=20,
        default_root_dir=pinn_dir
    )

    trainer.fit(model_system, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # save best model and last state of the model
    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        import shutil
        shutil.copy(best_path, os.path.join(pinn_dir, "best_pinn_model.ckpt"))

    trainer.save_checkpoint(os.path.join(pinn_dir, "final_pinn_model.ckpt"))
    torch.save(model_system.model.state_dict(), os.path.join(pinn_dir, "best_pinn_weights.pth"))
    print(f"Save PINN model condiguration {pinn_dir}")

    # ---------------- Evaluation ----------------

    model_system.eval()
    with torch.no_grad():
        x_test = torch.tensor(prob['doflocs'], dtype=torch.float32, device=model_system.device)
        u_pinn_pred = model_system(x_test).cpu().numpy().flatten()

    # Plots and metrics
    fig_comp = plot_comparison_with_pinn(model_system, prob['u'], prob['doflocs'])
    fig_comp.savefig(os.path.join(pinn_dir, "pinn_vs_fem_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_comp)

    fig_err = plot_error_analysis(u_fem=prob['u'], u_model=u_pinn_pred, model_name="PINN")
    fig_err.savefig(os.path.join(pinn_dir, "pinn_error_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_err)

    np.save(os.path.join(pinn_dir, "u_pinn_pred.npy"), u_pinn_pred)

    l2_rel = np.linalg.norm(prob['u'] - u_pinn_pred) / np.linalg.norm(prob['u'])

    with open(pinn_report_file, "a") as f:
        f.write(f"\n--- Final Results ---\n")
        f.write(f"L2 Relative Error (vs FEM): {l2_rel:.6e}\n")
        f.write(f"Max Absolute Error: {np.abs(u_pinn_pred - prob['u']).max():.6e}\n")

    print(f">>> PINN completed. Relative L2 error: {l2_rel:.2%}")


    # ======================================================
    # 3) GNN TRAINING (MeshGraphNet)
    # ======================================================
    print(f"\n[3/3] Configuring GNN for {config['problem']}...")

    gnn_dir = os.path.join(run_dir, "gnn")
    os.makedirs(gnn_dir, exist_ok=True)
    raw_gnn_params = {
        'enc_n_units': argconfig.get('enc_n_units'),
        'enc_n_layers': argconfig.get('enc_n_layers'),
        'enc_e_units': argconfig.get('enc_e_units'),
        'enc_e_layers': argconfig.get('enc_e_layers'),
        'dec_n_units': argconfig.get('dec_n_units'),
        'dec_n_layers': argconfig.get('dec_n_layers'),
        'proc_e_units': argconfig.get('proc_e_units'),
        'proc_e_layers': argconfig.get('proc_e_layers'),
        'proc_e_fn': argconfig.get('proc_e_fn'),
        'proc_n_units': argconfig.get('proc_n_units'),
        'proc_n_layers': argconfig.get('proc_n_layers'),
        'proc_n_fn': argconfig.get('proc_n_fn'),
    }
    gnn_params = {k: v for k, v in raw_gnn_params.items() if v is not None}

    # Update GNN configuration
    MGN_CONFIG = GNN_CONFIG.copy()
    MGN_CONFIG.update({
        'geometry_type': config['geometry_type'],
        'nx': config['nx'],
        'ny': config['ny'],
        'porder': config['porder'],
        'source_type': config['source_type'],
        'source_value': config['source_value'],
        'hidden': config['hidden'],
        'num_layers': config['num_layers'],
        'latent_dim': config['latent_dim'],
        'lr': config['lr'],
        'epochs': config['epochs'],
        'batch': config['batch'],
        'msg_passes': config['msg_passes'],
        'node_in': config['node_in'],
        'edge_in': config['edge_in'],
        'decoder_out': config['decoder_out'],
        'x_range': config['x_range'], 
        'y_range': config['y_range'],
    })
    MGN_CONFIG.update(gnn_params)

    # ---------------- Multiscale training graphs ----------------
    geom = geometry_factory(MGN_CONFIG['geometry_type'], 
                            x_range=MGN_CONFIG['y_range'], y_range=MGN_CONFIG['y_range'])

    train_resolutions = [(4, 4), (8, 8), (12, 12), (16, 16)]
    val_resolution = (50, 50)

    train_graphs = []
    for nx, ny in train_resolutions:
        prob = get_problem(geometry=geom, nx=nx, ny=ny,
                           porder=MGN_CONFIG['porder'],
                           source_type=MGN_CONFIG['source_type'])
        train_graphs.append(FEM_to_GraphData(prob))

    prob_val = get_problem(geometry=geom, nx=val_resolution[0], ny=val_resolution[1],
                           porder=MGN_CONFIG['porder'],
                           source_type=MGN_CONFIG['source_type'])

    val_graph = FEM_to_GraphData(prob_val)

    train_loader = GNNDataLoader(train_graphs, batch_size=1, shuffle=True)
    val_loader = GNNDataLoader([val_graph], batch_size=1)

    # ---------------- Model definition ----------------
    sample_graph = train_graphs[0]
    MGN_CONFIG['node_in'] = sample_graph.x.shape[1]
    MGN_CONFIG['edge_in'] = sample_graph.edge_attr.shape[1]
    save_json(MGN_CONFIG, gnn_dir, "gnn_config.json")

    model_mgn = MeshGraphNet(
        node_in=MGN_CONFIG['node_in'],
        edge_in=MGN_CONFIG['edge_in'],
        decoder_out=MGN_CONFIG['decoder_out'],
        latent_dim=MGN_CONFIG['latent_dim'],
        msg_passes=MGN_CONFIG['msg_passes'],
        activation=MGN_CONFIG['activation'],
        hidden=MGN_CONFIG['hidden'],
        num_layers=MGN_CONFIG['num_layers'],
        **gnn_params
    )

    system_gnn = MGNSystem(model=model_mgn, lr=MGN_CONFIG['lr'])

    total_params = sum(p.numel() for p in model_mgn.parameters())
    gnn_report_file = os.path.join(gnn_dir, "gnn_report.txt")
    with open(gnn_report_file, "w") as f:
        report_lines = [
            "--- FICHA TÉCNICA GNN (MeshGraphNet) ---",
            f"Resoluciones Entrenamiento: {train_resolutions}",
            f"Resolución Validación: {config['nx']}x{config['ny']}",
            f"Latent Dim: {MGN_CONFIG['latent_dim']}",
            f"Mensajes (Passes): {MGN_CONFIG['msg_passes']}",
            f"Capas MLP: {MGN_CONFIG['num_layers']} | Hidden: {MGN_CONFIG['hidden']}",
            f"Total Parámetros Entrenables: {total_params:,}",
            f"Física: {config['source_type']} (val={config['source_value']})",
            "---------------------------------------"
        ]
        f.write("\n".join(report_lines))
        print("\n".join(report_lines))

    # ---------------- Training ----------------
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=True),
        ModelCheckpoint(monitor='val_loss', filename='best_gnn_model'),
        GradientMonitor(verbose=False),
        LossPlotterCallback(model_name="Poisson GNN")
    ]

    trainer_gnn = pl.Trainer(
        max_epochs=MGN_CONFIG['epochs'],
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        log_every_n_steps=1
    )

    trainer_gnn.fit(system_gnn, train_loader, val_loader)
    # save best model and last state of the model
    best_path_gnn = trainer_gnn.checkpoint_callback.best_model_path
    if best_path_gnn:
        import shutil
        shutil.copy(best_path_gnn, os.path.join(gnn_dir, "best_gnn_model.ckpt"))

    trainer_gnn.save_checkpoint(os.path.join(gnn_dir, "final_gnn_model.ckpt"))
    torch.save(system_gnn.model.state_dict(), os.path.join(gnn_dir, "best_gnn_weights.pth"))
    print(f"SAved GNN model configuration {gnn_dir}")

    # ---------------- Evaluation ----------------
    system_gnn.eval()
    with torch.no_grad():
        u_gnn_pred = system_gnn(val_graph.to(system_gnn.device)).cpu().numpy().flatten()

    fig_com = plot_comparison_with_fem(prob_val['u_exact'], u_gnn_pred, prob_val['doflocs'], model_name="GNN (MeshGraphNet)")
    fig_com.savefig(os.path.join(gnn_dir, "gnn_vs_fem_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_com)

    fig_err = plot_error_analysis(prob_val['u_exact'], u_gnn_pred, model_name="GNN")
    fig_err.savefig(os.path.join(gnn_dir, "gnn_error_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_err)

    np.save(os.path.join(gnn_dir, "u_gnn_pred.npy"), u_gnn_pred)

    print(f">>> GNN completed. Report available at {gnn_dir}")

    # ======================================================
    # 4) Message Passing Graph TRAINING (only procesor)
    # ======================================================
    mpg_dir = os.path.join(run_dir, "mpg")
    os.makedirs(mpg_dir, exist_ok=True)

    raw_mpg_params = {
        'proc_e_units': argconfig.get('proc_e_units'),
        'proc_e_layers': argconfig.get('proc_e_layers'),
        'proc_e_fn': argconfig.get('proc_e_fn'),

        'proc_n_units': argconfig.get('proc_n_units'),
        'proc_n_layers': argconfig.get('proc_n_layers'),
        'proc_n_fn': argconfig.get('proc_n_fn'),
    }   
    mpg_params = {k: v for k, v in raw_mpg_params.items() if v is not None} 
    
    MPG_CONFIG = GNN_CONFIG.copy()
    MPG_CONFIG.update({
        'geometry_type': config['geometry_type'],
        'nx': config['nx'],
        'ny': config['ny'],
        'porder': config['porder'],
        'source_type': config['source_type'],
        'source_value': config['source_value'],
        'hidden': config['hidden'],
        'num_layers': config['num_layers'],
        'lr': config['lr'],
        'epochs': config['epochs'],
        'batch': config['batch'],
        'msg_passes': config['msg_passes'],
        'node_in': config['node_in'],
        'edge_in': config['edge_in'],
        'x_range': config['x_range'], 
        'y_range': config['y_range'],
    })
    MPG_CONFIG.update(mpg_params)

    sample_graph = train_graphs[0]
    MPG_CONFIG['node_in'] = sample_graph.x.shape[1]         # p.ej. 4 ([x, y, is_bc, u_ref])
    MPG_CONFIG['edge_in'] = sample_graph.edge_attr.shape[1]
    save_json(MPG_CONFIG, mpg_dir, "mpg_config.json")

    model_mpg = MessagePassing(
        node_dim=MPG_CONFIG['node_in'], 
        edge_dim=MPG_CONFIG['edge_in'], 
        **MPG_CONFIG 
    )
    system_direct = DirectSystem(
        mp_layer=model_mpg, 
        msg_passes=MPG_CONFIG['msg_passes'], 
        lr=MPG_CONFIG['lr']
    )
    mpg_callbacks = [
        EarlyStopping(monitor='val_loss', patience=100, mode='min'),
        ModelCheckpoint(dirpath=mpg_dir, monitor='val_loss', filename='best_mpg_model'),
        LossPlotterCallback(model_name="Poisson MPG")
    ]
    trainer_mpg = pl.Trainer(
        max_epochs=MPG_CONFIG['epochs'],
        accelerator="auto",
        devices=1,
        callbacks=mpg_callbacks,
        default_root_dir=mpg_dir
    )
    trainer_mpg.fit(system_direct, train_loader, val_loader)
    
    system_direct.eval()
    with torch.no_grad():
        u_gnn_pred = system_direct(val_graph.to(system_direct.device)).cpu().numpy().flatten()

    u_fem = prob_val['u_exact']
    coords = prob_val['doflocs']

    fig_com = plot_comparison_with_fem(u_fem, u_gnn_pred, coords, model_name="Message Passing Graph")
    fig_com.savefig(os.path.join(mpg_dir, "mpg_vs_fem_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_com)

    fig_err = plot_error_analysis(u_fem, u_gnn_pred, model_name="MPG")
    fig_err.savefig(os.path.join(mpg_dir, "mpg_error_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_err)

    np.save(os.path.join(mpg_dir, "u_mpg_pred.npy"), u_gnn_pred)

    print(f">>> MPG completed. Report available at {mpg_dir}")


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":
    """
    Command-line launcher for the full FEM vs PINN vs GNN comparison pipeline.
    """
    argconfig = {
        # organizacion carpetas
        'project_name': "test_run",
        'output_root': 'outputs',
        # geometria 
        'geometry_type': 'square',
        'nx': 2,
        'ny': 2,
        'porder': 2,
        'mesh': 'quad',       # Mesh type: 'tri' or 'quad'
        # fisica 
        'problem': "Poisson",
        'source_type': "sine",
        'source_value': 1.0,
        # Arquitectura y optimizacion        
        'epochs': 10,
        'batch': 32,
        'lr': 1e-3,
        'hidden': 32,
        'num_layers': 2,
        'activation': 'silu',
        'layer_norm': True,
        'dropout': 0.0,
        # Parámetros específicos de PINN que pediste ampliar:
        'lambda_bc': 10.0,              # Peso de la condición de contorno
        'use_fem_for_train': False,     # ¿Usa nodos FEM para entrenar?
        'use_fem_for_test': False,       # ¿Evalúa contra la malla FEM?
        'n_train': 2000,                # Puntos de colocación (interior)
        'n_test': 500,                   # Puntos de validación
        'n_bc': 400,                    # Puntos en el contorno
        # Parámetros GNN
        'node_in': 4,
        'edge_in': 3,
        'decoder_out': 1,
        'latent_dim': 64,       # Espacio latente para encoders y procesadores
        'msg_passes': 6,
        # Extra parameters MeshGraphNet/MessagPassingGraph
        'enc_n_units': 4, 
        'enc_n_layers': 3, 
        'enc_e_units': None, 
        'enc_e_layers': None,
        'dec_n_units': None, 
        'dec_n_layers': None, 
        'proc_e_units': None, 
        'proc_e_layers': None, 
        'proc_e_fn': None, 
        'proc_n_units': None, 
        'proc_n_layers': None, 
        'proc_n_fn': None,
    }

    # 2. Configuración de Argparse
    parser = argparse.ArgumentParser(description="Launcher avanzado para TFG: PINN vs GNN")
    
    # Organización
    parser.add_argument('--project_name', type=str, default=argconfig['project_name'])
    parser.add_argument('--output_root', type=str, default=argconfig['output_root'])

    # modelo y geometria:
    parser.add_argument('--geometry_type', type=str, default=argconfig['geometry_type'])
    parser.add_argument('--nx', type=int, default=argconfig['nx'])
    parser.add_argument('--ny', type=int, default=argconfig['ny'])
    parser.add_argument('--porder', type=int, default=argconfig['porder'])
    parser.add_argument('--mesh', type=str, default=argconfig['mesh'], choices=['tri', 'quad'],
                        help="Tipo de malla para el FEM: 'tri' o 'quad'")
    parser.add_argument('--x_range', type=float, nargs=2, default=[0.0, 1.0], help='Rango en X [min, max]')
    parser.add_argument('--y_range', type=float, nargs=2, default=[0.0, 1.0], help='Rango en Y [min, max]')
    
    # fisica
    parser.add_argument('--problem', type=str, default=argconfig['problem'])
    parser.add_argument('--source_type', type=str, default=argconfig['source_type'])
    parser.add_argument('--source_value', type=float, default=argconfig['source_value'])
    
    # Arquitectura y Optimización
    parser.add_argument('--epochs', type=int, default=argconfig['epochs'])
    parser.add_argument('--batch', type=int, default=argconfig['batch'])
    parser.add_argument('--lr', type=float, default=argconfig['lr'])
    parser.add_argument('--hidden', type=int, default=argconfig['hidden'])
    parser.add_argument('--num_layers', type=int, default=argconfig['num_layers'])
    parser.add_argument('--activation', type=str, default=argconfig['activation'], choices=['tanh', 'relu', 'sigmoid'],
                        help="Función de activación para la PINN")  
    parser.add_argument('--layer_norm', action='store_false', default=argconfig['layer_norm'])
    parser.add_argument('--dropout', type=float, default=argconfig['dropout'])
    
    # Argumentos Pinn
    parser.add_argument('--lambda_bc', type=float, default=argconfig['lambda_bc'], 
                        help="Peso de importancia para la pérdida en el contorno")
    parser.add_argument('--use_fem_train', action='store_true', default=argconfig['use_fem_for_train'],
                        help="Si se activa, usa los nodos del FEM para el entrenamiento")
    parser.add_argument('--use_fem_for_test', action='store_true', default=argconfig['use_fem_for_test'],
                        help="Si se activa, evalúa el modelo en los nodos del FEM (en lugar de puntos aleatorios)")
    parser.add_argument('--n_train', type=int, default=argconfig['n_train'])
    parser.add_argument('--n_bc', type=int, default=argconfig['n_bc'])
    parser.add_argument('--n_test', type=int, default=argconfig['n_test'])
        
    # Argumentos Gnn
    parser.add_argument('--node_in', type=int, default=argconfig['node_in'])
    parser.add_argument('--edge_in', type=int, default=argconfig['edge_in'])
    parser.add_argument('--decoder_out', type=int, default=argconfig['decoder_out'])
    parser.add_argument('--latent_dim', type=int, default=argconfig['latent_dim'])
    parser.add_argument('--msg_passes', type=int, default=argconfig['msg_passes'])
    
    # extra arguments: 
    if argconfig['enc_n_units'] is not None:
        parser.add_argument('--enc_n_units', type=int, default=argconfig['enc_n_units'])
    if argconfig['enc_n_layers'] is not None:
        parser.add_argument('--enc_n_layers', type=int, default=argconfig['enc_n_layers'])

    if argconfig['enc_e_units'] is not None:
        parser.add_argument('--enc_e_units', type=int, default=argconfig['enc_e_units'])
    if argconfig['enc_e_layers'] is not None:
        parser.add_argument('--enc_e_layers', type=int, default=argconfig['enc_e_layers'])

    if argconfig['dec_n_units'] is not None:
        parser.add_argument('--dec_n_units', type=int, default=argconfig['dec_n_units'])
    if argconfig['dec_n_layers'] is not None:
        parser.add_argument('--dec_n_layers', type=int, default=argconfig['dec_n_layers'])

    if argconfig['proc_e_units'] is not None:
        parser.add_argument('--proc_e_units', type=int, default=argconfig['proc_e_units'])
    if argconfig['proc_e_layers'] is not None:
        parser.add_argument('--proc_e_layers', type=int, default=argconfig['proc_e_layers'])
    if argconfig['proc_e_fn'] is not None:
        parser.add_argument('--proc_e_fn', type=str, default=argconfig['proc_e_fn'])

    if argconfig['proc_n_units'] is not None:
        parser.add_argument('--proc_n_units', type=int, default=argconfig['proc_n_units'])
    if argconfig['proc_n_layers'] is not None:
        parser.add_argument('--proc_n_layers', type=int, default=argconfig['proc_n_layers'])
    if argconfig['proc_n_fn'] is not None:
        parser.add_argument('--proc_n_fn', type=int, default=argconfig['proc_n_fn'])

    args = parser.parse_args()
    
    # Actualizamos el diccionario con los valores de la terminal
    argconfig.update(vars(args))

    # Mapeo manual para nombres que no coinciden exactamente (opcional pero recomendado)
    argconfig['layer_norm'] = args.layer_norm
    argconfig['use_fem_for_train'] = args.use_fem_train
    argconfig['use_fem_for_test'] = args.use_fem_for_test

    # Lanzamos el proceso
    main(argconfig)