import os
import json
import argparse
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt

# =========================
# Project-specific imports
# =========================

from config.physics import PoissonGeneral
from config.fem_config import FEM_CONFIG

from PINN.pinn_module import PINNSystem
from FEM.fem_solver import get_problem
from GNN.gnn_module import MessagePassing, DirectSystem
from GNN.MeshGraphNet import MeshGraphNet, MGNSystem
from PiGnn.pignn_module import PhysicsMessagePassing, PhysicsSystem
from PiGnn.piMeshGraphNet import PhysMeshGraphNet, PhysMGNSystem

from utils.geometry import geometry_factory
from utils.plotting import (plot_comparison_with_fem, plot_comparison_with_pinn,
                             plot_error_analysis, plot_flux_comparison)
from utils.metrics import calculate_fem_metrics
from utils.gnn_utils import FEM_to_GraphData
from utils.pignn_utils import FEM_to_PiGnnData, update_graph_geometry
from utils.mathOps import PhysValidation


# ==========================================================
# Helpers
# ==========================================================

def load_json(folder, filename):
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {path}")
    with open(path, "r") as f:
        return json.load(f)

def find_checkpoint(folder, preferred_name):
    """Devuelve el ckpt preferido o el primero que encuentre."""
    preferred = os.path.join(folder, preferred_name)
    if os.path.exists(preferred):
        return preferred
    candidates = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".ckpt")]
    if not candidates:
        raise FileNotFoundError(f"No hay checkpoints en {folder}")
    return candidates[0]

def append_metrics(run_dir, model_data):
    """Añade una fila al CSV de comparativa global."""
    comp_file = os.path.join(run_dir, "compare_models", "summary_metrics.csv")
    os.makedirs(os.path.dirname(comp_file), exist_ok=True)
    fieldnames = ['Modelo', 'Params', 'Error Max', 'Error Medio', 'Error Relat (%)']
    file_exists = os.path.isfile(comp_file)
    with open(comp_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: model_data.get(k, "N/A") for k in fieldnames})

def compute_metrics(u_fem, u_pred, model_name, run_dir, num_params):
    """Calcula métricas estándar y las registra."""
    error_abs = np.abs(u_pred - u_fem)
    rel_error = np.linalg.norm(u_pred - u_fem) / (np.linalg.norm(u_fem) + 1e-10)
    metrics = {
        'Modelo': model_name,
        'Params': f"{num_params:,}",
        'Error Max': f"{np.max(error_abs):.2e}",
        'Error Medio': f"{np.mean(error_abs):.2e}",
        'Error Relat (%)': f"{rel_error * 100:.4f}%"
    }
    append_metrics(run_dir, metrics)
    return metrics


# ==========================================================
# PINN
# ==========================================================

def run_pinn_inference(run_dir, output_dir, physics, geom, prob_val):
    print("--- Ejecutando Inferencia PINN ---")

    u_fem = prob_val.get('u_exact', prob_val.get('u')).flatten()
    doflocs = prob_val['doflocs']

    pinn_dir = os.path.join(run_dir, "pinn")
    config = load_json(pinn_dir, "pinn_config.json")
    ckpt_path = find_checkpoint(pinn_dir, "best_pinn.ckpt")

    model = PINNSystem.load_from_checkpoint(
        ckpt_path, physics=physics, config=config, geometry=geom, weights_only=False)
    model.eval()

    device = next(model.parameters()).device
    x_torch = torch.from_numpy(doflocs).float().to(device)
    with torch.no_grad():
        u_pred = model(x_torch).cpu().numpy().flatten()

    np.save(os.path.join(output_dir, "u_pinn_pred.npy"), u_pred)

    fig = plot_comparison_with_pinn(model, u_fem, doflocs)
    fig.savefig(os.path.join(output_dir, "pinn_vs_fem_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig = plot_error_analysis(u_fem, u_pred, model_name="PINN")
    fig.savefig(os.path.join(output_dir, "pinn_error_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    metrics = compute_metrics(u_fem, u_pred, "PINN", run_dir,
                              sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"    RelErr: {metrics['Error Relat (%)']}  |  Params: {metrics['Params']}")
    print("--- Fin Inferencia PINN ---")
    return u_pred


# ==========================================================
# GNN — MeshGraphNet
# ==========================================================

def run_gnn_inference(run_dir, output_dir, physics, geom, prob_val):
    print("--- Ejecutando Inferencia GNN/MGN ---")

    u_fem = prob_val.get('u_exact', prob_val.get('u')).flatten()
    val_graph = FEM_to_GraphData(prob_val)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gnn_dir = os.path.join(run_dir, "gnn")
    config = load_json(gnn_dir, "gnn_config.json")
    ckpt_path = find_checkpoint(gnn_dir, "best_gnn_model.ckpt")

    model_kwargs = {
        'node_in':     config['node_in'],
        'edge_in':     config['edge_in'],
        'decoder_out': config.get('decoder_out', 1),
        'latent_dim':  config['latent_dim'],
        'hidden':      config.get('hidden', 128),
        'num_layers':  config.get('num_layers', 2),
        'msg_passes':  config.get('msg_passes', 5),
        'layer_norm':  config.get('layer_norm', True),
        'dropout':     config.get('dropout', 0.0),
    }
    for key in ['enc_n_units', 'enc_n_layers', 'enc_e_units', 'enc_e_layers',
                'dec_n_units', 'dec_n_layers', 'proc_e_units', 'proc_e_layers',
                'proc_e_fn', 'proc_n_units', 'proc_n_layers', 'proc_n_fn']:
        if key in config:
            model_kwargs[key] = config[key]

    model = MeshGraphNet(**model_kwargs)
    system = MGNSystem.load_from_checkpoint(ckpt_path, model=model, weights_only=False)
    system.eval()

    with torch.no_grad():
        u_pred = system(val_graph.to(system.device)).cpu().numpy().flatten()

    np.save(os.path.join(output_dir, "u_gnn_pred.npy"), u_pred)

    fig = plot_comparison_with_fem(u_fem, u_pred, prob_val['doflocs'], model_name="GNN (MeshGraphNet)")
    fig.savefig(os.path.join(output_dir, "gnn_vs_fem_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig = plot_error_analysis(u_fem, u_pred, model_name="GNN")
    fig.savefig(os.path.join(output_dir, "gnn_error_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    metrics = compute_metrics(u_fem, u_pred, "GNN", run_dir,
                              sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"    RelErr: {metrics['Error Relat (%)']}  |  Params: {metrics['Params']}")
    print("--- Fin Inferencia GNN/MGN ---")
    return u_pred


# ==========================================================
# MPG — MessagePassing directo
# ==========================================================

def run_mpg_inference(run_dir, output_dir, physics, geom, prob_val):
    print("--- Ejecutando Inferencia GNN/MPG ---")

    u_fem = prob_val.get('u_exact', prob_val.get('u')).flatten()
    val_graph = FEM_to_GraphData(prob_val)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mpg_dir = os.path.join(run_dir, "mpg")
    config = load_json(mpg_dir, "mpg_config.json")
    ckpt_path = find_checkpoint(mpg_dir, "best_mpg_model.ckpt")

    model_kwargs = {
        'node_dim':   config['node_in'],
        'edge_dim':   config['edge_in'],
        'hidden':     config.get('hidden', 128),
        'num_layers': config.get('num_layers', 2),
        'msg_passes': config.get('msg_passes', 5),
        'layer_norm': config.get('layer_norm', True),
        'dropout':    config.get('dropout', 0.0),
    }
    for key in ['proc_e_units', 'proc_e_layers', 'proc_e_fn',
                'proc_n_units', 'proc_n_layers', 'proc_n_fn']:
        if key in config:
            model_kwargs[key] = config[key]

    model = MessagePassing(**model_kwargs)
    system = DirectSystem.load_from_checkpoint(ckpt_path, model=model, weights_only=False)
    system.eval()

    with torch.no_grad():
        u_pred = system(val_graph.to(system.device)).cpu().numpy().flatten()

    np.save(os.path.join(output_dir, "u_mpg_pred.npy"), u_pred)

    fig = plot_comparison_with_fem(u_fem, u_pred, prob_val['doflocs'], model_name="MPG")
    fig.savefig(os.path.join(output_dir, "mpg_vs_fem_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig = plot_error_analysis(u_fem, u_pred, model_name="MPG")
    fig.savefig(os.path.join(output_dir, "mpg_error_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    metrics = compute_metrics(u_fem, u_pred, "MPG", run_dir,
                              sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"    RelErr: {metrics['Error Relat (%)']}  |  Params: {metrics['Params']}")
    print("--- Fin Inferencia GNN/MPG ---")
    return u_pred


# ==========================================================
# PiMPG — PhysicsMessagePassing + PhysicsSystem
# ==========================================================

def run_pimpg_inference(run_dir, output_dir, physics, geom, prob_val):
    print("--- Ejecutando Inferencia PiMPG (PhysicsMessagePassing) ---")

    u_fem = prob_val.get('u_exact', prob_val.get('u')).flatten()
    coords = prob_val['doflocs']

    # dataset PiGNN (FEM_to_PiGnnData, no FEM_to_GraphData)
    val_graph = FEM_to_PiGnnData(prob_val)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pimpg_dir = os.path.join(run_dir, "pimpg")
    config     = load_json(pimpg_dir, "pimpg_config.json")
    optim      = load_json(pimpg_dir, "optim_config.json")
    ckpt_path  = find_checkpoint(pimpg_dir, "best_pignn_model.ckpt")

    # Reconstrucción del modelo — igual que en launcher sección 5
    model_kwargs = {
        'node_dim':   config['node_in'],
        'edge_dim':   config['edge_in'],
        'pos_dim':    config['dim'],
        'hidden':     config.get('hidden', 64),
        'num_layers': config.get('num_layers', 2),
        'activation': config.get('activation', 'silu'),
        'msg_passes': config.get('msg_passes', 8),
        'node_out':   config.get('node_out', 1),
        'edge_out':   config.get('edge_out', 1),
        'layer_norm': config.get('layer_norm', False),
        'dropout':    config.get('dropout', 0.0),
        'source':     config.get('source', False),
        'symmetric':  config.get('symmetric', False),
    }
    for key in ['proc_e_units', 'proc_e_layers', 'proc_e_fn',
                'proc_n_units', 'proc_n_layers', 'proc_n_fn']:
        if key in config:
            model_kwargs[key] = config[key]

    model = PhysicsMessagePassing(**model_kwargs)

    system = PhysicsSystem.load_from_checkpoint(
        ckpt_path,
        model=model,
        physics=physics,
        lr=config.get('lr', 1e-3),
        lambda_bc=config.get('lambda_bc', 100.0),
        lambda_pde=config.get('lambda_pde', 1.0),
        weights_only=False,
        **optim
    )
    system.eval()

    with torch.no_grad():
        u_pred_t, flux_pred_t = system(val_graph.to(system.device))

    u_pred   = u_pred_t.cpu().numpy().flatten()
    flux_pred = flux_pred_t.cpu().numpy().flatten() if flux_pred_t is not None else None

    np.save(os.path.join(output_dir, "u_pimpg_pred.npy"), u_pred)

    # Plot 1: comparación FEM
    fig = plot_comparison_with_fem(u_fem, u_pred, coords, model_name="PiMPG")
    fig.savefig(os.path.join(output_dir, "pimpg_vs_fem_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Plot 2: análisis de error
    fig = plot_error_analysis(u_fem, u_pred, model_name="PiMPG")
    fig.savefig(os.path.join(output_dir, "pimpg_error_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Plot 3: flujo predicho por la red (edge output)
    if flux_pred is not None:
        fig = plot_flux_comparison(
            flux_gnn=flux_pred,
            graph=val_graph.cpu(),
            physics=physics,
            title="PiMPG — Flujo predicho (edge output)"
        )
        fig.savefig(os.path.join(output_dir, "pimpg_flux_predicted.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Plot 4: flujo reconstruido desde grad(u) del nodo
    validador = PhysValidation(physics=physics)
    flujo_gradu_t = validador.math.gradient_edge_discrete(u_pred_t.cpu(), val_graph.cpu())
    flujo_gradu   = -flujo_gradu_t.detach().numpy().flatten()
    fig = plot_flux_comparison(
        flux_gnn=flujo_gradu,
        graph=val_graph.cpu(),
        physics=physics,
        title="PiMPG — Flujo desde grad(u)"
    )
    fig.savefig(os.path.join(output_dir, "pimpg_flux_gradu.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    metrics = compute_metrics(u_fem, u_pred, "PiMPG", run_dir,
                              sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"    RelErr: {metrics['Error Relat (%)']}  |  Params: {metrics['Params']}")
    print("--- Fin Inferencia PiMPG ---")
    return u_pred


# ==========================================================
# PiMGN — PhysMeshGraphNet + PhysMGNSystem
# ==========================================================

def run_pimgn_inference(run_dir, output_dir, physics, geom, prob_val):
    print("--- Ejecutando Inferencia PiMGN (PhysMeshGraphNet) ---")

    u_fem  = prob_val.get('u_exact', prob_val.get('u')).flatten()
    coords = prob_val['doflocs']

    # dataset PiGNN
    val_graph = FEM_to_PiGnnData(prob_val)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pimgn_dir = os.path.join(run_dir, "pimgn")
    config    = load_json(pimgn_dir, "pimgn_config.json")
    optim     = load_json(pimgn_dir, "optim_config.json")
    ckpt_path = find_checkpoint(pimgn_dir, "best_pimgn_model.ckpt")

    # Reconstrucción — igual que PhysMeshGraphNet.__init__ en piMeshGraphNet.py
    model_kwargs = {
        'node_dim':   config['node_in'],
        'edge_dim':   config['edge_in'],
        'pos_dim':    config['dim'],
        'latent_dim': config.get('latent_dim', 64),
        'node_out':   config.get('node_out', 1),
        'edge_out':   config.get('edge_out', 1),
        'activation': config.get('activation', 'silu'),
        'msg_passes': config.get('msg_passes', 8),
        'hidden':     config.get('hidden', 64),
        'num_layers': config.get('num_layers', 2),
        'layer_norm': config.get('layer_norm', False),
        'dropout':    config.get('dropout', 0.0),
        'source':     config.get('source', False),
        'symmetric':  config.get('symmetric', False),
        'decode_edges': config.get('decode_edges', True),
    }
    for key in ['enc_n_units', 'enc_n_layers', 'enc_e_units', 'enc_e_layers',
                'dec_n_units', 'dec_n_layers', 'proc_e_units', 'proc_e_layers',
                'proc_e_fn', 'proc_n_units', 'proc_n_layers', 'proc_n_fn']:
        if key in config:
            model_kwargs[key] = config[key]

    model = PhysMeshGraphNet(**model_kwargs)

    system = PhysMGNSystem.load_from_checkpoint(
        ckpt_path,
        model=model,
        physics=physics,
        lr=config.get('lr', 1e-3),
        lambda_bc=config.get('lambda_bc', 100.0),
        lambda_pde=config.get('lambda_pde', 1.0),
        weights_only=False,
        **optim
    )
    system.eval()

    with torch.no_grad():
        u_pred_t, flux_pred_t = system(val_graph.to(system.device))

    u_pred    = u_pred_t.cpu().numpy().flatten()
    flux_pred = flux_pred_t.cpu().numpy().flatten() if flux_pred_t is not None else None

    np.save(os.path.join(output_dir, "u_pimgn_pred.npy"), u_pred)

    # Plot 1: comparación FEM
    fig = plot_comparison_with_fem(u_fem, u_pred, coords, model_name="PiMGN")
    fig.savefig(os.path.join(output_dir, "pimgn_vs_fem_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Plot 2: análisis de error
    fig = plot_error_analysis(u_fem, u_pred, model_name="PiMGN")
    fig.savefig(os.path.join(output_dir, "pimgn_error_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Plot 3: flujo predicho (edge output del decoder)
    if flux_pred is not None:
        fig = plot_flux_comparison(
            flux_gnn=flux_pred,
            graph=val_graph.cpu(),
            physics=physics,
            title="PiMGN — Flujo predicho (edge output)"
        )
        fig.savefig(os.path.join(output_dir, "pimgn_flux_predicted.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Plot 4: flujo desde grad(u)
    validador = PhysValidation(physics=physics)
    flujo_gradu_t = validador.math.gradient_edge_discrete(u_pred_t.cpu(), val_graph.cpu())
    flujo_gradu   = -flujo_gradu_t.detach().numpy().flatten()
    fig = plot_flux_comparison(
        flux_gnn=flujo_gradu,
        graph=val_graph.cpu(),
        physics=physics,
        title="PiMGN — Flujo desde grad(u)"
    )
    fig.savefig(os.path.join(output_dir, "pimgn_flux_gradu.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    metrics = compute_metrics(u_fem, u_pred, "PiMGN", run_dir,
                              sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"    RelErr: {metrics['Error Relat (%)']}  |  Params: {metrics['Params']}")
    print("--- Fin Inferencia PiMGN ---")
    return u_pred


# ==========================================================
# MAIN
# ==========================================================

INFERENCE_REGISTRY = {
    'pinn':   run_pinn_inference,
    'gnn':    run_gnn_inference,
    'mpg':    run_mpg_inference,
    'pimpg':  run_pimpg_inference,   # PhysicsMessagePassing + PhysicsSystem
    'pimgn':  run_pimgn_inference,   # PhysMeshGraphNet + PhysMGNSystem
}

def main(args, models_to_run):
    run_dir = args.run_dir

    # ---- Reconstruir problema de validación desde fem_config.json ----
    fem_dir    = os.path.join(run_dir, "fem")
    fem_config = load_json(fem_dir, "fem_config.json")

    geom = geometry_factory(
        fem_config['geometry_type'],
        x_range=fem_config.get('x_range', [0.0, 1.0]),
        y_range=fem_config.get('y_range', [0.0, 1.0])
    )
    physics = PoissonGeneral(
        source_type=fem_config['source_type'],
        scale=fem_config['source_value'],
        bc_type=fem_config.get('bc_type', 'zero')
    )
    prob_val = get_problem(
        geometry=geom,
        physics=physics,
        nx=int(fem_config['nx']),
        ny=int(fem_config['ny']),
        porder=int(fem_config.get('porder', 1)),
        mesh=fem_config.get('mesh', 'quad')
    )

    # ---- Bucle de inferencia ----
    results = {}
    for m_type in models_to_run:
        model_root = os.path.join(run_dir, m_type)
        if not os.path.exists(model_root):
            print(f"WARN: Carpeta '{m_type}' no encontrada en {run_dir}. Saltando...")
            continue

        output_dir = os.path.join(model_root, args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nEvaluando {m_type.upper()} → {output_dir}")

        fn = INFERENCE_REGISTRY.get(m_type)
        if fn is None:
            print(f"WARN: Modelo '{m_type}' no tiene función de inferencia registrada.")
            continue

        results[m_type] = fn(run_dir, output_dir, physics, geom, prob_val)

    print("\n✓ Inferencia completada.")
    print(f"  Resumen de métricas: {os.path.join(run_dir, 'compare_models', 'summary_metrics.csv')}")


if __name__ == "__main__":
    # ---------------------------------------------------------
    # CONFIGURACIÓN PREDEFINIDA
    # ---------------------------------------------------------
    main_folder  = "outputs"
    system_name  = "PDE_Poisson"
    project_name = "test_run"
    case_name    = "out_001"
    DEFAULT_FOLDER = os.path.join(main_folder, system_name, project_name, case_name)
    # ---------------------------------------------------------

    inference_pinn  = False
    inference_gnn   = False
    inference_mpg   = False
    inference_pimpg = True    # PhysicsMessagePassing + PhysicsSystem
    inference_pimgn = False   # PhysMeshGraphNet + PhysMGNSystem

    model_list = []
    if inference_pinn:  model_list.append('pinn')
    if inference_gnn:   model_list.append('gnn')
    if inference_mpg:   model_list.append('mpg')
    if inference_pimpg: model_list.append('pimpg')
    if inference_pimgn: model_list.append('pimgn')

    parser = argparse.ArgumentParser(description="Inferencia modelos PDE")
    parser.add_argument('--run_dir',    type=str, default=DEFAULT_FOLDER)
    parser.add_argument('--output_dir', type=str, default="inference_results", nargs='?')
    parser.add_argument('--compare_dir',type=str, default="compare_models",    nargs='?')

    args = parser.parse_args()

    if not os.path.exists(args.run_dir):
        print(f"ERROR: La ruta '{args.run_dir}' no existe.")
    else:
        main(args, model_list)