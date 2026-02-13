import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import itertools
import skfem as fem

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
from utils.geometry import *
from utils.plotting import *
from utils.metrics import *
from utils.reporting import *
from utils.pinn_utils import *
from utils.gnn_utils import *


# ==========================================================
# Helper utilities
# ==========================================================

def load_json(folder, filename):
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {path}. ¿Es la carpeta correcta?")
    with open(path, "r") as f:
        return json.load(f)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def append_to_metrics_log(file_path, model_data):
    """
    Guarda métricas en un CSV. Si el archivo existe, añade fila. 
    model_data: dict con {'Modelo': ..., 'Error Max': ..., etc.}
    """
    file_exists = os.path.isfile(file_path)
    
    # Definimos el orden de las columnas para que siempre sea igual
    fieldnames = ['Modelo', 'Params', 'Error Max', 'Error Medio', 'Error Relat (%)']
    
    with open(file_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        # Solo escribimos los campos que tenemos definidos
        writer.writerow({k: model_data.get(k, "N/A") for k in fieldnames})

# ==========================================
# MOTORES DE INFERENCIA ESPECÍFICOS
# ==========================================

def run_pinn_inference(run_dir, output_dir, physics, geom, prob_val):
    doflocs = prob_val['doflocs']
    u_fem = prob_val.get('u_exact')
    if u_fem is None:
        u_fem = prob_val.get('u')

    u_fem = u_fem.flatten()

    print("--- Ejecutando Inferencia PINN ---")
    pinn_dir = os.path.join(run_dir, "pinn")
    config = load_json(pinn_dir, "pinn_config.json")
    
    # Buscar el mejor checkpoint
    ckpt_path = os.path.join(pinn_dir, "best_pinn_model.ckpt")
    if not os.path.exists(ckpt_path):
        # Fallback si el nombre es distinto
        ckpt_path = [os.path.join(pinn_dir, f) for f in os.listdir(pinn_dir) if f.endswith(".ckpt")][0]

    model = PINNSystem.load_from_checkpoint(ckpt_path, physics=physics, config=config, geometry=geom, weights_only=False)
    model.eval()

    device = next(model.parameters()).device
    x_torch = torch.from_numpy(doflocs).float().to(device)
    
    with torch.no_grad():
        u_pred = model(x_torch).cpu().numpy().flatten()

    fig_comp = plot_comparison_with_pinn(model, u_fem, doflocs) 
    fig_comp.savefig(os.path.join(output_dir, "pinn_vs_fem_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_comp)

    fig_err = plot_error_analysis(u_fem=u_fem, u_model=u_pred, model_name="PINN")
    fig_err.savefig(os.path.join(output_dir, "pinn_error_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_err)

    error_abs = np.abs(u_pred - u_fem)
    rel_error = np.linalg.norm(u_pred - u_fem) / (np.linalg.norm(u_fem) + 1e-10)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics = {
        'Modelo': 'PINN',
        'Params': f"{num_params:,}",
        'Error Max': f"{np.max(error_abs):.2e}",
        'Error Medio': f"{np.mean(error_abs):.2e}",
        'Error Relat (%)': f"{rel_error * 100:.4f}%"
    }
    comp_file = os.path.join(run_dir, "compare_models", "summary_metrics.csv")
    os.makedirs(os.path.dirname(comp_file), exist_ok=True)
    append_to_metrics_log(comp_file, metrics)

    print("--- Fin de Inferencia PINN ---")
    return u_pred

def run_gnn_inference(run_dir, output_dir, physics, geom, prob_val):
    print("--- Ejecutando Inferencia GNN/MGN ---")
    val_graph = FEM_to_GraphData(prob_val)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_graph = val_graph.to(device)
    u_fem = prob_val.get('u_exact')
    if u_fem is None:
        u_fem = prob_val.get('u')

    u_fem = u_fem.flatten()

    gnn_dir = os.path.join(run_dir, "gnn")
    config = load_json(gnn_dir, "gnn_config.json")
    
    ckpt_path = os.path.join(gnn_dir, "best_gnn_model.ckpt")
    if not os.path.exists(ckpt_path):
        ckpt_path = [os.path.join(gnn_dir, f) for f in os.listdir(gnn_dir) if f.endswith(".ckpt")][0]

    # Reconstrucción del modelo MGN
    model_kwargs = {
        'node_in': config['node_in'],
        'edge_in': config['edge_in'],
        'decoder_out': config.get('decoder_out', 1),
        'latent_dim': config['latent_dim'],
        'hidden': config.get('hidden', 128),
        'num_layers': config.get('num_layers', 2),
        'msg_passes': config.get('msg_passes', 5), 
        'layer_norm': config.get('layer_norm', True),
        'dropout': config.get('dropout', 0.0)
    }
    optional_keys = [
        'enc_n_units', 'enc_n_layers', 
        'enc_e_units', 'enc_e_layers',
        'dec_n_units', 'dec_n_layers', 
        'proc_e_units', 'proc_e_layers', 'proc_e_fn', 
        'proc_n_units', 'proc_n_layers', 'proc_n_fn'
    ]
    for key in optional_keys:
        if key in config:
            model_kwargs[key] = config[key]

    model = MeshGraphNet(**model_kwargs)
    '''
    model = MeshGraphNet(
        node_in=config['node_in'],
        edge_in=config['edge_in'],
        decoder_out=config.get('decoder_out', 1),
        latent_dim=config['latent_dim'],
        hidden=config['hidden'],
        num_layers=config['num_layers'],
        msg_passes=config['msg_passes']
    )
    '''
    
    system = MGNSystem.load_from_checkpoint(ckpt_path, model=model, weights_only=False)
    system.eval()
    with torch.no_grad():
        u_pred = system(val_graph.to(system.device)).cpu().numpy().flatten()

    np.save(os.path.join(output_dir, "u_gnn_pred.npy"), u_pred)

    fig_com = plot_comparison_with_fem(u_fem, u_pred, prob_val['doflocs'], model_name="GNN (MeshGraphNet)")
    fig_com.savefig(os.path.join(output_dir, "gnn_vs_fem_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_com)

    fig_err = plot_error_analysis(u_fem=u_fem, u_model=u_pred, model_name="GNN")
    fig_err.savefig(os.path.join(output_dir, "gnn_error.png"))
    plt.close(fig_err)

    error_abs = np.abs(u_pred - u_fem)
    rel_error = np.linalg.norm(u_pred - u_fem) / (np.linalg.norm(u_fem) + 1e-10)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics = {
        'Modelo': 'GNN',
        'Params': f"{num_params:,}",
        'Error Max': f"{np.max(error_abs):.2e}",
        'Error Medio': f"{np.mean(error_abs):.2e}",
        'Error Relat (%)': f"{rel_error * 100:.4f}%"
    }

    comp_file = os.path.join(run_dir, "compare_models", "summary_metrics.csv")
    os.makedirs(os.path.dirname(comp_file), exist_ok=True)
    append_to_metrics_log(comp_file, metrics)

    print("--- Fin de Inferencia GNN/MGN ---")
    return u_pred

def run_mpg_inference(run_dir, output_dir, physics, geom, prob_val):
    print("--- Ejecutando Inferencia GNN/MPG ---")
    val_graph = FEM_to_GraphData(prob_val)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_graph = val_graph.to(device)
    u_fem = prob_val.get('u_exact')
    if u_fem is None:
        u_fem = prob_val.get('u')

    u_fem = u_fem.flatten()

    mpg_dir = os.path.join(run_dir, "mpg")
    config = load_json(mpg_dir, "mpg_config.json")
    
    ckpt_path = os.path.join(mpg_dir, "best_mpg_model.ckpt")
    if not os.path.exists(ckpt_path):
        ckpt_path = [os.path.join(mpg_dir, f) for f in os.listdir(mpg_dir) if f.endswith(".ckpt")][0]

    # Reconstrucción del modelo MGN
    model_kwargs = {
        'node_dim': config['node_in'],
        'edge_dim': config['edge_in'],
        'hidden': config.get('hidden', 128),
        'num_layers': config.get('num_layers', 2),
        'msg_passes': config.get('msg_passes', 5),
        'layer_norm': config.get('layer_norm', True),
        'dropout': config.get('dropout', 0.0)
    }
    optional_keys = [
        'proc_e_units', 'proc_e_layers', 'proc_e_fn', 
        'proc_n_units', 'proc_n_layers', 'proc_n_fn'
    ]
    for key in optional_keys:
        if key in config:
            model_kwargs[key] = config[key]

    model = MessagePassing(**model_kwargs)

    '''
    model = MessagePassing(
        node_dim=config['node_in'], 
        edge_dim=config['edge_in'],
        hidden=config['hidden'],
        num_layers=config['num_layers'],
        msg_passes=config['msg_passes'],
    )
    '''
    
    system = DirectSystem.load_from_checkpoint(ckpt_path, model=model, weights_only=False)
    system.eval()
    with torch.no_grad():
        u_pred = system(val_graph.to(system.device)).cpu().numpy().flatten()

    np.save(os.path.join(output_dir, "u_mpg_pred.npy"), u_pred)

    fig_com = plot_comparison_with_fem(u_fem, u_pred, prob_val['doflocs'], model_name="MPG (MesPasGraph)")
    fig_com.savefig(os.path.join(output_dir, "mpg_vs_fem_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_com)

    fig_err = plot_error_analysis(u_fem=u_fem, u_model=u_pred, model_name="MPG")
    fig_err.savefig(os.path.join(output_dir, "mpg_error.png"))
    plt.close(fig_err)

    error_abs = np.abs(u_pred - u_fem)
    rel_error = np.linalg.norm(u_pred - u_fem) / (np.linalg.norm(u_fem) + 1e-10)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics = {
        'Modelo': 'MPG',
        'Params': f"{num_params:,}",
        'Error Max': f"{np.max(error_abs):.2e}",
        'Error Medio': f"{np.mean(error_abs):.2e}",
        'Error Relat (%)': f"{rel_error * 100:.4f}%"
    }

    comp_file = os.path.join(run_dir, "compare_models", "summary_metrics.csv")
    os.makedirs(os.path.dirname(comp_file), exist_ok=True)
    append_to_metrics_log(comp_file, metrics)

    print("--- Fin de Inferencia GNN/MPG ---")
    return u_pred

# ==========================================
# MAIN LOOP
# ==========================================

def main(args, models_to_run):
    run_dir = args.run_dir

    # 1. Cargar la "Verdad" (FEM y Física)
    # Buscamos en la carpeta fem los datos originales para comparar
    fem_dir = os.path.join(run_dir, "fem")
    u_fem = np.load(os.path.join(fem_dir, "u_fem.npy"))
    doflocs = np.load(os.path.join(fem_dir, "doflocs.npy"))
    
    # Reconstruimos el objeto problema para la GNN (necesita conectividad)
    # Asumimos que guardaste los parámetros en fem_config.json
    fem_config = load_json(fem_dir, "fem_config.json")
    geom = geometry_factory(
        fem_config['geometry_type'], 
        x_range=fem_config.get('x_range', [0.0, 1.0]), 
        y_range=fem_config.get('y_range', [0.0, 1.0])
    )
    prob_val = get_problem(
        geometry=geom, 
        nx=int(fem_config['nx']), 
        ny=int(fem_config['ny']),
        porder=int(fem_config['porder']),
        source_type=fem_config['source_type'],
        mesh_type=fem_config.get('mesh', 'tri') 
    )
    physics = PoissonPhysics(fem_config['source_type'], fem_config['source_value'])

    # listado de modelos para inferencia
    results = {}
    for m_type in models_to_run:
        model_root = os.path.join(run_dir, m_type)
        if os.path.exists(model_root):
            model_inf_dir = os.path.join(model_root, args.output_dir)
            os.makedirs(model_inf_dir, exist_ok=True)
                
            print(f"Evaluando {m_type.upper()} -> Guardando en {model_inf_dir}")
            if m_type == 'pinn':
                results['pinn'] = run_pinn_inference(
                    run_dir, model_inf_dir, physics, geom, prob_val)
            
            elif m_type == 'gnn':
                results['gnn'] = run_gnn_inference(
                    run_dir, model_inf_dir, physics, geom, prob_val)
                
            elif m_type == 'mpg':
                results['mpg'] = run_mpg_inference(
                    run_dir, model_inf_dir, physics, geom, prob_val)
        else:
            print(f"WARN: Carpeta de modelo '{m_type}' no existe. Saltando...")


    # 3. Comparativas Cruzadas (En la raíz del run_dir)
    # comp_dir = os.path.join(run_dir, args.compare_dir)
    print(f"WARN: Sistema de comparativas por definir. Saltando...")


if __name__ == "__main__":
    # ---------------------------------------------------------
    # CONFIGURACIÓN PREDEFINIDA (Cambia esto para "Run" rápido)
    # ---------------------------------------------------------
    main_folder = "outputs"
    System = "PDE_Poisson"
    project_name = "test_run"   
    case_name = "out_003"
    n_pts = 5000
    DEFAULT_FOLDER = os.path.join(main_folder, System, project_name, case_name)
    # DEFAULT_FOLDER = "outputs/PDE_Poisson/test_run/out_001"
    # ---------------------------------------------------------
    inference_pinn = True
    inference_gnn = True
    inference_mpg = True

    model_list = []
    if inference_pinn: model_list.append('pinn')
    if inference_gnn:  model_list.append('gnn')
    if inference_mpg:  model_list.append('mpg')
    # ---------------------------------------------------------

    parser = argparse.ArgumentParser(description="Inferencia PINN")
    parser.add_argument('--run_dir', type=str, default=DEFAULT_FOLDER, 
                        help="Carpeta del caso. Si no se da, usa la predefinida.")
    parser.add_argument('--output_dir', type=str, default="inference_results", nargs='?',
                        help="Subcarpeta dentro de run_dir para guardar resultados de inferencia.")
    parser.add_argument('--compare_dir', type=str, default="compare_models", nargs='?',
                        help="Subcarpeta dentro de run_dir para guardar comparativas de modelos.")
    parser.add_argument('--n_pts', type=int, default=n_pts, help="Número de puntos para la inferencia.")
    
    args = parser.parse_args()
    
    # Verificamos si la carpeta existe (sea la default o la de CLI)
    if os.path.exists(args.run_dir):
        main(args, model_list)
    else:
        print(f"ERROR: La ruta '{args.run_dir}' no existe. Revisa DEFAULT_FOLDER en el código.")