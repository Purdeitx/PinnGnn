import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

# Configs
from config.common_config import COMMON_CONFIG
from config.pinn_config import PINN_CONFIG
from config.gnn_config import GNN_CONFIG

# Modules
from FEM.fem_solver import get_problem
from PINN.pinn_module import PINNSystem
from GNN.gnn_module import GNNSystem
from GNN.dataset import PINNGraphDataset
from utils.plotting import plot_comparison_with_fem, plot_error_analysis
from utils.metrics import calculate_rrmse

def main():
    parser = argparse.ArgumentParser(description="Inference for PINN and GNN")
    parser.add_argument('--run_dir', type=str, required=True, help="Path to the run directory (e.g., outputs/PDE_Poisson/manual_launch/case_001)")
    parser.add_argument('--nelem', type=int, default=None, help="Overide nelem for inference (defaults to training config)")
    parser.add_argument('--porder', type=int, default=None, help="Overide porder for inference (defaults to training config)")
    
    args = parser.parse_args()
    
    # 0. Load Training Config
    import json
    config_path = os.path.join(args.run_dir, "config_active.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    nelem = args.nelem if args.nelem is not None else config['nelem']
    porder = args.porder if args.porder is not None else config['porder']
    
    # 1. Load Problem (Ground Truth)
    print(f"Generating FEM Ground Truth for inference (nelem={nelem}, porder={porder})...")
    prob = get_problem(nelem=nelem, porder=porder, source_type=config['source_type'], scale=config.get('scale', 1.0))
    u_exact = prob['u_exact']
    doflocs = prob['basis'].doflocs.T
    
    # 2. Load Models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models_dir = os.path.join(args.run_dir, "models")
    
    pinn_best = os.path.join(models_dir, "pinn_best.ckpt")
    pinn_final = os.path.join(models_dir, "pinn_final.ckpt")
    gnn_best = os.path.join(models_dir, "gnn_best.ckpt")
    gnn_final = os.path.join(models_dir, "gnn_final.ckpt")
    
    u_pinn = None
    u_gnn = None
    
    # Try PINN
    pinn_ckpt = pinn_best if os.path.exists(pinn_best) else (pinn_final if os.path.exists(pinn_final) else None)
    if pinn_ckpt:
        print(f"Loading PINN from {os.path.basename(pinn_ckpt)}...")
        model_pinn = PINNSystem.load_from_checkpoint(pinn_ckpt).to(device)
        model_pinn.eval()
        with torch.no_grad():
            x_eval = torch.tensor(doflocs, dtype=torch.float32).to(device)
            u_pinn = model_pinn(x_eval).cpu().numpy().flatten()
            
    # Try GNN
    gnn_ckpt = gnn_best if os.path.exists(gnn_best) else (gnn_final if os.path.exists(gnn_final) else None)
    if gnn_ckpt:
        print(f"Loading GNN from {os.path.basename(gnn_ckpt)}...")
        model_gnn = GNNSystem.load_from_checkpoint(gnn_ckpt).to(device)
        model_gnn.eval()
        
        # We need a dataset for the GNN structure (matching the inference mesh)
        dataset = PINNGraphDataset(nelem=nelem, porder=porder, source_type=config['source_type'])
        data = dataset[0]
        with torch.no_grad():
            u_gnn = model_gnn(data['x'].to(device), data['edge_index'].to(device)).cpu().numpy().flatten()
            
    # 3. Analyze & Plot
    print("\nInference Results Summary:")
    if u_pinn is not None:
        r_pinn = calculate_rrmse(u_exact, u_pinn)
        print(f"PINN RRMSE: {r_pinn:.4e}")
        plot_comparison_with_fem(u_exact, u_pinn, doflocs, os.path.join(args.run_dir, "inference_pinn.png"), "PINN")
        plot_error_analysis(u_exact, u_pinn, os.path.join(args.run_dir, "inference_error_pinn.png"), "PINN")
        
    if u_gnn is not None:
        r_gnn = calculate_rrmse(u_exact, u_gnn)
        print(f"GNN  RRMSE: {r_gnn:.4e}")
        plot_comparison_with_fem(u_exact, u_gnn, doflocs, os.path.join(args.run_dir, "inference_gnn.png"), "GNN")
        plot_error_analysis(u_exact, u_gnn, os.path.join(args.run_dir, "inference_error_gnn.png"), "GNN")
        
    print(f"\nInference completed. Results saved in {args.run_dir}")

if __name__ == "__main__":
    main()
