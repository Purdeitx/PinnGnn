import os

# =============================================================================
# ENVIRONMENT WORKAROUNDS FOR WINDOWS
# =============================================================================
# 1. Allow duplicate OpenMP runtime (avoids OMP: Error #15)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

# 2. Use non-interactive backend for stability (avoids QThreadStorage crashes)
plt.switch_backend('Agg')

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import sys

# Configs
from config.common_config import COMMON_CONFIG
from config.fem_config import FEM_CONFIG
from config.pinn_config import PINN_CONFIG
from config.gnn_config import GNN_CONFIG

# Modules
from FEM.fem_solver import get_problem
from PINN.pinn_module import PINNSystem
from GNN.gnn_module import GNNSystem
from GNN.dataset import PINNGraphDataset
from torch.utils.data import DataLoader
from utils.plotting import plot_loss, plot_comparison_with_fem, plot_error_analysis, save_simulation_gif
from utils.metrics import calculate_rrmse
from utils.io_utils import get_case_path, save_config_json, redirect_stdout_to_file

class ConsoleLoggerCallback(pl.Callback):
    """Callback for clean, periodic console logging."""
    def __init__(self, every_n_epochs=10):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0 or trainer.current_epoch == trainer.max_epochs - 1:
            loss = trainer.callback_metrics.get("train_loss")
            if loss is not None:
                print(f"Epoch {trainer.current_epoch:4d} | Loss: {loss:.6e}")

class GIFCallback(pl.Callback):
    """Callback to capture solution frames for GIF generation."""
    def __init__(self, points, model_type="PINN", every_n_epochs=50):
        super().__init__()
        self.points = points
        self.model_type = model_type
        self.every_n_epochs = every_n_epochs
        self.frames = []

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            pl_module.eval()
            with torch.no_grad():
                if self.model_type == "PINN":
                    x = torch.tensor(self.points, dtype=torch.float32).to(pl_module.device)
                    u = pl_module(x).cpu().numpy().flatten()
                else:
                    batch = next(iter(trainer.train_dataloader))
                    x_gnn = batch['x']
                    while x_gnn.dim() > 2: x_gnn = x_gnn[0]
                    ei_gnn = batch['edge_index']
                    while ei_gnn.dim() > 2: ei_gnn = ei_gnn[0]
                    
                    u = pl_module(x_gnn.to(pl_module.device), ei_gnn.to(pl_module.device)).cpu().numpy().flatten()
            self.frames.append(u)
            pl_module.train()

def train(config):
    """
    Main training function that accepts a configuration dictionary.
    """
    # 1. Setup Output Directory
    case_path = get_case_path(COMMON_CONFIG['output_root'], config['pde'], config['project'])
    
    # Wrap everything in stdout redirection to case.log
    with redirect_stdout_to_file(os.path.join(case_path, "case.log")):
        print(f"=== PinnGNN Benchmarking Session ===")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Case Path: {case_path}")
        print(f"Active Config: {config}")
        
        # Save exact configuration used for this run
        save_config_json(config, case_path, "config_active.json")

        # 2. FEM GROUND TRUTH
        print("\n[1/4] Generating FEM Ground Truth...")
        prob = get_problem(
            nelem=config['nelem'], 
            porder=config['porder'], 
            source_type=config['source_type'], 
            scale=config['scale'],
            source_value=config.get('source_value', 1.0)
        )
        u_exact = prob['u_exact']
        doflocs = prob['basis'].doflocs.T
        
        u_pinn = None
        u_gnn = None

        # 3. TRAIN PINN
        if config['mode'] in ['pinn', 'compare']:
            print("\n[2/4] Training PINN...")
            system_pinn = PINNSystem(
                hidden_dim=config['pinn_hidden_dim'],
                num_layers=config['pinn_layers'],
                lr=config['pinn_lr'],
                lambda_bc=config['pinn_lambda_bc'],
                n_collocation=config['pinn_n_collocation'],
                n_boundary=config['pinn_n_boundary']
            )
            
            # Setup Validation Data (Fixed off-grid points)
            x_val = torch.rand(1000, 2)
            if config['source_type'] == 'sine':
                u_val = prob['u_exact_fn'](x_val.numpy(), scale=config.get('scale', 1.0))
            else:
                # Interpolator for 'const' doesn't take scale argument
                u_val = prob['u_exact_fn'](x_val.numpy())
            
            u_val = torch.tensor(u_val, dtype=torch.float32)
            val_loader = DataLoader([(x_val, u_val)], batch_size=1)

            models_dir = os.path.join(case_path, "models")
            os.makedirs(models_dir, exist_ok=True)
            
            checkpoint_callback = ModelCheckpoint(
                dirpath=models_dir,
                filename="pinn_best",
                monitor="val_loss",
                mode="min",
                save_top_k=1
            )

            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                patience=50,
                mode="min",
                verbose=True
            )

            callbacks = [
                ConsoleLoggerCallback(every_n_epochs=10),
                checkpoint_callback,
                early_stop_callback
            ]

            gif_cb = None
            if config['generate_gif'] and config['pde'] != 'Poisson':
                gif_cb = GIFCallback(doflocs, model_type="PINN", every_n_epochs=max(1, config['epochs'] // 20))
                callbacks.append(gif_cb)
            elif config['generate_gif'] and config['pde'] == 'Poisson':
                print("Note: GIF generation skipped for static Poisson problem.")

            logger_list = [CSVLogger(save_dir=case_path, name="pinn_logs")]

            trainer = pl.Trainer(
                max_epochs=config['epochs'],
                accelerator='auto',
                devices=1,
                logger=logger_list,
                callbacks=callbacks,
                enable_checkpointing=True,
                enable_progress_bar=False,
                enable_model_summary=False,
                log_every_n_steps=1,
                check_val_every_n_epoch=min(10, config['epochs'])
            )
            
            train_loader = DataLoader([0], batch_size=1) 
            trainer.fit(system_pinn, train_loader, val_loader)
            
            # Save final as well
            trainer.save_checkpoint(os.path.join(models_dir, "pinn_final.ckpt"))

            # Results & Plots (Use best for plots)
            best_model_path = checkpoint_callback.best_model_path
            if best_model_path:
                print(f"Loading best PINN model from {best_model_path} for final evaluation.")
                system_pinn = PINNSystem.load_from_checkpoint(best_model_path)
            
            u_pinn = system_pinn(torch.tensor(doflocs, dtype=torch.float32).to(system_pinn.device)).detach().cpu().numpy().flatten()
            plot_comparison_with_fem(u_exact, u_pinn, doflocs, os.path.join(case_path, "fem_pinn.png"), "PINN")
            plot_error_analysis(u_exact, u_pinn, os.path.join(case_path, "error_pinn.png"), "PINN")
            
            metrics_path = os.path.join(case_path, "pinn_logs", "version_0", "metrics.csv")
            if os.path.exists(metrics_path):
                df = pd.read_csv(metrics_path)
                df_epoch = df.groupby('epoch').mean().ffill().reset_index()
                # Training Decomposition Plot (No val_loss)
                train_cols = [c for c in df_epoch.columns if (c.startswith('loss_') or c == 'train_loss') and c != 'val_loss']
                plot_loss(df_epoch[train_cols], os.path.join(case_path, "loss_pinn.png"), "PINN Decomposition")
                
                # Dedicated Train vs Val plot
                eval_cols = [c for c in ['train_loss', 'val_loss'] if c in df_epoch.columns]
                if len(eval_cols) >= 1:
                    plot_loss(df_epoch[eval_cols], os.path.join(case_path, "lossEval_pinn.png"), "PINN Train vs Val")

            if config['generate_gif'] and gif_cb:
                save_simulation_gif(gif_cb.frames, u_exact, doflocs, os.path.join(case_path, "gif_pinn.gif"), "PINN Training")

            print(f"PINN models saved to {models_dir}")


        # 4. TRAIN GNN
        if config['mode'] in ['gnn', 'compare']:
            print("\n[3/4] Training GNN...")
            system_gnn = GNNSystem(
                hidden_dim=config['gnn_hidden_dim'],
                num_layers=config['gnn_layers'],
                lr=config['gnn_lr'],
                lambda_bc=config['gnn_lambda_bc'],
                supervised=config['supervised_gnn'],
                use_chebconv=config.get('use_chebconv', False)
            )
            
            dataset = PINNGraphDataset(nelem=config['nelem'], porder=config['porder'], source_type=config['source_type'])
            loader = DataLoader(dataset, batch_size=1)
            # For GNN, we use the same graph in validation to monitor RRMSE
            val_loader = loader 

            models_dir = os.path.join(case_path, "models")
            os.makedirs(models_dir, exist_ok=True)

            checkpoint_callback = ModelCheckpoint(
                dirpath=models_dir,
                filename="gnn_best",
                monitor="val_loss",
                mode="min",
                save_top_k=1
            )

            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                patience=50,
                mode="min",
                verbose=True
            )

            callbacks = [
                ConsoleLoggerCallback(every_n_epochs=10),
                checkpoint_callback,
                early_stop_callback
            ]

            gif_cb = None
            if config['generate_gif'] and config['pde'] != 'Poisson':
                gif_cb = GIFCallback(doflocs, model_type="GNN", every_n_epochs=max(1, config['epochs'] // 20))
                callbacks.append(gif_cb)
            elif config['generate_gif'] and config['pde'] == 'Poisson':
                # Already printed note in PINN section or just skip silently here
                pass

            logger_list = [CSVLogger(save_dir=case_path, name="gnn_logs")]

            trainer = pl.Trainer(
                max_epochs=config['epochs'],
                accelerator='auto',
                devices=1,
                logger=logger_list,
                callbacks=callbacks,
                enable_checkpointing=True,
                enable_progress_bar=False,
                enable_model_summary=False,
                log_every_n_steps=1,
                check_val_every_n_epoch=min(10, config['epochs'])
            )
            
            trainer.fit(system_gnn, loader, val_loader)
            
            # Save final
            trainer.save_checkpoint(os.path.join(models_dir, "gnn_final.ckpt"))

            # Results & Plots (Use best for plots)
            best_model_path = checkpoint_callback.best_model_path
            if best_model_path:
                print(f"Loading best GNN model from {best_model_path} for final evaluation.")
                system_gnn = GNNSystem.load_from_checkpoint(best_model_path)

            u_gnn = system_gnn(dataset[0]['x'].to(system_gnn.device), dataset[0]['edge_index'].to(system_gnn.device)).detach().cpu().numpy().flatten()
            plot_comparison_with_fem(u_exact, u_gnn, doflocs, os.path.join(case_path, "fem_gnn.png"), "GNN")
            plot_error_analysis(u_exact, u_gnn, os.path.join(case_path, "error_gnn.png"), "GNN")
            
            metrics_path = os.path.join(case_path, "gnn_logs", "version_0", "metrics.csv")
            if os.path.exists(metrics_path):
                df = pd.read_csv(metrics_path)
                df_epoch = df.groupby('epoch').mean().ffill().reset_index()
                # Training Decomposition Plot (No val_loss)
                train_cols = [c for c in df_epoch.columns if (c.startswith('loss_') or c == 'train_loss') and c != 'val_loss']
                plot_loss(df_epoch[train_cols], os.path.join(case_path, "loss_gnn.png"), "GNN Decomposition")

                # Dedicated Train vs Val plot
                eval_cols = [c for c in ['train_loss', 'val_loss'] if c in df_epoch.columns]
                if len(eval_cols) >= 1:
                    plot_loss(df_epoch[eval_cols], os.path.join(case_path, "lossEval_gnn.png"), "GNN Train vs Val")

            if config['generate_gif'] and gif_cb:
                save_simulation_gif(gif_cb.frames, u_exact, doflocs, os.path.join(case_path, "gif_gnn.gif"), "GNN Training")

            print(f"GNN models saved to {models_dir}")

        # 5. SUMMARY
        if config['mode'] == 'compare' and u_pinn is not None and u_gnn is not None:
            print("\n[4/4] Final Metrics Benchmark:")
            r_pinn = calculate_rrmse(u_exact, u_pinn)
            r_gnn = calculate_rrmse(u_exact, u_gnn)
            print(f"PINN RRMSE: {r_pinn:.4e}")
            print(f"GNN  RRMSE: {r_gnn:.4e}")

    print(f"\nDone! Results are at: {case_path}")

def main():
    # =========================================================================
    # 0. CONFIGURACIÓN MANUAL (PRIORIDAD ALTA)
    # Aquí puedes poner "lo que te dé la gana". Estos valores sobrescriben
    # los ficheros config/*.py, pero pueden ser sobrescritos por CLI.
    # =========================================================================
    manual_config = {
        "epochs": 500,   # Aligned with Gao's maxit=500
        "pinn_lr": 1e-3, # Standard LR
        "gnn_lr": 1e-3,
        "project": "gao_alignment",
        "nelem": 2,
        "porder": 2, 
    }

    # =========================================================================
    # 1. CONSTRUCCIÓN DE ARGCONFIG (JERARQUÍA: Default -> Manual)
    # =========================================================================
    argconfig = {
        "pde": COMMON_CONFIG['pde_type'],
        "project": COMMON_CONFIG['project_name'],
        "epochs": COMMON_CONFIG['epochs'],
        "mode": 'compare',
        "scale": 1.0,
        "generate_gif": False,
        "supervised_gnn": GNN_CONFIG['supervised'],
        # Architecture and weights from files
        "pinn_hidden_dim": PINN_CONFIG['hidden_dim'],
        "pinn_layers": PINN_CONFIG['num_layers'],
        "pinn_lr": PINN_CONFIG['lr'],
        "pinn_lambda_bc": PINN_CONFIG['lambda_bc'],
        "pinn_n_collocation": PINN_CONFIG['n_collocation'],
        "pinn_n_boundary": PINN_CONFIG['n_boundary'],
        "gnn_hidden_dim": GNN_CONFIG['hidden_dim'],
        "gnn_layers": GNN_CONFIG['num_layers'],
        "gnn_lr": GNN_CONFIG['lr'],
        "gnn_lambda_bc": GNN_CONFIG['lambda_bc'],
        "nelem": COMMON_CONFIG['nelem'],
        "porder": COMMON_CONFIG['porder'],
        "source_type": COMMON_CONFIG['source_type'],
        "source_value": COMMON_CONFIG.get('source_value', 1.0),
        "use_chebconv": GNN_CONFIG.get('use_chebconv', False)
    }
    
    # Sobrescribir con manual_config
    argconfig.update(manual_config)

    # =========================================================================
    # 2. PARSE DE ARGUMENTOS (PRIORIDAD MÁXIMA)
    # Usamos los valores de argconfig como DEFAULTS en argparse.
    # =========================================================================
    parser = argparse.ArgumentParser(description="PinnGNN Benchmark Launcher (v3 - Manual Priority)")
    parser.add_argument('--mode', type=str, default=argconfig['mode'], choices=['pinn', 'gnn', 'compare'])
    parser.add_argument('--project', type=str, default=argconfig['project'])
    parser.add_argument('--pde', type=str, default=argconfig['pde'])
    parser.add_argument('--epochs', type=int, default=argconfig['epochs'])
    parser.add_argument('--supervised_gnn', action='store_true', default=argconfig['supervised_gnn'])
    parser.add_argument('--generate_gif', action='store_true', default=argconfig['generate_gif'])
    parser.add_argument('--scale', type=float, default=argconfig['scale'])
    
    # Argumentos adicionales para overrides rápidos de arquitectura/LR/Malla
    parser.add_argument('--pinn_lr', type=float, default=argconfig['pinn_lr'])
    parser.add_argument('--gnn_lr', type=float, default=argconfig['gnn_lr'])
    parser.add_argument('--nelem', type=int, default=argconfig['nelem'])
    parser.add_argument('--porder', type=int, default=argconfig['porder'])
    
    args = parser.parse_args()

    # Actualizar argconfig con lo que venga por CLI (que tiene los defaults de manual/static)
    # Convertimos Namespace a dict
    cli_overrides = vars(args)
    argconfig.update(cli_overrides)

    # =========================================================================
    # 3. LANZAMIENTO
    # =========================================================================
    train(argconfig)

if __name__ == "__main__":
    main()
