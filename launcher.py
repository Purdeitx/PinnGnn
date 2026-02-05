import os
import argparse
import torch
import json
import numpy as np
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

# Imports del Proyecto
from config.pinn_config import PINN_CONFIG
from config.physics import PoissonPhysics
from PINN.pinn_module import PINNSystem, PinnDataset, ValDataset
from FEM.fem_solver import get_problem
from utils.geometry import *
from utils.plotting import plot_fem_validation, plot_comparison_with_pinn, plot_error_analysis

def get_next_out_dir(base_path):
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
    """Guarda un diccionario en formato JSON de forma segura."""
    path = os.path.join(folder, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def main(config):
    # --- 1. CONFIGURACIÓN DINÁMICA ---
    # Usamos el config que entra por parámetro (que ya viene actualizado con CLI)
    active_config = config 

    # --- 2. GESTIÓN DE DIRECTORIOS ---
    # Estructura: outputs / PROBLEM / PROJECT / out_XXX
    root_project_dir = os.path.join("outputs", active_config['problem'], active_config['project'])
    case_name = get_next_out_dir(root_project_dir)
    base_dir = os.path.join(root_project_dir, case_name)
    
    fem_dir = os.path.join(base_dir, "fem")
    pinn_dir = os.path.join(base_dir, "pinn")
    os.makedirs(fem_dir, exist_ok=True)
    os.makedirs(pinn_dir, exist_ok=True)

    # Guardamos el maestro en la raíz
    save_json(config, base_dir, "config_active.json")
    # Pinn params (importante para reproducibilidad y análisis posterior)
    pinn_params = {k: config[k] for k in ['hidden_dim', 'num_layers', 'activation', 'epochs', 'lr', 'lambda_bc',
                                'use_fem_for_train', 'use_fem_for_test', 'n_train', 'n_test', 'n_bc']}
    save_json(pinn_params, pinn_dir, "pinn_architecture.json")
    # fem_params (importante para reproducibilidad y análisis posterior)
    fem_params = {k: config[k] for k in ['problem', 'nx', 'ny', 'porder', 'mesh_type']}
    save_json(fem_params, fem_dir, "fem_setup.json")
    print(f"\n>>> Lanzando {active_config['problem']} | Proyecto: {active_config['project']} | Caso: {case_name}")

    # --- 3. PREPARACIÓN DE DATOS Y FÍSICA ---
    geom = SquareGeometry(x_range=[0, 1], y_range=[0, 1])
    physics = PoissonPhysics()
    
    # Baseline FEM usando los parámetros del config
    prob = get_problem(geometry=geom, nx=active_config['nx'], ny=active_config['ny'], porder=active_config['porder']) 
    np.save(os.path.join(fem_dir, "u_fem.npy"), prob['u'])
    np.save(os.path.join(fem_dir, "coords.npy"), prob['doflocs'])
    fem_val = plot_fem_validation(prob, title=f"Validation: {active_config['mesh_type']} P{active_config['porder']} (Res: {active_config['nx']})")
    fem_val.savefig(os.path.join(fem_dir, "fem_validation.png"))

    if active_config['use_fem_for_train']:
        train_pts = torch.tensor(prob['doflocs'], dtype=torch.float32)
        train_label = "FEM Nodes"
    else:
        train_pts = geom.sample_interior(active_config['n_train'])
        train_label = "Random Collocation"

    # 2. Boundary points (BCs) - 2D Logics
    bc_coords = geom.sample_boundary(active_config['n_bc'])
    side_mask = torch.randint(0, 4, (active_config['n_bc'],))
    bc_coords[side_mask==0, 0] = 0.0; bc_coords[side_mask==1, 0] = 1.0
    bc_coords[side_mask==2, 1] = 0.0; bc_coords[side_mask==3, 1] = 1.0

    # 3. Test/Validation points
    if active_config['use_fem_for_test']:
        test_pts = torch.tensor(prob['doflocs'], dtype=torch.float32)
        test_val = torch.tensor(prob['u'], dtype=torch.float32).view(-1, 1)
        test_label = "FEM Mesh"
    else:
        test_pts = geom.sample_interior(active_config['n_test'])
        test_val = None             # Random points
        test_label = "Random Test"

    # 4. DataLoaders en PinnModule
    train_ds = PinnDataset(geometry=geom, pde_pts=train_pts, bc_pts=bc_coords)
    train_loader = DataLoader(train_ds, batch_size=active_config['batch_size'], shuffle=True, num_workers=0)

    val_ds = ValDataset(geometry=geom, pts=test_pts, vals=test_val)
    val_loader = DataLoader(val_ds, batch_size=len(test_pts))


    # --- 4. ENTRENAMIENTO ---
    pinn = PINNSystem(config=active_config, physics=physics)

    callbacks = [
        ModelCheckpoint(dirpath=pinn_dir, filename="best_model", monitor="val_loss", save_top_k=1),
        EarlyStopping(monitor="val_loss", patience=50)
    ]

    trainer = L.Trainer(
        max_epochs=active_config['epochs'],
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        default_root_dir=base_dir
    )

    trainer.fit(pinn, train_loader, val_loader)
    best_model_path = callbacks[0].best_model_path
    print(f"Cargando el mejor modelo desde: {best_model_path}")

    # --- 5. INFERENCIA Y GUARDADO ---
    # Cargamos el mejor checkpoint
    pinn = PINNSystem.load_from_checkpoint(best_model_path, physics=physics, config=active_config)
    pinn.eval()
    with torch.no_grad():
        x_test = torch.tensor(prob['doflocs'], dtype=torch.float32).to(pinn.device)
        u_pinn_pred = pinn(x_test).cpu().numpy().flatten()

    fig_comp = plot_comparison_with_pinn(pinn, prob['u'], prob['doflocs'])
    fig_comp.savefig(os.path.join(pinn_dir, "comparison_map.png"))

    fig_err = plot_error_analysis(u_fem=prob['u'], u_model=u_pinn_pred, model_name="PINN")
    fig_err.savefig(os.path.join(pinn_dir, "error_analysis.png"))

    print(f"\n>>> Finalizado. Resultados en {base_dir}")

if __name__ == "__main__":
    # Diccionario maestro de configuración
    argconfig = {
        'project': "test_run",
        'problem': "PDE_Poisson",
        'lr': 1e-3,
        'epochs': 1000,
        'hidden_dim': 32,
        'num_layers': 4,
        'batch_size': 32,
        'activation': 'tanh',
        # Parámetros específicos de PINN que pediste ampliar:
        'lambda_bc': 10.0,              # Peso de la condición de contorno
        'use_fem_for_train': False,     # ¿Usa nodos FEM para entrenar?
        'use_fem_for_test': False,       # ¿Evalúa contra la malla FEM?
        'n_train': 2000,                # Puntos de colocación (interior)
        'n_test': 500,                   # Puntos de validación
        'n_bc': 400,                    # Puntos en el contorno
        # Parámetros de la malla FEM
        'nx': 2,
        'ny': 2,
        'porder': 2,
        'mesh_type': 'quad'       # Mesh type: 'tri' or 'quad'
    }

    # 2. Configuración de Argparse
    parser = argparse.ArgumentParser(description="Launcher avanzado para TFG: PINN vs GNN")
    
    # Organización
    parser.add_argument('--project', type=str, default=argconfig['project'])
    parser.add_argument('--problem', type=str, default=argconfig['problem'])
    
    # Arquitectura y Optimización
    parser.add_argument('--lr', type=float, default=argconfig['lr'])
    parser.add_argument('--epochs', type=int, default=argconfig['epochs'])
    parser.add_argument('--hidden_dim', type=int, default=argconfig['hidden_dim'])
    parser.add_argument('--num_layers', type=int, default=argconfig['num_layers'])
    parser.add_argument('--batch_size', type=int, default=argconfig['batch_size'])
    parser.add_argument('--activation', type=str, default=argconfig['activation'], choices=['tanh', 'relu', 'sigmoid'],
                        help="Función de activación para la PINN")  
    
    # Pesos de Loss y Estrategia (Crucial para PINNs)
    parser.add_argument('--lambda_bc', type=float, default=argconfig['lambda_bc'], 
                        help="Peso de importancia para la pérdida en el contorno")
    parser.add_argument('--use_fem_train', action='store_true', default=argconfig['use_fem_for_train'],
                        help="Si se activa, usa los nodos del FEM para el entrenamiento")
    parser.add_argument('--use_fem_for_test', action='store_true', default=argconfig['use_fem_for_test'],
                        help="Si se activa, evalúa el modelo en los nodos del FEM (en lugar de puntos aleatorios)")
    
    # Volumen de puntos
    parser.add_argument('--n_train', type=int, default=argconfig['n_train'])
    parser.add_argument('--n_bc', type=int, default=argconfig['n_bc'])
    parser.add_argument('--n_test', type=int, default=argconfig['n_test'])
    
    # Malla FEM
    parser.add_argument('--nx', type=int, default=argconfig['nx'])
    parser.add_argument('--ny', type=int, default=argconfig['ny'])
    parser.add_argument('--porder', type=int, default=argconfig['porder'])
    parser.add_argument('--mesh_type', type=str, default=argconfig['mesh_type'], choices=['tri', 'quad'],
                        help="Tipo de malla para el FEM: 'tri' o 'quad'")
    
    args = parser.parse_args()
    
    # Actualizamos el diccionario con los valores de la terminal
    argconfig.update(vars(args))

    # Mapeo manual para nombres que no coinciden exactamente (opcional pero recomendado)
    argconfig['use_fem_for_train'] = args.use_fem_train
    argconfig['use_fem_for_test'] = args.use_fem_for_test

    # Lanzamos el proceso
    main(argconfig)