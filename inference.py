import os
import json
import argparse
import torch
import numpy as np

# Imports del Proyecto
from PINN.pinn_module import PINNSystem
from config.physics import PoissonPhysics
from utils.plotting import plot_comparison_with_pinn, plot_error_analysis, plot_error_boxplot
from utils.geometry import SquareGeometry

def load_json(folder, filename):
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {path}. ¿Es la carpeta correcta?")
    with open(path, "r") as f:
        return json.load(f)

def main(run_dir, output_dir=None, n_pts=5000):
    # 1. Configuración de rutas
    pinn_dir = os.path.join(run_dir, "pinn")
    fem_dir = os.path.join(run_dir, "fem")
    pinn_inf = os.path.join(run_dir, output_dir) if output_dir else os.path.join(run_dir, "inference_results")
    os.makedirs(pinn_inf, exist_ok=True)

    print(f"\n[MODO INFERENCIA] Analizando carpeta: {run_dir}")

    # 2. Reconstrucción de la PINN con su propia configuración guardada
    pinn_arch = load_json(pinn_dir, "pinn_architecture.json")
    physics = PoissonPhysics()    
    ckpt_path = os.path.join(pinn_dir, "best_model.ckpt")
    
    # Cargamos pesos y arquitectura
    model = PINNSystem.load_from_checkpoint(ckpt_path, physics=physics, config=pinn_arch)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 3. Evaluar datos de referencia (FEM/analitica)
    geom = SquareGeometry(x_range=[0, 1], y_range=[0, 1]) 
    coords_rand = geom.sample_interior(n_pts)
    if torch.is_tensor(coords_rand):
        x_tensor = coords_rand.detach().clone().float().to(device)
        coords_rand_np = coords_rand.detach().cpu().numpy()
    else:
        x_tensor = torch.from_numpy(coords_rand).float().to(device)
        coords_rand_np = coords_rand

    u_exact = physics.exact_solution(coords_rand).flatten()

    # 4. Inferencia   
    print("Generando mapa de comparación...")
    fig_comp = plot_comparison_with_pinn(model, u_exact, coords_rand_np, ref_name="Analytic")
    fig_comp.savefig(os.path.join(pinn_inf, "comparison_map.png"))
    
    with torch.no_grad():
        u_pinn = model(x_tensor).cpu().numpy().flatten()


    fig_err = plot_error_analysis(u_fem=u_exact, u_model=u_pinn, model_name="PINN")
    fig_err.savefig(os.path.join(pinn_inf, "error_analysis.png"))

    var_labels = ['u']
    fig_box = plot_error_boxplot(u_exact, u_pinn, var_names=var_labels, model_name="PINN")
    fig_box.savefig(os.path.join(pinn_inf, "error_boxplot.png"))

    l2_rel = np.linalg.norm(u_pinn - u_exact) / np.linalg.norm(u_exact)
    with open(os.path.join(pinn_inf, "metrics.txt"), "w") as f:
        f.write(f"L2_Relative_Error: {l2_rel:.6e}\n")
        f.write(f"Max_Abs_Error: {np.abs(u_pinn - u_exact).max():.6e}")

    print(f">>> Éxito. L2 Relativo: {l2_rel:.2%}")

if __name__ == "__main__":
    # ---------------------------------------------------------
    # CONFIGURACIÓN PREDEFINIDA (Cambia esto para "Run" rápido)
    # ---------------------------------------------------------
    main_folder = "outputs"
    System = "PDE_Poisson"
    project_name = "test_run"   
    case_name = "out_001"
    n_pts = 5000
    DEFAULT_FOLDER = os.path.join(main_folder, System, project_name, case_name)
    # DEFAULT_FOLDER = "outputs/PDE_Poisson/test_run/out_001"
    # ---------------------------------------------------------

    parser = argparse.ArgumentParser(description="Inferencia PINN")
    parser.add_argument('--run_dir', type=str, default=DEFAULT_FOLDER, 
                        help="Carpeta del caso. Si no se da, usa la predefinida.")
    parser.add_argument('--output_dir', type=str, default="inference_results", nargs='?',
                        help="Subcarpeta dentro de run_dir para guardar resultados de inferencia.")
    parser.add_argument('--n_pts', type=int, default=n_pts, help="Número de puntos para la inferencia.")
    
    args = parser.parse_args()
    
    # Verificamos si la carpeta existe (sea la default o la de CLI)
    if os.path.exists(args.run_dir):
        main(args.run_dir)
    else:
        print(f"ERROR: La ruta '{args.run_dir}' no existe. Revisa DEFAULT_FOLDER en el código.")