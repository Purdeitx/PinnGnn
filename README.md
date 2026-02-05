# Physics-Informed & Graph Neural Networks for PDEs

This repository contains a comprehensive framework for solving Partial Differential Equations (PDEs) using a comparative approach between classical numerical methods and modern Deep Learning techniques. 

The project evaluates the steady-state **Poisson Equation** on a 2D domain, establishing a baseline for future research into non-Fourier heat conduction models (Cattaneo-Vernotte).

## 🚀 Project Overview

The core of this research is to compare three distinct paradigms for solving PDEs:
1.  **Finite Element Method (FEM)**: The classical numerical baseline using `skfem`.
2.  **Physics-Informed Neural Networks (PINNs)**: A mesh-free approach that solves the PDE by minimizing residuals via Automatic Differentiation.
3.  **Graph Neural Networks (GNNs)**: A data-driven approach that exploits the topological structure of the mesh to predict physical fields.



## 🛠️ Repository Structure

The project is designed with a modular architecture to ensure scalability and reproducibility:

* `GraphPinns.ipynb`: The main tutorial and experimental notebook.
* `config/`:
    * `physics.py`: Definition of physical laws (Residuals, Source terms, BCs).
    * `pinn_config.py`: Architecture and training hyperparameters.
* `PINN/`:
    * `pinn_module.py`: The "Engine" (PyTorch Lightning System) that handles the training logic.
* `utils/`:
    * `geometry.py`: Domain definition and point sampling strategies.
    * `plotting.py`: Visualization tools for comparison and error analysis.

## 📋 Requirements & Setup

This project uses **Conda** for environment management. 

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/Purdeitx/PinnGNN.git](https://github.com/Purdeitx/PinnGNN.git)
    cd PinnGNN
    ```

2.  **Create and activate the environment**:
    ```bash
    conda activate /home/purdeitx/miniconda3/envs/PinnGnn
    ```

3.  **Key Libraries**:
    * `PyTorch` & `PyTorch Lightning` (Deep Learning)
    * `skfem` (Numerical Baseline)
    * `torch_geometric` (Graph Neural Networks)

## 🧪 Experiments: The Poisson Equation

Currently, the framework solves:
$$-\Delta u(x, y) = f(x, y) \quad \forall (x, y) \in \Omega$$
with Dirichlet boundary conditions $u = 0$ on $\partial\Omega$.

### Key Features:
* **Modular Physics**: Swap the Poisson equation for Fourier or Cattaneo models just by editing `physics.py`.
* **Hardware Acceleration**: Full support for CUDA/GPU training via PyTorch Lightning.
* **Error Quantization**: Automatic calculation of $L_2$ relative error against FEM ground truth.



## 📈 Future Work
- [ ] Implementation of **Cattaneo-Vernotte** thermal relaxation model.
- [ ] Integration of **PiGNN** (Physics-Informed Graph Neural Networks).
- [ ] Temporal evolution analysis for transient heat equations.

---
**Author**: [Tu Nombre/purdeitx]  
**Academic Context**: ...