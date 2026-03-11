import numpy as np
import scipy.sparse as sp
import skfem
from skfem.models.poisson import laplace, mass
from skfem.helpers import dot, grad

try:
    from skfem.assembly import asm, Basis, LinearForm
    from skfem.utils import solve, condense
except ImportError:
    from skfem import asm, solve, condense, Basis, LinearForm

try:
    from skfem.mesh import MeshTri, MeshQuad
except ImportError:
    from skfem import MeshTri, MeshQuad

try:
    from skfem.element import (ElementTriP1, ElementTriP2, 
                                ElementQuad1, ElementQuad2)
except ImportError:
    from skfem import (ElementTriP1, ElementTriP2, 
                        ElementQuad1, ElementQuad2)

# Local modules 
from config.physics import PoissonGeneral
from utils import geometry

def get_problem(geometry, physics=None, nx=4, ny=4, porder=1, mesh='tri', custom_mesh=None):
    """
    Defines a Poisson problem on a geometry:
    - Delta u = f
    u = 0 on boundary
    
    Args:
        nelem (int): Number of elements along one side (total elements = 2 * nelem^2).
        porder (int): Polynomial order (1 for P1, 2 for P2).
        mesh_type (str): Type of mesh ('tri' for triangular, 'quad' for quadrilateral).
        
    Returns:
        dict: Problem data including matrices and solution.
    """
    # MESH
    if custom_mesh is not None:
        m = custom_mesh
        print("Imported custom mesh.")
    elif geometry.name == "square":
        x = np.linspace(geometry.x_range[0], geometry.x_range[1], nx + 1)
        y = np.linspace(geometry.y_range[0], geometry.y_range[1], ny + 1)
        # m = MeshTri.init_tensor(x, y)

        if mesh == 'tri':
            m = MeshTri.init_tensor(x, y)
        elif mesh == 'quad':
            m = MeshQuad.init_tensor(x, y)
        else:
            raise ValueError(f"Tipo de malla '{mesh}' no soportado para Square.")
    elif geometry.name == "circle":
        # m = MeshTri.init_circle(geometry.radius)
        if mesh != 'tri':
            print(f"Warning: Circle only supports 'tri'. Using tri by defult.")
        m = MeshTri.init_circle(nx, geometry.radius) 
    else: 
        raise NotImplementedError(
            f"Interior sampling not implemented for geometry '{geometry.name}'."
        )
   
    # Element and Basis
    if isinstance(m, MeshTri):
        e = ElementTriP1() if porder == 1 else ElementTriP2()
    elif isinstance(m, MeshQuad):
        e = ElementQuad1() if porder == 1 else ElementQuad2()
    else:
        raise ValueError("Unknown polynomial order: {porder}")

    basis = Basis(m, e)
    
    # Physics and Assembly 
    # phys = PoissonPhysics(source_type=source_type, scale=scale)
    if physics is not None:
        print(f"Using custom physics: {physics}")
        phys = physics 
    else:
        print(f"Using default PoissonPhysics with source_type='sine' and scale=1.")
        phys = PoissonGeneral()

    K = asm(laplace, basis)
    M = asm(mass, basis)

    @LinearForm
    def F_form(v, w):
        original_shape = w.x[0].shape 
        x_flat = w.x[0].flatten()
        y_flat = w.x[1].flatten()
        x_input = np.stack([x_flat, y_flat], axis=-1)
        f_val = phys.source_term(x_input)
        f_val = f_val.reshape(original_shape)
        
        return f_val * v
    
    F = asm(F_form, basis)

    # Boundary Conditions (Dirichlet u=0)
    dofs = basis.get_dofs()
    D = dofs.all()
    x_dofs = basis.doflocs.T

    # Exact solution (for validation)
    u_exact = phys.exact_solution(x_dofs)
    if u_exact is not None:
        u_exact = u_exact.flatten()

    # Output of the problem
    return {
        "mesh": m,
        "basis": basis,
        "u": None,
        "u_exact": u_exact,
        "doflocs": x_dofs,
        "boundary_indices": D,
        "interior_indices": basis.complement_dofs(D),
        'F': F,
        'K': K,
        'M': M,
        "physics": phys
    }

def solve_problem(prob):
    K, F, D = prob['K'], prob['F'], prob['boundary_indices']
    u_bc = prob['u_exact']
    u = solve(*condense(K, F, x=u_bc, D=D))
    return u

def compute_fem_flux(prob, u_fem):
    """
    Calcula el flujo (gradiente proyectado) en las aristas de la malla FEM.
    
    Args:
        prob (dict): Diccionario retornado por get_problem.
        u_fem (ndarray): Solución calculada por solve_problem.
        
    Returns:
        ndarray: Flujo en cada arista del grafo (definido por prob['K'].tocoo()).
    """
    coords = prob['doflocs']
    K_coo = prob['K'].tocoo()
    senders, receivers = K_coo.row, K_coo.col
    
    # Diferencia de potencial u_j - u_i
    du = u_fem[receivers] - u_fem[senders]
    
    # Distancia euclídea entre nodos
    dist = np.linalg.norm(coords[receivers] - coords[senders], axis=1) + 1e-8
    
    # Flujo q = -grad(u) projected on edge
    # q_ij = -(u_j - u_i) / dist
    flux = -du / dist
    return flux

if __name__ == "__main__":
    from utils.geometry import geometry_factory
    geo = geometry_factory("square", x_range=[0, 1], y_range=[0, 1])
    prob = get_problem(geometry=geo, nx=10, ny=10, porder=2, mesh_type='tri')
    print(f"--- TEST FEM SOLVER ---")
    print(f"Geometría: {geo.name}")
    print(f"Nodes (DOFs): {prob['doflocs'].shape[0]}")
    print(f"Solution u mean: {prob['u'].mean():.6f}")
    # circle outputs
    print(f"\nProbando cambio a círculo...")
    geo_cir = geometry_factory("circle", center=(0, 0), radius=1.0)
    prob = get_problem(geometry=geo_cir, nx=10, ny=10, porder=2, mesh_type='tri')
    print(f"--- TEST FEM SOLVER ---")
    print(f"Geometría: {geo.name}")
    print(f"Nodes (DOFs): {prob['doflocs'].shape[0]}")
    print(f"Solution u mean: {prob['u'].mean():.6f}")