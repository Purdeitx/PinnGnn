import numpy as np
import scipy.sparse as sp
from skfem import *
from skfem.models.poisson import laplace, mass
from config.physics import PoissonPhysics
from skfem.helpers import dot, grad

def get_problem(nx=4, ny=4, porder=1, source_type='sine', mesh_type='tri', scale=1.0, custom_mesh=None):
    """
    Defines a Poisson problem on a Unit Square:
    - Delta u = f
    u = 0 on boundary
    
    Args:
        nelem (int): Number of elements along one side (total elements = 2 * nelem^2).
        porder (int): Polynomial order (1 for P1, 2 for P2).
        source_type (str): 'sine' or 'const'.
        scale (float): Scaling factor (default 1.0 for unit range).
        source_value (float): Source value for const case, matching Gao's f(x,el)=source_value.
        mesh_type (str): Type of mesh ('tri' for triangular, 'quad' for quadrilateral).
        
    Returns:
        dict: Problem data including matrices and solution.
    """
    
    # 1. Mesh: Structured Grid
    if custom_mesh is not None:
        m = custom_mesh
    else:
        x = np.linspace(0, 1, nx + 1)
        y = np.linspace(0, 1, ny + 1)
        if mesh_type == 'tri':
            m = MeshTri.init_tensor(x, y)
        else:
            m = MeshQuad.init_tensor(x, y)
    
    # 2. Element and Basis
    if isinstance(m, MeshTri):
        e = ElementTriP1() if porder == 1 else ElementTriP2()
    elif isinstance(m, MeshQuad):
        e = ElementQuad1() if porder == 1 else ElementQuad2()
    else:
        raise ValueError("Unknown polynomial order: {porder}")

    basis = Basis(m, e)
    
# 3. Física y Assembly
    phys = PoissonPhysics(source_type=source_type, scale=scale)
    K = asm(laplace, basis)
    
    @LinearForm
    def F_form(v, w):
        return phys.source_term(w.x[0], w.x[1], library='numpy') * v
    
    F = asm(F_form, basis)

    # 4. Boundary Conditions (Dirichlet u=0)
    dofs = basis.get_dofs()
    D = dofs.all()
    u = solve(*condense(K, F, D=D))

    # 5. Output Unificado
    return {
        "mesh": m,
        "basis": basis,
        "u": u,
        "u_exact": phys.exact_solution(basis.doflocs[0], basis.doflocs[1]),
        "doflocs": basis.doflocs.T,
        "boundary_indices": D,
        "interior_indices": basis.complement_dofs(D)
    }

if __name__ == "__main__":
    # Cambiado nelem por nx/ny para que coincida con tu firma de función
    prob = get_problem(nx=10, ny=10, porder=2, mesh_type='tri')
    print(f"Nodes (DOFs): {prob['doflocs'].shape[0]}")
    print(f"Solution u mean: {prob['u'].mean():.6f}")