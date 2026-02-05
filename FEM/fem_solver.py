import numpy as np
import scipy.sparse as sp
from skfem import *
from skfem.models.poisson import laplace, mass
from config.physics import PoissonPhysics
from skfem.helpers import dot, grad

from utils import geometry

def get_problem(geometry, nx=4, ny=4, porder=1, source_type='sine', mesh_type='tri', scale=1.0, custom_mesh=None):
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
    elif hasattr(geometry, 'get_skfem_mesh'): 
        m = geometry.get_skfem_mesh(nx, ny, mesh_type)
    else:
        x0, x1 = geometry.x_range
        y0, y1 = geometry.y_range
        x = np.linspace(x0, x1, nx + 1)
        y = np.linspace(y0, y1, ny + 1)
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
    

    '''
    # Ejemplo de forma lineal para el término fuente f(x):
    @LinearForm
    def F_form(v, w):
        if phys.source_type == 'sine':
            # f = scale * 2 * pi^2 * sin(pi*x) * sin(pi*y)
            f_val = phys.scale * 2 * (np.pi**2) * np.sin(np.pi * w.x[0]) * np.sin(np.pi * w.x[1])
        else:
            f_val = phys.scale 
        return f_val * v
    '''
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

    # 4. Boundary Conditions (Dirichlet u=0)
    dofs = basis.get_dofs()
    D = dofs.all()
    u = solve(*condense(K, F, D=D))

    x_dofs = basis.doflocs.T
    u_exact = phys.exact_solution(x_dofs)
    if u_exact is not None:
        u_exact = u_exact.flatten()

    # 5. Output Unificado
    return {
        "mesh": m,
        "basis": basis,
        "u": u,
        "u_exact": u_exact,
        "doflocs": x_dofs,
        "boundary_indices": D,
        "interior_indices": basis.complement_dofs(D)
    }

if __name__ == "__main__":
    # Cambiado nelem por nx/ny para que coincida con tu firma de función
    prob = get_problem(nx=10, ny=10, porder=2, mesh_type='tri')
    print(f"Nodes (DOFs): {prob['doflocs'].shape[0]}")
    print(f"Solution u mean: {prob['u'].mean():.6f}")