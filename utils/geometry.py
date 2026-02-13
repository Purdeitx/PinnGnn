import torch
import numpy as np

try:
    # Newer versions (v0.8.0+)
    from skfem.mesh import MeshTri, MeshQuad
except ImportError:
    # Old versions 
    from skfem import MeshTri, MeshQuad

try:
    # Newer versions (v0.8.0+)
    from skfem.element import (ElementTriP1, ElementTriP2, 
                                ElementQuad1, ElementQuad2)
except ImportError:
    # Old versions 
    from skfem import (ElementTriP1, ElementTriP2, 
                        ElementQuad1, ElementQuad2)

class BaseGeometry:
    def __init__(self, name):
        self.name = name

class SquareDomain(BaseGeometry):
    def __init__(self, x_range=[0.0, 1.0], y_range=[0.0, 1.0]):
        super().__init__("square")
        self.x_range = x_range
        self.y_range = y_range
    
    def is_inside(self, x, y):
        return (self.x_range[0] <= x <= self.x_range[1]) and \
               (self.y_range[0] <= y <= self.y_range[1])
    
    def get_skfem_mesh(self, nx, ny, mesh_type='tri'):
        """Genera una malla compatible con skfem usando sus rangos"""
        # x = torch.rand(nx, 1) * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        # y = torch.rand(ny, 1) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        x = np.linspace(self.x_range[0], self.x_range[1], nx + 1)
        y = np.linspace(self.y_range[0], self.y_range[1], ny + 1)
        if mesh_type == 'tri':
            return MeshTri.init_tensor(x, y)
        return MeshQuad.init_tensor(x, y)

class CircleDomain(BaseGeometry):
    def __init__(self, center=(0.0, 0.0), radius=1.0):
        super().__init__("circle")
        self.center = center
        self.radius = radius
        self.x_range = [center[0] - radius, center[0] + radius]
        self.y_range = [center[1] - radius, center[1] + radius]

    def is_inside(self, x, y):
        return np.sqrt((x - self.center[0])**2 + (y - self.center[1])**2) <= self.radius
    
    def get_skfem_mesh(self, nx, ny, mesh_type='tri'):
        """
        Genera una malla circular. 
        Nota: nx aquí define el nivel de refinamiento (densidad de triángulos).
        """
        # skfem genera un círculo unidad por defecto, luego lo escalamos y movemos
        m = MeshTri.init_circle(nrefs=nx) 
        
        # Escalado por el radio y traslación al centro
        m = m.scale(self.radius).translate(np.array(self.center))
        
        if mesh_type == 'quad':
            print("Warning: skfem no soporta mallas de cuadriláteros estructuradas en círculos fácilmente. Usando triángulos.")
            
        return m

# Constructor centralizado
def geometry_factory(geom_type="square", **kwargs):
    if geom_type == "square":
        return SquareDomain(**kwargs)
    elif geom_type == "circle":
        return CircleDomain(**kwargs)
    else:
        raise ValueError(f"Geometría {geom_type} no soportada.")
    
'''
# TODO: implementar carga de malla externa
class MeshGeometry:
    def __init__(self, mesh_file):
        # Aquí cargaría los nodos de un .vtu o .obj
        self.nodes = load_mesh(mesh_file) 
    
    def sample_interior(self, n_points):
        # Retorna puntos aleatorios de dentro de la malla
        pass
'''
