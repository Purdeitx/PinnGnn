import torch
from torch.utils.data import Dataset
import numpy as np
from FEM.fem_solver import get_problem

class PINNGraphDataset(Dataset):
    def __init__(self, nelem=4, porder=1, source_type='sine'):
        self.prob = get_problem(nelem=nelem, porder=porder, source_type=source_type)
        self.mesh = self.prob['mesh']
        self.basis = self.prob['basis']
        
        # Node features: [x, y]
        # Use basis.doflocs to capture all nodes
        self.x = torch.tensor(self.basis.doflocs.T, dtype=torch.float32)
        
        # K Connectivity
        K_coo = self.prob['K'].tocoo()
        self.edge_index = torch.tensor(np.vstack((K_coo.row, K_coo.col)), dtype=torch.long)
        
        # Physics Matrices (For Loss)
        self.K_values = torch.tensor(K_coo.data, dtype=torch.float32)
        self.K_indices = self.edge_index 
        self.F = torch.tensor(self.prob['F'], dtype=torch.float32)
        
        # Boundary Mask (For Dirichlet BC)
        # Find boundary nodes
        # basis.get_dofs(dict(...)) returns a dict {'key': DofsView}
        dofs_container = self.basis.get_dofs(dict(markers=[1, 2, 3, 4]))
        
        all_indices = []
        
        # Helper to extract from DofsView or dict of arrays
        def extract_from_view(view):
            inds = []
            # Check for flatten/all (scikit-fem >= 6.0)
            if hasattr(view, 'flatten'):
                return view.flatten().tolist()
            if hasattr(view, 'all'):
                return view.all().tolist()
                
            # Manual extraction (scikit-fem < 6.0 or custom)
            for attr in ['nodal', 'facet', 'edge', 'interior']:
                if hasattr(view, attr):
                    container = getattr(view, attr)
                    if isinstance(container, dict):
                        for c_key in container:
                            inds.extend(container[c_key])
            return inds

        if isinstance(dofs_container, dict):
            for k, v in dofs_container.items():
                if isinstance(v, (list, np.ndarray)):
                    all_indices.extend(v)
                else:
                    # Assume it's a DofsView
                    all_indices.extend(extract_from_view(v))
        else:
             # It might be a DofsView itself (if called differently)
             all_indices.extend(extract_from_view(dofs_container))
             
        boundary_dofs = np.unique(all_indices)
        self.boundary_mask = torch.tensor(boundary_dofs, dtype=torch.long)
        self.u_exact = torch.tensor(self.prob['u_exact'], dtype=torch.float32)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            'x': self.x,
            'edge_index': self.edge_index,
            'K_values': self.K_values,
            'K_indices': self.K_indices,
            'F': self.F,
            'boundary_mask': self.boundary_mask,
            'u_exact': self.u_exact
        }
