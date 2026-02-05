import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class MLP(nn.Module):
    def __init__(self, in_features=2, out_features=1, hidden_layers=[64, 64, 64, 64], activation=nn.Tanh()):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features, hidden_layers[0]))
        layers.append(activation)
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(activation)
        layers.append(nn.Linear(hidden_layers[-1], out_features))
        self.net = nn.Sequential(*layers)
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

def generate_pinn_data(geometry, n_pde, n_bc):
    """
    Función auxiliar para extraer tensores de la geometría
    y pasárselos a los Datasets clásicos.
    """
    # Extraemos los puntos usando los métodos que ya definimos en geometry.py
    pde_points = geometry.sample_interior(n_pde)
    bc_points = geometry.sample_boundary(n_bc)
    
    return pde_points, bc_points
  
class PinnDataset(torch.utils.data.Dataset):
    def __init__(self, geometry, pde_pts, bc_pts):
        self.geometry = geometry
        self.pde_pts = pde_pts
        self.bc_pts = bc_pts
        # the epoch is determined by the max lenght of the two sets
        self.length = max(len(pde_pts), len(bc_pts))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # If pde points are fewer than the current index, we will sample randomly from them
        if idx < len(self.pde_pts):
            x_pde = self.pde_pts[idx]
        else:
            x_pde = self.pde_pts[torch.randint(0, len(self.pde_pts), (1,)).item()]

        # If bc points are fewer than the current index, we will sample randomly from them
        if idx < len(self.bc_pts):
            x_bc = self.bc_pts[idx]
        else:
            x_bc = self.bc_pts[torch.randint(0, len(self.bc_pts), (1,)).item()]
        
        return {
            'pde': x_pde,
            'bc': x_bc
        }

class ValDataset(torch.utils.data.Dataset):
    def __init__(self, geometry, pts, vals=None):
        self.geometry = geometry
        self.pts = pts
        self.vals = vals

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, idx):
        item = {'coords': self.pts[idx]}
        if self.vals is not None:
            item['real'] = self.vals[idx]
        return item


class PINNSystem(pl.LightningModule):
    def __init__(self, hidden_dim=64, num_layers=4, lr=1e-3, lambda_bc=100.0, physics=None, **kwargs): 
        super().__init__()
        # Ignore collocation_points and physics in hparams
        self.save_hyperparameters(ignore=['collocation_points', 'physics'])
        self.physics = physics 
        hidden_dim = self.hparams.get('hidden_dim', 64)
        num_layers = self.hparams.get('num_layers', 4)
        input_dim = self.hparams.get('input_dim', 2)

        self.model = MLP(in_features=input_dim, out_features=1, 
                         hidden_layers=[hidden_dim]*num_layers)
        
        from utils.metrics import calculate_rrmse
        self.rrmse_fn = calculate_rrmse
        
    def forward(self, x):
        return self.model(x)
    
    def get_input_dim(self):
        return self.model.net[0].in_features
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    def on_before_optimizer_step(self, optimizer):
        # Group all MLP weights in a single value to compute the norm
        norms = grad_norm(self, norm_type=2)
        
        # Logging the gradient norms for monitoring vanishing/exploding gradients
        self.log_dict(norms)
        total_norm = norms.get('grad_2.0_norm_total', 0)
        if total_norm > 1000:
            print(f"🚨 EXPLODING: Gradient norm is too high ({total_norm:.3e}). Consider reducing the learning rate.")
        elif total_norm < 1e-7:
            print(f"❄️ VANISHING: Gradient norm is nearly zero ({total_norm:.3e}). Your network stopped learning.")
    
    def compute_pde_residual(self, x):
        """
        Take residual of PDE from physics module.
        This method is the core of the PINN training, where we compute the PDE residual at collocation points. 
        """
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        
        u = self.forward(x)
        
        # First order derivative: Jacobian (u_x, u_y, ...)
        # just in case we need neumann BCs
        grads = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u), 
            create_graph=True, 
            retain_graph=True
        )[0]
        
        # Take the residual from the physics module, which can be general for any PDE 
        # as long as the physics class implements the compute_pde_residual method
        return self.physics.compute_pde_residual(x, u, grads)

    def training_step(self, batch, batch_idx):        
        x_pde = batch['pde']    # batch['pde']: puntos internos del dominio
        x_bc = batch['bc']      # batch['bc']: puntos en la frontera
        
        # PDE Loss
        residual = self.compute_pde_residual(x_pde)
        loss_pde = torch.mean(residual**2)
        
        # BC Loss
        u_bc_pred = self.forward(x_bc)
        u_bc_target = self.physics.boundary_condition(x_bc, library='torch')
        loss_bc = torch.mean((u_bc_pred - u_bc_target)**2)
        
        total_loss = loss_pde + self.hparams.lambda_bc * loss_bc
        
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('loss_pde', loss_pde)
        self.log('loss_bc', loss_bc)
        
        return total_loss

    @torch.enable_grad()
    def validation_step(self, batch, batch_idx):
        x_val = batch['coords'] 
        u_real = batch.get('real', None)
        # Prediction
        u_pred = self.forward(x_val)
        
        if u_real is not None:
            # CASE A: Ground truth is available (Data-driven validation)
            loss_val = torch.mean((u_pred - u_real)**2)
            # Also compute the relative L2 error, which is very useful
            error_rel = torch.norm(u_real - u_pred) / (torch.norm(u_real) + 1e-8)
            self.log('val_error_rel', error_rel, prog_bar=True)
        else:
            # CASE B: Only physics is available (Physics-based validation)
            # Compute how far the network is from satisfying the equation
            residual = self.compute_pde_residual(x_val)
            loss_val = torch.mean(residual**2)
            
        self.log('val_loss', loss_val, prog_bar=True)
        return loss_val