from matplotlib.pylab import rint
import torch
import torch.nn as nn
import pytorch_lightning as pl
from utils.pignn_utils import decompose_graph, copy_geometric_data, update_graph_geometry
from torch_geometric.data import Data
from utils.models import MLP

from utils.mathOps import GraphMathOps, PhysValidation

from torch_geometric.utils import scatter
def scatter_add(src, index, dim=0, dim_size=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='sum')


class PhysicsEdgeProcessor(nn.Module):
    """Computes edge updates based on connected nodes and current edge features."""
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, 
                 activation='silu', dropout=0.0, layer_norm=False, 
                 symmetric=True,
                 **kwargs):
        super().__init__()
        self.symmetric = symmetric
        self.output_dim = output_dim

        # MLP configuration: hidden_dim repeated num_layers times
        self.model = MLP(in_features=input_dim, out_features=output_dim, 
                         hidden_layers=[hidden_dim]*num_layers, 
                         activation=activation, dropout=dropout, layer_norm=layer_norm)
        
    def _collected_features(self, g, x, edge_attr, s_idx, r_idx):

        # dynamic definition of information for nodes
        edges_to_collect = []
        edges_to_collect.append(x[s_idx])              # [E, 5]: crds, typeBC, props, u_current
        edges_to_collect.append(x[r_idx])              # [E, 5]: crds, typeBC, props, u_current
        edges_to_collect.append(edge_attr)             # [E, 6]: dist, |x|, BC, props, f_current 

        collected_edges = torch.cat(edges_to_collect, dim=1)
        return collected_edges
    
    @staticmethod
    def _build_reverse_index(edge_index):
        s = edge_index[0].cpu().numpy()
        r = edge_index[1].cpu().numpy()
        lookup = {(s[k], r[k]): k for k in range(len(s))}
        rev = torch.tensor(
            [lookup[(r[k], s[k])] for k in range(len(s))],
            dtype=torch.long,
            device=edge_index.device
        )
        return rev
    
    def forward(self, g, x, edge_attr):

        s_idx, r_idx = g.edge_index
        feat = self._collected_features(g, x, edge_attr, s_idx, r_idx)
        raw_flux = self.model(feat)  
        
        if self.symmetric:
            if not hasattr(g, '_rev_idx'):
                g._rev_idx = PhysicsEdgeProcessor._build_reverse_index(g.edge_index)
            rev = g._rev_idx
            delta_flux = (raw_flux - raw_flux[rev]) / 2.0       # [E, 1]
        else:
            delta_flux = raw_flux                               # [E, 1]

        new_flux = edge_attr[:, -self.output_dim:] + delta_flux
        edge_attr_ = torch.cat([edge_attr[:, :-self.output_dim], new_flux], dim=-1)

        return edge_attr_

class PhysicsNodeProcessor(nn.Module):
    """Computes node updates based on node features and aggregated edge messages."""
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers,
                 activation='silu', dropout=0.0, layer_norm=False, 
                 source=False, 
                 **kwargs):
        super().__init__()
        self.source = source
        self.output_dim = output_dim

        # MLP configuration: hidden_dim repeated num_layers times
        self.model = MLP(in_features=input_dim, out_features=output_dim, 
                         hidden_layers=[hidden_dim]*num_layers, 
                         activation=activation, dropout=dropout, layer_norm=layer_norm)

    def forward(self, g, x, edge_attr):

        _, r_idx = g.edge_index
        agg_edge = scatter_add(edge_attr, r_idx, dim=0, dim_size=g.num_nodes)

        nodes_to_collect = [x]                      # [N, 5]: crds, typeBC, props, u_current
        nodes_to_collect.append(agg_edge)           # [N, 6]: dist, |x|, BC, props, f_current 

        if self.source: 
            nodes_to_collect.append(g.f_pointwise)      # [N, 1]: source

        collected_nodes = torch.cat(nodes_to_collect, dim=1)

        delta_u = self.model(collected_nodes)
        new_u = x[:, -self.output_dim:] + delta_u
        x_ = torch.cat([x[:, :-self.output_dim], new_u], dim=-1)

        return x_

class PhysicsMessagePassing(nn.Module):
    """
    Message Passing Layer following the project skeleton signature.
    Orchestrates Edge and Node processors with residual connections.
    
    Independent configuration for edge and node MLPs can be provided via kwargs:
    - proc_e_units, proc_e_layers, proc_e_fn
    - proc_n_units, proc_n_layers, proc_n_fn
    """
    def __init__(self, node_dim, edge_dim, pos_dim,
                 node_out=1, edge_out=1, 
                 activation='silu', msg_passes=5,
                 hidden=64, num_layers=2, 
                 dropout=0.0, layer_norm=False, **kwargs):
        super().__init__()
        
        # Extract independent configurations or fall back to general defaults
        e_h_dim = kwargs.get('proc_e_units', hidden)
        e_n_lay = kwargs.get('proc_e_layers', num_layers)
        e_custom_fnc = kwargs.get('proc_e_fn', activation)
        n_h_dim = kwargs.get('proc_n_units', hidden)
        n_n_lay = kwargs.get('proc_n_layers', num_layers)
        n_custom_fnc = kwargs.get('proc_n_fn', activation)
        source = kwargs.get('source', False)
        symmetric = kwargs.get('symmetric', False)
        self.num_pases = msg_passes

        # dynamic input dimmensions 
        edge_input_dim = (2 * node_dim) + edge_dim
        node_input_dim = node_dim + edge_dim

        if source:
            node_input_dim += 1

        # Initialize processors using the skeleton style
        self.edge_proc = PhysicsEdgeProcessor(hidden_dim=e_h_dim, num_layers=e_n_lay, 
                                       input_dim=edge_input_dim, output_dim=edge_out,
                                       activation=e_custom_fnc, dropout=dropout, layer_norm=layer_norm,
                                       symmetric=symmetric)
        
        self.node_proc = PhysicsNodeProcessor(hidden_dim=n_h_dim, num_layers=n_n_lay, 
                                       input_dim=node_input_dim, output_dim=node_out,
                                       activation=n_custom_fnc, dropout=dropout, layer_norm=layer_norm,
                                       source=source) 

    def _single_pass(self, g, x, edge_attr):
        edge_attr = self.edge_proc(g, x, edge_attr)   # actualiza edge_attr con el nuevo flujo
        x = self.node_proc(g, x, edge_attr)                          # actualiza node_attr con el nuevo u

        return x, edge_attr

    def forward(self, g):
        x = g.x
        edge_attr = g.edge_attr

        for _ in range(self.num_pases):
            x, edge_attr = self._single_pass(g, x, edge_attr)

        g_out = copy_geometric_data(g)
        g_out.x = x
        g_out.edge_attr = edge_attr

        return g_out
    
class PhysicsSystem(pl.LightningModule):
    def __init__(self, model, lr=1e-3, physics=None, lambda_bc=1.0, lambda_pde=1.0, 
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.physics = physics
        self.model = model
        self.lr = lr
        self.lambda_bc = lambda_bc
        self.lambda_pde = lambda_pde
        self.pde_factor = 1e-4
        self.loss_fn = nn.MSELoss()
        # Opciones de residuo
        self.use_flux_residual = kwargs.get('use_flux_residual', False)
        self.strong_form = kwargs.get('strong_form', False) 
        self.autograd = kwargs.get('autograd', False)      # Activa derivadas automáticas
        self.lambda_Fick = kwargs.get('lambda_Fick', 1.0)
        self.strong_form = kwargs.get('strong_form', True)
        self.autograd = kwargs.get('autograd', False)
        # Opciones para rampa de activacion loss_pde progresiva
        self.min_pde_factor = kwargs.get('min_pde_factor', 1e-4)
        self.min_ramp = kwargs.get('min_ramp', 0.10)
        self.max_ramp = kwargs.get('max_ramp', 0.70)
        # scheluder config
        self.weight_decay = kwargs.get('weight_decay', 1e-6)
        self.scheduler_factor = kwargs.get('scheduler_factor', 0.8)
        self.scheduler_patience = kwargs.get('scheduler_patience', 50)
        # comparativa con operadores analíticos
        self.verify = kwargs.get('verify', False)
        self.has_verified = False

    def forward(self, batch, steps=None):
        g = batch
        g = update_graph_geometry(g)
        g = self.model(g) 

        # predition is the last feature
        u_pred = g.x[:, -1:]
        flux_pred = g.edge_attr[:, -1:]

        return u_pred, flux_pred

    def training_step(self, batch, batch_idx):
        try:
            batch = batch.to(self.device)
        except Exception:
            pass
        if self.autograd:
            batch.pos.requires_grad_(True)
        
        # update graph with message passing
        u_pred, flux_pred = self(batch) 

        if batch_idx == 0 and self.verify and not self.has_verified:
            grad_u = torch.autograd.grad(u_pred.sum(), batch.pos, retain_graph=True, allow_unused=True)[0]
            print(f"du/dpos max:  {grad_u.abs().max():.2e}")
            print(f"du/dpos mean: {grad_u.abs().mean():.2e}")
            lap = self.physics.ops.laplacian_autograd(u_pred, batch.pos)
            print(f"laplacian max:  {lap.abs().max():.2e}")
            print(f"laplacian mean: {lap.abs().mean():.2e}")

        if self.use_flux_residual:
            l_pde, l_Fick = self.physics.compute_graph_residual_with_flux(
                u_pred, flux_pred, batch, strong_form=self.strong_form, autograd=self.autograd
            )
            loss_fick = torch.mean(l_Fick**2)
            loss_div = torch.mean(l_pde**2)
            loss_pde = torch.mean(l_pde**2) + self.lambda_Fick * loss_fick
            self.log('- mse_cons', loss_div, prog_bar=True)
            self.log('- mse_fick', loss_fick, prog_bar=True)
        else:
            l_pde = self.physics.compute_graph_residual(
                u_pred, batch, strong_form=self.strong_form, autograd=self.autograd
            )
            loss_pde = torch.mean(l_pde**2)
            self.log('- mse_lapl', loss_pde, prog_bar=True)

        # BC loss: enforce u=0 at boundary nodes
        mask = batch.is_bc.bool()
        loss_bc = self.loss_fn(u_pred[mask], batch.u_bc[mask])

        loss = self.pde_factor * self.lambda_pde * loss_pde + self.lambda_bc * loss_bc 
        
        self.log('train_loss', loss, prog_bar=True, batch_size=1)
        self.log('loss_pde', loss_pde, prog_bar=True, batch_size=1)  
        self.log('loss_bc', loss_bc, prog_bar=True, batch_size=1)  
        if self.use_flux_residual:
            self.log('loss_fick', loss_fick, prog_bar=True, batch_size=1) 
  
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0 and self.verify and not self.has_verified: 
            with torch.enable_grad():
                print_bc=False
                verifyOps = PhysValidation(self.physics)
                verifyOps.verifica_laplacian(batch, print_bc=print_bc)
                verifyOps.verifica_gradiente(batch, print_bc=print_bc)
                verifyOps.verifica_divergencia(batch, print_bc=print_bc)
                verifyOps.verifica_operadores_gfd_grafo(batch, print_bc=print_bc)
                verifyOps.verifica_operadores_mixtos_grafo(batch, print_bc=print_bc)
                self.verify_multipass_gradient(batch)
            
            self.has_verified = True

        try:
            batch = batch.to(self.device)
        except Exception:
            pass
        u_pred, flux_pred = self(batch)
        loss_val = self.loss_fn(u_pred, batch.y)
        error_rel = torch.norm(batch.y - u_pred) / (torch.norm(batch.y) + 1e-8)
        self.log('val_loss', loss_val, prog_bar=True)
        self.log('val_rel', error_rel, prog_bar=True)

        return loss_val

    def on_train_epoch_start(self):
        # Ejemplo: De 0 a 125 épocas, casi nada. Luego sube linealmente.
        start_ramp = self.trainer.max_epochs * self.min_ramp
        end_ramp = self.trainer.max_epochs * self.max_ramp
        min_factor = self.min_pde_factor

        if self.current_epoch < start_ramp:
            self.pde_factor = min_factor
        elif self.current_epoch > end_ramp:
            self.pde_factor = 1.0
        else:
            # Interpolación entre min_factor y 1.0
            progress = (self.current_epoch - start_ramp) / (end_ramp - start_ramp)
            self.pde_factor = max(min_factor, 0.5 * (1.0 - torch.cos(torch.tensor(torch.pi * progress))))
            # self.pde_factor = max(1e-5, self.pde_factor - 1e-5 * progress)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.lr, 
                                     weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                               factor=self.scheduler_factor, 
                                                               patience=self.scheduler_patience)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss", "interval": "epoch"}
        }

    def verify_multipass_gradient(self, batch):
        """
        Verifica que el gradiente cambia con el número de pasos,
        lo que confirma que autograd atraviesa todos los message passing.
        """
        results = {}
        
        for n in [1, self.model.num_pases]:
            # Copia limpia para cada prueba
            g = copy_geometric_data(batch)
            g.pos = batch.pos.clone().requires_grad_(True)
            
            # Guardar num_pases original
            original = self.model.num_pases
            self.model.num_pases = n
            
            u_pred, _ = self(g)
            grad = torch.autograd.grad(
                u_pred.sum(), g.pos, retain_graph=False
            )[0]
            
            results[n] = grad.abs().mean().item()
            self.model.num_pases = original  # restaurar
        
        print(f"Grad con 1 paso:  {results[1]:.6f}")
        print(f"Grad con {self.model.num_pases} pasos: {results[self.model.num_pases]:.6f}")
        
        if abs(results[1] - results[self.model.num_pases]) < 1e-8:
            print("⚠️  Gradientes idénticos: autograd solo ve 1 paso")
        else:
            print("✓  Gradientes distintos: autograd atraviesa todos los pasos")
