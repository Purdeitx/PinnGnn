import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
from torch_geometric.data import Data
from PiGnn.pignn_module import PhysicsMessagePassing
from utils.pignn_utils import decompose_graph, copy_geometric_data, update_graph_geometry
from utils.models import MLP

from utils.mathOps import GraphMathOps, PhysValidation

class PhysEncoder(nn.Module):
    """
    Processes features with optional dimensionality augmentation.
    Defaults to maintaining input dimensions (node_in -> node_in).
    """
    def __init__(self, node_inp, edge_inp, activation='silu', 
                 node_out=None, n_units=64, n_layers=2, 
                 edge_out=None, e_units=64, e_layers=2, 
                 **kwargs):
        super().__init__()
        # If no target dimension is provided, we default to the input size
        n_out = node_out if node_out is not None else node_inp
        e_out = edge_out if edge_out is not None else edge_inp
        
        # Processing hidden layers match the output target dimension
        node_h = [n_units] * (n_layers - 1) if n_layers > 0 else []
        edge_h = [e_units] * (e_layers - 1) if e_layers > 0 else []
        dropout = kwargs.get('dropout', 0.0)
        layer_norm = kwargs.get('layer_norm', False)
        
        self.node_encoder = MLP(node_inp, n_out, hidden_layers=node_h, activation=activation, 
                                dropout=dropout, layer_norm=layer_norm)
        self.edge_encoder = MLP(edge_inp, e_out, hidden_layers=edge_h, activation=activation, 
                                dropout=dropout, layer_norm=layer_norm)

    def forward(self, graph):
        """
        Inputs: 
            Graph: Data object containing:
                Node attributes (num_nodes, node_input_size), 
                edge_index (2, num_edges), 
                edge_attr (num_edges, edge_input_size) 
        Outputs: 
            node_lat (num_nodes, n_out), 
            edge_index (2, num_edges),
            edge_lat (num_edges, e_out)
        """
        node_ = self.node_encoder(graph.x)
        edge_ = self.edge_encoder(graph.edge_attr)

        g_out = copy_geometric_data(graph)
        g_out.x = node_
        g_out.edge_attr = edge_

        return g_out

class PhysDecoder(nn.Module):
    """
    Projects from the internal processing state back to the target physical output dimension.
    """
    def __init__(self, node_inp, node_out, 
                 edge_inp, edge_out, 
                 decode_edges=False,
                 activation='silu', n_units=64, n_layers=2, **kwargs):
        super().__init__()
        # Hidden layers match the internal processing dimension
        node_h = [n_units] * (n_layers - 1) if n_layers > 0 else []
        # Layer norm is often disabled in the output stage to preserve scale
        dropout = kwargs.get('dropout', 0.0)
        layer_norm = kwargs.get('layer_norm', False)

        self.node_decoder = MLP(node_inp, node_out, hidden_layers=node_h, activation=activation, 
                           dropout=dropout, layer_norm=layer_norm)
        
        # Edge decoder es opcional — solo si se pide
        self.edge_decoder = None
        if decode_edges:
            edge_h = [n_units] * (n_layers - 1) if n_layers > 0 else []
            self.edge_decoder = MLP(edge_inp, edge_out, hidden_layers=edge_h,
                                    activation=activation, dropout=dropout, layer_norm=layer_norm)

    def forward(self, graph):
        """
        Inputs: x (num_nodes, node_in)
        Outputs: out (num_nodes, node_out)
        """
        u_pred = self.node_decoder(graph.x)

        flux_pred = None
        if self.edge_decoder is not None:
            flux_pred = self.edge_decoder(graph.edge_attr)    # [E, edge_out]
        
        return u_pred, flux_pred

class PhysMeshGraphNet(nn.Module):
    """
    Advanced Encoder-Processor-Decoder assembly.
    By default, it conserves input dimensions throughout the graph.
    Optional augmentation can be triggered by providing node_dim or edge_dim.
    
    Args:
        node_in (int): Dimension of input node features.
        edge_in (int): Dimension of input edge features.
        decoder_out (int): Dimension of the final output features for nodes.         (decoder output)
        latent_dim (int): Dimmension of latent features 
        msg_passes (int): Number of message passing layers.
        activation (str): Activation function name for all MLPs.
        hidden (int): Default value for hidden units
        num_layers (int): Default value for hidden layers

        Independent configuration for edge and node MLPs can be provided via kwargs:
        - enc_n_units, enc_n_layers (encoder, nodes)
        - enc_e_units, enc_e_layers (encoder, edges)
        - dec_n_units, dec_n_layers (decoder, nodes)
        - proc_e_units, proc_e_layers, proc_e_fn    (procesor, edges)
        - proc_n_units, proc_n_layers, proc_n_fn    (procesor, nodes)
    """
    def __init__(self, node_dim, edge_dim, pos_dim,
                 latent_dim=64, node_out=1, edge_out=1, 
                 activation='silu', msg_passes=5,
                 hidden=64, num_layers=2, 
                 dropout=0.0, layer_norm=False, **kwargs):
        super().__init__()
        
        # General setup: use provided target 
        self.node_out=node_out
        self.edge_out=edge_out
        self.num_pases = msg_passes
        self.n_dim = latent_dim 
        self.e_dim = latent_dim

        # 1. Encoder: transforms raw features to work space
        self.encoder = PhysEncoder(
            node_inp=node_dim,
            edge_inp=edge_dim,
            node_out=self.n_dim,
            edge_out=self.e_dim,
            activation=kwargs.get('enc_fn', activation), 
            n_units=kwargs.get('enc_n_units', hidden),
            n_layers=kwargs.get('enc_n_layers', num_layers),
            e_units=kwargs.get('enc_e_units', hidden),
            e_layers=kwargs.get('enc_e_layers', num_layers), 
            **kwargs)
        
        # 2. Processor: core message passing logic   
        self.processor = PhysicsMessagePassing(
                node_dim=self.n_dim, edge_dim=self.e_dim, pos_dim=pos_dim,
                node_out=self.n_dim, edge_out=self.e_dim, 
                activation=activation,
                msg_pases=self.num_pases,
                hidden=hidden, 
                num_layers=num_layers, 
                dropout=dropout, 
                layer_norm=layer_norm,
                **kwargs) 
        
        # 3. Decoder: project back to output physical space
        self.decoder = PhysDecoder(
            node_inp=self.n_dim, node_out=node_out,
            edge_inp=self.e_dim, edge_out=edge_out, 
            activation=kwargs.get('enc_fn', activation), 
            n_units=kwargs.get('dec_n_units', hidden),
            n_layers=kwargs.get('dec_n_layers', num_layers),
            **kwargs)

    def forward(self, graph, return_passes=False):
        """
        Standard forward pass:
            - encode node and edge features
            - perform message passing for a specified number of iterations
                - update edge $e_{ij}$ message passing 
                - compute edge-aggregated messages for each node
                - update node features based on aggregated messages
            - decode the final node features to the target output dimension
        """
        g = self.encoder(graph)                # → latent
        g = self.processor(g)                  # → latent procesado
        u_pred, flux_pred = self.decoder(g)    # → variables físicas
        return u_pred, flux_pred

class PhysMGNSystem(pl.LightningModule):
    """
    LightningModule for training MeshGraphNet in a supervised manner.
    """
    def __init__(self, model, lr=1e-3, physics=None, lambda_bc=1.0,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.physics = physics
        self.model = model
        self.lr = lr
        self.lambda_bc = lambda_bc
        self.lambda_pde = kwargs.get('lambda_pde', 1.0)
        self.pde_factor = kwargs.get('pde_factor', 1.0e-4)
        self.loss_fn = nn.MSELoss()
        # Opciones de residuo
        self.use_flux_residual = kwargs.get('use_flux_residual', False)
        self.autograd = kwargs.get('autograd', False)      # Activa derivadas automáticas
        self.lambda_Fick = kwargs.get('lambda_Fick', 1.0)
        self.strong_form = kwargs.get('strong_form', True)
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

    def forward(self, batch):
        g = update_graph_geometry(batch)
        u_pred, flux_pred = self.model(g) 

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
    
    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)