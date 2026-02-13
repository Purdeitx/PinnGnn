import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
from torch_geometric.data import Data
from GNN.gnn_module import MessagePassing
from utils.gnn_utils import decompose_graph

def get_activation(name):
    """
    Converts a string name into a PyTorch activation object.
    
    Args:
        name (str): The name of the activation function (e.g., 'relu', 'silu', 'tanh').
        
    Returns:
        nn.Module: The corresponding PyTorch activation function. Defaults to SiLU if name not found.
    """
    activations = {
        'tanh': nn.Tanh(),
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'elu': nn.ELU(),
        'silu': nn.SiLU(), # Highly recommended for PINNs
        'leakyrelu': nn.LeakyReLU(),
    }
    return activations.get(name.lower(), nn.SiLU())

class MLP(nn.Module):
    """
    A Multi-Layer Perceptron with customizable layers, activation, and normalization.
    
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        hidden_layers (list of int): Sizes of the hidden layers.
        activation (str): Name of the activation function to use.
        dropout (float): Dropout probability.
        layer_norm (bool): Whether to include LayerNorm after linear layers.
    """
    def __init__(self, in_features=2, out_features=1, hidden_layers=[64, 64], 
                 activation='silu', dropout=0.0, layer_norm=False):
        super().__init__()
        act_fn = get_activation(activation)
        layers = []
        dims = [in_features] + hidden_layers
        for i in range(len(dims) - 1):                              # stack of N hidden layers
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if layer_norm:
                layers.append(nn.LayerNorm(dims[i+1]))              # Layer norm
            
            layers.append(act_fn)                                   # Activation fn
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))                # dropout

        layers.append(nn.Linear(hidden_layers[-1], out_features))   # output layer
        self.net = nn.Sequential(*layers)
        self._init_weights()                                        # initialization
        
    def _init_weights(self):
        """Initializes weights using Xavier normal for linear layers and constants for LayerNorm."""
        for m in self.modules():
            if isinstance(m, nn.Linear):                # Xavier initialization for linear layers
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):           # Initialize LayerNorm weights to 1 and biases to 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (N, in_features).
            
        Returns:
            Tensor: Output tensor of shape (N, out_features).
        """
        return self.net(x)

class Encoder(nn.Module):
    """
    Processes features with optional dimensionality augmentation.
    Defaults to maintaining input dimensions (node_in -> node_in).
    """
    def __init__(self, node_in, edge_in, activation='silu', 
                 node_out=None, n_units=64, n_layers=2, 
                 edge_out=None, e_units=64, e_layers=2, 
                 **kwargs):
        super().__init__()
        # If no target dimension is provided, we default to the input size
        n_out = node_out if node_out is not None else node_in
        e_out = edge_out if edge_out is not None else edge_in
        
        # Processing hidden layers match the output target dimension
        node_h = [n_units] * (n_layers - 1) if n_layers > 0 else []
        edge_h = [e_units] * (e_layers - 1) if e_layers > 0 else []
        dropout = kwargs.get('dropout', 0.0)
        layer_norm = kwargs.get('layer_norm', True)
        
        self.node_encoder = MLP(node_in, n_out, hidden_layers=node_h, activation=activation, 
                                dropout=dropout, layer_norm=layer_norm)
        self.edge_encoder = MLP(edge_in, e_out, hidden_layers=edge_h, activation=activation, 
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
        node_attr, _, edge_attr = decompose_graph(graph)
        node_ = self.node_encoder(node_attr)
        edge_ = self.edge_encoder(edge_attr)

        return Data(x=node_, edge_index=graph.edge_index, edge_attr=edge_)

class Decoder(nn.Module):
    """
    Projects from the internal processing state back to the target physical output dimension.
    """
    def __init__(self, node_in, node_out, activation='silu', n_units=64, n_layers=2, **kwargs):
        super().__init__()
        # Hidden layers match the internal processing dimension
        node_h = [n_units] * (n_layers - 1) if n_layers > 0 else []
        # Layer norm is often disabled in the output stage to preserve scale
        dropout = kwargs.get('dropout', 0.0)
        layer_norm = kwargs.get('layer_norm', True)
        self.decoder = MLP(node_in, node_out, hidden_layers=node_h, activation=activation, 
                           dropout=dropout, layer_norm=layer_norm)

    def forward(self, graph):
        """
        Inputs: x (num_nodes, node_in)
        Outputs: out (num_nodes, node_out)
        """
        node_attr, _, _ = decompose_graph(graph)

        return self.decoder(node_attr)

class MeshGraphNet(nn.Module):
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
    def __init__(self, node_in, edge_in, decoder_out=1,
                 latent_dim=64, msg_passes=5, activation='silu',
                 hidden=64, num_layers=2, **kwargs):
        super().__init__()
        
        # Dimensions setup: use provided target 
        self.n_dim = latent_dim 
        self.e_dim = latent_dim

        # 1. Encoder: transforms raw features to work space
        self.encoder = Encoder(
            node_in=node_in,
            edge_in=edge_in,
            node_out=self.n_dim,
            edge_out=self.e_dim,
            activation=activation, 
            n_units=kwargs.get('enc_n_units', hidden),
            n_layers=kwargs.get('enc_n_layers', num_layers),
            e_units=kwargs.get('enc_e_units', hidden),
            e_layers=kwargs.get('enc_e_layers', num_layers), 
            **kwargs)
        
        # 2. Processor: core message passing logic   
        self.processor = nn.ModuleList(
            [MessagePassing(
                node_dim=self.n_dim, edge_dim=self.e_dim, activation=activation,
                e_layers=kwargs.get('proc_e_layers', num_layers),
                e_units=kwargs.get('proc_e_units', latent_dim),
                n_layers=kwargs.get('proc_n_layers', num_layers),
                n_units=kwargs.get('proc_n_units', latent_dim),
            **kwargs) for _ in range(msg_passes)]
            )
        
        # 3. Decoder: project back to output physical space
        # def __init__(self, hidden_dim=128, num_layers=2, **kwargs):
        self.decoder = Decoder(
            node_in=self.n_dim, 
            node_out=decoder_out, 
            activation=activation,
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
        # Encode
        graph = self.encoder(graph)
        # Process
        passes = [] if return_passes else None
        for i, model in enumerate(self.processor):
            graph = model(graph)
            if return_passes:
                passes.append(graph.x.clone())

        # Decode
        decoded = self.decoder(graph)

        if return_passes:
            return decoded, passes
        else:
            return decoded

class MGNSystem(pl.LightningModule):
    """
    LightningModule for training MeshGraphNet in a supervised manner.
    """
    def __init__(self, model, lr=1e-3, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.loss_fn = nn.MSELoss()

    def forward(self, graph):
        return self.model(graph)

    def training_step(self, batch, batch_idx):
        pred = self.model(batch)
        loss = self.loss_fn(pred, batch.y)
        self.log('train_loss', loss, prog_bar=True, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.model(batch)
        loss = self.loss_fn(pred, batch.y)
        self.log('val_loss', loss, prog_bar=True, batch_size=1)
        
        # Calculate relative error if possible
        if hasattr(batch, 'y'):
            error_rel = torch.norm(batch.y - pred) / (torch.norm(batch.y) + 1e-8)
            self.log('val_error_rel', error_rel, prog_bar=True, batch_size=1)
            
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)