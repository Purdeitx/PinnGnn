import torch
import torch.nn as nn
import pytorch_lightning as pl
from utils.gnn_utils import decompose_graph, copy_geometric_data
from torch_geometric.data import Data
try:
    from torch_scatter import scatter_add
except ImportError:
    from torch_geometric.utils import scatter
    def scatter_add(src, index, dim=0, dim_size=None):
        return scatter(src, index, dim=dim, dim_size=dim_size, reduce='sum')

def get_activation(name):
    """Converts a string name into a PyTorch activation object."""
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
    Standard MLP following the project's skeleton.
    
    Args:
        in_features (int): Input dimension.
        out_features (int): Output dimension.
        hidden_layers (list): List of dimensions for each hidden layer.
        activation (str): Activation function name.
        dropout (float): Dropout probability.
        layer_norm (bool): Whether to use Layer Normalization.
    """
    def __init__(self, in_features, out_features, hidden_layers=[64, 64], 
                 activation='silu', dropout=0.0, layer_norm=False):
        super().__init__()
        act_fn = get_activation(activation)
        layers = []
        dims = [in_features] + hidden_layers
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if layer_norm:
                layers.append(nn.LayerNorm(dims[i+1]))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(dims[-1], out_features))
        self.net = nn.Sequential(*layers)
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization for weights and zero for biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

class EdgeProcessor(nn.Module):
    """Computes edge updates based on connected nodes and current edge features."""
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, 
                 activation='silu', dropout=0.0, layer_norm=True, **kwargs):
        super().__init__()
        # MLP configuration: hidden_dim repeated num_layers times
        self.model = MLP(in_features=input_dim, out_features=output_dim, 
                         hidden_layers=[hidden_dim]*num_layers, 
                         activation=activation, dropout=dropout, layer_norm=layer_norm)

    def forward(self, graph):
        node_attr, edge_index, edge_attr = decompose_graph(graph)
        senders_idx, receivers_idx = edge_index
        edges_to_collect = []

        senders_attr   = node_attr[senders_idx]
        receivers_attr = node_attr[receivers_idx]

        edges_to_collect.append(senders_attr)
        edges_to_collect.append(receivers_attr)
        edges_to_collect.append(edge_attr)

        collected_edges = torch.cat(edges_to_collect, dim=1)

        edge_attr_ = self.model(collected_edges)

        return Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr_)

class NodeProcessor(nn.Module):
    """Computes node updates based on node features and aggregated edge messages."""
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers,
                 activation='silu', dropout=0.0, layer_norm=True, **kwargs):
        super().__init__()
        self.model = MLP(in_features=input_dim, out_features=output_dim, 
                         hidden_layers=[hidden_dim]*num_layers, 
                         activation=activation, dropout=dropout, layer_norm=layer_norm)

    def forward(self, graph):
        node_attr, edge_index, edge_attr = decompose_graph(graph)
        _, receivers_idx = edge_index
        nodes_to_collect = []
        
        num_nodes          = graph.num_nodes
        agg_received_edges = scatter_add(edge_attr, receivers_idx, dim=0, dim_size=num_nodes)

        nodes_to_collect.append(node_attr)
        nodes_to_collect.append(agg_received_edges)
        collected_nodes = torch.cat(nodes_to_collect, dim=-1)

        node_attr = self.model(collected_nodes)

        return Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)

class MessagePassing(nn.Module):
    """
    Message Passing Layer following the project skeleton signature.
    Orchestrates Edge and Node processors with residual connections.
    
    Independent configuration for edge and node MLPs can be provided via kwargs:
    - proc_e_units, proc_e_layers, proc_e_fn
    - proc_n_units, proc_n_layers, proc_n_fn
    """
    def __init__(self, node_dim, edge_dim, activation='silu',
                 hidden=64, num_layers=2, 
                 dropout=0.0, layer_norm=True, **kwargs):
        super().__init__()
        
        # Extract independent configurations or fall back to general defaults
        e_h_dim = kwargs.get('proc_e_units', hidden)
        e_n_lay = kwargs.get('proc_e_layers', num_layers)
        e_custom_fnc = kwargs.get('proc_e_fn', activation)
        n_h_dim = kwargs.get('proc_n_units', hidden)
        n_n_lay = kwargs.get('proc_n_layers', num_layers)
        n_custom_fnc = kwargs.get('proc_n_fn', activation)

        # Initialize processors using the skeleton style
        self.edge_proc = EdgeProcessor(hidden_dim=e_h_dim, num_layers=e_n_lay, 
                                       input_dim=2*node_dim + edge_dim, output_dim=edge_dim,
                                       activation=e_custom_fnc, dropout=dropout, layer_norm=layer_norm)
        
        self.node_proc = NodeProcessor(hidden_dim=n_h_dim, num_layers=n_n_lay, 
                                       input_dim=node_dim + edge_dim, output_dim=node_dim,
                                       activation=n_custom_fnc, dropout=dropout, layer_norm=layer_norm)

    def forward(self, graph):
        graph_cpy = copy_geometric_data(graph)

        graph = self.edge_proc(graph)
        graph = self.node_proc(graph)
        
        # Edge update
        edge_attr = graph_cpy.edge_attr + graph.edge_attr
        # Node update
        x = graph_cpy.x + graph.x
        
        return Data(x=x, edge_index=graph.edge_index, edge_attr=edge_attr)
    
class DirectSystem(pl.LightningModule):
    def __init__(self, mp_layer, msg_passes=5, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.mp_layer = mp_layer
        self.msg_passes = msg_passes
        self.lr = lr
        self.loss_fn = nn.MSELoss()

    def forward(self, graph):
        g = graph.clone()
        for _ in range(self.msg_passes):
            g = self.mp_layer(g) 

        # predition is the last feature
        u_pred = g.x[:, 3:4]
        return u_pred 

    def training_step(self, batch, batch_idx):
        pred = self(batch) 
        loss = self.loss_fn(pred, batch.y)
        self.log('train_loss', loss, prog_bar=True, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self.loss_fn(pred, batch.y)
        error_rel = torch.norm(batch.y - pred) / (torch.norm(batch.y) + 1e-8)
        self.log('val_loss', loss, prog_bar=True, batch_size=1)
        self.log('val_error_rel', error_rel, prog_bar=True, batch_size=1)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }
