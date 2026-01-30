import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class SimpleMPNN(nn.Module):
    """
    Simple Message Passing Neural Network without convolution.
    Uses basic neighbor aggregation: h_i = σ(W * Σ_j h_j)
    """
    def __init__(self, in_features, hidden_features, out_features, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.in_proj = nn.Linear(in_features, hidden_features)
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_features, hidden_features))
        self.out_proj = nn.Linear(hidden_features, out_features)
        self.act = nn.Tanh() 

    def forward(self, x, edge_index):
        # x: (N, in_features)
        # edge_index: (2, E)
        h = self.act(self.in_proj(x))
        
        # Ensure edge_index is (2, E)
        if edge_index.dim() == 3:
            edge_index = edge_index[0]
        row, col = edge_index[0].long(), edge_index[1].long()
        
        for layer in self.layers:
            # Aggregation: sum neighbors' features
            # Equivalent to A * h where A is adjacency matrix
            aggr = torch.zeros_like(h)
            aggr.index_add_(0, row, h[col])
            h = self.act(layer(aggr)) 
            
        return self.out_proj(h)


class ChebConvGNN(nn.Module):
    """
    GNN using Chebyshev Convolution (like Gao's PossionNet).
    Requires torch_geometric for ChebConv.
    """
    def __init__(self, in_features, hidden_features, out_features, num_layers=8, K=10):
        super().__init__()
        try:
            from torch_geometric.nn import ChebConv
        except ImportError:
            raise ImportError("torch_geometric is required for ChebConvGNN. Install with: pip install torch-geometric")
        
        self.layers = nn.ModuleList()
        self.layers.append(ChebConv(in_features, hidden_features, K=K))
        for _ in range(num_layers - 2):
            self.layers.append(ChebConv(hidden_features, hidden_features, K=K))
        self.layers.append(ChebConv(hidden_features, out_features, K=K))
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        # Ensure edge_index is (2, E)
        if edge_index.dim() == 3:
            edge_index = edge_index[0]
        
        for i, layer in enumerate(self.layers[:-1]):
            x = self.act(layer(x, edge_index))
        
        # Last layer without activation
        x = self.layers[-1](x, edge_index)
        return x


class GNNSystem(pl.LightningModule):
    def __init__(self, hidden_dim=32, num_layers=4, lr=1e-3, lambda_bc=10.0, supervised=False, use_chebconv=False):
        super().__init__()
        self.save_hyperparameters()
        
        # Select model based on use_chebconv flag
        if use_chebconv:
            self.model = ChebConvGNN(
                in_features=2, 
                hidden_features=hidden_dim, 
                out_features=1, 
                num_layers=num_layers,
                K=10  # Chebyshev polynomial order
            )
        else:
            self.model = SimpleMPNN(
                in_features=2, 
                hidden_features=hidden_dim, 
                out_features=1, 
                num_layers=num_layers
            )
        
        from utils.metrics import calculate_rrmse
        self.rrmse_fn = calculate_rrmse
        
    def forward(self, x, edge_index):
        return self.model(x, edge_index)

    def training_step(self, batch, batch_idx):
        # batch is dict from PINNGraphDataset
        x = batch['x'][0]
        edge_index = batch['edge_index'][0]
        u_true = batch['u_exact'][0]
        
        u_pred = self(x, edge_index).squeeze()
        
        if self.hparams.supervised:
            # Data-Driven (Supervised) Loss
            loss_data = torch.mean((u_pred - u_true)**2)
            self.log('train_loss', loss_data, prog_bar=True)
            self.log('loss_data', loss_data)
            return loss_data
        else:
            # Physics-Informed Loss
            K_indices = batch['K_indices'][0]
            K_values = batch['K_values'][0]
            F_vec = batch['F'][0]
            boundary_mask = batch['boundary_mask'][0]

            # Physics Loss: || K*u - F ||^2
            # Use index_add_ for robustness over sparse.mm
            if K_indices.dim() == 3: K_indices = K_indices[0]
            row, col = K_indices[0].long(), K_indices[1].long()
            if K_values.dim() == 2: K_values = K_values[0]
            
            u_vec = u_pred.unsqueeze(1) # (N, 1)
            # Ku[row] += K_values * u_pred[col]
            Ku = torch.zeros_like(u_vec)
            Ku.index_add_(0, row, K_values.unsqueeze(1) * u_vec[col])
            Ku = Ku.squeeze()
            loss_physics = torch.mean((Ku - F_vec)**2)
            
            # BC Loss
            u_bound = u_pred[boundary_mask]
            loss_bc = torch.mean(u_bound**2)
            
            loss = loss_physics + self.hparams.lambda_bc * loss_bc
            loss_ic = torch.tensor(0.0, device=self.device)

            self.log('train_loss', loss, prog_bar=True)
            self.log('loss_pde', loss_physics)
            self.log('loss_bc', loss_bc)
            self.log('loss_ic', loss_ic)
            return loss

    def validation_step(self, batch, batch_idx):
        x = batch['x'][0]
        edge_index = batch['edge_index'][0]
        u_true = batch['u_exact'][0]
        u_pred = self(x, edge_index).squeeze()
        
        rrmse = self.rrmse_fn(u_true.cpu().numpy(), u_pred.cpu().numpy())
        self.log('val_rrmse', rrmse, prog_bar=True)
        self.log('val_loss', rrmse)
        return rrmse

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
