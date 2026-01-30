import torch
import torch.nn as nn
import pytorch_lightning as pl
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

class PINNSystem(pl.LightningModule):
    def __init__(self, hidden_dim=64, num_layers=4, lr=1e-3, lambda_bc=100.0, 
                 n_collocation=2000, n_boundary=500, collocation_points=None, 
                 physics_engine=None, **kwargs): 
        super().__init__()
        # Ignore collocation_points and physics_engine in hparams
        self.save_hyperparameters(ignore=['collocation_points', 'physics_engine'])
        
        self.model = MLP(in_features=2, out_features=1, hidden_layers=[hidden_dim]*num_layers)
        self.train_physics_pts = collocation_points
        self.physics = physics_engine 
        
        from utils.metrics import calculate_rrmse
        self.rrmse_fn = calculate_rrmse
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # 1. Collocation Points (Train)
        if self.train_physics_pts is not None:
            x_col = self.train_physics_pts.clone().detach().to(self.device).requires_grad_(True)
        else:
            x_col = torch.rand(self.hparams.n_collocation, 2, device=self.device, requires_grad=True)
        
        u = self(x_col)
        
        # 2. Laplacian via AutoDiff
        grads = torch.autograd.grad(u, x_col, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x, u_y = grads[:, 0], grads[:, 1]
        grads2_x = torch.autograd.grad(u_x, x_col, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0]
        grads2_y = torch.autograd.grad(u_y, x_col, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1]
        laplacian = grads2_x + grads2_y

        # 3. PDE Loss using the central physics engine
        # use the physics engine to get f(x,y)
        f = self.physics.source_term(x_col[:, 0], x_col[:, 1], library='torch')
        loss_pde = torch.mean((laplacian + f)**2)

        # 4. Boundary Loss
        x_b = torch.rand(self.hparams.n_boundary, 2, device=self.device)
        side_mask = torch.randint(0, 4, (self.hparams.n_boundary,), device=self.device)
        x_b[side_mask==0, 0] = 0.0; x_b[side_mask==1, 0] = 1.0
        x_b[side_mask==2, 1] = 0.0; x_b[side_mask==3, 1] = 1.0
        
        u_b = self(x_b)
        loss_bc = torch.mean(u_b**2)

        loss = loss_pde + self.hparams.lambda_bc * loss_bc
        
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_val, u_val = batch
        u_pred = self(x_val).squeeze()
        rrmse = self.rrmse_fn(u_val.cpu().numpy(), u_pred.cpu().numpy())
        
        self.log('val_rrmse', rrmse, prog_bar=True)
        self.log('val_loss', rrmse) 
        return rrmse

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)