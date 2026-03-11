# utils/models.py
import torch
import torch.nn as nn

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
    A Multi-Layer Perceptron with customizable layers, activation, and normalization.
    
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        hidden_layers (list of int): Sizes of the hidden layers.
        activation (str): Name of the activation function to use.
        dropout (float): Dropout probability.
        layer_norm (bool): Whether to include LayerNorm after linear layers.
    """
    def __init__(self, in_features=2, out_features=1, hidden_layers=[64, 64, 64, 64], 
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

