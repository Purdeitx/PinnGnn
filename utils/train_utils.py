import torch
from pytorch_lightning.callbacks import Callback

class GradientMonitor(Callback):
    """
    Monitor gradient norms to detect vanishing or exploding gradients.
    Useful for PINNs (Tanh saturation) and GNNs (Deep Message Passing).
    """
    def __init__(self, verbose=True):
        super().__init__()
        self.verbose = verbose

    def on_after_backward(self, trainer, pl_module):
        total_grad_norm = 0.0
        max_grad_norm = 0.0
        
        for name, p in pl_module.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_grad_norm += param_norm ** 2
                max_grad_norm = max(max_grad_norm, param_norm)
        
        total_grad_norm = total_grad_norm ** 0.5
        
        # Log to the logger (TensorBoard/WandB/CSV)
        pl_module.log("grad/total_norm", total_grad_norm, prog_bar=True)
        pl_module.log("grad/max_norm", max_grad_norm)
        
        # Alert for vanishing gradients
        if total_grad_norm < 1e-8 and self.verbose:
            print(f"\n⚠️ CRITICAL: Vanishing gradients! Total Norm: {total_grad_norm:.2e}")
            
        # Alert for exploding gradients
        if total_grad_norm > 1e4 and self.verbose:
            print(f"\n⚠️ CRITICAL: Exploding gradients! Total Norm: {total_grad_norm:.2e}")