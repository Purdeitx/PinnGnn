import torch
import matplotlib.pyplot as plt
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


class LossPlotterCallback(Callback):
    def __init__(self, model_name="PINN"):
        super().__init__()
        self.model_name = model_name
        self.losses = {"train_loss": [], "loss_pde": [], "loss_bc": [], "val_loss": []}
        self.fig = None  

    def on_train_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        for key in ["train_loss", "loss_pde", "loss_bc"]:
            if key in logs:
                self.losses[key].append(logs[key].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        if "val_loss" in logs:
            self.losses["val_loss"].append(logs["val_loss"].item())

    def on_train_end(self, trainer, pl_module):
        from utils.plotting import plot_loss
        import matplotlib.pyplot as plt
        
        self.fig = plot_loss(self.losses, model_name=self.model_name)
        plt.show()