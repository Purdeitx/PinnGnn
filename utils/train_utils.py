import torch
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import EarlyStopping

class WarmupEarlyStopping(EarlyStopping):
    """No activa early stopping hasta que el pde_factor haya llegado a 1.0"""
    def __init__(self, warmup_epochs, **kwargs):
        super().__init__(**kwargs)
        self.warmup_epochs = warmup_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.warmup_epochs:
            return   # ignorar completamente durante la rampa
        super().on_validation_epoch_end(trainer, pl_module)

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
        pl_module.log("gradient/total_norm_sq", total_grad_norm, prog_bar=True)
        pl_module.log("grad/max_norm", max_grad_norm)
        
        # Alert for vanishing gradients
        if total_grad_norm < 1e-8 and self.verbose:
            print(f"\n⚠️ CRITICAL: Vanishing gradients! Total Norm: {total_grad_norm:.2e}")
            
        # Alert for exploding gradients
        if total_grad_norm > 1e4 and self.verbose:
            print(f"\n⚠️ CRITICAL: Exploding gradients! Total Norm: {total_grad_norm:.2e}")


class LossPlotterCallback(Callback):
    def __init__(self, model_name="PiGNN", print_interval=1):
        super().__init__()
        self.model_name = model_name
        self.print_interval = print_interval
        # Usamos listas separadas para asegurar que no fallen si tienen longitudes distintas
        self.losses = {
            "train_loss": [], 
            "loss_pde": [], 
            "loss_bc": [], 
            "val_loss": []
        }

    def on_train_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        # Verificamos que las keys existan (pueden variar según tus logs en el training_step)
        if "train_loss" in logs:
            self.losses["train_loss"].append(logs["train_loss"].item())
        if "loss_pde" in logs: 
            self.losses["loss_pde"].append(logs["loss_pde"].item())
        if "loss_bc" in logs:
            self.losses["loss_bc"].append(logs["loss_bc"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        # Evitamos guardar el sanity check inicial
        if trainer.sanity_checking:
            return
            
        logs = trainer.callback_metrics
        if "val_loss" in logs:
            self.losses["val_loss"].append(logs["val_loss"].item())

    def on_train_end(self, trainer, pl_module):
        # Solo intentamos plotear si tenemos datos
        if len(self.losses["train_loss"]) > 0:
            self.plot_final_losses()

    def plot_final_losses(self):
        import matplotlib.pyplot as plt
        import numpy as np

        plt.figure(figsize=(10, 6))
        
        # 1. Preparar eje X basado en train_loss (que siempre debería estar)
        train_loss = self.losses.get("train_loss", [])
        if len(train_loss) == 0:
            print("No hay datos de entrenamiento para graficar.")
            return
            
        epochs = np.arange(len(train_loss))
        plt.plot(epochs, train_loss, label="Total Train Loss", color="black", lw=2)

        # 2. Graficar PDE Residual SOLO SI existe
        pde_loss = self.losses.get("loss_pde", [])
        if len(pde_loss) == len(epochs):
            plt.plot(epochs, pde_loss, label="PDE Residual", color="blue", alpha=0.7, ls="--")

        # 3. Graficar BC Loss SOLO SI existe
        bc_loss = self.losses.get("loss_bc", [])
        if len(bc_loss) == len(epochs):
            plt.plot(epochs, bc_loss, label="BC Loss", color="red", alpha=0.7, ls=":")

        # 4. Graficar Validation Loss si existe
        val_loss = self.losses.get("val_loss", [])
        if len(val_loss) > 0:
            # Ajustamos el eje X para val_loss en caso de que no se valide en cada época
            val_epochs = np.linspace(0, epochs[-1], len(val_loss))
            plt.plot(val_epochs, val_loss, label="Val Loss", color="green", marker='o', markersize=4, ls="-")

        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log Scale)')
        plt.title(f'Training History: {self.model_name}')
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.show()

