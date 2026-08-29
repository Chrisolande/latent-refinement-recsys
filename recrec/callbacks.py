import torch
from pytorch_lightning.callbacks import Callback


class EMACallback(Callback):
    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}

    def trainable_parameters(self, module: torch.nn.Module):
        return ((name, p) for name, p in module.named_parameters() if p.requires_grad)

    def initialize(self, module: torch.nn.Module) -> None:
        if not self.shadow:
            self.shadow = {
                name: p.detach().clone()
                for name, p in self.trainable_parameters(module)
            }

    @torch.no_grad()
    def on_train_start(self, trainer, pl_module) -> None:
        self.initialize(pl_module)

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self.initialize(pl_module)

        for name, p in self.trainable_parameters(pl_module):
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, pl_module: torch.nn.Module) -> None:
        if not self.shadow:
            return

        self.backup = {}
        for name, p in self.trainable_parameters(pl_module):
            if name in self.shadow:
                self.backup[name] = p.detach().clone()
                p.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, pl_module: torch.nn.Module) -> None:
        if not self.backup:
            return

        for name, p in self.trainable_parameters(pl_module):
            if name in self.backup:
                p.copy_(self.backup[name])
        self.backup.clear()

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self.apply_shadow(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self.restore(pl_module)

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self.apply_shadow(pl_module)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        self.restore(pl_module)
