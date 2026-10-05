from .callbacks import EMACallback
from .diagnostics import smoke_test_batch_diagnostics, verify_architecture_invariants
from .trainer import main, train

__all__ = [
    "EMACallback",
    "main",
    "smoke_test_batch_diagnostics",
    "train",
    "verify_architecture_invariants",
]
