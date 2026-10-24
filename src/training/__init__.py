"""Training routines, loss functions, and optimization loops."""
from .losses import MultiTaskConsistencyLoss
from .trainer import MultimodalTrainer, BaselineTrainer

__all__ = ["MultiTaskConsistencyLoss", "MultimodalTrainer", "BaselineTrainer"]
