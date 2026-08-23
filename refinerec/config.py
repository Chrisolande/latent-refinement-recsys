from dataclasses import dataclass


@dataclass
class RefineRecConfig:
    embedding_dim: int = 384
    max_history_length: int = 50
    outer_steps: int = 7
    inner_steps: int = 3
    core_depth: int = 5
    preference_scale: float = 1.0  # L in Eq. (3)
    temperature: float = 1.0  # tau in Eq. (4)
    candidate_size: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    dropout: float = 0.0
    batch_size: int = 512
    max_epochs: int = 50
    ema_decay: float = 0.999
    freeze_item_embeddings: bool = False
    num_workers: int = 3
    exclude_history_items_from_negatives: bool = True


RecRecConfig = RefineRecConfig
