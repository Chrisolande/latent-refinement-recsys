import random

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from ..config import RefineRecConfig
from ..data import (
    RefineRecDataModule,
    generate_causal_interaction_pairs,
    load_user_sequences,
    resolve_data_paths,
    validate_item_id_continuity,
)
from ..models.lightning_module import RefineRecLightning
from .callbacks import EMACallback

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_wandb_logger(
    project_name: str = "refinerec", run_name: str = "refinerec-training"
) -> WandbLogger | None:
    """Set up Weights & Biases logger."""
    try:
        # Deferred import: hpo depends on this module for set_seed().
        from ..hpo import configure_wandb_auth

        configure_wandb_auth()
    except Exception as exc:
        print(f"W&B auth unavailable ({exc}). Continuing without logging.")
        return None
    return WandbLogger(project=project_name, name=run_name, log_model=True)


def train(config: RefineRecConfig | None = None) -> tuple[pl.Trainer, RefineRecLightning]:
    set_seed()
    if config is None:
        config = RefineRecConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    interaction_path, embedding_path = resolve_data_paths()
    user_sequences = load_user_sequences(interaction_path)
    item_embeddings = torch.load(embedding_path, map_location=device)

    num_items = item_embeddings.size(0)
    validate_item_id_continuity(user_sequences, num_items)

    train_pairs, val_pairs = generate_causal_interaction_pairs(user_sequences)

    datamodule = RefineRecDataModule(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        num_items=num_items,
        config=config,
    )

    model = RefineRecLightning(
        pretrained_sbert_embeddings=item_embeddings,
        config=config,
    ).to(device)

    callbacks = [
        EMACallback(decay=config.ema_decay),
        ModelCheckpoint(
            monitor="val_ndcg10",
            mode="max",
            save_top_k=1,
            filename="best-refinerec-{epoch:02d}-{val_ndcg10:.4f}",
        ),
        EarlyStopping(monitor="val_ndcg10", mode="max", patience=10),
    ]

    wandb_logger = setup_wandb_logger(project_name="refinerec", run_name="refinerec-training")

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        callbacks=callbacks,
        gradient_clip_val=config.grad_clip,
        logger=[wandb_logger] if wandb_logger is not None else True,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)

    return trainer, model


def main() -> None:
    train()
