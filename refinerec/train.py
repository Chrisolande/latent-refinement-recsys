import os
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from .callbacks import EMACallback
from .config import RefineRecConfig
from .data import (
    RefineRecDataModule,
    generate_causal_interaction_pairs,
    load_user_sequences,
    validate_item_id_continuity,
)
from .lightning_module import RefineRecLightning

SEED = 42


def resolve_data_paths() -> tuple[Path, Path]:
    """Resolves interaction and embedding paths, checking local directory first then Kaggle."""
    local_interaction = Path("data/Luxury_Beauty_5.txt")
    local_embeddings = Path("data/sbert_item_embeddings.pt")

    if local_interaction.exists() and local_embeddings.exists():
        return local_interaction, local_embeddings

    kaggle_interaction = Path("/kaggle/input/datasets/chrisolande2/recsys/data/Luxury_Beauty_5.txt")
    kaggle_embeddings = Path(
        "/kaggle/input/datasets/chrisolande2/recsys/data/sbert_item_embeddings.pt"
    )

    if kaggle_interaction.exists() and kaggle_embeddings.exists():
        return kaggle_interaction, kaggle_embeddings

    return local_interaction, local_embeddings


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_wandb_logger(
    project_name: str = "refinerec", run_name: str = "refinerec-run"
) -> WandbLogger | None:
    """Set up Weights & Biases logger. Fetches API key from Kaggle Secrets if available."""
    try:
        from kaggle_secrets import UserSecretsClient

        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret("WANDB_API_KEY")
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
    except Exception:
        pass

    try:
        return WandbLogger(project=project_name, name=run_name, log_model=True)
    except Exception as e:
        print(f"WandB not initialized: {e}. Continuing without it.")
        return None


def get_device_and_strategy() -> tuple[int | str, DDPStrategy | str]:
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            return num_gpus, DDPStrategy(find_unused_parameters=False)
        return 1, "auto"
    return "auto", "auto"


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

    ema_callback = EMACallback(decay=config.ema_decay)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_ndcg10",
        mode="max",
        save_top_k=1,
        filename="best-refinerec-{epoch:02d}-{val_ndcg10:.4f}",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_ndcg10",
        mode="max",
        patience=10,
    )

    wandb_logger = setup_wandb_logger(project_name="refinerec", run_name="refinerec-training")
    devices, strategy = get_device_and_strategy()

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices=devices,
        strategy=strategy,
        callbacks=[ema_callback, checkpoint_callback, early_stop_callback],
        gradient_clip_val=config.grad_clip,
        logger=[wandb_logger] if wandb_logger is not None else True,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)

    return trainer, model


def main() -> None:
    train()


if __name__ == "__main__":
    main()
