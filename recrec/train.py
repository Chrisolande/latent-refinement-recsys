import random

import numpy as np
import pytorch_lightning as pl
import torch

from .callbacks import EMACallback
from .config import RecRecConfig
from .data import (
    RecRecDataModule,
    load_user_sequences,
    make_train_val_pairs,
    validate_item_indexing,
)
from .diagnostics import inspect_one_batch, validate_paper_compliance
from .lightning_module import RecRecLightning

SEED = 42
INTERACTION_PATH = "/kaggle/input/datasets/chrisolande2/recsys/data/Luxury_Beauty_5.txt"
SBERT_EMBEDDING_PATH = "/kaggle/input/datasets/chrisolande2/recsys/data/sbert_item_embeddings.pt"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = RecRecConfig()

    user_sequences = load_user_sequences(INTERACTION_PATH)
    item_embeddings = torch.load(SBERT_EMBEDDING_PATH, map_location=device)

    num_items = item_embeddings.size(0)
    validate_item_indexing(user_sequences, num_items)

    print(f"Users: {len(user_sequences)}")
    print(f"Items: {num_items}")
    print(f"Interactions: {sum(len(s) for s in user_sequences.values())}")
    print(f"SBERT shape: {tuple(item_embeddings.shape)}")

    train_pairs, val_pairs = make_train_val_pairs(user_sequences)
    print(f"Train instances: {len(train_pairs)}")
    print(f"Val instances: {len(val_pairs)}")

    datamodule = RecRecDataModule(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        num_items=num_items,
        config=config,
    )

    model = RecRecLightning(
        pretrained_sbert_embeddings=item_embeddings,
        config=config,
    ).to(device)

    validate_paper_compliance(model, config)
    print("Paper-compliance checks passed.")

    datamodule.setup("fit")
    inspect_one_batch(model, datamodule.train_dataloader())

    ema_callback = EMACallback(decay=config.ema_decay)
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[ema_callback],
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
