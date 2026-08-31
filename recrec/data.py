import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .config import RecRecConfig


def load_user_sequences(interaction_path: str | Path) -> dict[int, list[int]]:
    sequences: dict[int, list[int]] = defaultdict(list)

    with open(interaction_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user_id, item_id = map(int, parts[:2])
            sequences[user_id].append(item_id)

    sequences = {uid: seq for uid, seq in sequences.items() if len(seq) >= 2}
    if not sequences:
        raise ValueError("No valid user sequences found.")
    return sequences


def validate_item_indexing(user_sequences: dict[int, list[int]], num_items: int) -> None:
    ids = [item_id for seq in user_sequences.values() for item_id in seq]
    if not ids:
        raise ValueError("No item IDs found.")
    if min(ids) < 0:
        raise ValueError(f"Found negative item ID: {min(ids)}")
    if max(ids) >= num_items:
        raise ValueError(
            f"Maximum item ID {max(ids)} exceeds embedding table size {num_items}."
        )


def make_train_val_pairs(
    user_sequences: dict[int, list[int]],
) -> tuple[list[tuple[list[int], int]], list[tuple[list[int], int]]]:
    train_pairs = []
    val_pairs = []

    for seq in user_sequences.values():
        if len(seq) < 3:
            continue

        # Leave-one-out validation target
        val_pairs.append((seq[:-1], seq[-1]))

        # Causal prefixes for training
        for i in range(1, len(seq) - 1):
            train_pairs.append((seq[:i], seq[i]))

    return train_pairs, val_pairs


def sample_candidate_set(
    target_item: int,
    history: Sequence[int],
    num_items: int,
    candidate_size: int,
    exclude_history: bool = True,
) -> tuple[list[int], int]:
    forbidden = {target_item}
    if exclude_history:
        forbidden.update(history)

    available = [i for i in range(num_items) if i not in forbidden]
    if len(available) < candidate_size - 1:
        raise ValueError("Not enough negatives to construct candidate set.")

    negatives = random.sample(available, candidate_size - 1)
    candidates = negatives + [target_item]
    random.shuffle(candidates)
    return candidates, candidates.index(target_item)


class RecRecDataset(Dataset):
    def __init__(self, pairs: Sequence[tuple[Sequence[int], int]]):
        self.pairs = [(list(h), int(t)) for h, t in pairs]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.pairs[index]


class RecRecCollator:
    def __init__(self, config: RecRecConfig, num_items: int):
        self.max_history_length = config.max_history_length
        self.candidate_size = config.candidate_size
        self.num_items = num_items
        self.exclude_history = config.exclude_history_items_from_negatives

    def __call__(
        self, batch: list[tuple[list[int], int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(batch)

        history_ids = torch.zeros(batch_size, self.max_history_length, dtype=torch.long)
        history_mask = torch.zeros(batch_size, self.max_history_length, dtype=torch.float32)
        candidate_ids = torch.zeros(batch_size, self.candidate_size, dtype=torch.long)
        target_index = torch.zeros(batch_size, dtype=torch.long)

        for row, (history, target) in enumerate(batch):
            history = history[-self.max_history_length:]
            length = len(history)

            history_ids[row, -length:] = torch.tensor(history, dtype=torch.long)
            history_mask[row, -length:] = 1.0

            candidates, target_position = sample_candidate_set(
                target_item=target,
                history=history,
                num_items=self.num_items,
                candidate_size=self.candidate_size,
                exclude_history=self.exclude_history,
            )

            candidate_ids[row] = torch.tensor(candidates, dtype=torch.long)
            target_index[row] = target_position

        return history_ids, history_mask, candidate_ids, target_index


class RecRecDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_pairs: Sequence[tuple[Sequence[int], int]],
        val_pairs: Sequence[tuple[Sequence[int], int]],
        num_items: int,
        config: RecRecConfig,
    ):
        super().__init__()
        self.train_pairs = list(train_pairs)
        self.val_pairs = list(val_pairs)
        self.num_items = num_items
        self.config = config
        self.collator = RecRecCollator(config, num_items)

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = RecRecDataset(self.train_pairs)
        self.val_dataset = RecRecDataset(self.val_pairs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self.collator,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self.collator,
            pin_memory=torch.cuda.is_available(),
        )
