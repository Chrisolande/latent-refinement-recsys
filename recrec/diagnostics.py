import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import RecRecConfig
from .lightning_module import RecRecLightning
from .modules import CoreRecursionMLP


def validate_paper_compliance(model: RecRecLightning, config: RecRecConfig) -> None:
    assert config.embedding_dim == 384
    assert config.max_history_length == 50
    assert config.outer_steps == 7
    assert config.inner_steps == 3
    assert config.candidate_size == 100
    assert config.learning_rate == 1e-3
    assert config.batch_size == 512
    assert config.max_epochs == 50
    assert config.ema_decay == 0.999
    assert config.freeze_item_embeddings is False

    item_table = model.model.item_embeddings
    assert item_table.weight.requires_grad is True

    refinement = model.model.preference_refinement
    assert len(refinement.correction_gates) == config.outer_steps
    assert isinstance(refinement.f_phi, CoreRecursionMLP)


@torch.no_grad()
def inspect_one_batch(model: RecRecLightning, loader: DataLoader) -> None:
    device = next(model.parameters()).device

    batch = next(iter(loader))
    batch = [x.to(device) for x in batch]
    history_ids, history_mask, candidate_ids, target_index = batch

    logits_per_step = model(history_ids, history_mask, candidate_ids)
    losses = [F.cross_entropy(logits, target_index).item() for logits in logits_per_step]

    final_logits = logits_per_step[-1]
    ranks = (
        (torch.argsort(final_logits, dim=1, descending=True) == target_index.unsqueeze(1))
        .nonzero(as_tuple=True)[1] + 1
    )

    print("One-batch diagnostic:")
    print(f"Step losses: {[round(x, 4) for x in losses]}")
    print(f"Mean loss: {sum(losses) / len(losses):.4f}")
    print(f"Mean target rank: {ranks.float().mean().item():.2f}")
    print(f"HR@1: {(ranks <= 1).float().mean().item():.4f}")
    print(f"HR@10: {(ranks <= 10).float().mean().item():.4f}")
    print(f"Uniform 100-way CE: {math.log(100):.4f}")
