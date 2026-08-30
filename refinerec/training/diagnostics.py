import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import RefineRecConfig
from ..models.lightning_module import RefineRecLightning
from ..models.modules import CoreRecursionMLP


def verify_architecture_invariants(model: RefineRecLightning, config: RefineRecConfig) -> None:
    assert config.embedding_dim > 0, "Embedding dim must be positive."
    assert config.max_history_length > 0, "History length must be positive."
    assert config.outer_steps >= 1, "Outer steps must be >= 1."
    assert config.inner_steps >= 1, "Inner steps must be >= 1."
    assert config.candidate_size >= 2, "Candidate size must be >= 2."

    item_table = model.model.item_embeddings
    if not config.freeze_item_embeddings:
        assert item_table.weight.requires_grad is True, "Item embeddings must be trainable."

    refinement = model.model.preference_refinement
    assert len(refinement.correction_gates) == config.outer_steps
    assert isinstance(refinement.f_phi, CoreRecursionMLP)


@torch.no_grad()
def smoke_test_batch_diagnostics(model: RefineRecLightning, loader: DataLoader) -> None:
    device = next(model.parameters()).device

    batch = next(iter(loader))
    batch = [x.to(device) for x in batch]
    history_ids, history_mask, candidate_ids, target_index = batch

    logits_per_step = model(history_ids, history_mask, candidate_ids)
    losses = [F.cross_entropy(logits, target_index).item() for logits in logits_per_step]

    final_logits = logits_per_step[-1]
    ranks = (
        torch.argsort(final_logits, dim=1, descending=True) == target_index.unsqueeze(1)
    ).nonzero(as_tuple=True)[1] + 1

    print("Batch diagnostics:")
    print(f"Step losses: {[round(x, 4) for x in losses]}")
    print(f"Mean loss: {sum(losses) / len(losses):.4f}")
    print(f"Mean target rank: {ranks.float().mean().item():.2f}")
    print(f"HR@1: {(ranks <= 1).float().mean().item():.4f}")
    print(f"HR@10: {(ranks <= 10).float().mean().item():.4f}")
    cand_k = candidate_ids.size(1)
    print(f"Uniform {cand_k}-way CE: {math.log(cand_k):.4f}")


