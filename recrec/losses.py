import torch
import torch.nn.functional as F


def deep_supervision_loss(
    logits_per_step: list[torch.Tensor], target_index: torch.Tensor
) -> torch.Tensor:
    # Eq. (4): L_total = (1 / T) * sum_t L_CE(y_(t+1), target)
    return torch.stack(
        [F.cross_entropy(logits, target_index) for logits in logits_per_step]
    ).mean()
