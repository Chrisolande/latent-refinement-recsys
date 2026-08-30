import torch


def compute_ranking_metrics(ranks: torch.Tensor) -> dict[str, float]:
    ranks = ranks.float()
    metrics = {}

    for k in (1, 5, 10):
        hit = ranks <= k
        metrics[f"HR@{k}"] = hit.float().mean().item()
        metrics[f"NDCG@{k}"] = torch.where(
            hit, 1.0 / torch.log2(ranks + 1.0), torch.zeros_like(ranks)
        ).mean().item()
        metrics[f"Prec@{k}"] = (hit.float() / float(k)).mean().item()

    return metrics
