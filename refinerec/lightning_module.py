import pytorch_lightning as pl
import torch

from .config import RefineRecConfig
from .losses import deep_supervision_loss
from .metrics import compute_ranking_metrics
from .modules import RefineRec


class RefineRecLightning(pl.LightningModule):
    def __init__(self, pretrained_sbert_embeddings: torch.Tensor, config: RefineRecConfig):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained_sbert_embeddings"])
        self.config = config
        self.model = RefineRec(pretrained_sbert_embeddings, config)
        self.validation_ranks: list[torch.Tensor] = []

    def forward(
        self,
        history_ids: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_ids: torch.Tensor | None = None,
    ) -> list[torch.Tensor] | torch.Tensor:
        return self.model(history_ids, history_mask, candidate_ids)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        history_ids, history_mask, candidate_ids, target_index = batch
        logits = self(history_ids, history_mask, candidate_ids)
        loss = deep_supervision_loss(logits, target_index)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=history_ids.size(0),
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        self.validation_ranks = []

    @torch.no_grad()
    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        history_ids, history_mask, candidate_ids, target_index = batch
        logits = self(history_ids, history_mask, candidate_ids)

        final_logits = logits[-1]
        ranking = torch.argsort(final_logits, dim=1, descending=True)
        ranks = (ranking == target_index.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
        self.validation_ranks.append(ranks.to(self.device))

    def on_validation_epoch_end(self) -> None:
        if not self.validation_ranks:
            return

        ranks = torch.cat(self.validation_ranks)
        metrics = compute_ranking_metrics(ranks)
        self.validation_ranks.clear()

        for name, value in metrics.items():
            self.log(f"val_{name.lower().replace('@', '')}", value, prog_bar=(name == "NDCG@10"))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )


# Backward-compatible alias
RecRecLightning = RefineRecLightning
