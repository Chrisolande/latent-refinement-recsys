"""Run one HPO trial: best-metric tracking, config typing, and the W&B agent train function."""

import dataclasses
import math
from typing import Any

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from ..config import RefineRecConfig
from ..data import (
    VALIDATION_CANDIDATE_SEED,
    RefineRecDataModule,
)
from ..models.lightning_module import RefineRecLightning
from ..training.callbacks import EMACallback
from ..training.trainer import set_seed

ENTITY = "olandechris-"
DEFAULT_PROJECT = "refinerec"


class BestValNDCGCallback(pl.Callback):
    """Track the best validation NDCG and epoch without saving a checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.best_score = float("-inf")
        self.best_epoch = -1

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        score = float(trainer.callback_metrics["val_ndcg10"])
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = int(trainer.current_epoch)


def finalized_trial_metrics(
    callback_best_score: float,
    callback_best_epoch: int,
    final_score: float,
    trainer_current_epoch: int,
) -> dict[str, float | int]:
    """Reconcile the callback with Lightning's final validation result."""
    completed_epochs = max(int(trainer_current_epoch), 1)
    final_epoch = completed_epochs - 1
    if not math.isfinite(callback_best_score) or final_score > callback_best_score:
        best_score = final_score
        best_epoch = final_epoch
    else:
        best_score = callback_best_score
        best_epoch = int(callback_best_epoch)
    return {
        "best_val_ndcg10": float(best_score),
        "best_epoch": best_epoch,
        "final_val_ndcg10": float(final_score),
        "epochs_completed": completed_epochs,
    }


def _config_from_wandb(run: Any) -> dict[str, Any]:
    """Convert sweep values to the exact types expected by RefineRecConfig."""
    return {
        "learning_rate": float(run.config["learning_rate"]),
        "weight_decay": float(run.config["weight_decay"]),
        "grad_clip": float(run.config["grad_clip"]),
        "inner_steps": int(run.config["inner_steps"]),
        "outer_steps": int(run.config["outer_steps"]),
        "core_depth": int(run.config["core_depth"]),
        "dropout": float(run.config["dropout"]),
        "temperature": float(run.config["temperature"]),
        "preference_scale": float(run.config["preference_scale"]),
        "ema_decay": float(run.config["ema_decay"]),
        "batch_size": int(run.config["batch_size"]),
    }


def create_sweep_trial(
    train_pairs: list,
    val_pairs: list,
    item_embeddings: torch.Tensor,
    base_config: RefineRecConfig,
    project_name: str = DEFAULT_PROJECT,
):
    """Create the training function called once for every W&B agent run."""
    num_items = item_embeddings.size(0)

    def train_trial() -> None:
        with wandb.init(
            entity=ENTITY,
            project=project_name,
            job_type="hpo-trial",
            tags=["wandb-sweep", "bayes", "hyperband"],
            settings=wandb.Settings(save_code=True),
        ) as run:
            # Vary training randomness by trial seed while holding validation
            # candidates fixed across every run.
            seed = int(run.config.get("seed", 42))
            set_seed(seed)
            pl.seed_everything(seed, workers=True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if bool(run.config.get("require_cuda", True)) and not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for this research trial")
            run.config.update(
                {
                    "seed": seed,
                    "evaluation_protocol": "fixed-validation-candidates-v1",
                    "validation_candidate_seed": VALIDATION_CANDIDATE_SEED,
                },
                allow_val_change=True,
            )

            hparams = _config_from_wandb(run)
            search_epochs = int(run.config["search_epochs"])

            trial_config = dataclasses.replace(
                base_config,
                **hparams,
                max_epochs=search_epochs,
                num_workers=int(run.config.get("num_workers", 0)),
            )

            datamodule = RefineRecDataModule(
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                num_items=num_items,
                config=trial_config,
            )

            model = RefineRecLightning(
                pretrained_sbert_embeddings=item_embeddings,
                config=trial_config,
            )

            # Connect this existing sweep run to PyTorch Lightning.
            # Every metric logged with self.log/self.log_dict is forwarded.
            wandb_logger = WandbLogger(
                experiment=run,
                log_model=False,
            )

            best_metric_callback = BestValNDCGCallback()
            early_stopping = EarlyStopping(
                monitor="val_ndcg10",
                mode="max",
                patience=int(run.config.get("early_stopping_patience", 5)),
                min_delta=float(run.config.get("early_stopping_min_delta", 0.0001)),
            )
            callbacks = [
                EMACallback(decay=trial_config.ema_decay),
                best_metric_callback,
                early_stopping,
            ]

            trainer = pl.Trainer(
                max_epochs=search_epochs,
                accelerator=(
                    "gpu"
                    if torch.cuda.is_available()
                    else "cpu"
                ),
                devices=1,
                callbacks=callbacks,
                gradient_clip_val=trial_config.grad_clip,
                enable_progress_bar=False,
                logger=wandb_logger,
                enable_checkpointing=False,
                log_every_n_steps=10,
                deterministic=True,
            )

            try:
                trainer.fit(
                    model,
                    datamodule=datamodule,
                )

                val_ndcg = trainer.callback_metrics.get("val_ndcg10")

                if val_ndcg is None:
                    raise RuntimeError(
                        "RefineRecLightning did not produce val_ndcg10. "
                        "Log it with self.log("
                        "'val_ndcg10', ..., "
                        "on_epoch=True, logger=True)."
                    )

                score = float(val_ndcg)

                metrics = finalized_trial_metrics(
                    callback_best_score=best_metric_callback.best_score,
                    callback_best_epoch=best_metric_callback.best_epoch,
                    final_score=score,
                    trainer_current_epoch=int(trainer.current_epoch),
                )
                run.summary["trial_state"] = "COMPLETE"
                run.summary["val_ndcg10"] = score
                for key, value in metrics.items():
                    run.summary[key] = value
                run.summary["evaluation_protocol"] = "fixed-validation-candidates-v1"

            except Exception as exc:
                run.summary["trial_state"] = "FAILED"
                run.summary["error_type"] = type(exc).__name__
                run.summary["error_message"] = str(exc)[:1000]
                raise

    return train_trial
