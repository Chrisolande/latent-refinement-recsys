import dataclasses
import os
from typing import Any

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch

try:
    from optuna.integration import PyTorchLightningPruningCallback, WeightsAndBiasesCallback
except ImportError:
    from optuna_integration import PyTorchLightningPruningCallback, WeightsAndBiasesCallback

from .callbacks import EMACallback
from .config import RecRecConfig
from .data import (
    RecRecDataModule,
    load_user_sequences,
    make_train_val_pairs,
    validate_item_indexing,
)
from .lightning_module import RecRecLightning
from .train import INTERACTION_PATH, SBERT_EMBEDDING_PATH, set_seed, setup_wandb_logger


def get_wandb_api_key() -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("WANDB_API_KEY")
    except Exception:
        return os.environ.get("WANDB_API_KEY")


def sample_hyperparameters(trial: optuna.Trial) -> dict[str, Any]:
    """Defines search space for RecRec hyperparameters."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "temperature": trial.suggest_float("temperature", 0.5, 2.0, step=0.1),
        "preference_scale": trial.suggest_float("preference_scale", 0.5, 2.0, step=0.1),
        "inner_steps": trial.suggest_int("inner_steps", 1, 4),
        "outer_steps": trial.suggest_int("outer_steps", 3, 8),
        "core_depth": trial.suggest_int("core_depth", 3, 6),
        "ema_decay": trial.suggest_categorical("ema_decay", [0.99, 0.999, 0.9995]),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512]),
    }


def create_objective(
    train_pairs: list,
    val_pairs: list,
    item_embeddings: torch.Tensor,
    base_config: RecRecConfig,
    search_epochs: int = 15,
):
    num_items = item_embeddings.size(0)

    def objective(trial: optuna.Trial) -> float:
        hparams = sample_hyperparameters(trial)
        trial_config = dataclasses.replace(
            base_config,
            **hparams,
            max_epochs=search_epochs,
        )

        datamodule = RecRecDataModule(
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            num_items=num_items,
            config=trial_config,
        )

        model = RecRecLightning(
            pretrained_sbert_embeddings=item_embeddings,
            config=trial_config,
        )

        # Official Optuna PyTorch Lightning pruning callback
        pruning_cb = PyTorchLightningPruningCallback(trial=trial, monitor="val_ndcg10")
        ema_cb = EMACallback(decay=trial_config.ema_decay)

        trainer = pl.Trainer(
            max_epochs=search_epochs,
            accelerator="auto",
            devices="auto",
            callbacks=[ema_cb, pruning_cb],
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )

        trainer.fit(model, datamodule=datamodule)

        val_ndcg = trainer.callback_metrics.get("val_ndcg10")
        if val_ndcg is None:
            return 0.0
        return val_ndcg.item() if hasattr(val_ndcg, "item") else float(val_ndcg)

    return objective


def run_hparam_search_and_train(
    n_trials: int = 20,
    search_epochs: int = 15,
    final_epochs: int = 50,
    project_name: str = "recrec-recsys",
) -> tuple[optuna.Study, pl.Trainer, RecRecLightning]:
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_config = RecRecConfig()

    user_sequences = load_user_sequences(INTERACTION_PATH)
    item_embeddings = torch.load(SBERT_EMBEDDING_PATH, map_location=device)
    num_items = item_embeddings.size(0)
    validate_item_indexing(user_sequences, num_items)

    train_pairs, val_pairs = make_train_val_pairs(user_sequences)

    # 1. Configure Optuna TPESampler & HyperbandPruner
    sampler = TPESampler(seed=42)
    pruner = HyperbandPruner(
        min_resource=3,
        max_resource=search_epochs,
        reduction_factor=3,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="recrec_tpe_hyperband",
    )

    # 2. Official Optuna WandB integration callback for the study
    api_key = get_wandb_api_key()
    study_callbacks = []
    if api_key:
        wandb_cb = WeightsAndBiasesCallback(
            metric_name="val_ndcg10",
            wandb_kwargs={"project": project_name, "name": "optuna-hpo-study"},
            as_multirun=False,
        )
        study_callbacks.append(wandb_cb)

    print(f"Starting HPO search: {n_trials} trials, max {search_epochs} epochs per trial.")
    objective = create_objective(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        item_embeddings=item_embeddings,
        base_config=base_config,
        search_epochs=search_epochs,
    )

    study.optimize(objective, n_trials=n_trials, callbacks=study_callbacks)

    print(f"HPO complete. Best trial #{study.best_trial.number} (Val NDCG@10: {study.best_value:.5f})")
    print(f"Best hyperparameters: {study.best_params}")

    # 3. Automatically feed best parameters into final full training loop
    best_config = dataclasses.replace(
        base_config,
        **study.best_params,
        max_epochs=final_epochs,
    )

    print(f"Starting final training for {final_epochs} epochs with best config.")
    final_datamodule = RecRecDataModule(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        num_items=num_items,
        config=best_config,
    )

    final_model = RecRecLightning(
        pretrained_sbert_embeddings=item_embeddings,
        config=best_config,
    ).to(device)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_ndcg10",
        mode="max",
        save_top_k=1,
        filename="best-recrec-{epoch:02d}-{val_ndcg10:.4f}",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_ndcg10",
        mode="max",
        patience=10,
    )
    ema_callback = EMACallback(decay=best_config.ema_decay)
    wandb_logger = setup_wandb_logger(
        project_name=project_name,
        run_name=f"best-trial-{study.best_trial.number}-final",
    )

    if wandb_logger is not None and hasattr(wandb_logger, "experiment"):
        wandb_logger.experiment.config.update({"hpo_best_trial": study.best_trial.number, **study.best_params})

    trainer = pl.Trainer(
        max_epochs=final_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[ema_callback, checkpoint_callback, early_stop_callback],
        logger=[wandb_logger] if wandb_logger is not None else True,
        enable_progress_bar=True,
    )

    trainer.fit(final_model, datamodule=final_datamodule)
    trainer.validate(final_model, datamodule=final_datamodule)

    return study, trainer, final_model


if __name__ == "__main__":
    run_hparam_search_and_train(n_trials=15, search_epochs=12, final_epochs=50)
