import dataclasses
import os
from typing import Any

import optuna
import pytorch_lightning as pl
import torch
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna_integration import PyTorchLightningPruningCallback

from .callbacks import EMACallback
from .config import RefineRecConfig
from .data import (
    RefineRecDataModule,
    generate_causal_interaction_pairs,
    load_user_sequences,
    validate_item_id_continuity,
)
from .lightning_module import RefineRecLightning
from .train import resolve_data_paths, set_seed


def get_wandb_study_callback(
    metric_name: str = "val_ndcg10",
    project_name: str = "refinerec",
) -> Any | None:
    """Returns W&B callback for Optuna study."""
    try:
        from kaggle_secrets import UserSecretsClient

        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret("WANDB_API_KEY")
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
    except Exception:
        pass

    try:
        from optuna_integration.wandb import WeightsAndBiasesCallback

        return WeightsAndBiasesCallback(
            metric_name=metric_name,
            wandb_kwargs={"project": project_name, "name": "optuna-hpo-study"},
            as_multirun=False,
        )
    except Exception as e:
        print(f"WandB study callback not initialized: {e}. Continuing without it.")
        return None


def suggest_refinerec_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Defines search space for RefineRec hyperparameters."""
    return {
        # Optimization
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "grad_clip": trial.suggest_float("grad_clip", 0.5, 5.0, step=0.5),
        # Model Architecture
        "inner_steps": trial.suggest_int("inner_steps", 1, 4),
        "outer_steps": trial.suggest_int("outer_steps", 3, 8),
        "core_depth": trial.suggest_int("core_depth", 3, 6),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
        # Loss / Scaling
        "temperature": trial.suggest_float("temperature", 0.05, 2.0, log=True),
        "preference_scale": trial.suggest_float("preference_scale", 0.1, 2.0, log=True),
        # Regularization / Hardware
        "ema_decay": trial.suggest_categorical("ema_decay", [0.99, 0.999, 0.9995]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
    }


# Backward-compatible alias
sample_hyperparameters = suggest_refinerec_search_space


def create_objective(
    train_pairs: list,
    val_pairs: list,
    item_embeddings: torch.Tensor,
    base_config: RefineRecConfig,
    search_epochs: int = 15,
):
    num_items = item_embeddings.size(0)

    def objective(trial: optuna.Trial) -> float:
        hparams = suggest_refinerec_search_space(trial)
        trial_config = dataclasses.replace(
            base_config,
            **hparams,
            max_epochs=search_epochs,
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

        callbacks = [
            EMACallback(decay=trial_config.ema_decay),
            PyTorchLightningPruningCallback(trial=trial, monitor="val_ndcg10"),
        ]

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        trial_device = (
            [trial.number % num_gpus] if torch.cuda.is_available() and num_gpus > 1 else "auto"
        )

        trainer = pl.Trainer(
            max_epochs=search_epochs,
            accelerator="auto",
            devices=trial_device,
            callbacks=callbacks,
            gradient_clip_val=trial_config.grad_clip,
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


def run_hparam_search(
    n_trials: int = 40,
    search_epochs: int = 15,
    project_name: str = "refinerec",
) -> optuna.Study:
    """Runs Optuna hyperparameter optimization for RefineRec."""
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_config = RefineRecConfig()

    interaction_path, embedding_path = resolve_data_paths()
    user_sequences = load_user_sequences(interaction_path)
    item_embeddings = torch.load(embedding_path, map_location=device)
    num_items = item_embeddings.size(0)
    validate_item_id_continuity(user_sequences, num_items)

    train_pairs, val_pairs = generate_causal_interaction_pairs(user_sequences)

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
        study_name="refinerec_tpe_hyperband",
    )

    wandb_cb = get_wandb_study_callback(metric_name="val_ndcg10", project_name=project_name)
    callbacks = [wandb_cb] if wandb_cb is not None else []

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    n_jobs = num_gpus if num_gpus > 1 else 1

    print(
        f"Starting HPO search: {n_trials} trials, {n_jobs} parallel GPU worker(s), "
        f"max {search_epochs} epochs per trial."
    )
    objective = create_objective(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        item_embeddings=item_embeddings,
        base_config=base_config,
        search_epochs=search_epochs,
    )

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=callbacks)

    print(
        f"HPO complete. Best trial #{study.best_trial.number} (Val NDCG@10: {study.best_value:.5f})"
    )
    print(f"Best hyperparameters: {study.best_params}")

    return study


# Backward-compatible alias
run_hparam_search_and_train = run_hparam_search


if __name__ == "__main__":
    run_hparam_search(n_trials=40, search_epochs=12)
