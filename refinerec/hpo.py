import dataclasses
import os
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import yaml

try:
    import wandb
    from pytorch_lightning.loggers import WandbLogger
except ImportError:
    wandb = None
    WandbLogger = None

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


ENTITY = "olandechris-"
DEFAULT_PROJECT = "refinerec"
SWEEP_NAME = "refinerec-bayes-hyperband"


def configure_wandb_auth() -> None:
    """Load the W&B API key from Kaggle Secrets."""
    from kaggle_secrets import UserSecretsClient

    api_key = UserSecretsClient().get_secret("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY is missing from Kaggle Secrets")

    os.environ["WANDB_API_KEY"] = api_key


def load_sweep_config(
    config_path: str | Path | None = None,
    search_epochs: int | None = None,
) -> dict[str, Any]:
    """Loads the W&B Sweep configuration from a YAML file."""
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Sweep config file not found: {path}")
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    else:
        candidates = [
            Path("sweep.yaml"),
            Path("refinerec/sweep.yaml"),
            Path(__file__).parent / "sweep.yaml",
            Path(__file__).parent.parent / "sweep.yaml",
            Path("/kaggle/working/sweep.yaml"),
        ]
        config = None
        for candidate in candidates:
            if candidate.exists():
                with open(candidate, "r") as f:
                    config = yaml.safe_load(f)
                break
        if config is None:
            raise FileNotFoundError(
                "Could not find sweep.yaml. Please provide a valid config_path or place sweep.yaml in the project root."
            )

    if (
        search_epochs is not None
        and "parameters" in config
        and "search_epochs" in config["parameters"]
    ):
        config["parameters"]["search_epochs"] = {"value": search_epochs}

    return config


def build_sweep_config(
    search_epochs: int = 15,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Loads the sweep configuration from YAML (backward compatibility)."""
    return load_sweep_config(config_path=config_path, search_epochs=search_epochs)


suggest_refinerec_search_space = load_sweep_config


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
    if wandb is None or WandbLogger is None:
        raise ImportError(
            "W&B and WandbLogger are required for HPO. "
            "Please install wandb (e.g. pip install wandb)."
        )

    num_items = item_embeddings.size(0)

    def train_trial() -> None:
        with wandb.init(
            entity=ENTITY,
            project=project_name,
            job_type="hpo-trial",
            tags=["wandb-sweep", "bayes", "hyperband"],
            settings=wandb.Settings(save_code=True),
        ) as run:
            # Reset model/data-loader randomness for fair trial comparisons.
            set_seed()

            hparams = _config_from_wandb(run)
            search_epochs = int(run.config["search_epochs"])

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

            # Connect this existing sweep run to PyTorch Lightning.
            # Every metric logged with self.log/self.log_dict is forwarded.
            wandb_logger = WandbLogger(
                experiment=run,
                log_model=False,
            )

            callbacks = [
                EMACallback(decay=trial_config.ema_decay),
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

                score = (
                    float(val_ndcg.item())
                    if hasattr(val_ndcg, "item")
                    else float(val_ndcg)
                )

                # Summary metadata does not add another history point.
                run.summary["trial_state"] = "COMPLETE"
                run.summary["val_ndcg10"] = score

            except Exception as exc:
                run.summary["trial_state"] = "FAILED"
                run.summary["error_type"] = type(exc).__name__
                run.summary["error_message"] = str(exc)[:1000]
                raise

    return train_trial


def run_hparam_search(
    n_trials: int = 40,
    search_epochs: int = 15,
    project_name: str = DEFAULT_PROJECT,
    sweep_id: str | None = None,
    config_path: str | Path | None = None,
) -> str:
    """
    Create or resume a native W&B Bayesian Sweep loaded from YAML and run its agent.

    To create a new sweep:
        run_hparam_search(n_trials=40, search_epochs=12)

    To use a custom sweep YAML:
        run_hparam_search(config_path="custom_sweep.yaml", n_trials=40)

    To continue an existing sweep:
        run_hparam_search(
            n_trials=20,
            search_epochs=12,
            sweep_id="<existing-sweep-id>",
        )
    """
    configure_wandb_auth()
    set_seed()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    base_config = RefineRecConfig()

    interaction_path, embedding_path = resolve_data_paths()

    user_sequences = load_user_sequences(
        interaction_path
    )

    item_embeddings = torch.load(
        embedding_path,
        map_location=device,
    )

    num_items = item_embeddings.size(0)

    validate_item_id_continuity(
        user_sequences,
        num_items,
    )

    train_pairs, val_pairs = (
        generate_causal_interaction_pairs(
            user_sequences
        )
    )

    if sweep_id is None:
        sweep_config = load_sweep_config(
            config_path=config_path,
            search_epochs=search_epochs,
        )
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            entity=ENTITY,
            project=project_name,
        )

    train_trial = create_sweep_trial(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        item_embeddings=item_embeddings,
        base_config=base_config,
        project_name=project_name,
    )

    print(
        f"Starting native W&B Bayesian sweep {sweep_id}: "
        f"{n_trials} agent run(s), "
        f"up to {search_epochs} epochs each."
    )

    wandb.agent(
        sweep_id=sweep_id,
        function=train_trial,
        count=n_trials,
        entity=ENTITY,
        project=project_name,
    )

    print(f"Sweep complete: {sweep_id}")

    return sweep_id


run_hparam_search_and_train = run_hparam_search



#
# if __name__ == "__main__":
#     run_hparam_search(
#         n_trials=40,
#         search_epochs=12,
#     )