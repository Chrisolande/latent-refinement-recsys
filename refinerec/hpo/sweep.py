"""W&B sweep orchestration: auth, sweep config loading, and agent lifecycle."""

import os
from pathlib import Path
from typing import Any

import torch
import wandb
import yaml

from ..config import RefineRecConfig
from ..data import (
    generate_causal_interaction_pairs,
    load_user_sequences,
    resolve_data_paths,
    validate_item_id_continuity,
)
from ..training.trainer import set_seed
from .trial import (
    DEFAULT_PROJECT,
    ENTITY,
    create_sweep_trial,
)


def configure_wandb_auth() -> None:
    """Use an existing W&B key or load it from Kaggle Secrets."""
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        try:
            from kaggle_secrets import UserSecretsClient

            api_key = UserSecretsClient().get_secret("WANDB_API_KEY")
        except ImportError:
            api_key = None
    if not api_key:
        raise RuntimeError(
            "WANDB_API_KEY is missing. Set it in the environment or Kaggle Secrets."
        )

    os.environ["WANDB_API_KEY"] = api_key
    wandb.login(key=api_key, relogin=False)


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
        path = next(
            (candidate for candidate in candidates if candidate.exists()), None
        )
        if path is None:
            raise FileNotFoundError(
                "Could not find sweep.yaml. Provide a valid config_path or place "
                "sweep.yaml in the project root."
            )
        config = yaml.safe_load(path.read_text(encoding="utf-8"))

    if (
        search_epochs is not None
        and "parameters" in config
        and "search_epochs" in config["parameters"]
    ):
        config["parameters"]["search_epochs"] = {"value": search_epochs}

    return config


def run_hparam_search(
    n_trials: int = 40,
    search_epochs: int = 15,
    project_name: str = DEFAULT_PROJECT,
    sweep_id: str | None = None,
    config_path: str | Path | None = None,
) -> str:
    """Create or resume a native W&B Bayesian Sweep and run its agent."""
    configure_wandb_auth()
    set_seed()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_config = RefineRecConfig()
    interaction_path, embedding_path = resolve_data_paths()
    user_sequences = load_user_sequences(interaction_path)
    item_embeddings = torch.load(embedding_path, map_location=device)
    num_items = item_embeddings.size(0)
    validate_item_id_continuity(user_sequences, num_items)
    train_pairs, val_pairs = generate_causal_interaction_pairs(user_sequences)

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

