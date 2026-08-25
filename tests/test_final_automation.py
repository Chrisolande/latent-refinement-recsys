from pathlib import Path
from types import SimpleNamespace

import yaml

from automation.constants import SEARCH_KEYS
from automation.sweeps import confirmation_config
from refinerec.hpo.trial import finalized_trial_metrics

ROOT = Path(__file__).resolve().parents[1]


def test_final_metric_reconciles_last_validation_point() -> None:
    metrics = finalized_trial_metrics(
        callback_best_score=0.5616,
        callback_best_epoch=28,
        final_score=0.5648,
        trainer_current_epoch=30,
    )
    assert metrics == {
        "best_val_ndcg10": 0.5648,
        "best_epoch": 29,
        "final_val_ndcg10": 0.5648,
        "epochs_completed": 30,
    }


def test_final_metric_preserves_earlier_peak() -> None:
    metrics = finalized_trial_metrics(
        callback_best_score=0.5868,
        callback_best_epoch=23,
        final_score=0.5845,
        trainer_current_epoch=29,
    )
    assert metrics["best_val_ndcg10"] == 0.5868
    assert metrics["best_epoch"] == 23
    assert metrics["epochs_completed"] == 29


def test_focused_sweep_budget_and_objective() -> None:
    config = yaml.safe_load(
        (ROOT / "automation" / "sweep-final-focused.yaml").read_text(encoding="utf-8")
    )
    assert config["method"] == "bayes"
    assert config["metric"] == {"name": "best_val_ndcg10", "goal": "maximize"}
    assert config["parameters"]["search_epochs"]["value"] == 30
    assert config["parameters"]["seed"]["value"] == 42
    assert "early_terminate" not in config


def test_confirmation_is_five_seed_40_epoch_grid() -> None:
    winner = SimpleNamespace(
        id="winner",
        sweep=SimpleNamespace(id="focused"),
        config={key: index + 1 for index, key in enumerate(SEARCH_KEYS)},
    )
    config = confirmation_config(winner, [42, 43, 44, 45, 46])
    assert config["method"] == "grid"
    assert config["parameters"]["seed"]["values"] == [42, 43, 44, 45, 46]
    assert config["parameters"]["search_epochs"]["value"] == 40
    assert config["parameters"]["early_stopping_patience"]["value"] == 6
