"""W&B sweep lifecycle: durable state, trial queueing, ranking, and aggregation."""

import json
import math
import statistics
from pathlib import Path
from typing import Any

import wandb

from automation.constants import CONFIRMATION_EPOCHS, ENTITY, PROJECT, SEARCH_KEYS


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def valid_metric(run: Any) -> float | None:
    value = run.summary.get("best_val_ndcg10")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    if run.summary.get("trial_state") != "COMPLETE":
        return None
    return float(value)


def ranked_finished_runs(sweep: Any) -> list[Any]:
    candidates = [(valid_metric(run), run) for run in sweep.runs]
    ranked = [(score, run) for score, run in candidates if score is not None]
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [run for _, run in ranked]


def ensure_sweep(
    state: dict[str, Any],
    state_path: Path,
    state_key: str,
    config: dict[str, Any],
) -> str:
    sweep_id = state.get(state_key)
    if sweep_id:
        print(f"resuming_{state_key}={sweep_id}", flush=True)
        return str(sweep_id)
    sweep_id = wandb.sweep(config, entity=ENTITY, project=PROJECT)
    state[state_key] = sweep_id
    write_state(state_path, state)
    print(f"created_{state_key}={sweep_id}", flush=True)
    return sweep_id


def run_remaining_trials(sweep_id: str, target_count: int, epochs: int) -> Any:
    from refinerec.hpo import run_hparam_search

    api = wandb.Api(timeout=120)
    sweep = api.sweep(f"{ENTITY}/{PROJECT}/{sweep_id}")
    existing = len(sweep.runs)
    remaining = max(0, target_count - existing)
    print(
        f"sweep={sweep_id} existing={existing} target={target_count} remaining={remaining}",
        flush=True,
    )
    if remaining:
        run_hparam_search(
            sweep_id=sweep_id,
            n_trials=remaining,
            search_epochs=epochs,
            project_name=PROJECT,
        )
    return api.sweep(f"{ENTITY}/{PROJECT}/{sweep_id}")


def confirmation_config(winner: Any, seeds: list[int]) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        key: {"value": winner.config[key]} for key in SEARCH_KEYS
    }
    parameters.update(
        {
            "seed": {"values": seeds},
            "search_epochs": {"value": CONFIRMATION_EPOCHS},
            "early_stopping_patience": {"value": 6},
            "early_stopping_min_delta": {"value": 0.0001},
            "num_workers": {"value": 0},
            "require_cuda": {"value": True},
            "log_model": {"value": False},
            "purpose": {"value": "final-five-seed-confirmation"},
            "research_stage": {"value": "final-automation-confirmation"},
            "evaluation_protocol": {"value": "fixed-validation-candidates-v1"},
            "validation_candidate_seed": {"value": 1729},
            "source_sweep_id": {"value": winner.sweep.id},
            "source_run_id": {"value": winner.id},
        }
    )
    return {
        "name": "refinerec-final-five-seed-confirmation-40ep",
        "method": "grid",
        "metric": {"name": "best_val_ndcg10", "goal": "maximize"},
        "parameters": parameters,
    }


def summarize_confirmation(runs: list[Any]) -> dict[str, Any]:
    scores = [float(run.summary["best_val_ndcg10"]) for run in runs]
    finals = [float(run.summary["final_val_ndcg10"]) for run in runs]
    best = max(runs, key=lambda run: float(run.summary["best_val_ndcg10"]))
    return {
        "complete_runs": len(runs),
        "mean_best_val_ndcg10": statistics.mean(scores),
        "std_best_val_ndcg10": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "min_best_val_ndcg10": min(scores),
        "max_best_val_ndcg10": max(scores),
        "mean_final_val_ndcg10": statistics.mean(finals),
        "best_run_id": best.id,
        "best_run_name": best.name,
        "best_seed": best.config.get("seed"),
        "best_val_ndcg10": float(best.summary["best_val_ndcg10"]),
        "best_epoch": int(best.summary["best_epoch"]),
    }
