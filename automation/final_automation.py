# ruff: noqa: E402 -- the sys.path preamble below must run before project imports.
"""End-to-end RefineRec final search, confirmation, aggregation, and report."""

import argparse
import sys
from pathlib import Path

import torch
import wandb
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from automation.constants import (
    CONFIRMATION_EPOCHS,
    DEFAULT_CONFIRMATION_SEEDS,
    DEFAULT_HPO_TRIALS,
    ENTITY,
    HPO_EPOCHS,
    PROJECT,
)
from automation.report import create_report
from automation.sweeps import (
    confirmation_config,
    ensure_sweep,
    ranked_finished_runs,
    read_state,
    run_remaining_trials,
    summarize_confirmation,
    write_state,
)
from refinerec.hpo import configure_wandb_auth


def preflight() -> None:
    configure_wandb_auth()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Enable a Kaggle GPU accelerator.")
    print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(f"torch={torch.__version__} cuda={torch.version.cuda}", flush=True)


def execute(args: argparse.Namespace) -> None:
    state_path = Path(args.state_file)
    state = read_state(state_path)
    hpo_config = yaml.safe_load(
        (ROOT / "automation" / "sweep-final-focused.yaml").read_text(encoding="utf-8")
    )

    if args.dry_run:
        assert hpo_config["method"] == "bayes"
        assert hpo_config["metric"]["name"] == "best_val_ndcg10"
        assert args.hpo_trials == 15
        assert args.confirmation_seeds == DEFAULT_CONFIRMATION_SEEDS
        print("dry_run=passed", flush=True)
        return

    preflight()

    hpo_sweep_id = ensure_sweep(
        state, state_path, "hpo_sweep_id", hpo_config
    )
    hpo_sweep = run_remaining_trials(hpo_sweep_id, args.hpo_trials, HPO_EPOCHS)
    hpo_runs = ranked_finished_runs(hpo_sweep)
    if len(hpo_runs) < args.hpo_trials:
        raise RuntimeError(
            f"Only {len(hpo_runs)}/{args.hpo_trials} HPO runs completed with valid metrics."
        )
    winner = hpo_runs[0]
    state["hpo_winner_run_id"] = winner.id
    write_state(state_path, state)
    print(
        f"hpo_winner={winner.id} best_val_ndcg10={winner.summary['best_val_ndcg10']}",
        flush=True,
    )

    confirm_config = confirmation_config(winner, args.confirmation_seeds)
    confirmation_sweep_id = ensure_sweep(
        state, state_path, "confirmation_sweep_id", confirm_config
    )
    confirmation_sweep = run_remaining_trials(
        confirmation_sweep_id,
        len(args.confirmation_seeds),
        CONFIRMATION_EPOCHS,
    )
    confirmation_runs = ranked_finished_runs(confirmation_sweep)
    if len(confirmation_runs) < len(args.confirmation_seeds):
        raise RuntimeError(
            f"Only {len(confirmation_runs)}/{len(args.confirmation_seeds)} "
            "confirmation runs completed with valid metrics."
        )
    aggregate = summarize_confirmation(confirmation_runs)

    # A sweep agent can leave its last trial attached to the process. Finish
    # that run explicitly so the aggregate table and report are recorded on a
    # distinct research-summary run rather than overwriting the last seed.
    if wandb.run is not None:
        wandb.run.finish()

    with wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name="refinerec-final-automation-summary",
        job_type="research-summary",
        reinit="finish_previous",
        config={
            "purpose": "final-automation-summary",
            "hpo_sweep_id": hpo_sweep_id,
            "confirmation_sweep_id": confirmation_sweep_id,
            "hpo_trial_budget": args.hpo_trials,
            "confirmation_seeds": args.confirmation_seeds,
            "winner_run_id": winner.id,
        },
    ) as summary_run:
        table = wandb.Table(
            columns=[
                "run_id",
                "seed",
                "best_val_ndcg10",
                "final_val_ndcg10",
                "best_epoch",
                "epochs_completed",
            ]
        )
        for run in sorted(confirmation_runs, key=lambda item: item.config.get("seed", -1)):
            table.add_data(
                run.id,
                run.config.get("seed"),
                run.summary.get("best_val_ndcg10"),
                run.summary.get("final_val_ndcg10"),
                run.summary.get("best_epoch"),
                run.summary.get("epochs_completed"),
            )
        summary_run.log({"confirmation/results": table})
        for key, value in aggregate.items():
            summary_run.summary[f"confirmation/{key}"] = value
        report_url = create_report(
            hpo_runs=hpo_runs,
            confirmation_runs=confirmation_runs,
            winner=winner,
            aggregate=aggregate,
            hpo_sweep_id=hpo_sweep_id,
            confirmation_sweep_id=confirmation_sweep_id,
        )
        summary_run.summary["report_url"] = report_url
        summary_run.summary["automation_state"] = "COMPLETE"
        print(f"summary_run={summary_run.url}", flush=True)
        print(f"report={report_url}", flush=True)

    state.update(
        {
            "automation_state": "COMPLETE",
            "report_url": report_url,
            "aggregate": aggregate,
        }
    )
    write_state(state_path, state)
    print("final_automation_complete=true", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpo-trials", type=int, default=DEFAULT_HPO_TRIALS)
    parser.add_argument(
        "--confirmation-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_CONFIRMATION_SEEDS,
    )
    parser.add_argument(
        "--state-file",
        default="/kaggle/working/refinerec_final_automation_state.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    execute(parse_args())


if __name__ == "__main__":
    main()
