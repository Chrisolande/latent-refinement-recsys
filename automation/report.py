"""Build and publish the W&B research summary report."""

import json
from typing import Any

from automation.constants import ENTITY, PROJECT, SEARCH_KEYS


def create_report(
    hpo_runs: list[Any],
    confirmation_runs: list[Any],
    winner: Any,
    aggregate: dict[str, Any],
    hpo_sweep_id: str,
    confirmation_sweep_id: str,
) -> str:
    import wandb_workspaces.expr as expr
    import wandb_workspaces.reports.v2 as wr

    hpo_ids = [run.id for run in hpo_runs]
    confirmation_ids = [run.id for run in confirmation_runs]
    hpo_runset = wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name="Focused Bayesian search",
        filters=[expr.Metric("name").isin(hpo_ids)],
        order=[wr.OrderBy(name=wr.SummaryMetric("best_val_ndcg10"), ascending=False)],
        visible_columns=[
            "run:name",
            "run:state",
            "summary:best_val_ndcg10",
            "summary:best_epoch",
            "config:learning_rate.value",
            "config:weight_decay.value",
            "config:dropout.value",
            "config:temperature.value",
            "config:preference_scale.value",
        ],
        lock_columns=True,
    )
    confirmation_runset = wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name="Five-seed confirmation",
        filters=[expr.Metric("name").isin(confirmation_ids)],
        order=[wr.OrderBy(name=wr.Config("seed"), ascending=True)],
        visible_columns=[
            "run:name",
            "run:state",
            "config:seed.value",
            "summary:best_val_ndcg10",
            "summary:final_val_ndcg10",
            "summary:best_epoch",
            "summary:epochs_completed",
        ],
        lock_columns=True,
    )
    winner_config = {key: winner.config.get(key) for key in SEARCH_KEYS}
    decision = (
        f"Best focused-search run: `{winner.id}`. Confirmation mean best NDCG@10: "
        f"**{aggregate['mean_best_val_ndcg10']:.4f} ± "
        f"{aggregate['std_best_val_ndcg10']:.4f}** across "
        f"{aggregate['complete_runs']} seeds. Best confirmation: "
        f"**{aggregate['best_val_ndcg10']:.4f}** at epoch "
        f"{aggregate['best_epoch']} (seed {aggregate['best_seed']})."
    )
    marker = f"Final automation sweeps: `{hpo_sweep_id}` and `{confirmation_sweep_id}`."
    report = wr.Report(
        entity=ENTITY,
        project=PROJECT,
        title=f"RefineRec Final Automated Search - {confirmation_sweep_id}",
        description="Focused Bayesian optimization, five-seed confirmation, and final decision.",
        width="fluid",
        blocks=[
            wr.H1(text="RefineRec final automated research result"),
            wr.MarkdownBlock(text=marker),
            wr.MarkdownBlock(text=decision),
            wr.H2(text="Focused Bayesian search"),
            wr.PanelGrid(
                runsets=[hpo_runset],
                panels=[
                    wr.LinePlot(
                        title="Validation NDCG@10 by epoch",
                        x="epoch",
                        y=["val_ndcg10"],
                        layout=wr.Layout(w=24, h=10),
                    ),
                    wr.BarPlot(
                        title="Best NDCG@10 by trial",
                        metrics=["best_val_ndcg10"],
                        orientation="h",
                        max_bars_to_show=20,
                        layout=wr.Layout(w=24, h=10),
                    ),
                ],
            ),
            wr.H2(text="Five-seed confirmation"),
            wr.PanelGrid(
                runsets=[confirmation_runset],
                panels=[
                    wr.LinePlot(
                        title="Confirmation learning curves",
                        x="epoch",
                        y=["val_ndcg10"],
                        layout=wr.Layout(w=24, h=10),
                    ),
                    wr.BarPlot(
                        title="Best and final NDCG@10 by seed",
                        metrics=["best_val_ndcg10", "final_val_ndcg10"],
                        orientation="v",
                        max_bars_to_show=10,
                        layout=wr.Layout(w=24, h=10),
                    ),
                ],
            ),
            wr.H2(text="Winner configuration"),
            wr.CodeBlock(code=[json.dumps(winner_config, indent=2)], language="json"),
        ],
    )
    report.save(draft=False)
    return report.url
