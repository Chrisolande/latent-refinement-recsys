from .config import RecRecConfig, RefineRecConfig
from .data import (
    CandidateSamplingCollator,
    RecRecCollator,
    RecRecDataModule,
    RecRecDataset,
    RefineRecDataModule,
    SequentialRecDataModule,
    SequentialRecDataset,
    generate_causal_interaction_pairs,
    load_user_sequences,
    sample_negative_candidates,
    validate_item_id_continuity,
)
from .diagnostics import smoke_test_batch_diagnostics, verify_architecture_invariants
from .hpo import run_hparam_search, run_hparam_search_and_train, suggest_refinerec_search_space
from .lightning_module import RecRecLightning, RefineRecLightning
from .losses import deep_supervision_loss
from .metrics import compute_ranking_metrics
from .modules import (
    CandidateScoring,
    CoreRecursionMLP,
    InputEncoding,
    RecRec,
    RecursivePreferenceRefinement,
    RefineRec,
)
from .train import main, setup_wandb_logger, train

__all__ = [
    "CandidateSamplingCollator",
    "CandidateScoring",
    "CoreRecursionMLP",
    "InputEncoding",
    "RecRec",
    "RecRecCollator",
    "RecRecConfig",
    "RecRecDataModule",
    "RecRecDataset",
    "RecRecLightning",
    "RecursivePreferenceRefinement",
    "RefineRec",
    "RefineRecConfig",
    "RefineRecDataModule",
    "RefineRecLightning",
    "SequentialRecDataModule",
    "SequentialRecDataset",
    "compute_ranking_metrics",
    "deep_supervision_loss",
    "generate_causal_interaction_pairs",
    "load_user_sequences",
    "main",
    "run_hparam_search",
    "run_hparam_search_and_train",
    "sample_negative_candidates",
    "setup_wandb_logger",
    "smoke_test_batch_diagnostics",
    "suggest_refinerec_search_space",
    "train",
    "validate_item_id_continuity",
    "verify_architecture_invariants",
]
