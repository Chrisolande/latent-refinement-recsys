"""Shared settings for the RefineRec automation package."""

ENTITY = "olandechris-"
PROJECT = "refinerec"
HPO_EPOCHS = 30
CONFIRMATION_EPOCHS = 40
DEFAULT_HPO_TRIALS = 15
DEFAULT_CONFIRMATION_SEEDS = [42, 43, 44, 45, 46]
SEARCH_KEYS = [
    "learning_rate",
    "weight_decay",
    "dropout",
    "temperature",
    "preference_scale",
    "batch_size",
    "core_depth",
    "ema_decay",
    "grad_clip",
    "inner_steps",
    "outer_steps",
]
