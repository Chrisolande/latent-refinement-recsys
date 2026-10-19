# RefineRec: Recursive Preference Refinement with Semantic Evidence Anchoring

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/Lightning-2.0%2B-792ee5.svg)](https://lightning.ai/)
[![Weights & Biases](https://img.shields.io/badge/W%26B-Report-FFBE00.svg?logo=weightsandbiases&logoColor=black)](https://wandb.ai/olandechris-/refinerec/reports/RefineRec-Bayesian-Optimization-and-Five-Seed-Validation--VmlldzoxNzc5NjQwNA?accessToken=egmmekquwgr4ygitbfcg578evciwogl5q9br61ooksbnaenypd4qznvduncqizzh)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-4%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What this does

Every day, millions of people browse digital marketplaces, streaming platforms, and e-commerce stores looking for products, media, and services they will love. Delivering timely, accurate next-item recommendations is essential for helping users discover relevant items without getting overwhelmed by choice.

Two major challenges make sequential recommendation exceptionally difficult in the real world:

1. **User intent is fluid, noisy, and easily distracted.** A user searching for outdoor camping equipment might briefly click an unrelated trending gadget or an accidental link before continuing their shopping journey. Traditional sequential recommendation models often suffer from extreme "recency bias" or catastrophic drift—either overreacting to the single latest click or failing to track how underlying tastes gradually evolve across a session.
2. **Cold-start items and massive item catalogs.** In large dynamic catalogs, thousands of items have few or no prior user interactions (the classic cold-start problem). Standard collaborative filtering and ID-based models treat items as arbitrary integer tokens, failing completely when encountering new products and ignoring rich textual metadata like product titles, categories, and descriptions.

This project solves both problems with **RefineRec (Recursive Preference Refinement with Semantic Evidence Anchoring)**. Instead of attempting to guess user interest in a single rushed forward pass, RefineRec models recommendation as an *iterative preference refinement process*. By combining dual-loop recurrent latent dynamics with rich natural language item semantics, the system converges on accurate, noise-resistant recommendations that bridge both established and brand-new items.

---

## How it works

RefineRec models sequential recommendation as an iterative trajectory optimization problem in latent space. A few key design choices make its predictions robust in practice:

- **It refines user intent step-by-step across two dynamic loops.** Rather than relying on a static one-shot prediction, RefineRec decouples recommendation into a fast inner loop (synthesizing immediate contextual evidence) and a macro outer loop (progressively steering the user's latent preference trajectory toward their target interest).
- **It stays grounded with evidence-anchored correction.** During iterative updates, unconstrained neural representations can drift away from reality. An input-dependent sigmoid gate dynamically anchors intermediate latent states back to the user's historical context, preventing preference hallucinations.
- **It understands items through natural language (Zero Cold-Start).** Item titles and descriptions are projected into dense 384-dimensional semantic embeddings using SentenceTransformers (`all-MiniLM-L6-v2`), enabling zero-shot recommendation for newly added products without prior click histories.
- **It enforces monotonic improvements through multi-step deep supervision.** The model is supervised across *every* intermediate refinement step ($t = 1 \dots T$), forcing ranking accuracy to improve smoothly at each stage and stabilizing gradient flow during training.
- **It keeps parameters ultra-compact with a shared core MLP.** All recursive steps across both inner and outer loops reuse a single, parameter-efficient multi-layer perceptron ($f_\phi$), delivering deep reasoning capabilities with minimal computational overhead.
- **Automated Bayesian optimization finds peak hyperparameters.** Integrates Weights & Biases Bayesian sweeps with Hyperband early stopping to discover the optimal refinement depths, temperature scaling, and learning rates.
- **Validated rigorously across multiple random seeds.** Evaluated across 5 independent random seeds (`42, 43, 44, 45, 46`) under a strict 100-candidate ranking protocol (1 ground truth + 99 sampled negatives) to guarantee reproducibility and empirical stability.

---

## Empirical benchmark performance

Evaluated on the Amazon **Luxury Beauty** sequential recommendation dataset across 5 independent random seeds (`42, 43, 44, 45, 46`) using the optimal Bayesian configuration under a 100-candidate ranking protocol (1 ground truth + 99 sampled negatives):

### Aggregate Results (Mean $\pm$ Std)

| Metric | RefineRec Performance | Range [Min - Max] |
| :--- | :---: | :---: |
| **Best NDCG@10** | **$0.5859 \pm 0.0076$** | **0.5711 - 0.5919** |
| **Terminal NDCG@10** | **$0.5840 \pm 0.0068$** | **0.5711 - 0.5906** |
| **Best-to-Terminal Drop** | **$0.0019 \pm 0.0014$** | **0.0000 - 0.0043** |
| **Best Epoch** | **$21.4 \pm 10.9$** | **12 - 39** |

### Per-Seed Detailed Breakdown

| Seed | Run ID | Best NDCG@10 | Terminal NDCG@10 | Best Epoch | Epochs Completed |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | `wulq23cc` | 0.57114 | 0.57114 | 39 | 40 |
| **43** | `66ozl5pr` | 0.58612 | 0.58421 | 24 | 30 |
| **44** | `25s48rlx` | 0.59091 | 0.58660 | 17 | 23 |
| **45** | `rovxc1wb` | **0.59188** | **0.59064** | 12 | 18 |
| **46** | `04051a7l` | 0.58926 | 0.58736 | 15 | 21 |

> **Checkpoint Retention Stability:** The mean best-to-terminal performance delta is only **0.00187** (maximum drop of **0.00431**), confirming that early stopping checkpoints are highly reliable and resistant to overfitting.
> 
> Explore the live interactive run curves, loss surfaces, and Bayesian search dynamics in the [W&B Study Report](https://wandb.ai/olandechris-/refinerec/reports/RefineRec-Bayesian-Optimization-and-Five-Seed-Validation--VmlldzoxNzc5NjQwNA?accessToken=egmmekquwgr4ygitbfcg578evciwogl5q9br61ooksbnaenypd4qznvduncqizzh).

---

## Mathematical formulations

<details>
<summary><b>Click to expand: metric derivations and model math</b></summary>

<br>

```mermaid
flowchart LR
    A["Interaction History S"] --> B["SBERT Semantic Embeddings e_i"]
    B --> C["Context Mean Vector x"]
    C --> D["Inner Loop: Evidence Synthesis z_t^(j)"]
    D --> E["Correction Gate: g_t Anchor"]
    E --> F["Outer Loop: Preference Trajectory y_(t+1)"]
    F --> G["Candidate Dot-Product Scoring s_(t,j)"]
    G --> H["Multi-Step Deep Supervision Loss L_total"]
```

### 1. Input Context Encoding & Semantic Item Embeddings

Given an interaction history sequence $S = (i_1, i_2, \dots, i_{|S|})$ and item semantic embeddings $\mathbf{e}_i \in \mathbb{R}^d$ extracted from text metadata:

$$\mathbf{x} = \frac{\sum_{i \in S} m_i \mathbf{e}_i}{\sum_{i \in S} m_i}, \quad \mathbf{y}_0 = \mathbf{x}, \quad \mathbf{z}_0 = \mathbf{0}$$

where $m_i \in \{0, 1\}$ denotes sequence validity mask indicators and $d = 384$.

### 2. Dual-Loop Latent Refinement Dynamics

For outer refinement step $t \in \{1, \dots, T\}$:

- **Inner Evidence Synthesis ($j = 1, \dots, n$):**
  $$\mathbf{z}_t^{(j)} = f_\phi([\mathbf{x} \parallel \mathbf{y}_t \parallel \mathbf{z}_t^{(j-1)}])$$

- **Evidence-Anchored Correction Gate:**
  $$\mathbf{g}_t = \sigma(\mathbf{W}_t [\mathbf{x} \parallel \mathbf{y}_t])$$
  $$\mathbf{z}_t = (1 - \mathbf{g}_t) \odot \mathbf{z}_t^{(n)} + \mathbf{g}_t \odot \mathbf{x}$$

- **Residual Preference Trajectory Update:**
  $$\mathbf{y}_{t+1} = \mathbf{y}_t + L \cdot \tanh(f_\phi([\mathbf{x} \parallel \mathbf{y}_t \parallel \mathbf{z}_t]))$$

Where:
- $f_\phi: \mathbb{R}^{3d} \to \mathbb{R}^d$ is a depth-$D$ multilayer perceptron with LayerNorm and ReLU activations, shared across all recursive evaluations.
- $\mathbf{W}_t \in \mathbb{R}^{d \times 2d}$ is a step-specific linear gating transformation.
- $L$ denotes the preference residual scaling factor ($L = 0.1036$ in the winning config).

### 3. Candidate Scoring & Multi-Step Deep Supervision

For a candidate set of item embeddings $\{\mathbf{e}_j\}_{j=1}^K$ ($1$ ground truth target + $K-1$ sampled negatives) and temperature scaling factor $\tau$:

$$s_{t, j} = \frac{\mathbf{y}_{t+1}^\top \mathbf{e}_j}{\tau}$$

$$\mathcal{L}_{\text{total}} = \frac{1}{T} \sum_{t=1}^T \mathcal{L}_{\text{CE}}(\mathbf{s}_t, \text{target})$$

### 4. Ranking Evaluation Metrics

Evaluation uses the standard leave-one-out ranking protocol over sampled 100-item candidate pools (1 ground truth target + 99 sampled negatives):

- **Normalized Discounted Cumulative Gain (NDCG@K):** Evaluates ranking quality with logarithmic position discounting:
  $$\text{DCG}@K = \sum_{r=1}^K \frac{2^{\mathbb{I}(r = r_{\text{target}})} - 1}{\log_2(r + 1)}, \quad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$
- **Hit Ratio (HR@K):** Binary indicator measuring whether the target item appears in the top-$K$ recommendations ($\mathbb{I}(r_{\text{target}} \le K)$).
- **Precision (Prec@K):** Proportion of top-$K$ recommendations that match the ground-truth target.

</details>

---

## Project structure

```text
latent-refinement-recsys/
├── refinerec/                            # Core modular framework
│   ├── config.py                         # Dataclass hyperparameters (RefineRecConfig)
│   ├── sweep.yaml                        # Packaged exploratory Bayesian sweep specification
│   ├── data/
│   │   ├── loader.py                     # Sequence loading, causal pairs, negative sampling, DataModule
│   │   ├── embeddings.py                 # Offline SBERT metadata embedding extraction
│   │   └── __init__.py
│   ├── models/
│   │   ├── modules.py                    # Core PyTorch modules (InputEncoding, CoreRecursionMLP, RefineRec)
│   │   ├── losses.py                     # Multi-step deep supervision cross-entropy loss
│   │   ├── metrics.py                    # Top-K ranking metrics (HR@k, NDCG@k, Prec@k)
│   │   ├── lightning_module.py           # PyTorch Lightning module (RefineRecLightning)
│   │   └── __init__.py
│   ├── training/
│   │   ├── callbacks.py                  # Exponential Moving Average callback (EMACallback)
│   │   ├── trainer.py                    # End-to-end training pipeline and CLI entrypoint
│   │   ├── diagnostics.py                # Invariant audit & single-batch sanity check
│   │   └── __init__.py
│   ├── hpo/
│   │   ├── sweep.py                      # Sweep orchestration: auth, config loading, agent lifecycle
│   │   ├── trial.py                      # Single-trial machinery: best-metric tracking, train function
│   │   └── __init__.py
│   └── __init__.py                       # Clean public API exports
├── automation/                           # End-to-end research study pipeline
│   ├── constants.py                      # Shared entity/project, budgets, and search keys
│   ├── sweeps.py                         # Durable state, sweep creation/resume, ranking, aggregation
│   ├── report.py                         # W&B report construction & publication
│   ├── final_automation.py               # Orchestration CLI (search → confirm → summarize → report)
│   ├── sweep-final-focused.yaml          # Focused 15-trial Bayesian search configuration
│   └── __init__.py
├── tests/                                # Automated PyTest test suite
│   └── test_final_automation.py          # Unit tests for trial metrics, sweep configs, budgets
├── RefineRecLightning.ipynb              # Interactive portfolio walkthrough notebook
├── pyproject.toml                        # Build system, package metadata, and dependency specs
├── uv.lock                               # Deterministic environment lockfile
└── .gitignore                            # Git artifact and cache exclusions
```

---

## Getting started

### 1. Installation

Set up the environment with `uv` (recommended) or standard `pip`:

```bash
# Clone the repository
git clone https://github.com/Chrisolande/latent-refinement-recsys.git
cd latent-refinement-recsys

# Install via uv (including optional extras)
uv sync --all-extras
```

Or using standard `pip`:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[automation,dev]"
```

### 2. Pretrained item feature extraction

Extract 384-dimensional dense semantic vectors from text metadata:

```python
from refinerec.embeddings import extract_sbert_item_embeddings

extract_sbert_item_embeddings(
    interaction_path="data/Luxury_Beauty_5.txt",
    metadata_path="data/Luxury_Beauty_5_text_name_dict.pkl",
    output_path="data/sbert_item_embeddings.pt",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
```

---

## Workflows & usage

### 1. Command Line Interface (CLI)

Execute full training with pre-flight invariant checks and validation:

```bash
# Using installed entrypoint
refinerec-train

# Or directly via python module
python -m refinerec.training.trainer
```

### 2. Programmatic Python API

Train and evaluate RefineRec directly within Python scripts using PyTorch Lightning:

```python
import pytorch_lightning as pl
import torch
from refinerec import (
    EMACallback,
    RefineRecConfig,
    RefineRecDataModule,
    RefineRecLightning,
    generate_causal_interaction_pairs,
    load_user_sequences,
)

# 1. Initialize configuration with winning Bayesian hyperparameters
config = RefineRecConfig(
    outer_steps=3,
    inner_steps=4,
    core_depth=6,
    preference_scale=0.1036,
    temperature=0.5695,
    learning_rate=0.000233,
    weight_decay=0.0000077,
    grad_clip=3.0,
    dropout=0.1,
    batch_size=128,
    max_epochs=40,
    ema_decay=0.999,
)

# 2. Ingest Data & Setup DataModule
user_sequences = load_user_sequences("data/Luxury_Beauty_5.txt")
item_embeddings = torch.load("data/sbert_item_embeddings.pt")
train_pairs, val_pairs = generate_causal_interaction_pairs(user_sequences)

datamodule = RefineRecDataModule(
    train_pairs=train_pairs,
    val_pairs=val_pairs,
    num_items=item_embeddings.size(0),
    config=config,
)

# 3. Instantiate Lightning Model & Trainer
model = RefineRecLightning(
    pretrained_sbert_embeddings=item_embeddings,
    config=config,
)

trainer = pl.Trainer(
    max_epochs=config.max_epochs,
    callbacks=[EMACallback(decay=config.ema_decay)],
    accelerator="auto",
)

# 4. Train and Validate
trainer.fit(model, datamodule=datamodule)
trainer.validate(model, datamodule=datamodule)
```

### 3. Interactive notebook & visualization

Explore the step-by-step tensor transformations, architecture, and evaluation charts interactively:

```bash
# Launch interactive portfolio walkthrough
jupyter lab RefineRecLightning.ipynb
```

### 4. Exploratory Bayesian Hyperparameter Optimization

Launch native Weights & Biases Bayesian sweeps with Hyperband early stopping:

```python
from refinerec.hpo import run_hparam_search

# Launch native W&B Bayesian Sweep
sweep_id = run_hparam_search(
    n_trials=40,
    search_epochs=15,
    project_name="refinerec",
)
print(f"Sweep completed. ID: {sweep_id}")
```

### 5. Autonomous end-to-end study pipeline (Kaggle T4 + W&B)

The `automation` module runs a complete, resumable research study in one Kaggle GPU session:

```bash
# Console entrypoint (installed with [automation] extra)
refinerec-final-automation

# Or directly via python module
python automation/final_automation.py
```

The pipeline:
1. Executes 15 focused Bayesian trials ([`automation/sweep-final-focused.yaml`](automation/sweep-final-focused.yaml)) capped at 30 epochs with early stopping.
2. Selects the winning configuration based on peak validation NDCG@10.
3. Validates the winner across a 5-seed confirmation grid (seeds 42–46, 40 epochs).
4. Logs an aggregate summary run (mean ± std, min/max, best seed, best epoch) and results table.
5. Publishes a complete W&B Report with search curves, parameter rankings, and confirmation charts.

---

## Key configurable parameters

All hyperparameters are centralized in `RefineRecConfig`. The **Best-Run Value** column reports the winning configuration discovered by the automated Bayesian optimization study:

| Parameter | Type | Default | Best-Run Value | Description |
| :--- | :---: | :---: | :---: | :--- |
| `embedding_dim` | `int` | `384` | `384` *(default)* | Item semantic feature dimension (SBERT). |
| `max_history_length` | `int` | `50` | `50` *(default)* | Maximum historical interaction sequence length. |
| `outer_steps` | `int` | `7` | **`3`** | Number of outer preference refinement iterations ($T$). |
| `inner_steps` | `int` | `3` | **`4`** | Number of inner recursive evidence updates ($n$). |
| `core_depth` | `int` | `5` | **`6`** | Number of layers in the shared recursive MLP ($f_\phi$). |
| `preference_scale` | `float` | `1.0` | **`0.1036`** | Preference residual update step scale ($L$). |
| `temperature` | `float` | `1.0` | **`0.5695`** | Candidate dot-product logit scaling factor ($\tau$). |
| `candidate_size` | `int` | `100` | `100` *(default)* | Candidate evaluation pool size ($1 \text{ pos} + 99 \text{ neg}$). |
| `learning_rate` | `float` | `1e-3` | **`0.000233`** | Adam optimizer learning rate. |
| `weight_decay` | `float` | `0.0` | **`7.68e-6`** | Adam optimizer $L_2$ weight decay. |
| `grad_clip` | `float` | `1.0` | **`3.0`** | Gradient clipping norm threshold. |
| `dropout` | `float` | `0.0` | **`0.1`** | Dropout probability in the shared core MLP. |
| `batch_size` | `int` | `512` | **`128`** | Mini-batch size for training and validation. |
| `max_epochs` | `int` | `50` | `40` | Maximum training epochs. |
| `ema_decay` | `float` | `0.999` | `0.999` *(default)* | Exponential Moving Average decay factor. |
| `freeze_item_embeddings` | `bool` | `False` | `False` *(default)* | Freeze pretrained item embeddings during training. |
| `num_workers` | `int` | `3` | `3` *(default)* | DataLoader multiprocessing workers. |
| `exclude_history_items_from_negatives` | `bool` | `True` | `True` *(default)* | Filter past user interactions when sampling negatives. |

---

## Testing & quality assurance

```bash
# Run test suite
pytest tests/ -v

# Run code style & lint checks
ruff check refinerec automation tests
```

---

## License

Distributed under the MIT License. See `LICENSE` for details.