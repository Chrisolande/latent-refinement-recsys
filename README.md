# RecRec: Recursive Refinement for Sequential Recommendation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Lightning 2.0+](https://img.shields.io/badge/Lightning-2.0+-792EE5.svg?logo=lightning&logoColor=white)](https://lightning.ai/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A clean, modular PyTorch & PyTorch Lightning implementation of **RecRec** (*Recursive Refinement for Sequential Recommendation*), featuring a production-ready package structure alongside a self-contained interactive Jupyter Notebook ([`RecRecLightning.ipynb`](RecRecLightning.ipynb)).

---

## Highlights & Core Concepts

* **Recursive Latent Refinement**: Replaces computationally heavy autoregressive transformers with iterative residual state updates over a compact, shared parameter-efficient MLP core.
* **Dual-Loop Dynamics**: Employs an inner recursion loop ($n$ iterations) for evidence synthesis and an outer loop ($T$ iterations) for progressive preference vector trajectory evolution.
* **Evidence-Anchored Correction Gate**: Dynamically modulates state transitions via an input-dependent sigmoid gate $\mathbf{g}_t$, mitigating drift and anchoring predictions to historical user context.
* **Multi-Step Deep Supervision**: Optimizes predictions across all intermediate refinement steps ($t = 1 \dots T$), enforcing monotonic ranking improvements and stable gradient flow.
* **Pretrained Semantic Alignment**: Seamlessly integrates 384-dimensional item semantic embeddings extracted from raw metadata via SentenceTransformers (`all-MiniLM-L6-v2`).
* **Production-Grade Lightning Integration**: Complete with `RecRecDataModule`, `RecRecLightning` module, Exponential Moving Average (`EMACallback`, $\beta = 0.999$), and numerical diagnostic suites.

---

## Architectural Workflow

```mermaid
flowchart TD
    subgraph Data["1. Input & Sequence Encoding"]
        H["User Interaction History [i_1, i_2, ..., i_L]"]
        E["Semantic Item Embeddings (SBERT, d=384)"]
        H -->|Lookup & Mask| HE["History Embeddings Matrix"]
        HE -->|Masked Mean Pooling| X["Context Vector x"]
        X --> Y0["Initial Preference y_0 = x"]
        X --> Z0["Initial Evidence z_0 = 0"]
    end

    subgraph Refinement["2. Outer Refinement Loop (t = 1 ... T)"]
        Y0 & Z0 --> StepT["Step t"]
        
        subgraph InnerLoop["Inner Evidence Loop (j = 1 ... n)"]
            ZPrev["z_t^(j-1)"] --> ConcatInner["[x || y_t || z_t^(j-1)]"]
            ConcatInner --> FPhiInner["Shared Core MLP f_phi"]
            FPhiInner --> ZNext["z_t^(j)"]
        end
        
        StepT --> InnerLoop
        InnerLoop --> ZEvidence["Final Inner Evidence z_t^(n)"]
        
        ZEvidence & X & Y0 --> Gate["Correction Gate: g_t = σ(W_t [x || y_t])"]
        Gate --> ZAnchored["Anchored Evidence: z_t = (1 - g_t) ⊙ z_t^(n) + g_t ⊙ x"]
        
        ZAnchored & X & Y0 --> CoreUpdate["tanh(f_phi([x || y_t || z_t]))"]
        CoreUpdate --> Residual["Preference Update: y_(t+1) = y_t + L · Δ"]
    end

    subgraph Scoring["3. Candidate Scoring & Objectives"]
        Residual --> Logits["Logits s_(t, j) = (y_(t+1) · e_j) / τ"]
        Logits --> Loss["Deep Supervision Loss: (1/T) Σ CrossEntropy"]
        Logits --> Eval["Top-K Ranking: HR@K, NDCG@K, Prec@K"]
    end
```

---

## Mathematical Formulation

### 1. Input Context Encoding
Given an interaction history sequence $S = (i_1, i_2, \dots, i_{|S|})$ and item semantic embeddings $\mathbf{e}_i \in \mathbb{R}^d$:

$$\mathbf{x} = \frac{\sum_{i \in S} m_i \mathbf{e}_i}{\sum_{i \in S} m_i}, \quad \mathbf{y}_0 = \mathbf{x}, \quad \mathbf{z}_0 = \mathbf{0}$$

where $m_i \in \{0, 1\}$ denotes sequence validity mask indicators and $d = 384$.

### 2. Dual-Loop Latent Refinement
For outer refinement step $t \in \{1, \dots, T\}$:

* **Inner Evidence Synthesis ($j = 1, \dots, n$):**
  $$\mathbf{z}_t^{(j)} = f_\phi([\mathbf{x} \parallel \mathbf{y}_t \parallel \mathbf{z}_t^{(j-1)}])$$

* **Evidence-Anchored Correction Gate:**
  $$\mathbf{g}_t = \sigma(\mathbf{W}_t [\mathbf{x} \parallel \mathbf{y}_t])$$
  $$\mathbf{z}_t = (1 - \mathbf{g}_t) \odot \mathbf{z}_t^{(n)} + \mathbf{g}_t \odot \mathbf{x}$$

* **Residual Preference Trajectory Update:**
  $$\mathbf{y}_{t+1} = \mathbf{y}_t + L \cdot \tanh(f_\phi([\mathbf{x} \parallel \mathbf{y}_t \parallel \mathbf{z}_t]))$$

Where:
* $f_\phi: \mathbb{R}^{3d} \to \mathbb{R}^d$ is a depth-$D$ multilayer perceptron with LayerNorm and ReLU activations, shared across all recursive evaluations.
* $\mathbf{W}_t \in \mathbb{R}^{d \times 2d}$ is a step-specific linear gating transformation.
* $L$ denotes the preference residual scaling factor ($L = 1.0$).

### 3. Candidate Scoring & Multi-Step Deep Supervision
For a candidate set of item embeddings $\{\mathbf{e}_j\}_{j=1}^K$ ($1$ ground truth target + $K-1$ sampled negatives) and temperature $\tau$:

$$s_{t, j} = \frac{\mathbf{y}_{t+1}^\top \mathbf{e}_j}{\tau}$$

$$\mathcal{L}_{\text{total}} = \frac{1}{T} \sum_{t=1}^T \mathcal{L}_{\text{CE}}(\mathbf{s}_t, \text{target})$$

---

## Repository Layout

```
.
├── .gitattributes                       # Linguist language override (*.ipynb -> Python)
├── pyproject.toml                       # Build system, package metadata, and dependency specs
├── README.md                            # Comprehensive architectural documentation
├── RecRecLightning.ipynb                # Self-contained, executable Jupyter notebook
└── recrec/                              # Modular production package
    ├── __init__.py                      # Public API exports
    ├── callbacks.py                     # Exponential Moving Average callback (EMACallback)
    ├── config.py                        # Dataclass hyperparameters (RecRecConfig)
    ├── data.py                          # Causal pair generation, negative sampling, DataModule
    ├── diagnostics.py                   # Paper assertions & single-batch sanity check
    ├── embeddings.py                    # Offline SBERT metadata embedding extraction
    ├── lightning_module.py              # PyTorch Lightning module (RecRecLightning)
    ├── losses.py                        # Multi-step deep supervision cross-entropy loss
    ├── metrics.py                       # Top-K ranking evaluation metrics (HR@k, NDCG@k, Prec@k)
    ├── modules.py                       # Core PyTorch modules (InputEncoding, CoreRecursionMLP, RecRec)
    └── train.py                         # End-to-end training pipeline and CLI entrypoint
```

---

## Getting Started

### Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/Chrisolande/recursive-refinement-recsys.git
cd recursive-refinement-recsys

# Install via pip
pip install -e .

# Or install via uv
uv pip install -e .
```

### Pretrained Item Feature Extraction

To extract 384-dimensional dense semantic vectors from text metadata:

```python
from recrec.embeddings import extract_sbert_item_embeddings

extract_sbert_item_embeddings(
    interaction_path="data/Luxury_Beauty_5.txt",
    metadata_path="data/Luxury_Beauty_5_text_name_dict.pkl",
    output_path="data/sbert_item_embeddings.pt",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
```

---

## Training & Evaluation Workflows

### 1. Command Line Interface (CLI)
Execute full training with diagnostics and validation using either the console script or module runner:

```bash
# Using installed entrypoint
recrec-train

# Or directly via python module
python -m recrec.train
```

### 2. Standalone Jupyter Notebook
An end-to-end executable notebook containing data ingestion, model architecture, training loop, and evaluation charts is available at [`RecRecLightning.ipynb`](RecRecLightning.ipynb).

### 3. Programmatic Lightning Pipeline

```python
import pytorch_lightning as pl
import torch
from recrec import (
    EMACallback,
    RecRecConfig,
    RecRecDataModule,
    RecRecLightning,
    load_user_sequences,
    make_train_val_pairs,
)

# 1. Hyperparameter Configuration
config = RecRecConfig(
    embedding_dim=384,
    outer_steps=7,
    inner_steps=3,
    core_depth=5,
    candidate_size=100,
    learning_rate=1e-3,
    batch_size=512,
    max_epochs=50,
    ema_decay=0.999,
)

# 2. Ingest Data & Setup DataModule
user_sequences = load_user_sequences("data/Luxury_Beauty_5.txt")
item_embeddings = torch.load("data/sbert_item_embeddings.pt")
train_pairs, val_pairs = make_train_val_pairs(user_sequences)

datamodule = RecRecDataModule(
    train_pairs=train_pairs,
    val_pairs=val_pairs,
    num_items=item_embeddings.size(0),
    config=config,
)

# 3. Instantiate Lightning Model & Trainer
model = RecRecLightning(
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

---

## Configuration Reference

All hyperparameters are centralized in `RecRecConfig`:

| Parameter | Type | Default | Math Symbol | Description |
| :--- | :---: | :---: | :---: | :--- |
| `embedding_dim` | `int` | `384` | $d$ | Item semantic feature dimension |
| `max_history_length` | `int` | `50` | $L_{\max}$ | Maximum historical interaction sequence length |
| `outer_steps` | `int` | `7` | $T$ | Number of outer refinement iterations |
| `inner_steps` | `int` | `3` | $n$ | Number of inner recursive evidence updates |
| `core_depth` | `int` | `5` | $D$ | Depth of shared MLP core network $f_\phi$ |
| `preference_scale` | `float` | `1.0` | $L$ | Preference residual update step scale |
| `temperature` | `float` | `1.0` | $\tau$ | Candidate dot-product logit scaling factor |
| `candidate_size` | `int` | `100` | $K$ | Candidate evaluation pool size ($1 \text{ pos} + 99 \text{ neg}$) |
| `learning_rate` | `float` | `1e-3` | $\eta$ | Adam optimizer learning rate |
| `batch_size` | `int` | `512` | $B$ | Mini-batch size for training and validation |
| `max_epochs` | `int` | `50` | — | Maximum training epochs |
| `ema_decay` | `float` | `0.999` | $\beta$ | Exponential Moving Average decay factor |
| `freeze_item_embeddings` | `bool` | `False` | — | Freeze pretrained item embeddings during training |
| `num_workers` | `int` | `3` | — | DataLoader multiprocessing workers |
| `exclude_history_items_from_negatives` | `bool` | `True` | — | Filter past user interactions when sampling negatives |

---

## Evaluation Protocol

Evaluation adheres strictly to standard leave-one-out ranking protocol over sampled 100-item candidate pools ($1$ ground truth + $99$ uniform negatives):

* **Hit Ratio ($HR@K$)**:
  $$\text{HR}@K = \frac{1}{|U|} \sum_{u \in U} \mathbb{I}(\text{rank}_u \le K)$$

* **Normalized Discounted Cumulative Gain ($NDCG@K$)**:
  $$\text{NDCG}@K = \frac{1}{|U|} \sum_{u \in U} \frac{\mathbb{I}(\text{rank}_u \le K)}{\log_2(\text{rank}_u + 1)}$$

* **Precision ($Prec@K$)**:
  $$\text{Prec}@K = \frac{1}{|U|} \sum_{u \in U} \frac{\mathbb{I}(\text{rank}_u \le K)}{K}$$

All metrics are evaluated and reported across top-$K$ cutoffs $K \in \{1, 5, 10\}$.

---

## License

This project is open-source under the [MIT License](LICENSE).
