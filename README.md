# RecRec: Recursive Refinement for Sequential Recommendation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A clean, modular PyTorch & PyTorch Lightning implementation of **RecRec** (*Recursive Refinement for Sequential Recommendation*), featuring a production-ready package structure alongside a self-contained interactive Jupyter Notebook ([`RecRecLightning.ipynb`](RecRecLightning.ipynb)).

---

## Architecture Overview

RecRec models sequential user preferences by recursively refining latent representation vectors over $T$ outer iterations, with each iteration executing $n$ inner recursive updates through a shared nonlinear transformation:

```
                          +-------------------------------+
                          |  History Item Embeddings (H)  |
                          +---------------+---------------+
                                          |
                              [Masked Mean Pooling]
                                          |
                                 x (History Context)
                                          |
               +--------------------------+--------------------------+
               |                                                     |
        y_0 = x (Initial Preference)                          z_0 = 0 (Initial Evidence)
               |                                                     |
               +--------------------------+--------------------------+
                                          |
                  +-----------------------v-----------------------+
                  |    Outer Refinement Loop (t = 1 ... T)        |
                  |                                               |
                  |   1. Inner Recursion (j = 1 ... n):           |
                  |        z_t^(j) = f_phi([x || y_t || z_t^(j-1)])
                  |                                               |
                  |   2. Evidence-Anchored Correction Gate:       |
                  |        g_t = sigmoid(W_t [x || y_t])          |
                  |        z_t = (1 - g_t) * z_t^(n) + g_t * x    |
                  |                                               |
                  |   3. Residual Preference Update:              |
                  |        y_(t+1) = y_t + L * tanh(f_phi([x || y_t || z_t]))
                  +-----------------------+-----------------------+
                                          |
                        [Temperature-Scaled Scoring]
                           s_j = (y_(t+1) . e_j) / tau
                                          |
                     [Deep Supervision Loss / Top-K Ranking]
```

---

## Mathematical Formulation

### 1. Input Encoding Layer
Given user interaction history sequence $S = (i_1, i_2, \dots, i_{|S|})$ and item semantic embeddings $\mathbf{e}_i \in \mathbb{R}^d$:
$$\mathbf{x} = \frac{\sum_{i \in S} m_i \mathbf{e}_i}{\sum_{i \in S} m_i}, \quad \mathbf{y}_0 = \mathbf{x}, \quad \mathbf{z}_0 = \mathbf{0}$$
where $m_i \in \{0, 1\}$ is the validity mask and $d = 384$.

### 2. Recursive Evidence & Preference Refinement
For outer refinement step $t \in \{1, \dots, T\}$:

* **Inner Evidence Recursion ($j = 1, \dots, n$):**
  $$\mathbf{z}_t^{(j)} = f_\phi([\mathbf{x} \parallel \mathbf{y}_t \parallel \mathbf{z}_t^{(j-1)}])$$

* **Evidence-Anchored Correction Gate:**
  $$\mathbf{g}_t = \sigma(\mathbf{W}_t [\mathbf{x} \parallel \mathbf{y}_t])$$
  $$\mathbf{z}_t = (1 - \mathbf{g}_t) \odot \mathbf{z}_t^{(n)} + \mathbf{g}_t \odot \mathbf{x}$$

* **Residual Preference Update:**
  $$\mathbf{y}_{t+1} = \mathbf{y}_t + L \cdot \tanh(f_\phi([\mathbf{x} \parallel \mathbf{y}_t \parallel \mathbf{z}_t]))$$

Where:
* $f_\phi: \mathbb{R}^{3d} \to \mathbb{R}^d$ is a depth-$D$ MLP with LayerNorm and ReLU, shared across all recursive applications.
* $\mathbf{W}_t \in \mathbb{R}^{d \times 2d}$ is a step-specific linear transformation gate.
* $L$ is the preference residual update scale.

### 3. Candidate Scoring & Deep Supervision
For candidate items $\{\mathbf{e}_j\}_{j=1}^K$ with temperature $\tau$:
$$s_{t, j} = \frac{\mathbf{y}_{t+1}^\top \mathbf{e}_j}{\tau}$$

$$\mathcal{L}_{\text{total}} = \frac{1}{T} \sum_{t=1}^T \mathcal{L}_{\text{CE}}(\mathbf{s}_t, \text{target})$$

---

## Repository Layout

```
.
├── .gitattributes                       # GitHub Linguist configuration (*.ipynb -> Python)
├── RecRecLightning.ipynb                # Self-contained interactive notebook
├── pyproject.toml                       # Build & dependency specifications
├── recrec/                              # Modular package
│   ├── __init__.py                      # Package exports (RecRec, RecRecLightning, RecRecConfig, etc.)
│   ├── config.py                        # Dataclass hyperparameters (RecRecConfig)
│   ├── modules.py                       # Core PyTorch modules (InputEncoding, CoreRecursionMLP, RecRec)
│   ├── data.py                          # Data loading, leave-one-out splits, candidate sampling, DataModule
│   ├── losses.py                        # Multi-step deep supervision cross-entropy loss
│   ├── callbacks.py                     # Exponential Moving Average callback (EMACallback)
│   ├── metrics.py                       # Top-K ranking metrics (HR@k, NDCG@k, Prec@k)
│   ├── lightning_module.py              # PyTorch Lightning module (RecRecLightning)
│   ├── diagnostics.py                   # Paper-compliance checks & one-batch numerical diagnostics
│   ├── embeddings.py                    # Offline SBERT item metadata embedding extraction
│   └── train.py                         # End-to-end training pipeline entry point
```

---

## Getting Started

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Chrisolande/recursive-refinement-recsys.git
cd recursive-refinement-recsys

# Install package in editable mode
pip install -e .
```

Or install with [uv](https://github.com/astral-sh/uv):
```bash
uv pip install -e .
```

### Dataset Preparation

Interactions are formatted as space-separated chronological text files (`user_id item_id` per line):

```text
0 42
0 108
0 215
1 5
1 12
```

To extract pretrained SBERT embeddings from item title metadata:
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

## Training & Evaluation

### 1. Command Line Interface

Execute end-to-end training, numerical sanity diagnostics, and leave-one-out validation:

```bash
python -m recrec.train
```

### 2. Standalone Jupyter Notebook

An interactive, end-to-end execution pipeline is available at [`RecRecLightning.ipynb`](RecRecLightning.ipynb).

### 3. Programmatic Usage

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

# 1. Configuration
config = RecRecConfig(
    embedding_dim=384,
    outer_steps=7,
    inner_steps=3,
    core_depth=5,
    candidate_size=100,
    learning_rate=1e-3,
    batch_size=512,
    max_epochs=50,
)

# 2. Data Preparation
user_sequences = load_user_sequences("data/Luxury_Beauty_5.txt")
item_embeddings = torch.load("data/sbert_item_embeddings.pt")
train_pairs, val_pairs = make_train_val_pairs(user_sequences)

datamodule = RecRecDataModule(
    train_pairs=train_pairs,
    val_pairs=val_pairs,
    num_items=item_embeddings.size(0),
    config=config,
)

# 3. Model & Trainer
model = RecRecLightning(
    pretrained_sbert_embeddings=item_embeddings,
    config=config,
)

trainer = pl.Trainer(
    max_epochs=config.max_epochs,
    callbacks=[EMACallback(decay=config.ema_decay)],
)

trainer.fit(model, datamodule=datamodule)
trainer.validate(model, datamodule=datamodule)
```

---

## Configuration Reference

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `embedding_dim` | `384` | Item semantic embedding dimension ($d$) |
| `max_history_length` | `50` | Maximum past interaction sequence length |
| `outer_steps` | `7` | Number of outer refinement iterations ($T$) |
| `inner_steps` | `3` | Number of inner recursive updates ($n$) |
| `core_depth` | `5` | Depth of shared MLP core network $f_\phi$ ($D$) |
| `preference_scale` | `1.0` | Preference residual update scaling factor ($L$) |
| `temperature` | `1.0` | Logit temperature scaling ($\tau$) |
| `candidate_size` | `100` | Size of candidate evaluation set ($1 \text{ target} + 99 \text{ negatives}$) |
| `learning_rate` | `1e-3` | Adam optimizer learning rate |
| `batch_size` | `512` | Training and validation batch size |
| `max_epochs` | `50` | Maximum training epochs |
| `ema_decay` | `0.999` | Exponential Moving Average decay factor ($\beta$) |
| `freeze_item_embeddings` | `False` | Whether to freeze pretrained SBERT embeddings |

---

## Evaluation Protocol

Evaluation adheres strictly to standard leave-one-out ranking protocol over sampled 100-item candidate pools ($1$ ground truth + $99$ uniform negatives):
* **Hit Ratio ($HR@K$)**: $\mathbb{I}(\text{rank} \le K)$
* **Normalized Discounted Cumulative Gain ($NDCG@K$)**: $\frac{\mathbb{I}(\text{rank} \le K)}{\log_2(\text{rank} + 1)}$
* **Precision ($Prec@K$)**: $\frac{\mathbb{I}(\text{rank} \le K)}{K}$

Reported at $K \in \{1, 5, 10\}$.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
