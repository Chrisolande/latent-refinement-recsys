# RecRec: Recursive Refinement for Sequential Recommendation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clean, modular PyTorch & PyTorch Lightning implementation of **RecRec** (*Recursive Refinement for Sequential Recommendation*), alongside a self-contained interactive Jupyter Notebook (`RecRecLightning.ipynb`).

---

## Architecture Overview

RecRec models sequential user preferences through iterative recursive refinement across $T$ outer steps, each with $n$ inner recursive updates:

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

### 1. Input Representation
Given a user interaction history $H = (i_1, i_2, \dots, i_{|H|})$ with item embeddings $e_i \in \mathbb{R}^d$:
$$\mathbf{x} = \frac{\sum_{i=1}^{|H|} m_i e_i}{\sum_{i=1}^{|H|} m_i}, \quad \mathbf{y}_0 = \mathbf{x}, \quad \mathbf{z}_0 = \mathbf{0}$$
where $m_i \in \{0, 1\}$ denotes sequence validity mask.

### 2. Recursive Evidence & Preference Refinement
For outer step $t \in \{1, \dots, T\}$:
* **Inner Evidence Recursion ($j = 1 \dots n$):**
  $$\mathbf{z}_t^{(j)} = f_\phi([\mathbf{x} \parallel \mathbf{y}_t \parallel \mathbf{z}_t^{(j-1)}])$$
* **Evidence-Anchored Gate:**
  $$\mathbf{g}_t = \sigma(\mathbf{W}_t [\mathbf{x} \parallel \mathbf{y}_t])$$
  $$\mathbf{z}_t = (1 - \mathbf{g}_t) \odot \mathbf{z}_t^{(n)} + \mathbf{g}_t \odot \mathbf{x}$$
* **Residual Preference Update:**
  $$\mathbf{y}_{t+1} = \mathbf{y}_t + L \cdot \tanh(f_\phi([\mathbf{x} \parallel \mathbf{y}_t \parallel \mathbf{z}_t]))$$

Where $f_\phi: \mathbb{R}^{3d} \to \mathbb{R}^d$ is a shared MLP with LayerNorm and ReLU, and $\mathbf{W}_t \in \mathbb{R}^{d \times 2d}$ is a step-specific linear projection.

### 3. Candidate Scoring & Deep Supervision Loss
Given candidate items $\{e_j\}_{j=1}^K$ with temperature $\tau$:
$$s_{t, j} = \frac{\mathbf{y}_{t+1}^\top \mathbf{e}_j}{\tau}$$

$$\mathcal{L}_{\text{total}} = \frac{1}{T} \sum_{t=1}^T \mathcal{L}_{\text{CE}}(\mathbf{s}_t, \text{target})$$

---

## Repository Layout

```
.
├── RecRecLightning                      # Self-contained executable notebook
├── recrec/                              # Modular package
│   ├── __init__.py                      # Public package exports
│   ├── config.py                        # Dataclass hyperparameters (RecRecConfig)
│   ├── modules.py                       # Pure PyTorch modules (InputEncoding, CoreRecursionMLP, RecRec)
│   ├── data.py                          # Sequences, causal pairs, candidate sampling, DataModule
│   ├── losses.py                        # Deep supervision averaged cross-entropy
│   ├── callbacks.py                     # Exponential Moving Average (EMACallback, decay=0.999)
│   ├── metrics.py                       # Evaluation ranking metrics (HR@k, NDCG@k, Prec@k)
│   ├── lightning_module.py              # PyTorch Lightning module (RecRecLightning)
│   ├── diagnostics.py                   # Compliance assertions & single-batch sanity check
│   ├── embeddings.py                    # SBERT offline item embedding extraction
│   └── train.py                         # End-to-end training pipeline entry point
```

---

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/Chrisolande/recursive-refinement-recsys.git
cd recursive-refinement-recsys

# Install dependencies with pip or uv
pip install torch pytorch-lightning sentence-transformers numpy tqdm
```

### Dataset Preparation

Interactions should be formatted as space-separated text files (`user_id item_id` per line, sorted chronologically):

```text
0 42
0 108
0 215
1 5
1 12
```

To extract pretrained SBERT embeddings:
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

Run end-to-end training and evaluation:
```bash
python -m recrec.train
```

### 2. Standalone Jupyter Notebook

An interactive, all-in-one execution pipeline is available at [`RecRecLightning`](RecRecLightning.ipynb).

### 3. Programmatic Usage

```python
import torch
from recrec import RecRec, RecRecConfig, RecRecLightning
from recrec.data import RecRecDataModule

# Initialize configuration
config = RecRecConfig(
    embedding_dim=384,
    outer_steps=7,
    inner_steps=3,
    core_depth=5,
    candidate_size=100,
)

# Load SBERT embeddings
embeddings = torch.load("data/sbert_item_embeddings.pt")

# Initialize Lightning model
model = RecRecLightning(pretrained_sbert_embeddings=embeddings, config=config)
```

---

## Evaluation Metrics

Models are evaluated under strict leave-one-out recommendation with sampled negative candidate sets:
* **Hit Ratio (HR@k)**: Proportion of times target item appears in top-$k$.
* **NDCG@k**: Position-discounted ranking metric.
* **Precision@k**: $\frac{\mathbb{I}(\text{rank} \le k)}{k}$.

Evaluated at $k \in \{1, 5, 10\}$.
