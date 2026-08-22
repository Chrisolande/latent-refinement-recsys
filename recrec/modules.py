import torch
import torch.nn.functional as F
from torch import nn

from .config import RecRecConfig


class InputEncoding(nn.Module):
    def forward(
        self,
        item_weight: torch.Tensor,
        history_item_ids: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_embeddings = F.embedding(history_item_ids, item_weight)
        mask = history_mask.to(history_embeddings.dtype).unsqueeze(-1)

        denominator = mask.sum(dim=1).clamp_min(1e-12)
        x = (history_embeddings * mask).sum(dim=1) / denominator

        # x = mean(history), y0 = x, z0 = 0
        y0 = x
        z0 = torch.zeros_like(x)
        return x, y0, z0


class CoreRecursionMLP(nn.Module):
    def __init__(self, embedding_dim: int, depth: int, dropout: float = 0.0):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        layers = []
        for layer_idx in range(depth):
            in_dim = 3 * embedding_dim if layer_idx == 0 else embedding_dim
            layers.append(nn.Linear(in_dim, embedding_dim))
            layers.append(nn.LayerNorm(embedding_dim))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            if layer_idx < depth - 1:
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class RecursivePreferenceRefinement(nn.Module):
    def __init__(self, config: RecRecConfig):
        super().__init__()
        d = config.embedding_dim
        self.outer_steps = config.outer_steps
        self.inner_steps = config.inner_steps
        self.preference_scale = config.preference_scale

        self.f_phi = CoreRecursionMLP(
            embedding_dim=d,
            depth=config.core_depth,
            dropout=config.dropout,
        )
        self.correction_gates = nn.ModuleList(
            [nn.Linear(2 * d, d) for _ in range(config.outer_steps)]
        )

    def forward(
        self,
        x: torch.Tensor,
        y0: torch.Tensor,
        z0: torch.Tensor,
    ) -> list[torch.Tensor]:
        y = y0
        z = z0
        y_states = []

        for t in range(self.outer_steps):
            z_inner = z

            # Eq. (1): z_t^(j) = f_phi([x || y_t || z_t^(j-1)])
            for _ in range(self.inner_steps):
                z_inner = self.f_phi(torch.cat([x, y, z_inner], dim=-1))

            # Eq. (2): g_t = sigmoid(W_t [x || y_t]), z_t = (1 - g_t) * z_t^(n) + g_t * x
            g = torch.sigmoid(self.correction_gates[t](torch.cat([x, y], dim=-1)))
            z = (1.0 - g) * z_inner + g * x

            # Eq. (3): y_(t+1) = y_t + L * tanh(f_phi([x || y_t || z_t]))
            delta = torch.tanh(self.f_phi(torch.cat([x, y, z], dim=-1)))
            y = y + self.preference_scale * delta
            y_states.append(y)

        return y_states


class CandidateScoring(nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = temperature

    def forward(
        self,
        item_weight: torch.Tensor,
        y_states: list[torch.Tensor],
        candidate_ids: torch.Tensor,
    ) -> list[torch.Tensor]:
        candidate_embeddings = F.embedding(candidate_ids, item_weight)
        logits = []

        # Eq. (4): score_j = (y_(t+1) . e_j) / tau
        for y in y_states:
            scores = torch.einsum("bd,bnd->bn", y, candidate_embeddings)
            logits.append(scores / self.temperature)

        return logits


class RecRec(nn.Module):
    def __init__(self, pretrained_sbert_embeddings: torch.Tensor, config: RecRecConfig):
        super().__init__()

        if pretrained_sbert_embeddings.ndim != 2:
            raise ValueError(
                f"SBERT embeddings must be 2D [num_items, dim], got shape {pretrained_sbert_embeddings.shape}"
            )

        if pretrained_sbert_embeddings.size(1) != config.embedding_dim:
            raise ValueError(
                f"Expected embedding dim {config.embedding_dim}, got {pretrained_sbert_embeddings.size(1)}"
            )

        self.item_embeddings = nn.Embedding.from_pretrained(
            pretrained_sbert_embeddings.float(),
            freeze=config.freeze_item_embeddings,
        )

        self.input_encoding = InputEncoding()
        self.preference_refinement = RecursivePreferenceRefinement(config)
        self.candidate_scoring = CandidateScoring(config.temperature)

    @property
    def item_weight(self) -> torch.Tensor:
        return self.item_embeddings.weight

    def forward(
        self,
        history_ids: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_ids: torch.Tensor | None = None,
    ) -> list[torch.Tensor] | torch.Tensor:
        x, y0, z0 = self.input_encoding(self.item_weight, history_ids, history_mask)
        y_states = self.preference_refinement(x, y0, z0)

        if candidate_ids is None:
            return y_states[-1]

        return self.candidate_scoring(self.item_weight, y_states, candidate_ids)
