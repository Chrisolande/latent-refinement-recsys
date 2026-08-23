import pickle
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


def extract_sbert_item_embeddings(
    interaction_path: str | Path,
    metadata_path: str | Path,
    output_path: str | Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 512,
    device: str | None = None,
) -> torch.Tensor:
    from .data import load_user_sequences

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    user_sequences = load_user_sequences(interaction_path)
    item_ids = sorted({i for seq in user_sequences.values() for i in seq})

    expected_ids = list(range(len(item_ids)))
    if item_ids != expected_ids:
        raise ValueError("Item IDs must be contiguous 0..N-1 before SBERT extraction.")

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    id_to_title = (
        metadata["title"] if isinstance(metadata, dict) and "title" in metadata else metadata
    )

    texts = []
    for item_id in item_ids:
        text = id_to_title.get(item_id, "unknown item")
        if text is None or not str(text).strip():
            text = "unknown item"
        texts.append(str(text))

    sbert = SentenceTransformer(model_name).to(device)

    chunks = []
    for start in tqdm(range(0, len(texts), batch_size), desc="SBERT"):
        batch_texts = texts[start : start + batch_size]
        chunks.append(
            sbert.encode(
                batch_texts,
                convert_to_tensor=True,
                device=device,
                normalize_embeddings=True,
            )
        )

    item_embeddings = torch.cat(chunks, dim=0).to(device)

    if item_embeddings.shape != (len(item_ids), 384):
        raise RuntimeError(f"Expected ({len(item_ids)}, 384), got {tuple(item_embeddings.shape)}")

    if not torch.isfinite(item_embeddings).all():
        raise RuntimeError("SBERT embedding matrix contains non-finite values.")

    torch.save(item_embeddings, output_path)
    return item_embeddings
