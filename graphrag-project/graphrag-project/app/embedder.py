"""Step 3: local embeddings with sentence-transformers (no API key needed)."""
from typing import List

from .config import EMBED_MODEL

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    vecs = _get_model().encode(texts, batch_size=32, normalize_embeddings=True,
                               show_progress_bar=False)
    return [v.tolist() for v in vecs]


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]
