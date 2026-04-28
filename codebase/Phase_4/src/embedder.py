"""Embedding wrapper for evidence ranking."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


class EvidenceEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-dot-v1") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def rank(self, query: str, texts: List[str], top_k: int | None = None) -> List[Tuple[int, float]]:
        if not texts:
            return []
        top_k = top_k or len(texts)
        q = self.model.encode([query], normalize_embeddings=True)
        x = self.model.encode(texts, normalize_embeddings=True)
        scores = np.dot(x, q[0])
        order = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in order]
