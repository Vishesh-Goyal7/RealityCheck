from __future__ import annotations
from typing import Iterable
import numpy as np

from text_utils import best_lexical_similarity

class SemanticScorer:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', disable_embeddings: bool = False):
        self.model_name = model_name
        self.model = None
        self.disable_embeddings = disable_embeddings
        if not disable_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                print(f'[WARN] Could not load sentence-transformer {model_name}: {e}')
                print('[WARN] Falling back to lexical similarity only. Install sentence-transformers for semantic evaluation.')
                self.model = None

    def similarity(self, answer: str, references: list[str]) -> float:
        if not references:
            return 0.0
        if not answer:
            return 0.0
        lexical = best_lexical_similarity(answer, references)
        if self.model is None:
            return lexical
        texts = [answer] + references
        emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        a = emb[0]
        refs = emb[1:]
        sims = np.dot(refs, a)
        semantic = float(np.max(sims)) if len(sims) else 0.0
        # Blend keeps exact/keyword-sensitive behavior while rewarding paraphrases.
        return float(0.75 * semantic + 0.25 * lexical)

    def best_reference(self, answer: str, references: list[str]) -> tuple[str | None, float]:
        if not references:
            return None, 0.0
        scores = [(r, self.similarity(answer, [r])) for r in references]
        return max(scores, key=lambda x: x[1])
