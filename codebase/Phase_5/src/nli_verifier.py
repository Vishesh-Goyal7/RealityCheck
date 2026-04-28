"""NLI verifier wrapper for RealityCheck Phase 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sentence_transformers import CrossEncoder


@dataclass
class NLIScores:
    entailment: float
    contradiction: float
    neutral: float


class NLIVerifier:
    """Runs NLI over evidence/claim pairs using a sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base") -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
        self.label_mapping = self._detect_label_mapping()

    def _detect_label_mapping(self) -> Dict[str, int]:
        """Best-effort label mapping for common NLI cross-encoders."""
        config = getattr(self.model.model, "config", None)
        id2label = getattr(config, "id2label", None) or {}
        normalized = {int(i): str(label).lower() for i, label in id2label.items()}

        mapping: Dict[str, int] = {}
        for idx, label in normalized.items():
            if "entail" in label:
                mapping["entailment"] = idx
            elif "contrad" in label:
                mapping["contradiction"] = idx
            elif "neutral" in label:
                mapping["neutral"] = idx

        # Fallback for many MNLI-style models: contradiction, entailment, neutral
        # Some models differ, but config usually catches it.
        if set(mapping) != {"entailment", "contradiction", "neutral"}:
            mapping = {"contradiction": 0, "entailment": 1, "neutral": 2}

        return mapping

    @staticmethod
    def _softmax(logits: Sequence[float]) -> np.ndarray:
        arr = np.asarray(logits, dtype=np.float64)
        arr = arr - np.max(arr)
        exp = np.exp(arr)
        return exp / np.sum(exp)

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[NLIScores]:
        """Return NLI probabilities for premise/hypothesis pairs."""
        if not pairs:
            return []

        raw = self.model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
        results: List[NLIScores] = []

        for row in raw:
            probs = self._softmax(row)
            results.append(
                NLIScores(
                    entailment=float(probs[self.label_mapping["entailment"]]),
                    contradiction=float(probs[self.label_mapping["contradiction"]]),
                    neutral=float(probs[self.label_mapping["neutral"]]),
                )
            )
        return results
