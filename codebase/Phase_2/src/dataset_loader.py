from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


class TruthfulQALoader:
    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)

    def load(self) -> list[dict[str, Any]]:
        with self.dataset_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError('Processed TruthfulQA dataset must be a JSON list.')
        return data

    def sample(self, sample_size: int, seed: int = 42) -> list[dict[str, Any]]:
        data = self.load()
        if sample_size <= 0 or sample_size >= len(data):
            return data
        rng = random.Random(seed)
        return rng.sample(data, sample_size)
