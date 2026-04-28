"""Utilities for loading TruthfulQA source URLs by ID."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict


def load_source_map(path: str | None) -> Dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {str(r.get("id")): str(r.get("source", "")) for r in data if r.get("id")}
        if isinstance(data, dict):
            out = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    out[str(k)] = str(v.get("source", ""))
                else:
                    out[str(k)] = str(v)
            return out
    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return {str(row.get("id")): str(row.get("source", "")) for row in reader if row.get("id")}
    raise ValueError(f"Unsupported source map file: {p}")
