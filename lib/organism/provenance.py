"""Reproducibility metadata for activation-steering experiments."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentProvenance:
    model_id: str
    model_revision: str
    tokenizer_revision: str
    dtype: str
    backend: str
    vector_source: str
    vector_sha256: str
    layer: int
    dose_ratio: float
    seed: int
    prompt_id: str
    code_commit: str
    hardware: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
