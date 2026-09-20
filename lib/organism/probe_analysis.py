"""Geometry helpers for comparing learned activation directions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SubspaceProjection:
    projected: np.ndarray
    residual: np.ndarray
    coefficients: dict[str, float]
    residual_fraction: float


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def project_into_subspace(
    target: np.ndarray,
    basis_vectors: Mapping[str, np.ndarray],
) -> SubspaceProjection:
    if not basis_vectors:
        raise ValueError("basis_vectors must not be empty")

    names = list(basis_vectors)
    target64 = np.asarray(target, dtype=np.float64)
    matrix = np.column_stack(
        [np.asarray(basis_vectors[name], dtype=np.float64) for name in names]
    )
    coefficients, *_ = np.linalg.lstsq(matrix, target64, rcond=None)
    projected = matrix @ coefficients
    residual = target64 - projected
    target_norm = float(np.linalg.norm(target64))
    residual_fraction = (
        float(np.linalg.norm(residual)) / target_norm
        if target_norm > 1e-12
        else 0.0
    )
    return SubspaceProjection(
        projected=projected.astype(np.float32),
        residual=residual.astype(np.float32),
        coefficients={name: float(value) for name, value in zip(names, coefficients)},
        residual_fraction=residual_fraction,
    )


def same_layer_vad_projection(
    fatigue_vector: np.ndarray,
    vad_vectors: Mapping[str, np.ndarray],
) -> SubspaceProjection:
    expected = {"V", "A", "D"}
    missing = expected - set(vad_vectors)
    if missing:
        raise KeyError(f"Missing V/A/D basis axes: {sorted(missing)}")
    return project_into_subspace(
        fatigue_vector,
        {axis: vad_vectors[axis] for axis in ("V", "A", "D")},
    )
