"""Axis-agnostic latent-direction helpers for synthetic interoception.

The denoising recipe is adapted conceptually from Tagliabue, Dung, and Berg
(2026), "The Pain Axis: LLMs Represent Self-Directed Harm and Act to Relieve
It" (arXiv:2609.16247). Their released MIT-licensed implementation computes a
difference of means and projects out principal components of the control
activation cloud. This module reimplements that method generically in NumPy
so it can be used for FATIGUE, HUNGER, THIRST, THERMAL, and reference PAIN
axes without hard-coding any one construct.

This module operates on already-extracted residual-stream activations. Model
hooks and tokenization remain backend-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class DenoisedDirection:
    """Result of target-minus-control direction extraction."""

    direction: np.ndarray
    raw_direction: np.ndarray
    removed_component: np.ndarray
    control_mean: np.ndarray
    target_mean: np.ndarray
    n_removed_components: int
    control_variance_removed: float
    residual_fraction: float


def _as_matrix(name: str, values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix of shape (samples, hidden_dim)")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one sample and one feature")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return matrix


def denoised_difference_of_means(
    target_activations: np.ndarray,
    control_activations: np.ndarray,
    *,
    variance_to_remove: float = 0.50,
) -> DenoisedDirection:
    """Return a unit latent direction after removing dominant control variance.

    The raw direction is mean(target) - mean(control). PCA is performed only
    on the centered control cloud. Enough leading control principal components
    are removed to account for variance_to_remove of control variance.

    variance_to_remove=0 disables denoising.
    """

    if not 0.0 <= variance_to_remove < 1.0:
        raise ValueError("variance_to_remove must be in [0, 1)")

    target = _as_matrix("target_activations", target_activations)
    control = _as_matrix("control_activations", control_activations)
    if target.shape[1] != control.shape[1]:
        raise ValueError("target and control hidden dimensions must match")

    target_mean = target.mean(axis=0)
    control_mean = control.mean(axis=0)
    raw = target_mean - control_mean
    residual = raw.copy()
    removed_variance = 0.0
    n_removed = 0

    centered = control - control_mean
    if variance_to_remove > 0.0 and control.shape[0] > 1 and np.linalg.norm(centered) > 1e-12:
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        variances = singular_values ** 2
        total_variance = float(variances.sum())
        if total_variance > 1e-12:
            fractions = variances / total_variance
            cumulative = np.cumsum(fractions)
            n_removed = min(
                int(np.searchsorted(cumulative, variance_to_remove, side="left")) + 1,
                len(vt),
            )
            removed_variance = float(cumulative[n_removed - 1])
            basis = vt[:n_removed]
            residual = residual - basis.T @ (basis @ residual)

    raw_norm = float(np.linalg.norm(raw))
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm <= 1e-12:
        raise ValueError("denoising removed the entire candidate direction")

    direction = residual / residual_norm
    removed_component = raw - residual
    residual_fraction = residual_norm / raw_norm if raw_norm > 1e-12 else 0.0

    return DenoisedDirection(
        direction=direction.astype(np.float32),
        raw_direction=raw.astype(np.float32),
        removed_component=removed_component.astype(np.float32),
        control_mean=control_mean.astype(np.float32),
        target_mean=target_mean.astype(np.float32),
        n_removed_components=n_removed,
        control_variance_removed=removed_variance,
        residual_fraction=float(residual_fraction),
    )


def projection_scores(activations: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Project activation rows onto a direction."""

    acts = _as_matrix("activations", activations)
    vec = np.asarray(direction, dtype=np.float64)
    if vec.ndim != 1 or vec.shape[0] != acts.shape[1]:
        raise ValueError("direction must be a 1D vector matching hidden_dim")
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise ValueError("direction must be non-zero")
    return (acts @ (vec / norm)).astype(np.float64)


def direction_cosines(
    direction: np.ndarray,
    controls: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Cosine similarity between one candidate axis and named control axes."""

    target = np.asarray(direction, dtype=np.float64)
    target_norm = float(np.linalg.norm(target))
    if target.ndim != 1 or target_norm <= 1e-12:
        raise ValueError("direction must be a non-zero 1D vector")

    out: dict[str, float] = {}
    for name, value in controls.items():
        vec = np.asarray(value, dtype=np.float64)
        if vec.shape != target.shape:
            raise ValueError(f"control direction {name!r} has incompatible shape")
        denom = target_norm * float(np.linalg.norm(vec))
        out[name] = 0.0 if denom <= 1e-12 else float(np.dot(target, vec) / denom)
    return out
