"""MLX bridge from generic steering coefficients to npc-steering probes."""

from __future__ import annotations

from typing import Mapping, TYPE_CHECKING

from ..probes import ProbeBundle

if TYPE_CHECKING:
    import mlx.core as mx


def build_mlx_offsets(
    probes: ProbeBundle,
    steering_alphas: Mapping[str, float],
) -> dict[int, "mx.array"]:
    import mlx.core as mx

    missing = sorted(axis for axis in steering_alphas if axis not in probes.selected_layers)
    if missing:
        raise KeyError(
            "Steering profile requested probe axes that are not selected in "
            f"this ProbeBundle: {missing}"
        )

    offsets: dict[int, mx.array] = {}
    for axis, alpha in steering_alphas.items():
        if float(alpha) == 0.0:
            continue
        layer = probes.selected_layers[axis]
        vec = mx.array(probes.vec(axis, layer), dtype=mx.bfloat16)
        contribution = float(alpha) * vec
        offsets[layer] = offsets.get(layer, mx.zeros_like(vec)) + contribution
    return offsets
