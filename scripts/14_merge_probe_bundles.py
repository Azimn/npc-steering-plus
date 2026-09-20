#!/usr/bin/env python3
"""Merge validated probe axes into one model-specific ProbeBundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib.probes import ProbeBundle


def parse_selection(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        try:
            axis, layer = value.split("=", 1)
            out[axis] = int(layer)
        except ValueError as exc:
            raise ValueError(f"Invalid --select value {value!r}; expected AXIS=LAYER") from exc
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("addition")
    parser.add_argument("output")
    parser.add_argument("--select", action="append", default=[])
    args = parser.parse_args()

    base = ProbeBundle.load(args.base)
    addition = ProbeBundle.load(args.addition)
    if base.model_id != addition.model_id:
        raise ValueError(f"Model mismatch: {base.model_id!r} != {addition.model_id!r}")

    overlap = set(base.axes) & set(addition.axes)
    if overlap:
        raise ValueError(f"Refusing to overwrite existing axes: {sorted(overlap)}")

    base.axes.update(addition.axes)
    base.diagnostics.update(addition.diagnostics)
    base.candidate_layers = sorted(set(base.candidate_layers) | set(addition.candidate_layers))
    base.n_pairs_per_axis = min(base.n_pairs_per_axis, addition.n_pairs_per_axis)

    selections = dict(addition.selected_layers)
    selections.update(parse_selection(args.select))
    for axis, layer in selections.items():
        if axis not in base.axes:
            raise KeyError(f"Cannot select unknown axis {axis!r}")
        if layer not in base.axes[axis]:
            raise KeyError(f"Axis {axis!r} has no extracted layer {layer}")
        base.selected_layers[axis] = layer

    base.save(args.output)
    print(f"saved merged bundle: {args.output}")
    print(f"axes: {sorted(base.axes)}")
    print(f"selected_layers: {base.selected_layers}")


if __name__ == "__main__":
    main()
