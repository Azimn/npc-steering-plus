#!/usr/bin/env python3
"""Analyze FATIGUE probe geometry against V/A/D without loading an LLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib.organism.probe_analysis import cosine, same_layer_vad_projection
from lib.probes import ProbeBundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fatigue-probes", required=True)
    parser.add_argument("--vad-probes", required=True)
    parser.add_argument("--report", default="artifacts/fatigue_probe_geometry.md")
    parser.add_argument("--json", default="artifacts/fatigue_probe_geometry.json")
    parser.add_argument("--min-consistency", type=float, default=0.15)
    parser.add_argument("--min-residual-fraction", type=float, default=0.55)
    args = parser.parse_args()

    fatigue = ProbeBundle.load(args.fatigue_probes)
    vad = ProbeBundle.load(args.vad_probes)
    if fatigue.model_id != vad.model_id:
        raise ValueError(f"Model mismatch: {fatigue.model_id!r} != {vad.model_id!r}")
    if "FATIGUE" not in fatigue.axes:
        raise KeyError("Fatigue bundle does not contain FATIGUE")

    common_layers = sorted(
        set(fatigue.axes["FATIGUE"])
        & set(vad.axes.get("V", {}))
        & set(vad.axes.get("A", {}))
        & set(vad.axes.get("D", {}))
    )
    if not common_layers:
        raise ValueError("No layer contains FATIGUE plus V/A/D directions")

    rows = []
    for layer in common_layers:
        fvec = fatigue.axes["FATIGUE"][layer]
        vad_vectors = {axis: vad.axes[axis][layer] for axis in ("V", "A", "D")}
        projection = same_layer_vad_projection(fvec, vad_vectors)
        diag = fatigue.diagnostics.get("FATIGUE", {}).get(layer, {})
        consistency = float(diag.get("cosine_consistency", 0.0))
        row = {
            "layer": layer,
            "fatigue_cosine_consistency": consistency,
            "fatigue_norm_unnormalised": float(diag.get("norm_unnormalised", 0.0)),
            "cosine_V": cosine(fvec, vad_vectors["V"]),
            "cosine_A": cosine(fvec, vad_vectors["A"]),
            "cosine_D": cosine(fvec, vad_vectors["D"]),
            "vad_residual_fraction": projection.residual_fraction,
            "vad_projection_coefficients": projection.coefficients,
        }
        row["passes_geometry_gate"] = (
            consistency >= args.min_consistency
            and projection.residual_fraction >= args.min_residual_fraction
        )
        row["geometry_score"] = max(0.0, consistency) * projection.residual_fraction
        rows.append(row)

    passing = [row for row in rows if row["passes_geometry_gate"]]
    pool = passing or rows
    recommendation = max(pool, key=lambda row: row["geometry_score"])

    payload = {
        "model_id": fatigue.model_id,
        "fatigue_bundle": str(args.fatigue_probes),
        "vad_bundle": str(args.vad_probes),
        "operational_thresholds": {
            "min_consistency": args.min_consistency,
            "min_residual_fraction": args.min_residual_fraction,
            "note": "These are experiment gating heuristics, not established scientific cutoffs.",
        },
        "rows": rows,
        "geometry_recommendation": {
            "layer": recommendation["layer"],
            "passed_gate": recommendation["passes_geometry_gate"],
            "geometry_score": recommendation["geometry_score"],
        },
    }

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FATIGUE probe geometry",
        "",
        f"Model: {fatigue.model_id}",
        "",
        "This report asks whether the learned FATIGUE direction is largely reducible to the V/A/D subspace at the same transformer layer.",
        "",
        "The residual fraction is the norm of FATIGUE remaining after least-squares projection into the span of V, A, and D, divided by the original FATIGUE norm. A larger value indicates more of the direction lies outside that three-axis subspace.",
        "",
        "| layer | consistency | cos(F,V) | cos(F,A) | cos(F,D) | residual outside VAD | gate |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['layer']} | {row['fatigue_cosine_consistency']:+.3f} | "
            f"{row['cosine_V']:+.3f} | {row['cosine_A']:+.3f} | "
            f"{row['cosine_D']:+.3f} | {row['vad_residual_fraction']:.3f} | "
            f"{'pass' if row['passes_geometry_gate'] else 'hold'} |"
        )
    lines.extend([
        "",
        f"Geometry-only recommendation: layer {recommendation['layer']}.",
        "",
        "This recommendation is not enough to promote the probe. The steering calibration must also show a monotonic FATIGUE readback, acceptable coherence, and tolerable V/A/D leakage.",
        "",
        "The default geometry gate requires contrast-pair cosine consistency >= "
        f"{args.min_consistency:.2f} and V/A/D residual fraction >= {args.min_residual_fraction:.2f}. "
        "Those thresholds are operational heuristics for this experiment, not literature-derived cutoffs.",
    ])
    report_path.write_text("\n".join(lines) + "\n")

    print(f"report: {report_path}")
    print(f"json: {json_path}")
    print(
        "geometry recommendation: "
        f"layer={recommendation['layer']} "
        f"passed={recommendation['passes_geometry_gate']} "
        f"score={recommendation['geometry_score']:.3f}"
    )


if __name__ == "__main__":
    main()
