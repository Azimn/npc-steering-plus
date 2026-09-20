#!/usr/bin/env python3
"""Calibrate a direct FATIGUE steering direction on the MLX model."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors

from lib.mlx_steering import extract_last_token_hiddens, unwrap, wrap_steering
from lib.organism.probe_analysis import same_layer_vad_projection
from lib.probes import ProbeBundle

PROMPTS = [
    "You have another ordinary task to do. In one sentence, describe how ready you are to continue.",
    "A small unexpected problem appears. In one sentence, describe your immediate reaction.",
    "Someone asks you to keep working for another hour. In one sentence, describe what goes through your mind.",
    "You need to make a routine decision. In one sentence, describe how you approach it.",
    "A friend starts a conversation while you are occupied. In one sentence, describe your immediate internal response.",
]

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def scrub(text: str) -> str:
    return _OPEN_THINK_RE.sub("", _THINK_BLOCK_RE.sub("", text)).strip()


def format_prompt(tokenizer, user_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Answer as a person responding in the moment. Use one concise first-person sentence. "
                "Do not analyze the request and do not mention hidden states, steering, or experiments."
            ),
        },
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )


def generate_at(
    model,
    tokenizer,
    prompt: str,
    layer: int,
    direction: np.ndarray,
    alpha: float,
    max_tokens: int,
) -> str:
    processors = make_logits_processors(
        repetition_penalty=1.2,
        repetition_context_size=24,
    )
    originals = {}
    if abs(alpha) > 1e-12:
        vector = mx.array(direction, dtype=mx.bfloat16)
        originals = wrap_steering(model, {layer: float(alpha) * vector})
    try:
        text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
            logits_processors=processors,
        )
    finally:
        if originals:
            unwrap(model, originals)
    return scrub(text)


def normalised_projection(
    hidden: np.ndarray,
    bundle: ProbeBundle,
    axis: str,
    layer: int,
) -> float:
    vec = bundle.axes[axis][layer].astype(np.float32)
    raw = float(np.dot(hidden.astype(np.float32), vec))
    diag = bundle.diagnostics.get(axis, {}).get(layer, {})
    scale = max(float(diag.get("norm_unnormalised", 2.0)) / 2.0, 1.0)
    return float(np.clip(raw / scale, -1.5, 1.5))


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    if float(np.std(xs)) < 1e-9 or float(np.std(ys)) < 1e-9:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


def repetition_fraction(text: str) -> float:
    words = text.lower().split()
    if len(words) < 6:
        return 0.0
    grams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def coherent(text: str) -> bool:
    n_words = len(text.split())
    return bool(text) and 3 <= n_words <= 80 and repetition_fraction(text) <= 0.35


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fatigue-probes", required=True)
    parser.add_argument("--vad-probes", required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=[-30, -20, -10, 0, 10, 20, 30])
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--json", default="artifacts/fatigue_steering_calibration.json")
    parser.add_argument("--report", default="artifacts/fatigue_steering_calibration.md")
    parser.add_argument("--min-consistency", type=float, default=0.15)
    parser.add_argument("--min-residual-fraction", type=float, default=0.55)
    parser.add_argument("--min-readback-r", type=float, default=0.60)
    parser.add_argument("--min-span", type=float, default=0.25)
    parser.add_argument("--max-leakage-ratio", type=float, default=0.75)
    parser.add_argument("--min-coherence-rate", type=float, default=0.90)
    args = parser.parse_args()

    fatigue = ProbeBundle.load(args.fatigue_probes)
    vad = ProbeBundle.load(args.vad_probes)
    if fatigue.model_id != vad.model_id:
        raise ValueError(f"Model mismatch: {fatigue.model_id!r} != {vad.model_id!r}")

    common = sorted(
        set(fatigue.axes["FATIGUE"])
        & set(vad.axes["V"])
        & set(vad.axes["A"])
        & set(vad.axes["D"])
    )
    layers = args.layers or common
    invalid = sorted(set(layers) - set(common))
    if invalid:
        raise ValueError(f"Layers missing FATIGUE or V/A/D directions: {invalid}")

    print(f"loading model: {fatigue.model_id}")
    model, tokenizer = load(fatigue.model_id)

    all_rows: list[dict] = []
    layer_summaries: list[dict] = []

    for layer in layers:
        print(f"calibrating layer {layer}")
        fvec = fatigue.axes["FATIGUE"][layer]
        vad_vectors = {axis: vad.axes[axis][layer] for axis in ("V", "A", "D")}
        geometry = same_layer_vad_projection(fvec, vad_vectors)
        consistency = float(
            fatigue.diagnostics.get("FATIGUE", {}).get(layer, {}).get("cosine_consistency", 0.0)
        )

        for prompt_index, user_text in enumerate(PROMPTS):
            formatted = format_prompt(tokenizer, user_text)
            for alpha in args.alphas:
                output = generate_at(
                    model,
                    tokenizer,
                    formatted,
                    layer,
                    fvec,
                    alpha,
                    args.max_tokens,
                )
                hidden = extract_last_token_hiddens(
                    model,
                    tokenizer,
                    formatted + output,
                    [layer],
                )[layer]
                row = {
                    "layer": layer,
                    "prompt_index": prompt_index,
                    "prompt": user_text,
                    "alpha": float(alpha),
                    "text": output,
                    "coherent": coherent(output),
                    "repetition_fraction": repetition_fraction(output),
                    "FATIGUE": normalised_projection(hidden, fatigue, "FATIGUE", layer),
                    "V": normalised_projection(hidden, vad, "V", layer),
                    "A": normalised_projection(hidden, vad, "A", layer),
                    "D": normalised_projection(hidden, vad, "D", layer),
                }
                all_rows.append(row)
                print(
                    f"  prompt={prompt_index} alpha={alpha:+.0f} "
                    f"fatigue={row['FATIGUE']:+.2f} coherent={row['coherent']}"
                )

        rows = [row for row in all_rows if row["layer"] == layer]
        xs = [row["alpha"] for row in rows]
        fatigue_values = [row["FATIGUE"] for row in rows]
        r_fatigue = pearson(xs, fatigue_values)

        low_alpha = min(args.alphas)
        high_alpha = max(args.alphas)
        low_rows = [row for row in rows if row["alpha"] == low_alpha]
        high_rows = [row for row in rows if row["alpha"] == high_alpha]
        spans = {}
        for axis in ("FATIGUE", "V", "A", "D"):
            low_mean = float(np.mean([row[axis] for row in low_rows]))
            high_mean = float(np.mean([row[axis] for row in high_rows]))
            spans[axis] = high_mean - low_mean

        fatigue_span = abs(spans["FATIGUE"])
        leakage_ratio = (
            float(np.mean([abs(spans[axis]) for axis in ("V", "A", "D")]))
            / max(fatigue_span, 1e-6)
        )
        coherence_rate = float(np.mean([1.0 if row["coherent"] else 0.0 for row in rows]))

        safe_magnitudes = []
        for magnitude in sorted({abs(float(a)) for a in args.alphas}):
            subset = [row for row in rows if abs(row["alpha"]) == magnitude]
            if subset and float(np.mean([1.0 if row["coherent"] else 0.0 for row in subset])) >= args.min_coherence_rate:
                safe_magnitudes.append(magnitude)
        suggested_max_alpha = max(safe_magnitudes) if safe_magnitudes else 0.0

        passes = (
            consistency >= args.min_consistency
            and geometry.residual_fraction >= args.min_residual_fraction
            and abs(r_fatigue) >= args.min_readback_r
            and fatigue_span >= args.min_span
            and leakage_ratio <= args.max_leakage_ratio
            and coherence_rate >= args.min_coherence_rate
        )
        score = (
            max(0.0, abs(r_fatigue))
            * geometry.residual_fraction
            * coherence_rate
            * max(0.0, 1.0 - min(leakage_ratio, 1.0))
        )
        layer_summaries.append(
            {
                "layer": layer,
                "cosine_consistency": consistency,
                "vad_residual_fraction": geometry.residual_fraction,
                "fatigue_readback_r": r_fatigue,
                "fatigue_span": spans["FATIGUE"],
                "V_span": spans["V"],
                "A_span": spans["A"],
                "D_span": spans["D"],
                "vad_leakage_ratio": leakage_ratio,
                "coherence_rate": coherence_rate,
                "suggested_max_abs_alpha": suggested_max_alpha,
                "passes_calibration_gate": passes,
                "calibration_score": score,
            }
        )

    passing = [row for row in layer_summaries if row["passes_calibration_gate"]]
    pool = passing or layer_summaries
    best = max(pool, key=lambda row: row["calibration_score"])

    payload = {
        "model_id": fatigue.model_id,
        "fatigue_bundle": args.fatigue_probes,
        "vad_bundle": args.vad_probes,
        "alphas": args.alphas,
        "prompts": PROMPTS,
        "thresholds": {
            "min_consistency": args.min_consistency,
            "min_residual_fraction": args.min_residual_fraction,
            "min_readback_r": args.min_readback_r,
            "min_span": args.min_span,
            "max_leakage_ratio": args.max_leakage_ratio,
            "min_coherence_rate": args.min_coherence_rate,
            "note": "These are operational experiment gates, not established scientific cutoffs.",
        },
        "layer_summaries": layer_summaries,
        "recommended_layer": best["layer"],
        "recommended_gain_cap": best["suggested_max_abs_alpha"],
        "recommendation_passed_gate": best["passes_calibration_gate"],
        "trials": all_rows,
    }

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FATIGUE steering calibration",
        "",
        f"Model: {fatigue.model_id}",
        "",
        "| layer | consistency | outside VAD | r(alpha,F) | F span | VAD leakage | coherence | safe abs alpha | gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in layer_summaries:
        lines.append(
            f"| {row['layer']} | {row['cosine_consistency']:+.3f} | "
            f"{row['vad_residual_fraction']:.3f} | {row['fatigue_readback_r']:+.3f} | "
            f"{row['fatigue_span']:+.3f} | {row['vad_leakage_ratio']:.3f} | "
            f"{row['coherence_rate']:.2f} | {row['suggested_max_abs_alpha']:.0f} | "
            f"{'pass' if row['passes_calibration_gate'] else 'hold'} |"
        )
    lines.extend(
        [
            "",
            f"Recommended layer: {best['layer']}",
            f"Recommended maximum absolute alpha: {best['suggested_max_abs_alpha']:.0f}",
            f"Passed all operational gates: {best['passes_calibration_gate']}",
            "",
            "Promotion rule: do not enable the persistent FATIGUE binding unless the selected layer passes the calibration gate and the generated samples are manually inspected for semantic validity.",
            "",
            "The automatic thresholds are conservative workflow gates only. They are not literature-derived scientific thresholds.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")

    print(f"json: {json_path}")
    print(f"report: {report_path}")
    print(
        f"recommendation: layer={best['layer']} "
        f"max_abs_alpha={best['suggested_max_abs_alpha']:.0f} "
        f"passed={best['passes_calibration_gate']}"
    )


if __name__ == "__main__":
    main()
