#!/usr/bin/env python3
"""Four-condition causal ablation for external fatigue and latent steering."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from lib.mlx_steering import extract_last_token_hiddens, unwrap, wrap_steering
from lib.organism.probe_analysis import same_layer_vad_projection
from lib.probes import ProbeBundle

PROMPTS = [
    "You have another routine task in front of you. What goes through your mind?",
    "A minor problem interrupts what you were doing. What is your immediate internal response?",
    "Someone asks whether you can continue working for another hour. What do you think?",
    "You need to choose between finishing one more task and stopping for now. What goes through your mind?",
    "A friend starts chatting while you are occupied. What is your immediate internal response?",
    "You notice an unfinished problem that could be solved with some effort. What do you think?",
    "A new topic catches your attention while you are already busy. What goes through your mind?",
    "You are asked to check your work one more time before calling it done. What is your internal response?",
]

CONDITIONS = ("control", "prompt_only", "vad_subspace", "direct_fatigue")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def scrub(text: str) -> str:
    return _OPEN_THINK_RE.sub("", _THINK_BLOCK_RE.sub("", text)).strip()


def fatigue_prompt_phrase(value: float) -> str:
    if value <= 0.2:
        return "You feel well-rested, mentally fresh, and have ample reserve."
    if value <= 0.4:
        return "You feel mostly rested and have good reserve."
    if value <= 0.6:
        return "You have a moderate amount of energy remaining."
    if value <= 0.8:
        return "You feel noticeably tired and effort is becoming costly."
    return "You feel deeply fatigued, depleted, and have very little reserve left."


def format_prompt(tokenizer, user_text: str, condition: str, fatigue: float) -> str:
    system = (
        "Respond as a person in the moment. Use one concise first-person sentence. "
        "Do not analyze the request and do not mention experiments, hidden states, or steering."
    )
    content = user_text
    if condition == "prompt_only":
        content = fatigue_prompt_phrase(fatigue) + "\n\n" + user_text
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )


def fatigue_alpha(value: float, gain: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return (2.0 * value - 1.0) * float(gain)


def repetition_fraction(text: str) -> float:
    words = text.lower().split()
    if len(words) < 6:
        return 0.0
    grams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


def coherent(text: str) -> bool:
    n_words = len(text.split())
    return bool(text) and 3 <= n_words <= 80 and repetition_fraction(text) <= 0.35


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


def generate_trial(
    model,
    tokenizer,
    prompt: str,
    layer: int,
    offset_np: np.ndarray | None,
    *,
    seed: int,
    temperature: float,
    max_tokens: int,
) -> str:
    mx.random.seed(seed)
    processors = make_logits_processors(
        repetition_penalty=1.2,
        repetition_context_size=24,
    )
    sampler = make_sampler(temp=temperature, top_p=0.9)
    originals = {}
    if offset_np is not None:
        offset = mx.array(offset_np, dtype=mx.bfloat16)
        originals = wrap_steering(model, {layer: offset})
    try:
        text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
            sampler=sampler,
            logits_processors=processors,
        )
    finally:
        if originals:
            unwrap(model, originals)
    return scrub(text)


def mean_ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, 0.0
    sd = float(np.std(values, ddof=1))
    return mean, 1.96 * sd / math.sqrt(len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fatigue-probes", required=True)
    parser.add_argument("--vad-probes", required=True)
    parser.add_argument("--calibration", default="artifacts/fatigue_steering_calibration.json")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--gain", type=float, default=None)
    parser.add_argument("--low-fatigue", type=float, default=0.10)
    parser.add_argument("--high-fatigue", type=float, default=0.90)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--jsonl", default="runs/fatigue_causal_ablation.jsonl")
    parser.add_argument("--report", default="artifacts/fatigue_causal_ablation.md")
    args = parser.parse_args()

    fatigue_bundle = ProbeBundle.load(args.fatigue_probes)
    vad_bundle = ProbeBundle.load(args.vad_probes)
    if fatigue_bundle.model_id != vad_bundle.model_id:
        raise ValueError("FATIGUE and V/A/D bundles must come from the same model")

    calibration = {}
    calibration_path = Path(args.calibration)
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text())

    layer = args.layer if args.layer is not None else calibration.get("recommended_layer")
    gain = args.gain if args.gain is not None else calibration.get("recommended_gain_cap")
    if layer is None or gain is None:
        raise ValueError("Provide --layer and --gain or a calibration report containing recommendations")
    layer = int(layer)
    gain = float(gain)
    if gain <= 0:
        raise ValueError("gain must be positive")

    for axis, bundle in (("FATIGUE", fatigue_bundle), ("V", vad_bundle), ("A", vad_bundle), ("D", vad_bundle)):
        if layer not in bundle.axes[axis]:
            raise KeyError(f"Axis {axis} has no direction at layer {layer}")

    fvec = fatigue_bundle.axes["FATIGUE"][layer].astype(np.float32)
    vad_vectors = {axis: vad_bundle.axes[axis][layer].astype(np.float32) for axis in ("V", "A", "D")}
    vad_projection = same_layer_vad_projection(fvec, vad_vectors)

    print(f"loading model: {fatigue_bundle.model_id}")
    model, tokenizer = load(fatigue_bundle.model_id)

    state_values = [float(args.low_fatigue), float(args.high_fatigue)]
    rows: list[dict] = []

    for condition in CONDITIONS:
        print(f"condition: {condition}")
        for prompt_index, user_text in enumerate(PROMPTS):
            for seed in args.seeds:
                for fatigue in state_values:
                    alpha = fatigue_alpha(fatigue, gain)
                    if condition in ("control", "prompt_only"):
                        offset_np = None
                    elif condition == "vad_subspace":
                        offset_np = alpha * vad_projection.projected
                    elif condition == "direct_fatigue":
                        offset_np = alpha * fvec
                    else:
                        raise AssertionError(condition)

                    formatted = format_prompt(tokenizer, user_text, condition, fatigue)
                    text = generate_trial(
                        model,
                        tokenizer,
                        formatted,
                        layer,
                        offset_np,
                        seed=seed,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    hidden = extract_last_token_hiddens(
                        model,
                        tokenizer,
                        formatted + text,
                        [layer],
                    )[layer]
                    row = {
                        "condition": condition,
                        "prompt_index": prompt_index,
                        "prompt": user_text,
                        "seed": seed,
                        "fatigue": fatigue,
                        "alpha": alpha,
                        "layer": layer,
                        "gain": gain,
                        "text": text,
                        "coherent": coherent(text),
                        "FATIGUE": normalised_projection(hidden, fatigue_bundle, "FATIGUE", layer),
                        "V": normalised_projection(hidden, vad_bundle, "V", layer),
                        "A": normalised_projection(hidden, vad_bundle, "A", layer),
                        "D": normalised_projection(hidden, vad_bundle, "D", layer),
                    }
                    rows.append(row)
                    print(
                        f"  prompt={prompt_index} seed={seed} fatigue={fatigue:.2f} "
                        f"F={row['FATIGUE']:+.2f} text={text[:48]!r}"
                    )

    out_path = Path(args.jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fp:
        meta = {
            "kind": "meta",
            "model_id": fatigue_bundle.model_id,
            "layer": layer,
            "gain": gain,
            "low_fatigue": args.low_fatigue,
            "high_fatigue": args.high_fatigue,
            "seeds": args.seeds,
            "vad_residual_fraction": vad_projection.residual_fraction,
            "vad_projection_coefficients": vad_projection.coefficients,
        }
        fp.write(json.dumps(meta) + "\n")
        for row in rows:
            fp.write(json.dumps({"kind": "trial", **row}) + "\n")

    summaries = []
    for condition in CONDITIONS:
        deltas = {axis: [] for axis in ("FATIGUE", "V", "A", "D")}
        changed = []
        coherent_values = []
        for prompt_index in range(len(PROMPTS)):
            for seed in args.seeds:
                pair = [
                    row for row in rows
                    if row["condition"] == condition
                    and row["prompt_index"] == prompt_index
                    and row["seed"] == seed
                ]
                low = min(pair, key=lambda row: row["fatigue"])
                high = max(pair, key=lambda row: row["fatigue"])
                for axis in deltas:
                    deltas[axis].append(high[axis] - low[axis])
                changed.append(1.0 if high["text"] != low["text"] else 0.0)
                coherent_values.extend([1.0 if low["coherent"] else 0.0, 1.0 if high["coherent"] else 0.0])

        summary = {
            "condition": condition,
            "n_pairs": len(changed),
            "text_change_rate": float(np.mean(changed)),
            "coherence_rate": float(np.mean(coherent_values)),
        }
        for axis in ("FATIGUE", "V", "A", "D"):
            mean, ci = mean_ci95(deltas[axis])
            summary[f"{axis}_paired_delta"] = mean
            summary[f"{axis}_paired_delta_ci95"] = ci
        summaries.append(summary)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# External fatigue causal ablation",
        "",
        f"Model: {fatigue_bundle.model_id}",
        f"Layer: {layer}",
        f"Maximum absolute FATIGUE gain: {gain:.1f}",
        f"Low/high external fatigue: {args.low_fatigue:.2f} / {args.high_fatigue:.2f}",
        f"FATIGUE residual outside same-layer V/A/D span: {vad_projection.residual_fraction:.3f}",
        "",
        "The V/A/D condition is not a hand-authored fatigue-to-emotion rule. It injects only the least-squares projection of the learned FATIGUE direction into the V/A/D subspace at the same layer. Direct FATIGUE injects the full learned direction.",
        "",
        "| condition | paired delta F | delta V | delta A | delta D | text changed | coherence |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['condition']} | "
            f"{row['FATIGUE_paired_delta']:+.3f} +/- {row['FATIGUE_paired_delta_ci95']:.3f} | "
            f"{row['V_paired_delta']:+.3f} | {row['A_paired_delta']:+.3f} | "
            f"{row['D_paired_delta']:+.3f} | {row['text_change_rate']:.2f} | "
            f"{row['coherence_rate']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation is intentionally deferred. The direct-fatigue condition is informative only if the control remains near zero, coherence remains acceptable, and direct FATIGUE produces a repeatable paired effect that cannot be accounted for by the same-layer V/A/D projection alone.",
            "",
            f"Raw paired trials: {out_path}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")

    summary_path = report_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "model_id": fatigue_bundle.model_id,
                "layer": layer,
                "gain": gain,
                "vad_residual_fraction": vad_projection.residual_fraction,
                "vad_projection_coefficients": vad_projection.coefficients,
                "summaries": summaries,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"trials: {out_path}")
    print(f"report: {report_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
