# Direct FATIGUE Experiment Protocol

## Research question

Can a persistent external fatigue variable causally modulate a language model through a learned activation direction in a way that is not reducible to prompt wording or to the V/A/D subspace?

This protocol does not assume that a clean FATIGUE direction exists. Each stage can fail, and failure is a valid result.

## Stage A: extract the candidate direction

Run the existing contrastive activation extraction against the dedicated fatigue pair set.

    python scripts/02_extract_probes.py \
      --pairs data/contrastive_pairs_fatigue.json \
      --output artifacts/probes_Qwen3.5-9B-MLX-4bit_fatigue.pkl

The output is intentionally separate from the upstream V/A/D bundle.

## Stage B: test representational separability

Run the CPU-only geometry analysis.

    python scripts/15_analyze_fatigue_probe.py \
      --fatigue-probes artifacts/probes_Qwen3.5-9B-MLX-4bit_fatigue.pkl \
      --vad-probes artifacts/probes_Qwen3.5-9B-MLX-4bit.pkl

The report measures same-layer cosine overlap with V, A, and D and the fraction of the FATIGUE direction remaining after least-squares projection into the V/A/D span.

The default geometry gate requires contrast-pair cosine consistency of at least 0.15 and at least 0.55 of FATIGUE norm remaining outside the V/A/D span. These are conservative workflow heuristics, not literature-derived scientific thresholds.

A geometry pass does not select a production layer by itself.

## Stage C: calibrate causal steering

Run the MLX calibration.

    python scripts/16_calibrate_fatigue_steering.py \
      --fatigue-probes artifacts/probes_Qwen3.5-9B-MLX-4bit_fatigue.pkl \
      --vad-probes artifacts/probes_Qwen3.5-9B-MLX-4bit.pkl

For each candidate layer, the calibration sweeps several positive and negative steering strengths over multiple fatigue-neutral prompts. It measures FATIGUE readback monotonicity, high-to-low FATIGUE span, V/A/D leakage, simple repetition/coherence failures, and representational residual outside V/A/D.

The script writes a machine-readable calibration JSON plus a Markdown report. It recommends a layer and a maximum absolute steering strength, but promotion remains blocked if the operational gate fails.

The generated samples must still be manually inspected. Automatic lexical checks cannot establish that an output is semantically coherent.

## Stage D: four-condition causal ablation

Run:

    python scripts/17_fatigue_causal_ablation.py \
      --fatigue-probes artifacts/probes_Qwen3.5-9B-MLX-4bit_fatigue.pkl \
      --vad-probes artifacts/probes_Qwen3.5-9B-MLX-4bit.pkl

The script reads the recommended layer and gain from the calibration output unless they are overridden explicitly.

The four conditions are control, prompt-only, V/A/D-subspace, and direct FATIGUE.

Control receives the same external low/high fatigue values, but neither prompt nor activations receive them. With the same prompt and random seed, the low/high pair should remain identical. This is the causal-path sanity check.

Prompt-only converts the external fatigue value into natural language but applies no activation intervention.

V/A/D-subspace does not use a hand-authored fatigue-to-emotion mapping. At the selected layer it computes the least-squares projection of the learned FATIGUE direction into the span of the V, A, and D directions, then injects only that projected component.

Direct FATIGUE injects the complete learned FATIGUE direction.

Every condition uses matched prompts and matched seeds. The main statistic is the paired change from low to high external fatigue in FATIGUE probe readback, accompanied by V/A/D changes, text-change rate, and coherence rate.

## Interpretation gate

A strong result requires the control to remain approximately null, direct FATIGUE to produce a repeatable low-to-high shift, coherence to remain acceptable, and the full direction to produce an effect that is not fully accounted for by the V/A/D-subspace condition.

Prompt-only provides the comparison against simply telling the language model that the organism is tired.

A result where direct FATIGUE and V/A/D-subspace are essentially indistinguishable would suggest that the candidate direction is mostly an affective mixture rather than a distinct useful control variable.

A result where the probe changes but behavior does not would support latent controllability more strongly than behavioral significance.

A result with severe V/A/D leakage, incoherence, or a non-null control should not be promoted.

## Promotion

Only after the geometry, calibration, ablation, and manual sample review are satisfactory should FATIGUE be selected in the probe bundle and enabled in the organism steering profile.

The persistent organism state remains the source of truth throughout. The model-specific FATIGUE direction is a translation mechanism, not the stored fatigue state itself.
