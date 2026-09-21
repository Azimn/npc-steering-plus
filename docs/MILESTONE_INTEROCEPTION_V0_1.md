# Milestone: Virtual Body Interoception v0.1

This milestone establishes the study boundary and the minimum substrate for the synthetic interoception experiments.

## Added

- New research branch: virtual-body-interoception-v0.1
- Generalized denoised difference-of-means helper inspired by the Pain Axis methodology
- Unit tests for denoising, projection, cosine overlap, and fail-closed behavior
- Virtual-body axis manifest
- First-class thirst state
- First-class bipolar thermal state
- Research plan separating this study from the later Game AI deployment paper
- Pain Axis attribution and adaptation notes

## Axis priority

1. FATIGUE
2. HUNGER
3. THIRST
4. THERMAL
5. PAIN_REFERENCE as a positive control

Pain is intentionally not the first novelty target. The first use of pain is to verify that our intervention harness can reproduce an established result on an exact supported model.

## Current state

The persistent organism now stores the body variables needed for the planned study. The latent translation layer is still validated only for the original V/A/D directions. Direct FATIGUE has an existing candidate pipeline but previously failed its calibration gate and should be re-extracted using the stronger target-vs-controls method before promotion.

HUNGER, THIRST, and THERMAL latent directions do not yet exist.

## Next acceptance gate

Phase 0 is complete only when we can:

1. run an exact Pain Axis reference model, initially Qwen2.5-7B-Instruct;
2. load or reproduce the published pain direction;
3. apply the intervention through our own harness;
4. reproduce a clear causal effect without putting pain state into the active prompt;
5. record the exact model, layer, vector provenance, coefficient, seed, prompt, and output.

Only then should we begin interpreting new FATIGUE, HUNGER, THIRST, or THERMAL vectors.

## Next implementation tasks

- Add a PyTorch/Hugging Face backend suitable for the Pain Axis reference models.
- Add an adapter that can import published Pain Axis vectors into our SteeringProfile abstraction.
- Add a reference-pain reproduction script.
- Build target and control datasets for FATIGUE.
- Add held-out layer selection rather than choosing a layer on the same examples used to fit a vector.
- Generalize the causal ablation script from FATIGUE-specific to arbitrary body axes.
