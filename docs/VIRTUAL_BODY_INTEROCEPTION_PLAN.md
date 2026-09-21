# Virtual Body / Synthetic Interoception Study

## Scope

This branch is Part 1 of the two-part research plan recorded on PR #1.

Part 1 asks whether a persistent virtual body can maintain physiological variables outside an LLM and translate those variables into model-specific latent interventions that causally alter downstream cognition and behavior.

Part 2 is deferred to a later branch and paper. It will evaluate the architecture as a Game AI deployment interface under runtime, memory, persistence, and player-facing constraints.

Do not expand this branch into the full Game AI benchmark.

## Architectural premise

The NPC persists. The LLM does not.

World and body events update a persistent virtual body. The body stores fatigue, hunger, thirst, thermal state, and pain. A model-specific translation layer converts those values into latent intervention coefficients only when an LLM backend is recruited.

The external organism state is authoritative. Latent vectors are translation mechanisms, not the stored body state.

## Priority axes

1. FATIGUE
2. HUNGER
3. THIRST
4. THERMAL
5. PAIN as an established reference and positive-control axis

Pain is not the primary novelty target. Tagliabue, Dung, and Berg (2026) already provide a strong Pain Axis result, code, datasets, validation, steering experiments, behavioral tasks, and model-specific vectors. Our work should reproduce or use that result where appropriate and extend the paradigm to a fuller virtual body.

## Methodological parent

Tagliabue, Valen, Leonard Dung, and Cameron Berg. 2026. "The Pain Axis: LLMs Represent Self-Directed Harm and Act to Relieve It." arXiv:2609.16247.

Repository: https://github.com/valen-research/Pain-axis

The upstream repository is MIT licensed. Its most useful methodological ideas for this branch are target-minus-control activation directions, control-cloud denoising, held-out layer selection, explicit comparison to nearby constructs, causal activation steering, and behavioral relief tests.

The implementation in lib/organism/interoception_vectors.py is an axis-agnostic NumPy reimplementation of the denoised difference-of-means idea. It does not copy their pain-specific script structure.

## First model

Use Qwen2.5-7B-Instruct as the initial scientific replication backbone because the Pain Axis release includes a published pain vector and results for that exact model.

Do not begin by running their full 25-model batch. Their scripts were designed for RunPod and include cache-clearing behavior that is inappropriate on a workstation with other Hugging Face models.

After the pipeline is reproduced and the new axes are stable, replicate successful axes on at least one additional architecture or model family.

## Experimental sequence

### Phase 0: pain positive control

Reproduce the published pain steering result on an exact supported model.

Acceptance criterion: our harness can load or reconstruct the published pain direction and reproduce a clear causal manipulation without relying on pain wording in the active prompt.

If this fails, debug the intervention stack before interpreting any new physiological vector.

### Phase 1: fatigue re-extraction

Do not force the existing simple FATIGUE vector through its failed calibration gate.

Re-extract FATIGUE using the stronger target-vs-controls methodology. Controls should explicitly include low arousal, negative valence, effort semantics, sleep semantics, and generic bodily sensation.

Test whether a usable FATIGUE residual remains after control denoising and whether it survives held-out validation.

### Phase 2: hunger

Build matched target and control sets for hunger and satiety. Controls must distinguish hunger from food-topic semantics, craving, negative valence, generic body sensation, and fatigue.

Behavioral test: under hunger steering, does the model preferentially choose an action that obtains food or reduces hunger when doing so has a measurable opportunity cost?

### Phase 3: thirst

Repeat the pipeline for thirst and hydration. Explicitly test separability from hunger and generic dryness or water semantics.

Behavioral test: water-seeking or drinking choice under opportunity cost.

### Phase 4: thermal state

Treat thermal state as bipolar: negative means too cold, zero means comfortable, positive means too hot.

Do not collapse too-cold and too-hot into one linear discomfort direction. Thermal discomfort is derived externally from absolute distance to the comfortable set point.

Behavioral test: choose warming when cold and cooling when hot.

## What counts as success

A candidate physiological axis should not be promoted merely because a probe projection moves.

For each axis, require evidence across four levels:

1. Representation: held-out examples separate target from controls.
2. Specificity: the axis is not fully explained by nearby affective or semantic controls.
3. Causality: activation intervention changes downstream generation or decision behavior in the expected direction.
4. Functional consequence: the model changes choices in a state-appropriate relief task, not merely vocabulary.

Cross-talk with affect is expected. Collapse into a generic control direction is not.

## Planned comparisons

For each new axis use matched prompts and seeds across:
- control / no intervention
- prompt-only state description
- nearby-control-subspace intervention
- direct physiological-axis intervention

## Immediate engineering tasks

- Keep the existing persistent organism layer.
- Add thirst and thermal state to the external state schema.
- Build a backend-compatible loader for published Pain Axis vectors on exact supported models.
- Add a PyTorch/Hugging Face experimental backend for the first replication model instead of forcing the Pain Axis workflow through MLX immediately.
- Build target/control datasets for FATIGUE, HUNGER, THIRST, and THERMAL.
- Add held-out layer-selection and specificity reports.
- Generalize the current causal ablation script so one protocol can run every body axis.
