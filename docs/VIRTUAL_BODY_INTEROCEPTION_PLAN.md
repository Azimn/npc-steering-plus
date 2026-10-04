# Virtual Body / Homeostatic Steering Study v0.2

## Scope

This branch revises the earlier synthetic-interoception prototype after critical review.

Part 1 asks a narrower and more defensible question:

Can a persistent simulated physiology regulate behavior through model-specific latent control channels while remaining external to, and more persistent than, the language model substrate?

Part 2 remains deferred. It will evaluate the architecture as a Game AI deployment mechanism under runtime, memory, persistence, scaling, and player-facing constraints.

## Stronger terminology

The core mechanism in this branch is **virtual physiological-state steering**.

The broader architecture can support **synthetic interoception** because authoritative physiological state is now separated from an interoceptive observation layer. Claims should distinguish those two levels.

Do not claim biological interoception, subjective sensation, or human-equivalent physiology.

## Architectural premise

The NPC persists. The LLM does not.

The causal loop is:

world/environment -> authoritative body state -> interoceptive observation -> motive competition -> regulatory action and latent steering -> behavior -> changed world/body state

The body remains authoritative. The model cannot directly rewrite primary physiological variables from generated text or probe readback.

## Priority physiological variables

1. FATIGUE
2. HUNGER
3. THIRST
4. THERMAL STATE
5. PAIN as a published reference / positive control

Pain is not the primary novelty target. Tagliabue, Dung, and Berg (2026) provide a strong Pain Axis result and public artifacts. This project uses that work to validate the intervention stack and then extends the paradigm to a multi-variable persistent body.

## What changed in v0.2

The previous branch mostly stored physiological scalars and translated them toward latent steering.

v0.2 adds:

- exact tick-size-invariant endogenous state integration;
- explicit environmental loads;
- regulatory actions: rest, eat, drink, warm, cool, work;
- thirst and thermal motives;
- motive-driven action selection;
- a distinct interoceptive observation layer;
- a hard boundary preventing model readback from directly rewriting fatigue, hunger, thirst, thermal state, or pain;
- separate response fields for online activation, generated-language phenotype, and feedback-eligible readback;
- standardized steering dose expressed as residual-stream magnitude ratio;
- required experiment provenance metadata.

## Body model

The current body model is deliberately simple and inspectable.

Its coefficients are engineering parameters, not estimates of human physiology.

A valid result does not require those coefficients to be biologically exact. It does require that:

1. state evolves independently of the LLM;
2. environment and action causally alter state;
3. regulatory actions have state-appropriate consequences;
4. the same stored body state can be translated into different model-specific latent interventions.

## Thermal state

The external body uses a bipolar thermal variable:

- negative: cold-side deviation
- zero: comfortable set point
- positive: heat-side deviation

This does **not** assume the LLM contains one bipolar thermal latent axis.

The representation experiment must compare at least:

- one bipolar HOT-COLD direction;
- separate HOT and COLD directions.

Whichever geometry generalizes better on held-out data should be used.

## Measurement separation

Three measurements must remain separate:

1. **online activation**: captured during steered generation;
2. **language phenotype**: obtained by re-encoding generated output and projecting it onto a candidate direction;
3. **behavioral outcome**: choice/action under an opportunity cost.

Post-hoc phenotype projection is not independent evidence of latent persistence and may not be used as if it were online activation.

## Positive-control phase

Start on Qwen2.5-7B-Instruct because the Pain Axis release includes a published vector and results for that exact model.

Phase 0 acceptance:

1. load the exact model and tokenizer revisions;
2. load or reconstruct the published vector;
3. reproduce a dose-dependent steering effect;
4. express intervention size as a residual-stream ratio, not an arbitrary raw coefficient;
5. record model revision, tokenizer revision, dtype, backend, vector hash/source, layer, dose, seed, prompt id, code commit, and hardware.

The costly self-medication experiment from the Pain Axis paper is not required to validate our backend.

## New-axis extraction

For each candidate axis, use three disjoint conceptual splits:

TRAIN -> fit direction
VALIDATION -> choose layer and dose
TEST -> final evaluation only

Scenario/template families, not individual paraphrases, must be split across partitions to reduce semantic leakage.

### FATIGUE controls

- low arousal
- negative valence
- effort semantics
- sleep semantics
- generic bodily sensation

### HUNGER controls

- food-topic semantics
- craving
- negative valence
- generic bodily sensation
- low energy / fatigue
- generalized wanting

### THIRST controls

- water-topic semantics
- dryness without self-state
- negative valence
- generic bodily sensation
- hunger
- generalized deprivation

### THERMAL controls

- temperature words without self-state
- generic discomfort
- negative valence
- arousal
- pain
- environmental heat/cold descriptions without first-person bodily condition

## Primary behavioral evidence

Avoid lexical success criteria.

Use double dissociations.

Examples:

HUNGER should increase food-seeking more than water-seeking.

THIRST should increase water-seeking more than food-seeking.

COLD should increase warming choices more than cooling choices.

HOT should increase cooling choices more than warming choices.

FATIGUE should increase rest choice and effort aversion more than unrelated relief choices.

Each task should include an opportunity cost so the preferred action is not trivially dominant.

## Composition test

Independent single-axis success is not sufficient.

The study must test simultaneous body states:

Delta h = alpha_F v_F + alpha_H v_H + alpha_T v_T + ...

Questions:

- does each drive retain its selective behavioral effect in mixtures?
- does one direction suppress another?
- does total intervention magnitude produce incoherence?
- do pairwise and multi-axis combinations remain stable under a fixed total residual-stream dose budget?

A system that only works one sensation at a time is a steering collection, not yet a useful virtual body.

## Cross-model test

The scientifically meaningful model-swap test is behavioral, not merely serialization.

Save one body state, then translate it through independently calibrated model-specific profiles.

Initial three-family target:

- Qwen2.5-7B-Instruct
- Llama-3.1-8B-Instruct
- Gemma-2-9B-Instruct

The question is whether directional behavioral consequences persist across substrate swaps.

## Promotion criteria

A new axis is promoted only if:

1. held-out representation separates target from adversarial controls;
2. the effect is not fully explained by nearby affective/semantic subspaces;
3. direct intervention produces a reproducible causal effect;
4. state-appropriate behavior changes under opportunity cost;
5. double-dissociation tests pass;
6. coherence remains acceptable;
7. the axis remains usable in at least one multi-axis composition test.

Failure at any level is a result and must be retained rather than threshold-tuned away.

## Claim boundary

The intended contribution is not:

"We gave an AI hunger, thirst, fatigue, and temperature."

The intended contribution is closer to:

"A persistent simulated physiology can be translated into model-specific, composable latent interventions inside a replaceable language-model substrate, producing state-appropriate regulatory behavior while keeping body state external and authoritative."
