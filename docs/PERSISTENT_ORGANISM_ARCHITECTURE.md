# Persistent Organism Architecture

## Objective

npc-steering-plus now treats the organism as the persistent entity and the language model as a replaceable execution substrate.

The data flow is:

```text
world, body, social input
        |
        v
persistent organism state
        |
        v
motive competition
        |
        v
model-specific steering profile
        |
        v
activation coefficients
        |
        v
replaceable LLM backend
        |
        v
language and probe readback
        |
        v
optional bounded feedback
        |
        +----> persistent organism state
```

The organism remains authoritative. Its state advances and persists without an LLM. Replacing one model with another changes only the steering profile and backend, not the organism state.

## State

lib/organism/state.py stores affect, stress, fatigue, hunger, pain, threat, affiliation need, curiosity, and competence need. Each variable has explicit bounds, a baseline, a time constant, and optional endogenous drift. State is serialized atomically to JSON.

## Steering

lib/organism/steering.py maps organism variables to model-specific learned directions. The supplied Qwen profile enables the original V/A/D axes. Fatigue, pain, threat, affiliation, curiosity, and competence bindings are present but disabled until their directions are experimentally validated.

## Backends

lib/organism/backend.py defines the backend contract. lib/organism/mlx_backend.py is the first concrete implementation and uses the existing MLX residual-stream intervention code. MLX imports are lazy so the organism core remains importable without MLX.

lib/organism/mlx_bridge.py is axis-agnostic. Any validated direction in a ProbeBundle can receive an organism-derived coefficient.

## Motives

lib/organism/motives.py derives safety, rest, eating, affiliation, exploration, and mastery motives from persistent state. These are readouts over state, not a second hidden state store.

## First new experiment

The repository includes data/contrastive_pairs_fatigue.json for a direct FATIGUE direction. The current fatigue binding remains disabled until extraction, validation, layer selection, and gain calibration are complete.

The intended test is whether changing only the external fatigue variable produces a reliable change in model activations and downstream behavior under direct steering, compared with prompt-only, V/A/D-mediated, and unsteered controls.

## Upstream compatibility

The original driver simulator remains untouched and runnable as the reference baseline. The persistent organism package is parallel infrastructure, so the upstream result can be reproduced without the new architecture.
