# Milestone: Virtual Body Homeostasis v0.2

This milestone is the architectural revision produced after critical review of the v0.1 interoception branch.

## Problems addressed

v0.1 could be criticized as latent-state steering with physiological labels because the body variables mostly existed as scalars.

v0.2 adds the missing causal structure:

- environment affects physiology;
- exertion has fatigue/thirst/hunger costs;
- injury can increase pain;
- body state competes for motives;
- thirst and thermal state can win motive competition;
- winning motives select regulatory actions;
- eating, drinking, resting, warming, and cooling alter body state;
- body dynamics are integrated exactly so results are not scheduler-step artifacts;
- physiological steering is routed through an interoceptive observation rather than directly from the authoritative body dictionary;
- model readback cannot directly rewrite primary physiology.

## Measurement repair

CognitiveResponse now separates:

- online_activation
- phenotype_projection
- feedback-eligible readback

The existing MLX post-hoc projection is explicitly stored as phenotype_projection and no longer masquerades as independent internal readback.

## Dose repair

Latent intervention strength can now be expressed as a residual-stream ratio:

rho = ||alpha v|| / E[||h||]

This avoids treating the same numeric coefficient as comparable across vectors with different norms.

## Reproducibility repair

ExperimentProvenance records:

- model id and revision
- tokenizer revision
- dtype
- backend
- vector source and SHA-256
- layer
- residual-stream dose ratio
- seed
- prompt id
- code commit
- hardware

## Experimental standard

New axes require:

1. train / validation / test conceptual separation;
2. adversarial semantic and affective controls;
3. online activation measurement;
4. behavioral opportunity-cost tests;
5. double dissociations;
6. multi-axis composition;
7. cross-model replication after single-model validation.

## Immediate next gate

Phase 0 remains a Pain Axis reproduction on Qwen2.5-7B-Instruct.

After that succeeds, FATIGUE is re-extracted under the v0.2 methodology before HUNGER, THIRST, and THERMAL are added.
