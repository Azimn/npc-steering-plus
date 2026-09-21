# Pain Axis Method Adaptation and Attribution

## Purpose

The virtual-body study uses the Pain Axis work as a methodological predecessor and as a positive-control representation.

Reference: Tagliabue, Valen, Leonard Dung, and Cameron Berg. 2026. "The Pain Axis: LLMs Represent Self-Directed Harm and Act to Relieve It." arXiv:2609.16247.

Paper: https://arxiv.org/abs/2609.16247

Code and released results: https://github.com/valen-research/Pain-axis

The released repository is MIT licensed.

## What we intend to reuse

Where an exact model match exists, the project may directly load the authors' published pain vector as a reference axis, preserving attribution and any required license notice for redistributed source or artifacts.

We also adapt these methodological ideas:

1. fit target-minus-control directions rather than relying only on antonym sentence pairs;
2. estimate dominant variation inside the control activation cloud;
3. project those high-variance control components out of the candidate direction;
4. select layers using held-out data;
5. compare candidate axes against nearby semantic and affective control directions;
6. test causal steering and behavioral consequences.

## What we are changing

The Pain Axis pipeline is pain-specific. This branch generalizes the method to externally maintained virtual-body variables: FATIGUE, HUNGER, THIRST, THERMAL, and PAIN_REFERENCE.

The body variable exists independently of the model. The learned direction is only a model-specific translation from persistent body state into an intervention on the current LLM substrate.

## Pain's role in this project

Pain is not the first novelty target.

The published Pain Axis result is used to answer a narrower engineering question first: can our intervention harness reproduce an established latent-body result on an exact supported backbone?

Once that positive control works, development effort moves to fatigue, hunger, thirst, and thermal state.

A later physical-pain-specific axis may be useful if the study needs to distinguish somatic pain from the broader published Pain Axis. That is not required for the first milestone.

## Implementation note

The current helper lib/organism/interoception_vectors.py is a small NumPy implementation of the denoised difference-of-means idea. It was written for this repository rather than copied from the Pain Axis source.

If later work copies or substantially modifies upstream source files, add the upstream MIT copyright and permission notice alongside those files.
