# Milestone: Persistent Organism v0.1

This milestone establishes the continuity boundary before claiming any new learned physiological axis.

The code now supports persistent external state, motive competition, model-specific steering profiles, profile swapping without state reset, optional bounded readback feedback, a replaceable backend contract, a concrete MLX backend, and generalized residual-stream offsets for arbitrary validated probe axes.

The Qwen profile currently enables only V/A/D. Direct fatigue, pain, threat, affiliation, curiosity, and competence bindings remain disabled until their probe directions have been extracted and validated. This keeps implemented architecture separate from empirical evidence.

The CPU-only acceptance check is:

```bash
python -m unittest tests.test_organism
```

The persistence smoke check is:

```bash
python scripts/13_organism_smoke.py
```

Repeated smoke runs must resume the same state file.

The direct fatigue experiment begins with:

```bash
python scripts/02_extract_probes.py \
  --pairs data/contrastive_pairs_fatigue.json \
  --output artifacts/probes_Qwen3.5-9B-MLX-4bit_fatigue.pkl
```

Validation can then use:

```bash
python scripts/03_validate_probes.py \
  --probes artifacts/probes_Qwen3.5-9B-MLX-4bit_fatigue.pkl \
  --axes FATIGUE \
  --prompt "You have another ordinary task to do. Describe in one sentence how ready you are to continue."
```

After a layer is selected, scripts/14_merge_probe_bundles.py can merge the new axis into the original V/A/D bundle without overwriting existing directions.

The v0.2 acceptance criterion is a controlled ablation where changing only the external fatigue variable causes a repeatable activation and behavioral change under direct steering, with prompt-only, V/A/D-mediated, and unsteered conditions as controls.
