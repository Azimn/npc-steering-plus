#!/usr/bin/env python3
"""CPU-only smoke run for the persistent organism substrate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib.organism import OrganismState, PersistentOrganism, SteeringProfile, selected_motive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="runs/organism_state.json")
    parser.add_argument("--profile", default="data/organism/qwen35_extended.template.json")
    args = parser.parse_args()

    state_path = _ROOT / args.state
    profile = SteeringProfile.load(_ROOT / args.profile)
    if state_path.exists():
        state = OrganismState.load(state_path)
        origin = "resumed"
    else:
        state = OrganismState("aster")
        origin = "created"

    organism = PersistentOrganism(state, profile)
    organism.tick(60.0)
    organism.perturb({"threat": 0.55, "arousal": 0.30, "valence": -0.20})
    organism.save(state_path)

    print(f"{origin}: {organism.state.describe()}")
    print(f"selected motive: {selected_motive(organism.state).name}")
    print(f"active steering: {organism.steering_alphas()}")
    print(f"saved: {state_path}")


if __name__ == "__main__":
    main()
