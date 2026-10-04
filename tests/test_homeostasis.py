from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lib.organism import (
    BodyAction,
    BodyEnvironment,
    InteroceptiveSensor,
    OrganismState,
    PersistentOrganism,
    SteeringBinding,
    SteeringProfile,
)
from lib.organism.body import regulate_body


class HomeostasisTests(unittest.TestCase):
    def test_endogenous_integration_is_tick_size_invariant(self):
        a = OrganismState("a", values={"fatigue": 0.2, "hunger": 0.3, "thirst": 0.4})
        b = OrganismState("b", values={"fatigue": 0.2, "hunger": 0.3, "thirst": 0.4})

        a.step(3600.0)
        for _ in range(60):
            b.step(60.0)

        for axis in ("fatigue", "hunger", "thirst"):
            self.assertAlmostEqual(a.get(axis), b.get(axis), places=10)

    def test_high_thirst_selects_drink_and_drinking_reduces_thirst(self):
        state = OrganismState(
            "aster",
            values={
                "thirst": 0.95,
                "hunger": 0.10,
                "fatigue": 0.10,
                "curiosity": 0.10,
                "competence_need": 0.10,
            },
        )
        env = BodyEnvironment(water_available=True)
        before = state.get("thirst")
        step = regulate_body(state, env, 60.0)

        self.assertEqual(step.motive.name, "drink")
        self.assertEqual(step.action, BodyAction.DRINK)
        self.assertLess(state.get("thirst"), before)

    def test_high_hunger_selects_eat_and_eating_reduces_hunger(self):
        state = OrganismState(
            "aster",
            values={
                "hunger": 0.95,
                "thirst": 0.05,
                "fatigue": 0.05,
                "curiosity": 0.10,
                "competence_need": 0.10,
            },
        )
        env = BodyEnvironment(food_available=True)
        before = state.get("hunger")
        step = regulate_body(state, env, 120.0)

        self.assertEqual(step.motive.name, "eat")
        self.assertEqual(step.action, BodyAction.EAT)
        self.assertLess(state.get("hunger"), before)

    def test_cold_state_selects_warming_action(self):
        state = OrganismState(
            "aster",
            values={
                "thermal": -0.95,
                "curiosity": 0.10,
                "competence_need": 0.10,
            },
        )
        env = BodyEnvironment(warmth_available=True, thermal_exposure=-0.4)
        before = state.get("thermal")
        step = regulate_body(state, env, 60.0)

        self.assertEqual(step.motive.name, "warm")
        self.assertEqual(step.action, BodyAction.WARM)
        self.assertGreater(state.get("thermal"), before)

    def test_exertion_has_body_costs(self):
        active = OrganismState("active")
        idle = OrganismState("idle")

        regulate_body(
            active,
            BodyEnvironment(exertion=0.8),
            300.0,
            action=BodyAction.NONE,
        )
        regulate_body(
            idle,
            BodyEnvironment(exertion=0.0),
            300.0,
            action=BodyAction.NONE,
        )

        self.assertGreater(active.get("fatigue"), idle.get("fatigue"))
        self.assertGreater(active.get("thirst"), idle.get("thirst"))

    def test_steering_uses_interoceptive_observation_not_true_value(self):
        state = OrganismState("aster", values={"fatigue": 0.83})
        profile = SteeringProfile(
            "fake/model",
            (SteeringBinding("fatigue", "FATIGUE", 10.0),),
        )
        sensor = InteroceptiveSensor(quantization=0.5)
        organism = PersistentOrganism(state, profile, sensor=sensor)

        observation = organism.observe_body()
        self.assertEqual(observation.values["fatigue"], 1.0)
        self.assertAlmostEqual(organism.steering_alphas()["FATIGUE"], 10.0)

    def test_model_feedback_cannot_rewrite_primary_physiology(self):
        state = OrganismState("aster", values={"fatigue": 0.8})
        profile = SteeringProfile(
            "fake/model",
            (
                SteeringBinding(
                    "fatigue",
                    "FATIGUE",
                    1.0,
                    feedback_gain=0.5,
                ),
            ),
        )
        with self.assertRaises(ValueError):
            profile.feedback_delta(state, {"FATIGUE": -1.0})

    def test_schema_v1_state_migrates_to_v2(self):
        blob = {
            "schema_version": 1,
            "organism_id": "legacy",
            "age_s": 10.0,
            "revision": 2,
            "values": {"fatigue": 0.7},
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "legacy.json"
            path.write_text(json.dumps(blob))
            state = OrganismState.load(path)

        self.assertEqual(state.schema_version, 2)
        self.assertEqual(state.snapshot()["schema_version"], 2)
        self.assertAlmostEqual(state.get("fatigue"), 0.7)
        self.assertIn("thirst", state.values)
        self.assertIn("thermal", state.values)


if __name__ == "__main__":
    unittest.main()
