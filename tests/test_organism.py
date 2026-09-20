from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lib.organism import (
    CognitiveResponse,
    OrganismState,
    PersistentOrganism,
    SteeringBinding,
    SteeringProfile,
)


class FakeBackend:
    model_id = "fake/model"

    def __init__(self) -> None:
        self.last_alphas = None

    def generate(self, messages, steering_alphas):
        self.last_alphas = dict(steering_alphas)
        return CognitiveResponse(text="ok", readback={"V": -0.2})


class PersistentOrganismTests(unittest.TestCase):
    def test_state_persists_across_reload(self):
        state = OrganismState("aster")
        state.apply({"fatigue": 0.4, "threat": 0.7})
        state.step(5.0)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "state.json"
            state.save(path)
            loaded = OrganismState.load(path)
        self.assertEqual(loaded.organism_id, "aster")
        self.assertEqual(loaded.revision, state.revision)
        self.assertAlmostEqual(loaded.get("fatigue"), state.get("fatigue"))
        self.assertAlmostEqual(loaded.get("threat"), state.get("threat"))

    def test_model_swap_keeps_state(self):
        state = OrganismState("aster", values={"fatigue": 0.8})
        p1 = SteeringProfile("model/a", (SteeringBinding("fatigue", "FATIGUE", 10.0),))
        p2 = SteeringProfile("model/b", (SteeringBinding("fatigue", "TIREDNESS", 5.0),))
        organism = PersistentOrganism(state, p1)
        before = organism.state.snapshot()
        organism.bind_profile(p2)
        after = organism.state.snapshot()
        self.assertEqual(before, after)
        self.assertAlmostEqual(organism.steering_alphas()["TIREDNESS"], 4.0)

    def test_turn_steers_backend_and_feedback_is_opt_in(self):
        state = OrganismState("aster", values={"valence": -0.6})
        profile = SteeringProfile(
            "fake/model",
            (
                SteeringBinding(
                    "valence",
                    "V",
                    20.0,
                    feedback_gain=0.25,
                    readback_center=0.0,
                    readback_scale=1.0,
                ),
            ),
        )
        organism = PersistentOrganism(state, profile)
        backend = FakeBackend()
        turn = organism.turn(backend, [{"role": "user", "content": "hello"}])
        self.assertAlmostEqual(backend.last_alphas["V"], -12.0)
        self.assertEqual(turn.response.text, "ok")
        self.assertAlmostEqual(organism.state.get("valence"), -0.5)

    def test_profile_mismatch_fails_closed(self):
        organism = PersistentOrganism(
            OrganismState("aster"),
            SteeringProfile("other/model", ()),
        )
        with self.assertRaises(ValueError):
            organism.turn(FakeBackend(), [])


if __name__ == "__main__":
    unittest.main()
