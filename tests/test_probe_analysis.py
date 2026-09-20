from __future__ import annotations

import unittest

import numpy as np

from lib.organism.probe_analysis import cosine, project_into_subspace, same_layer_vad_projection


class ProbeAnalysisTests(unittest.TestCase):
    def test_cosine_handles_orthogonal_vectors(self):
        self.assertAlmostEqual(cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 0.0)

    def test_projection_reports_independent_residual(self):
        result = project_into_subspace(
            np.array([0.0, 0.0, 1.0]),
            {
                "x": np.array([1.0, 0.0, 0.0]),
                "y": np.array([0.0, 1.0, 0.0]),
            },
        )
        self.assertAlmostEqual(result.residual_fraction, 1.0)
        np.testing.assert_allclose(result.projected, np.zeros(3), atol=1e-7)

    def test_vad_projection_reconstructs_in_span_component(self):
        fatigue = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        result = same_layer_vad_projection(
            fatigue,
            {
                "V": np.array([1.0, 0.0, 0.0]),
                "A": np.array([0.0, 1.0, 0.0]),
                "D": np.array([0.0, 0.0, 1.0]),
            },
        )
        self.assertLess(result.residual_fraction, 1e-6)
        np.testing.assert_allclose(result.projected, fatigue, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
