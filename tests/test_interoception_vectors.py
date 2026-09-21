from __future__ import annotations

import unittest

import numpy as np

from lib.organism.interoception_vectors import (
    denoised_difference_of_means,
    direction_cosines,
    projection_scores,
)


class InteroceptionVectorTests(unittest.TestCase):
    def test_denoising_removes_dominant_control_component(self):
        controls = np.array(
            [
                [-3.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [3.0, 0.0],
            ]
        )
        targets = np.array(
            [
                [1.0, 2.0],
                [1.0, 2.0],
            ]
        )

        result = denoised_difference_of_means(
            targets,
            controls,
            variance_to_remove=0.50,
        )

        self.assertEqual(result.n_removed_components, 1)
        self.assertGreater(result.control_variance_removed, 0.99)
        np.testing.assert_allclose(
            np.abs(result.direction),
            np.array([0.0, 1.0]),
            atol=1e-6,
        )

    def test_zero_denoising_returns_normalized_mean_difference(self):
        controls = np.array([[0.0, 0.0], [0.0, 0.0]])
        targets = np.array([[3.0, 4.0], [3.0, 4.0]])

        result = denoised_difference_of_means(
            targets,
            controls,
            variance_to_remove=0.0,
        )

        np.testing.assert_allclose(result.direction, np.array([0.6, 0.8]), atol=1e-6)
        self.assertEqual(result.n_removed_components, 0)

    def test_projection_scores_order_samples(self):
        acts = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        scores = projection_scores(acts, np.array([0.0, 1.0]))
        self.assertTrue(np.all(np.diff(scores) > 0))

    def test_direction_cosines_reports_overlap(self):
        result = direction_cosines(
            np.array([1.0, 0.0]),
            {
                "same": np.array([2.0, 0.0]),
                "orthogonal": np.array([0.0, 1.0]),
            },
        )
        self.assertAlmostEqual(result["same"], 1.0)
        self.assertAlmostEqual(result["orthogonal"], 0.0)

    def test_fails_closed_if_controls_explain_entire_direction(self):
        controls = np.array([[-2.0, 0.0], [2.0, 0.0]])
        targets = np.array([[1.0, 0.0], [1.0, 0.0]])
        with self.assertRaises(ValueError):
            denoised_difference_of_means(targets, controls, variance_to_remove=0.50)


if __name__ == "__main__":
    unittest.main()
