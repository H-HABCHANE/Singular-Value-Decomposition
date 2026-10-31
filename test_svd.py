"""Tests de svd.py. Exécuter : python -m unittest -v"""

import unittest

import numpy as np

from svd import energy, rank_for_energy, reconstruct, svd, svd_from_eigen


class TestSVD(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_small_example(self):
        A = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        _, s, _ = svd(A)
        np.testing.assert_allclose(s, [np.sqrt(3.0), 1.0], atol=1e-12)

    def test_reconstruction_and_orthogonality(self):
        for shape in [(7, 3), (3, 7), (20, 20)]:
            A = self.rng.standard_normal(shape)
            U, s, Vt = svd(A)
            np.testing.assert_allclose(reconstruct(U, s, Vt), A, atol=1e-12)
            r = min(shape)
            np.testing.assert_allclose(U.T @ U, np.eye(r), atol=1e-12)
            np.testing.assert_allclose(Vt @ Vt.T, np.eye(r), atol=1e-12)
            self.assertTrue(np.all(np.diff(s) <= 0))  # valeurs décroissantes

    def test_eckart_young(self):
        A = self.rng.standard_normal((30, 15))
        U, s, Vt = svd(A)
        for k in range(1, 15):
            err = np.linalg.norm(A - reconstruct(U, s, Vt, k), 2)
            self.assertAlmostEqual(err, s[k], places=10)

    def test_exact_low_rank(self):
        A = self.rng.standard_normal((40, 3)) @ self.rng.standard_normal((3, 25))
        U, s, Vt = svd(A)
        self.assertLess(s[3] / s[0], 1e-12)
        self.assertEqual(rank_for_energy(s, 1 - 1e-12), 3)

    def test_energy(self):
        e = energy(np.array([2.0, 1.0, 1.0]))
        np.testing.assert_allclose(e, [4 / 6, 5 / 6, 1.0])

    def test_eigen_route_matches(self):
        A = self.rng.standard_normal((12, 5))
        _, s_ref, _ = svd(A)
        U, s, Vt = svd_from_eigen(A)
        np.testing.assert_allclose(s, s_ref, rtol=1e-8)
        np.testing.assert_allclose(reconstruct(U, s, Vt), A, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
