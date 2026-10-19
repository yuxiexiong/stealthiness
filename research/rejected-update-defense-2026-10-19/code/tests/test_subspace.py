"""Pure CPU tests; no models, datasets, CUDA, or optional dependencies."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subspace import orthonormal_basis, project_update


class SubspaceTests(unittest.TestCase):
    def test_orthogonality_and_rank_four_cap(self):
        generator = torch.Generator().manual_seed(711)
        vectors = [torch.randn(41, generator=generator) for _ in range(8)]
        basis = orthonormal_basis(vectors, max_rank=8)
        self.assertEqual(basis.shape, (4, 41))
        self.assertEqual(basis.dtype, torch.float32)
        torch.testing.assert_close(basis @ basis.T, torch.eye(4), atol=3e-6, rtol=0)

    def test_row_magnitude_and_sign_do_not_weight_the_subspace(self):
        generator = torch.Generator().manual_seed(713)
        vectors = [torch.randn(29, generator=generator) for _ in range(3)]
        scaled = [v * s for v, s in zip(vectors, (1e-20, -3e10, .125))]
        a = orthonormal_basis(vectors)
        b = orthonormal_basis(scaled)
        delta = torch.randn(29, generator=generator)
        first, _ = project_update(delta, a)
        second, _ = project_update(delta, b)
        torch.testing.assert_close(first, second, atol=3e-6, rtol=3e-6)

    def test_no_input_mutation(self):
        vectors = [torch.tensor([1., 2., 3., 4.]), torch.tensor([0., 1., 1., 0.])]
        copies = [v.clone() for v in vectors]
        basis = orthonormal_basis(vectors)
        before_basis = basis.clone()
        delta = torch.tensor([4., 3., 2., 1.])
        before_delta = delta.clone()
        output, _ = project_update(delta, basis)
        for a, b in zip(vectors, copies):
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(basis, before_basis))
        self.assertTrue(torch.equal(delta, before_delta))
        self.assertNotEqual(output.data_ptr(), delta.data_ptr())

    def test_zero_and_numerically_invalid_sources(self):
        self.assertEqual(orthonormal_basis([torch.zeros(7),torch.zeros(7)]).shape, (0, 7))
        with self.assertRaises(ValueError):
            orthonormal_basis([torch.tensor([float('nan')]*7)])
        self.assertEqual(orthonormal_basis([]).shape, (0, 0))
        empty = orthonormal_basis([torch.ones(7)], max_rank=0)
        delta = torch.arange(7, dtype=torch.float32)
        result, metrics = project_update(delta, empty)
        self.assertTrue(torch.equal(result, delta))
        self.assertEqual(metrics["fraction"], 0.)
        self.assertEqual(metrics["removed_norm"], 0.)
        self.assertEqual(metrics["original_norm"], metrics["projected_norm"])

    def test_duplicate_vectors_preserve_reduced_rank(self):
        vector = torch.tensor([1., 2., 3., 4.])
        basis = orthonormal_basis([vector, vector * 3, vector * -2, torch.zeros(4)])
        self.assertEqual(basis.shape, (1, 4))
        result, metrics = project_update(vector, basis)
        torch.testing.assert_close(result, torch.zeros(4), atol=1e-6, rtol=0)
        self.assertAlmostEqual(metrics["fraction"], 1., places=6)

    def test_projection_idempotence_and_residual_orthogonality(self):
        generator = torch.Generator().manual_seed(719)
        basis = orthonormal_basis([torch.randn(53, generator=generator) for _ in range(5)])
        delta = torch.randn(53, generator=generator)
        first, metrics = project_update(delta, basis)
        second, _ = project_update(first, basis)
        torch.testing.assert_close(first, second, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(basis @ first, torch.zeros(4), atol=3e-6, rtol=0)
        self.assertLessEqual(metrics["projected_norm"], metrics["original_norm"] + 1e-6)
        self.assertAlmostEqual(metrics["original_norm"] ** 2,
                               metrics["removed_norm"] ** 2 + metrics["projected_norm"] ** 2,
                               delta=1e-4)

    def test_known_projection_and_zero_update_metrics(self):
        basis = orthonormal_basis([torch.tensor([1., 0., 0.])])
        result, metrics = project_update(torch.tensor([3., 4., 0.]), basis)
        torch.testing.assert_close(result, torch.tensor([0., 4., 0.]))
        self.assertEqual(metrics, {"removed_norm": 3., "original_norm": 5.,
                                   "projected_norm": 4., "fraction": .6})
        result, metrics = project_update(torch.zeros(3), basis)
        self.assertTrue(torch.equal(result, torch.zeros(3)))
        self.assertEqual(metrics["fraction"], 0.)

    def test_invalid_shapes_types_and_bound(self):
        with self.assertRaises(ValueError):
            orthonormal_basis([torch.ones(3)] * 9)
        with self.assertRaises(ValueError):
            orthonormal_basis([torch.ones(3), torch.ones(4)])
        with self.assertRaises(ValueError):
            orthonormal_basis([torch.ones(3, dtype=torch.float64)])
        with self.assertRaises(ValueError):
            project_update(torch.ones(3), torch.ones(1, 4))


if __name__ == "__main__":
    unittest.main()
