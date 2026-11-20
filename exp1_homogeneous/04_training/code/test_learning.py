"""Structural checks only; synthetic arrays here never enter research outputs."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp1_paths as L  # noqa: E402  (step folders on sys.path; L.at() resolves stored paths)
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from lut_dataset import generate
from train import DepthCNN, metrics


class LearningChecks(unittest.TestCase):
    def test_identical_initialization_and_output(self):
        torch.manual_seed(42)
        a = DepthCNN()
        torch.manual_seed(42)
        b = DepthCNN()
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters())))
        out = a(torch.ones(4, 1, 128))
        self.assertEqual(tuple(out.shape), (4,))
        out.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in a.parameters()))

    def test_paired_manifest_and_scene_splits(self):
        # A constant field is a numerical fixture, not a substitute for MC.
        depths = np.array([0.5, 5.0])
        x = np.arange(-29.95, 30, 0.1)
        tables = {m: (depths, x, np.ones((2, len(x))) * (1-m/10)) for m in (0.15, 0.30, 0.508, 1.0)}
        with tempfile.TemporaryDirectory() as temp:
            path = generate(tables, temp, sizes=(16, 8, 8))
            with np.load(path, allow_pickle=False) as z:
                self.assertEqual(z["X_F"].shape, (32, 128))
                self.assertEqual(len(set(z["scene_id"])), 32)
                standard = np.isclose(z["mua_F"], 0.508)
                np.testing.assert_array_equal(z["X_F"][standard], z["X_K"][standard])
                for split, expected in (("train", 4), ("validation", 2), ("test", 2)):
                    for mua in tables:
                        self.assertEqual(int(np.sum((z["split"] == split) & np.isclose(z["mua_F"], mua))), expected)

    def test_signed_bias_convention(self):
        result = metrics(np.array([2, 4]), np.array([1, 2]))
        self.assertEqual(result["bias_mm"], 1.5)
        self.assertEqual(result["mae_mm"], 1.5)


if __name__ == "__main__":
    unittest.main()
