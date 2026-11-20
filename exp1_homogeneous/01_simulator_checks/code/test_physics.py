"""Small physical-invariant checks before launching expensive Monte Carlo runs."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp1_paths as L  # noqa: E402  (step folders on sys.path; L.at() resolves stored paths)
import unittest
import json
import numpy as np
from simulate import load_config, make_volume, properties, geometry_summary
from readout import bin_average, profile, diagnostics, compare


class PhysicsChecks(unittest.TestCase):
    def test_background_and_channel_geometry(self):
        config = load_config()
        blank, _, _ = make_volume(config)
        self.assertEqual(set(np.unique(blank)), {0, 1})
        for depth in config["depths_mm"]:
            vol, _, z = make_volume(config, depth)
            summary = geometry_summary(config, vol, z)
            self.assertTrue(summary["top_air_layer"])
            self.assertLessEqual(abs(summary["voxelized_top_mm"] - depth), config["geometry"]["voxel_mm"])
            self.assertAlmostEqual(summary["voxelized_width_mm"], 3.0)
            self.assertTrue(np.all(vol[:, 0, :] == vol[:, -1, :]))

    def test_optical_units(self):
        config = load_config()
        prop = properties(config, 0.508)
        self.assertAlmostEqual(float(prop[1, 1] * (1 - prop[1, 2])), 1.0, places=6)
        self.assertEqual(float(prop[2, 1]), 0.0)
        self.assertAlmostEqual(float(prop[2, 0]), 0.508, places=6)

    def test_readout_preserves_constant_and_integral(self):
        x = np.arange(-30 + 0.05, 30, 0.1)
        edges = np.linspace(-16, 16, 129)
        flat = bin_average(x, np.ones_like(x), edges)
        np.testing.assert_allclose(flat, 1, atol=1e-10)
        signal = 1 - 0.2 * np.exp(-x*x/8)
        result = bin_average(x, signal, edges)
        self.assertEqual(result.shape, (128,))
        np.testing.assert_allclose(result, result[::-1], atol=1e-10)
        self.assertGreater(float(result.min()), 0.79)
        self.assertLess(float(result.min()), 0.81)

    def test_diagnostics_reject_unresolved_width(self):
        x = np.linspace(-15.875, 15.875, 128)
        result = diagnostics(x, np.ones(128))
        self.assertIsNone(result["fwhm_mm"])
        self.assertFalse(compare(result, result)["pass"])

    def test_gaussian_diagnostic_width(self):
        x = np.linspace(-15.875, 15.875, 128)
        result = diagnostics(x, 1 - 0.2 * np.exp(-x*x/8), sigma_mm=0.25)
        self.assertLess(abs(result["fwhm_mm"] - 4.75), 0.1)

    def test_divide_each_pixel_before_averaging(self):
        config = load_config()
        x = np.arange(-29.95, 30, 0.1)
        common = {"x_mm": x, "metadata_json": json.dumps({"config": config})}
        background = {**common, "R_xy": np.tile([1.0, 2.0], (len(x), 1))}
        channel = {**common, "R_xy": np.tile([1.0, 4.0], (len(x), 1))}
        _, curve = profile(channel, background)
        np.testing.assert_allclose(curve, 1.5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
