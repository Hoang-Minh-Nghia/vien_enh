"""Check the locked V0 manifest without reusing the generator's logic."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp1_paths as L  # noqa: E402  (step folders on sys.path; L.at() resolves stored paths)
import hashlib
import json
import unittest
from pathlib import Path

ROOT = L.EXP1
MANIFEST = L.at("configs/validation_independent.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(MANIFEST.exists(), "V0 manifest has not been locked")
class ManifestChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.channel = cls.m["channel_runs"]
        cls.n0 = cls.m["n0_runs"]

    def test_sidecar_matches_manifest(self):
        recorded = MANIFEST.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        self.assertEqual(recorded, digest(MANIFEST))

    def test_full_factorial_design(self):
        self.assertEqual(len(self.channel), 180)
        self.assertEqual(len(self.n0), 3)
        triples = {(j["depth_mm"], j["mua_mm_inv"], j["repeat"]) for j in self.channel}
        self.assertEqual(len(triples), 180)
        self.assertEqual({t[0] for t in triples}, {0.6, 0.9, 1.4, 1.8, 2.3, 2.8, 3.3, 3.8, 4.3, 4.8})
        self.assertEqual({t[1] for t in triples}, {0.15, 0.22, 0.30, 0.40, 0.508, 1.0})
        self.assertEqual({t[2] for t in triples}, {1, 2, 3})

    def test_depths_are_new(self):
        used = set(json.loads((L.at("configs/simulation.json")).read_text(encoding="utf-8"))["depths_mm"])
        used |= {0.625, 2.25, 4.75}
        self.assertFalse({j["depth_mm"] for j in self.channel} & used)

    def test_unseen_absorption_is_not_a_training_level(self):
        levels = json.loads((L.at("configs/simulation.json")).read_text(encoding="utf-8"))["optics"]["channel_mua_levels_mm_inv"]
        unseen = {j["mua_mm_inv"] for j in self.channel if j["mua_group"] == "unseen"}
        self.assertEqual(unseen, {0.22, 0.40})
        self.assertFalse(unseen & set(levels))
        self.assertEqual(self.m["analysis_plan"]["primary_endpoint"]["mua_mm_inv"], [0.22, 0.40])

    def test_pilot_selection(self):
        pilot = {(j["depth_mm"], j["mua_mm_inv"], j["repeat"]) for j in self.channel if j["pilot"]}
        expected = {(z, m, r) for z in (0.6, 3.3, 4.8) for m in (0.15, 0.508) for r in (1, 2, 3)}
        self.assertEqual(pilot, expected)

    def test_channel_uses_n0_of_same_repeat(self):
        by_id = {j["run_id"]: j for j in self.n0}
        for job in self.channel:
            self.assertEqual(by_id[job["n0_run_id"]]["repeat"], job["repeat"])

    def test_seeds_and_outputs_are_unique_and_new(self):
        tiers = [t for j in self.channel for t in j["photon_tiers"]]
        self.assertTrue(all([t["photons"] for t in j["photon_tiers"]] == [60_000_000, 120_000_000, 240_000_000]
                            for j in self.channel))
        self.assertTrue(all(j["photons"] == 120_000_000 for j in self.n0))
        seeds = [t["seed"] for t in tiers] + [j["seed"] for j in self.n0]
        outputs = [t["output"] for t in tiers] + [j["output"] for j in self.n0]
        self.assertEqual(len(set(seeds)), 543)
        self.assertEqual(len(set(outputs)), 543)
        self.assertGreater(min(seeds), 13002, "must lie above every earlier MC run seed")
        self.assertTrue(all(o.startswith("results/validation_independent/") for o in outputs))

    def test_locked_hashes_match_disk(self):
        for path, value in self.m["locked_files"].items():
            self.assertEqual(digest(L.at(path)), value, path)
        for item in self.m["training_data"].values():
            self.assertEqual(digest(L.at(item["path"])), item["sha256"], item["path"])
        self.assertEqual(len(self.m["checkpoints"]), 6)
        self.assertEqual({(c["branch"], c["seed"]) for c in self.m["checkpoints"]},
                         {(b, s) for b in "FK" for s in (42, 43, 44)})
        for c in self.m["checkpoints"]:
            self.assertEqual(digest(L.at(c["path"])), c["sha256"], c["path"])
            self.assertEqual(digest(L.at(c["training_log"])), c["training_log_sha256"], c["training_log"])


if __name__ == "__main__":
    unittest.main()
