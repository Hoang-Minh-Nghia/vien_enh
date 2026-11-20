"""Verify completed artifacts and independently recompute reported weak-group MAE."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp1_paths as L  # noqa: E402  (step folders on sys.path; L.at() resolves stored paths)
import csv
import hashlib
import json

import numpy as np

from simulate import ROOT
from readout import READOUT_VERSION


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    folder = L.at("04_training/results/FK")
    for stage in (1, 2):
        report = json.loads((L.at(f"results/stage{stage}/report.json")).read_text(encoding="utf-8"))
        require(report["gate_pass"], f"Stage {stage} gate did not pass")
        require(report["readout_version"] == READOUT_VERSION, f"Stage {stage} extraction mismatch")
    mc_report = json.loads((L.at("03_dataset/data/simulation_report.json")).read_text(encoding="utf-8"))
    require(mc_report["gate_pass"], "Expanded MC checks failed")
    require(len(mc_report["sources"]) == 48, "Expected 48 main channel MC configurations")
    data_path = L.at("03_dataset/data/dataset_FK/dataset.npz")
    data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    with np.load(data_path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    for key in ("X_F", "X_K"):
        require(data[key].shape == (6000, 128), f"Unexpected {key} shape")
        require(data[key].dtype == np.float32, f"Unexpected {key} dtype")
        require(np.isfinite(data[key]).all(), f"Nonfinite {key}")
    require(len(set(data["scene_id"])) == 6000, "Scene IDs are not unique")
    require(np.all((data["y_mm"] >= 0.5) & (data["y_mm"] <= 5)), "Depth labels out of bounds")
    counts = {}
    for split, expected in (("train", 4800), ("validation", 600), ("test", 600)):
        mask = data["split"] == split
        require(int(mask.sum()) == expected, f"Wrong {split} size")
        counts[split] = {}
        for level in (0.15, 0.30, 0.508, 1.0):
            n = int(np.sum(mask & np.isclose(data["mua_F"], level)))
            require(n == expected // 4, f"Unbalanced {split}/{level}")
            counts[split][str(level)] = n
    standard = np.isclose(data["mua_F"], 0.508)
    require(np.array_equal(data["X_F"][standard], data["X_K"][standard]), "Standard F/K inputs differ")
    manifest = json.loads((L.at("03_dataset/data/dataset_FK/manifest.json")).read_text(encoding="utf-8"))
    lut_path = L.at(manifest["provenance"]["lut_path"])
    require(hashlib.sha256(lut_path.read_bytes()).hexdigest() == manifest["provenance"]["lut_sha256"], "LUT provenance mismatch")
    comparison = json.loads((L.at("04_training/results/FK/comparison.json")).read_text(encoding="utf-8"))
    test_ids = list(data["scene_id"][data["split"] == "test"])
    independent = []
    for seed_report in comparison["seeds"]:
        seed = seed_report["seed"]
        for branch in ("F", "K"):
            model = L.at(f"04_training/results/FK/models/cnn_{branch}_seed{seed}.pt")
            training = json.loads((L.at(f"04_training/results/FK/models/training_{branch}_seed{seed}.json")).read_text(encoding="utf-8"))
            require(model.exists() and model.stat().st_size > 0, f"Missing model {branch}/{seed}")
            require(training["dataset_sha256"] == data_sha, f"Model dataset mismatch: {branch}/{seed}")
        with (L.at(f"04_training/results/FK/predictions_seed{seed}.csv")).open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        require([row["scene_id"] for row in rows] == test_ids, "Prediction/test row mismatch")
        weak = [row for row in rows if float(row["mua_mm_inv"]) < 0.4]
        require(len(weak) == 300, "Expected 300 common weak-absorption test scenes")
        record = {"seed": seed}
        for branch in ("F", "K"):
            mae = float(np.mean([abs(float(row[f"pred_{branch}_mm"]) - float(row["depth_mm"])) for row in weak]))
            require(abs(mae - seed_report["groups"]["weak"][branch]["mae_mm"]) < 1e-6, f"MAE mismatch {branch}/{seed}")
            record[f"MAE_{branch}_mm"] = mae
        independent.append(record)
    for branch in ("F", "K"):
        require(abs(np.mean([r[f"MAE_{branch}_mm"] for r in independent]) - comparison[f"mean_weak_mae_{branch}_mm"]) < 1e-6,
                f"Mean MAE mismatch {branch}")
    result = {"pass": True, "main_MC_configurations": len(mc_report["sources"]), "samples_per_branch": 6000,
              "points_per_sample": 128, "model_count": 6, "split_counts_by_absorption": counts,
              "recomputed_weak_metrics_from_CSV": independent, "dataset_sha256": data_sha,
              "readout_version": READOUT_VERSION,
              "scope": "Artifact consistency checks; no physical generalization claim"}
    (L.at("maintenance/audit.json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
