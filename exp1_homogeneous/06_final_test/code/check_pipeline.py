"""Check the V2-V6 code without opening any prediction on the validation runs.

Uses only data already seen in stage 4, the MC-quality pilot report, and
synthetic predictions. Pilot profiles are computed but never given to a model.
Not named test_*.py so the package's documented unit-test count stays unchanged.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp1_paths as L  # noqa: E402  (step folders on sys.path; L.at() resolves stored paths)
import csv
import json
import sys
import tempfile

import numpy as np

from readout import load_run, profile
from simulate import ROOT
from run_independent_mc import OUTPUT, load_locked_manifest
from predict_independent import load_model, predict, select
from report_independent import MUAS, build


def check_checkpoint_loading():
    with np.load(L.at("03_dataset/data/dataset_FK/dataset.npz"), allow_pickle=False) as z:
        test = z["split"] == "test"
        x = z["X_F"][test].astype(np.float32)
        scene = z["scene_id"][test]
    with open(L.at("04_training/results/FK/predictions_seed42.csv"), newline="", encoding="utf-8") as f:
        stored = {r["scene_id"]: r for r in csv.DictReader(f)}
    for branch in "FK":
        pred = predict(load_model(L.at(f"04_training/results/FK/models/cnn_{branch}_seed42.pt"), branch, 42), x)
        gap = max(abs(float(pred[i]) - float(stored[str(scene[i])][f"pred_{branch}_mm"])) for i in range(len(pred)))
        assert gap < 1e-4, (branch, gap)
        print(f"checkpoint {branch} seed 42 reproduces stage 4 predictions, max gap {gap:.2e} mm")


def check_selection_on_pilot():
    manifest = load_locked_manifest()
    report = json.loads((OUTPUT / "quality_pilot.json").read_text(encoding="utf-8"))
    runs = select(report, manifest)
    assert len(runs) == 18 and all(r["tier"] == 0 and r["quality_status"] == "accepted" for r in runs)
    fake = json.loads(json.dumps(report))
    fake["conditions"][0].update(accepted=False, accepted_tier=None)
    fake["conditions"][0]["attempts"][-1].update(valid=True, converged=False, tier=2)
    fake["conditions"][1].update(accepted=False, accepted_tier=None)
    fake["conditions"][1]["attempts"][-1].update(valid=False, tier=1)
    runs = select(fake, manifest)
    by_condition = {r["condition_id"]: r for r in runs}
    assert by_condition[fake["conditions"][0]["condition_id"]]["quality_status"] == "valid_not_converged"
    assert by_condition[fake["conditions"][0]["condition_id"]]["output"].endswith("ph240M.npz")
    assert by_condition[fake["conditions"][1]["condition_id"]]["quality_status"] == "invalid"
    n0 = load_run(L.at(manifest["n0_runs"][0]["output"]))
    item = select(report, manifest)[0]
    assert item["n0_run_id"] == "N0_r1"
    assert profile(load_run(L.at(item["output"])), n0)[1].shape == (128,)
    print("selection rule and analysis-environment file loading are correct on the pilot report")


def synthetic_rows():
    depths = (0.6, 0.9, 1.4, 1.8, 2.3, 2.8, 3.3, 3.8, 4.3, 4.8)
    offset_k = {42: 0.3, 43: 0.4, 44: -0.05}
    rows = []
    for depth in depths:
        for mua in MUAS:
            condition = f"z{depth:.3f}_m{mua:.3f}"
            status = "valid_not_converged" if condition == "z3.300_m0.220" else "accepted"
            for repeat in (1, 2, 3):
                for seed in (42, 43, 44):
                    ef, ek = 0.1, offset_k[seed]
                    rows.append({"condition_id": condition, "model_seed": seed, "repeat": repeat, "depth_mm": depth, "mua_mm_inv": mua,
                                 "quality_status": status, "true_depth_mm": depth, "pred_F_mm": depth + ef, "pred_K_mm": depth + ek,
                                 "error_F_mm": ef, "error_K_mm": ek, "abs_error_K_minus_F_mm": abs(ek) - abs(ef)})
    return rows


def check_report():
    with tempfile.TemporaryDirectory() as folder:
        result = build(synthetic_rows(), 2.7229, 0.5, folder)
        accepted, every = result["analysis_sets"]["accepted"], result["analysis_sets"]["all_valid"]
        unseen = accepted["groups"]["unseen_pooled"]
        assert unseen["n_conditions"] == 19 and every["groups"]["unseen_pooled"]["n_conditions"] == 20
        assert abs(unseen["by_seed"][42]["delta_MAE_K_minus_F_mm"] - 0.2) < 1e-9
        assert abs(unseen["by_seed"][44]["delta_MAE_K_minus_F_mm"] + 0.05) < 1e-9
        assert abs(unseen["mean"]["delta_MAE_K_minus_F_mm"] - (0.2 + 0.3 - 0.05) / 3) < 1e-9
        assert unseen["F_lower_MAE_all_seeds"] is False
        assert all(abs(r["slope_F"] - 1) < 1e-9 and abs(r["intercept_F_mm"] - 0.1) < 1e-9 for r in accepted["depth_discrimination"])
        assert all(abs(r["R_F_mm"]) < 1e-9 and abs(r["R_K_mm"]) < 1e-9 for r in accepted["absorption_stability"])
        text = open(f"{folder}/KET_LUAN_V2_V6.md", encoding="utf-8").read()
        assert "không phải cả ba seed" in text and "Kiểm độ nhạy" in text
    print("V3-V6 aggregation, figures and report text are correct on synthetic predictions")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    check_checkpoint_loading()
    check_selection_on_pilot()
    check_report()
    print("PIPELINE_CHECK_PASS")
