"""V2: freeze the MC-quality data selection, then score the six locked CNNs on it.

The selection is written and hashed before torch or any checkpoint is touched, so
it cannot depend on a prediction. Tier choice uses quality_matrix.json only:
accepted conditions use their accepted tier; a condition that stayed unconverged
is kept at its highest tier and flagged, and a condition with invalid MC is not
scored. Run with the analysis environment.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp1_paths as L  # noqa: E402  (step folders on sys.path; L.at() resolves stored paths)
import csv
import json
import sys
import time

import numpy as np

from readout import READOUT_VERSION, load_run, profile
from simulate import ROOT
from run_independent_mc import MANIFEST, OUTPUT, load_locked_manifest, sha256

QUALITY = OUTPUT / "quality_matrix.json"
PRED_DIR = OUTPUT / "predictions"
SELECTION = PRED_DIR / "selection.json"
PREDICTIONS = PRED_DIR / "predictions.csv"
SELECTION_RULE = ("Set by MC quality only, before any CNN prediction: accepted -> accepted tier; valid but "
                  "unconverged after the highest tier -> highest tier, flagged insufficiently reliable; invalid -> not scored.")
FIELDS = ["run_id", "condition_id", "depth_mm", "mua_mm_inv", "mua_group", "repeat", "tier", "photons", "mc_seed",
          "n0_run_id", "quality_status", "model_seed", "true_depth_mm", "pred_F_mm", "pred_K_mm", "error_F_mm",
          "error_K_mm", "abs_error_K_minus_F_mm", "source_npz"]


def check_quality(report, manifest):
    """Refuse to open predictions unless the full locked matrix was assessed."""
    problems = []
    if report["scope"] != "matrix":
        problems.append(f"scope is {report['scope']}")
    if report["manifest_sha256"] != sha256(MANIFEST):
        problems.append("quality report used a different manifest")
    if report["n_conditions"] != manifest["design"]["n_conditions"]:
        problems.append(f"{report['n_conditions']} conditions assessed")
    if report["max_tier_allowed"] != len(manifest["mc"]["photon_tiers"]) - 1:
        problems.append("highest photon tier was not allowed")
    if report["cnn_predictions_opened"]:
        problems.append("report says predictions were already opened")
    if not all(n["validity"]["pass"] for n in report["n0_runs"]):
        problems.append("an N0 run is invalid")
    if problems:
        raise RuntimeError("V1 gate not met: " + "; ".join(problems))


def select(report, manifest):
    jobs = {}
    for job in manifest["channel_runs"]:
        jobs.setdefault(job["condition_id"], []).append(job)
    runs = []
    for condition in report["conditions"]:
        last = condition["attempts"][-1]
        status = "accepted" if condition["accepted"] else ("valid_not_converged" if last["valid"] else "invalid")
        tier = condition["accepted_tier"] if condition["accepted"] else last["tier"]
        for job in sorted(jobs[condition["condition_id"]], key=lambda j: j["repeat"]):
            item = job["photon_tiers"][tier]
            runs.append({"run_id": job["run_id"], "condition_id": job["condition_id"], "depth_mm": job["depth_mm"],
                         "mua_mm_inv": job["mua_mm_inv"], "mua_group": job["mua_group"], "repeat": job["repeat"],
                         "n0_run_id": job["n0_run_id"], "tier": tier, "photons": item["photons"], "mc_seed": item["seed"],
                         "output": item["output"], "quality_status": status})
    return runs


def freeze_selection(report, manifest):
    runs = select(report, manifest)
    for run in runs:
        run["sha256"] = sha256(L.at(run["output"]))
    n0 = {job["run_id"]: {"output": job["output"], "sha256": sha256(L.at(job["output"]))} for job in manifest["n0_runs"]}
    record = {"rule": SELECTION_RULE, "manifest_sha256": sha256(MANIFEST), "quality_report_sha256": sha256(QUALITY),
              "readout_version": READOUT_VERSION, "cnn_predictions_opened_at_selection": False, "n0": n0, "runs": runs}
    if SELECTION.exists():
        old = json.loads(SELECTION.read_text(encoding="utf-8"))
        if {k: v for k, v in old.items() if k != "frozen_local"} != record:
            raise RuntimeError("selection.json differs from the MC-quality selection; not overwriting it")
        return old
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    record["frozen_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    SELECTION.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def load_model(path, branch, seed):
    import torch
    from train import DepthCNN
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["branch"] != branch or checkpoint["seed"] != seed:
        raise RuntimeError(f"Checkpoint identity mismatch: {path}")
    model = DepthCNN()
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def predict(model, x):
    import torch
    with torch.no_grad():
        return model(torch.from_numpy(x[:, None, :])).numpy()


def main():
    manifest = load_locked_manifest()
    report = json.loads(QUALITY.read_text(encoding="utf-8"))
    check_quality(report, manifest)
    selection = freeze_selection(report, manifest)
    print(f"SELECTION_FROZEN {SELECTION} sha256={sha256(SELECTION)}", flush=True)
    if PREDICTIONS.exists():
        print(f"PREDICTIONS_ALREADY_EXIST {PREDICTIONS}", flush=True)
        return

    runs = [r for r in selection["runs"] if r["quality_status"] != "invalid"]
    background = {name: load_run(L.at(item["output"])) for name, item in selection["n0"].items()}
    x = np.stack([profile(load_run(L.at(r["output"])), background[r["n0_run_id"]])[1] for r in runs]).astype(np.float32)

    import torch
    torch.set_num_threads(4)
    outputs = {}
    for c in manifest["checkpoints"]:
        if sha256(L.at(c["path"])) != c["sha256"]:
            raise RuntimeError(f"Locked checkpoint changed: {c['path']}")
        outputs[(c["branch"], c["seed"])] = predict(load_model(L.at(c["path"]), c["branch"], c["seed"]), x)
    if not all(np.isfinite(v).all() for v in outputs.values()):
        raise RuntimeError("Non-finite CNN prediction")

    seeds = sorted({c["seed"] for c in manifest["checkpoints"]})
    with PREDICTIONS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        for seed in seeds:
            for i, run in enumerate(runs):
                truth = run["depth_mm"]
                f, k = float(outputs[("F", seed)][i]), float(outputs[("K", seed)][i])
                writer.writerow({**{key: run[key] for key in FIELDS if key in run}, "mc_seed": run["mc_seed"],
                                 "model_seed": seed, "true_depth_mm": truth, "pred_F_mm": f, "pred_K_mm": k,
                                 "error_F_mm": f - truth, "error_K_mm": k - truth,
                                 "abs_error_K_minus_F_mm": abs(k - truth) - abs(f - truth), "source_npz": run["output"]})
    np.savez_compressed(PRED_DIR / "profiles.npz", X=x, run_id=np.array([r["run_id"] for r in runs]),
                        y_mm=np.array([r["depth_mm"] for r in runs], dtype=np.float32),
                        mua_mm_inv=np.array([r["mua_mm_inv"] for r in runs], dtype=np.float32),
                        repeat=np.array([r["repeat"] for r in runs]))
    meta = {"selection_sha256": sha256(SELECTION), "predictions_sha256": sha256(PREDICTIONS), "n_inputs": len(runs),
            "n_model_predictions": len(runs) * len(outputs), "readout_version": READOUT_VERSION, "python": sys.version.split()[0],
            "numpy": np.__version__, "torch": torch.__version__, "completed_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "note": "Same inputs scored by six models; not independent samples."}
    (PRED_DIR / "predictions_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"PREDICTIONS_WRITTEN {PREDICTIONS} inputs={len(runs)} predictions={meta['n_model_predictions']}", flush=True)


if __name__ == "__main__":
    main()
