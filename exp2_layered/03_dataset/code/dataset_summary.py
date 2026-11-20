"""Aggregate sanity check of step 03 train/validation files (never opens locked_test).
Writes exp2_layered/03_dataset/data/check_train_validation.json."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp2_paths  # noqa: E402,F401  (all step code folders on sys.path)
import json
from pathlib import Path

import numpy as np

D = exp2_paths.EXP2 / "03_dataset/data"
m = json.loads((D / "manifest.json").read_text(encoding="utf-8"))
out = {}
for split in ("train", "validation"):
    ids = [c["id"] for c in m["splits"][split]["configs"]]
    files = [D / split / f"{i}.npz" for i in ids]
    missing = [f.name for f in files if not f.exists()]
    far, contrast, ripple, n0, nonfinite, mismatch, depth, argmin = [], [], [], [], 0, 0, [], []
    for f, cm in zip(files, m["splits"][split]["configs"]):
        if not f.exists():
            continue
        z = np.load(f)
        c = json.loads(str(z["config_json"]))
        if c != cm:
            mismatch += 1
        x, P, s = z["x_mm"], z["profiles_y_mean"].astype(float), float(z["n0_scalar"])
        if not (np.isfinite(P).all() and np.isfinite(s) and s > 0):
            nonfinite += 1
            continue
        n0.append(s)
        lat = z["n0_y_mean"].astype(float) / s
        ripple.append(lat.std())
        for v, p in zip(c["vessels"], P):
            r = p / s
            far.append(r[np.abs(x) > 20].mean())
            sm = np.convolve(r, np.ones(9) / 9, mode="same")
            contrast.append(1 - sm[np.abs(x) < 10].min())
            argmin.append(x[np.abs(x) < 10][np.argmin(sm[np.abs(x) < 10])])
            depth.append(v["depth_mm"])
    far, contrast, depth, argmin = map(np.array, (far, contrast, depth, argmin))
    rho = float(np.corrcoef(depth, np.log(contrast))[0, 1])
    out[split] = {
        "configs_expected": len(ids), "configs_present": len(ids) - len(missing), "missing": missing,
        "config_json_mismatch_vs_manifest": mismatch, "nonfinite_files": nonfinite, "vessels": int(len(far)),
        "far_field_ratio_mean": float(far.mean()), "far_field_ratio_std": float(far.std()),
        "far_field_ratio_min_max": [float(far.min()), float(far.max())],
        "n0_scalar_min_max": [float(min(n0)), float(max(n0))],
        "n0_lateral_ripple_median_max": [float(np.median(ripple)), float(max(ripple))],
        "contrast_smoothed_min_median_max": [float(contrast.min()), float(np.median(contrast)), float(contrast.max())],
        "corr_depth_vs_log_contrast": rho,
        "argmin_x_abs_median_max_mm": [float(np.median(np.abs(argmin))), float(np.abs(argmin).max())],
    }
    print(split, json.dumps(out[split], indent=1))
(D / "check_train_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
