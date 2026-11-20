"""Compare regenerated Monte-Carlo data with the data stored in the repository.

Usage:
    python tools/compare_regenerated.py <original_repo> <regenerated_repo>

Every MC data file that exists in both folders is compared (files missing in the regenerated copy are skipped):

  experiment 1   01_simulator_checks/results/*.npz, 02_forward_model/results/*.npz, 03_dataset/data/mc/*.npz
                 (one MC run each: reflectance map R_xy, profile R), lookup tables, dataset_FK/dataset.npz
  experiment 2   03_dataset/data/manifest*.json and 03_dataset/data/<split>/<id>.npz
                 (one skin each: vein profiles and N0)

Monte Carlo is reproducible run by run on the same GPU and driver (same seeds). On another GPU the photon
streams differ, so files agree only to Monte-Carlo noise. The checks below therefore compare against the
noise of each file:
  * mean reflectance / N0 within 0.5 %;
  * profile difference (RMS of 1 mm bins over the central +-16 mm) below 3 x the MC noise expected for the
    difference of two independent runs.
Exit code 0 when every compared file passes.
"""
import json
import sys
from pathlib import Path

import numpy as np


def far_noise(x, p):
    """MC noise of one 0.1 mm point: spread of point-to-point differences far from the vein (|x| > 12 mm)."""
    m = np.abs(x) > 12
    return float(np.std(np.diff(p[m])) / np.sqrt(2))


def mm_bins(x, p):
    """Average the central +-16 mm into 1 mm bins (10 points each)."""
    c = np.abs(x) <= 16
    q = p[c][: (c.sum() // 10) * 10]
    return q.reshape(-1, 10).mean(1)


def profile_check(x, pa, pb):
    """RMS difference of 1 mm bins vs. the expected Monte-Carlo difference of two independent runs."""
    rms = float(np.sqrt(np.mean((mm_bins(x, pa) - mm_bins(x, pb)) ** 2)))
    lim = 3 * np.sqrt(2) * far_noise(x, pa) / np.sqrt(10)
    return rms, lim


def cmp_run(a, b):
    """Experiment 1: one MC run (R_xy, R, x_mm)."""
    za, zb = np.load(a), np.load(b)
    same_id = str(za["identity_hash"]) == str(zb["identity_hash"])
    mean_rel = abs(float(zb["R"].mean()) / float(za["R"].mean()) - 1)
    x = za["x_mm"]
    rms, lim = profile_check(x, za["R"] / za["R"].mean(), zb["R"] / zb["R"].mean())
    ok = same_id and mean_rel < 0.005 and rms <= max(lim, 1e-12)
    return ok, f"same config/seed {same_id}, mean diff {mean_rel:.2e}, profile RMS diff {rms:.2e} (limit {lim:.2e})"


def cmp_skin(a, b):
    """Experiment 2: one skin (profiles_y_mean per vein, n0_scalar)."""
    za, zb = np.load(a), np.load(b)
    same_cfg = json.loads(str(za["config_json"])) == json.loads(str(zb["config_json"]))
    n0a, n0b = float(za["n0_scalar"]), float(zb["n0_scalar"])
    n0_rel = abs(n0b / n0a - 1)
    x = za["x_mm"]
    worst, worst_lim = 0.0, 0.0
    ok = same_cfg and n0_rel < 0.005
    for pa, pb in zip(za["profiles_y_mean"] / n0a, zb["profiles_y_mean"] / n0b):
        rms, lim = profile_check(x, pa, pb)
        ok &= rms <= max(lim, 1e-12)
        if rms / max(lim, 1e-12) >= worst / max(worst_lim, 1e-12):
            worst, worst_lim = rms, lim
    return ok, f"same config {same_cfg}, N0 diff {n0_rel:.2e}, worst vein RMS diff {worst:.2e} (limit {worst_lim:.2e})"


def cmp_arrays(a, b, keys, rel_tol):
    za, zb = np.load(a), np.load(b)
    out, ok = [], True
    for k in keys:
        if k not in za.files:
            continue
        va, vb = np.asarray(za[k], float), np.asarray(zb[k], float)
        if va.shape != vb.shape:
            return False, f"{k}: shape {va.shape} vs {vb.shape}"
        d = float(np.max(np.abs(va - vb)) / (np.max(np.abs(va)) + 1e-30))
        ok &= d <= rel_tol
        out.append(f"{k} max rel diff {d:.2e}")
    return ok, ", ".join(out)


def main(orig, new):
    jobs = []
    for exp, sub, kind in (("exp1_homogeneous", "01_simulator_checks/results", "run"),
                           ("exp1_homogeneous", "02_forward_model/results", "run"),
                           ("exp1_homogeneous", "03_dataset/data/mc", "run")):
        for f in sorted((new / exp / sub).glob("*.npz")):
            if f.name.endswith(".partial.npz") or "identity_hash" not in np.load(f).files:
                continue                                  # only MC runs (lookup tables are compared below)
            jobs.append((kind, orig / exp / sub / f.name, f))
    for rel in ("exp1_homogeneous/02_forward_model/results/lut_K.npz", "exp1_homogeneous/03_dataset/data/lut_FK.npz"):
        jobs.append(("lut", orig / rel, new / rel))
    jobs.append(("dataset", orig / "exp1_homogeneous/03_dataset/data/dataset_FK/dataset.npz",
                 new / "exp1_homogeneous/03_dataset/data/dataset_FK/dataset.npz"))
    d2o, d2n = orig / "exp2_layered/03_dataset/data", new / "exp2_layered/03_dataset/data"
    for f in sorted(d2n.glob("manifest*.json")):
        jobs.append(("manifest", d2o / f.name, f))
    for split in sorted(p.name for p in d2n.iterdir() if p.is_dir()) if d2n.exists() else []:
        for f in sorted((d2n / split).glob("*.npz")):
            if not f.name.endswith(".partial.npz"):
                jobs.append(("skin", d2o / split / f.name, f))

    n_ok = n_fail = n_skip = 0
    for kind, a, b in jobs:
        if not (a.exists() and b.exists()):
            n_skip += 1
            continue
        if kind == "run":
            ok, msg = cmp_run(a, b)
        elif kind == "skin":
            ok, msg = cmp_skin(a, b)
        elif kind == "lut":
            ok, msg = cmp_arrays(a, b, ["native_profiles", "profiles"], 0.02)
        elif kind == "dataset":
            ok, msg = cmp_arrays(a, b, ["y_mm", "mua_F", "shift_mm", "X_F", "X_K"], 0.02)
        else:
            ok = json.loads(a.read_text(encoding="utf-8")) == json.loads(b.read_text(encoding="utf-8"))
            msg = "identical" if ok else "DIFFERENT"
        n_ok += ok; n_fail += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {b.relative_to(new).as_posix()}: {msg}")
    print(f"\n{n_ok} passed, {n_fail} failed, {n_skip} not present in both (skipped)")
    return 0 if n_fail == 0 and n_ok > 0 else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()))
