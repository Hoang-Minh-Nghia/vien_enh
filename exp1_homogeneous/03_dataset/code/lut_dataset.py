"""Generate paired scene profiles from MC lookup tables, without camera data."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp1_paths as L  # noqa: E402  (step folders on sys.path; L.at() resolves stored paths)
import json
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from simulate import ROOT
from readout import READOUT_VERSION, bin_average


def generate(luts, output, seed=20261103, sizes=(4800, 600, 600), provenance=None):
    """luts maps mua to (depths, native x, native ratio profiles).

    All scenes derive from shared MC LUTs. Scene splitting prevents augmented
    duplicates crossing splits, but is not an independent physical MC test.
    """
    levels = np.array([0.15, 0.30, 0.508, 1.0])
    available = np.array(sorted(luts))
    paired = len(luts) == 4
    if paired and not np.allclose(available, levels):
        raise ValueError("Unexpected absorption levels")
    rng = np.random.default_rng(seed)
    scenes = []
    next_id = 0
    for split, count in zip(("train", "validation", "test"), sizes):
        if paired and count % 4:
            raise ValueError("Each split must be divisible by four")
        labels = np.tile(levels, count // 4) if paired else np.full(count, 0.508)
        rng.shuffle(labels)
        for absorption in labels:
            scenes.append({"scene_id": f"s{seed}_{next_id:06d}", "split": split,
                           "depth_mm": float(rng.uniform(0.5, 5.0)),
                           "shift_mm": float(rng.uniform(-2, 2)),
                           "mua_F_mm_inv": float(absorption), "mua_K_mm_inv": 0.508})
            next_id += 1
    interpolators = {m: interp1d(d, curves, axis=0, kind="linear", bounds_error=True)
                     for m, (d, _, curves) in luts.items()}
    edges = np.linspace(-16, 16, 129)
    Xk, Xf, depths, split_ids = [], [], [], []
    for scene in scenes:
        depth, shift = scene["depth_mm"], scene["shift_mm"]
        native_x = luts[0.508][1]
        k = bin_average(native_x, interpolators[0.508](depth), edges - shift)
        f = bin_average(native_x, interpolators[scene["mua_F_mm_inv"]](depth), edges - shift) if paired else k.copy()
        Xk.append(k)
        Xf.append(f)
        depths.append(depth)
        split_ids.append(scene["split"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {"X_K": np.asarray(Xk, np.float32), "X_F": np.asarray(Xf, np.float32),
              "y_mm": np.asarray(depths, np.float32), "split": np.asarray(split_ids),
              "scene_id": np.asarray([s["scene_id"] for s in scenes]),
              "mua_F": np.asarray([s["mua_F_mm_inv"] for s in scenes], np.float32),
              "shift_mm": np.asarray([s["shift_mm"] for s in scenes], np.float32)}
    if not all(np.isfinite(arrays[k]).all() for k in ("X_K", "X_F", "y_mm")):
        raise ValueError("Non-finite generated data")
    ids = [set(arrays["scene_id"][arrays["split"] == k]) for k in ("train", "validation", "test")]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError("Scene leakage")
    np.savez_compressed(output / "dataset.npz", **arrays)
    manifest = {"seed": seed, "sizes": dict(zip(("train", "validation", "test"), sizes)),
                "provenance": provenance,
                "camera": "ideal; no extra noise or blur", "scenes": scenes,
                "scope": "Interpolated shared MC LUT. Unique scene IDs do not imply independent physical simulations.",
                "interpolation": "linear in depth; spatial bin average of shifted native profile"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output / "dataset.npz"


def trial_dataset():
    learning = json.loads((L.at("configs/learning.json")).read_text(encoding="utf-8"))
    trial_cfg = learning["trial_dataset"]
    folder = L.at("02_forward_model/results")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if report.get("readout_version") != READOUT_VERSION:
        # Re-extract from immutable MC maps if extraction changed during a long run.
        # Analyze-only does not require pmcxcl and does not launch GPU simulations.
        subprocess.run([sys.executable, str(L.at("02_forward_model/code/depth_sweep.py")), "--analyze-only"], check=True)
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if not report["gate_pass"]:
        raise RuntimeError("Stage 2 interpolation checks did not pass")
    with np.load(folder / "lut_K.npz", allow_pickle=False) as lut:
        table = (lut["depths_mm"], lut["native_x_mm"], lut["native_profiles"])
    provenance = {"lut_path": "results/stage2/lut_K.npz",
                  "readout_version": READOUT_VERSION,
                  "lut_sha256": hashlib.sha256((folder / "lut_K.npz").read_bytes()).hexdigest()}
    return generate({0.508: table}, L.at("04_training/results/pilot_K/data"), seed=trial_cfg["seed"],
                    sizes=tuple(trial_cfg[k] for k in ("train", "validation", "test")), provenance=provenance)


if __name__ == "__main__":
    print(trial_dataset())
