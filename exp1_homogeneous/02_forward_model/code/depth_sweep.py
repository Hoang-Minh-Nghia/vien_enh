"""Build the standard-absorption depth LUT, then verify held-out interpolation."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp1_paths as L  # noqa: E402  (step folders on sys.path; L.at() resolves stored paths)
import argparse
import json

import numpy as np
from scipy.interpolate import interp1d

from simulate import ROOT, load_config, config_hash, run
from readout import READOUT_VERSION, load_run, profile, native_profile, diagnostics, compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    stage1 = L.at("01_simulator_checks/results")
    gate = json.loads((stage1 / "report.json").read_text(encoding="utf-8"))
    if not gate["gate_pass"]:
        raise RuntimeError("Stage 1 checks have not passed; refusing to expand the dataset")
    out = L.at("02_forward_model/results")
    out.mkdir(parents=True, exist_ok=True)
    bg = load_run(stage1 / "N0.npz")
    xs, profiles, sources, native_profiles = None, [], [], []
    for i, depth in enumerate(cfg["depths_mm"]):
        path = stage1 / "channel_2mm.npz" if depth == 2.0 else out / f"depth_{depth:.3f}.npz"
        if not args.analyze_only and depth != 2.0:
            run(cfg, path, depth, 0.508, seed=6000 + i)
        ch = load_run(path)
        xs, curve = profile(ch, bg)
        profiles.append(curve)
        native_profiles.append(native_profile(ch, bg))
        sources.append({"path": str(path.relative_to(ROOT)), "identity_hash": str(ch["identity_hash"])})
        print(f"DEPTH_COMPLETE {depth}", flush=True)
    depths = np.asarray(cfg["depths_mm"])
    profiles = np.asarray(profiles, np.float32)
    interpolator = interp1d(depths, profiles, axis=0, kind="linear", bounds_error=True)
    checks = []
    # These are validation of numerical interpolation, not the final CNN test.
    for i, depth in enumerate((0.625, 2.25, 4.75)):
        path = out / f"interpolation_check_{depth:.3f}.npz"
        if not args.analyze_only:
            run(cfg, path, depth, 0.508, seed=7000 + i)
        observed = profile(load_run(path), bg)[1]
        predicted = interpolator(depth)
        observed_diag = diagnostics(xs, observed)
        predicted_diag = diagnostics(xs, predicted)
        checks.append({"depth_mm": depth, "observed": observed_diag, "interpolated": predicted_diag,
                       **compare(observed_diag, predicted_diag),
                       "profile_mae": float(np.mean(np.abs(observed - predicted)))})
    report = {"config_hash": config_hash(cfg), "readout_version": READOUT_VERSION, "interpolation_checks": checks,
              "gate_pass": all(c["pass"] for c in checks),
              "profiles": [{"depth_mm": d, **diagnostics(xs, p)} for d, p in zip(depths.tolist(), profiles)],
              "scope": "Ideal simulation; training data from one shared MC LUT; held-out direct MC checks are not LUT knots.",
              "interpolation": "linear in depth; no extrapolation", "sources": sources}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(out / "lut_K.npz", depths_mm=depths, x_mm=xs,
                        profiles=profiles, native_x_mm=bg["x_mm"], native_profiles=np.asarray(native_profiles),
                        config_hash=config_hash(cfg), readout_version=READOUT_VERSION, sources_json=json.dumps(sources))
    print(json.dumps(report, indent=2), flush=True)
    if not report["gate_pass"]:
        raise RuntimeError("Depth interpolation checks failed; do not generate main datasets")


if __name__ == "__main__":
    main()
