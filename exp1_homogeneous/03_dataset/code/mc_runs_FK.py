"""Expand to four absorptions only after numerical and development gates pass."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp1_paths as L  # noqa: E402  (step folders on sys.path; L.at() resolves stored paths)
import argparse
import json
import hashlib

import numpy as np
from scipy.interpolate import interp1d

from simulate import ROOT, load_config, config_hash, run
from readout import READOUT_VERSION, load_run, profile, native_profile, diagnostics, compare
from lut_dataset import generate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()
    stage3 = L.at("04_training/results/pilot_K/trial_report.json")
    trial = json.loads(stage3.read_text(encoding="utf-8"))
    if trial["cnn"]["mae_mm"] >= trial["constant_training_mean"]["mae_mm"]:
        raise RuntimeError("Development CNN has not surpassed constant baseline")
    for stage in (1, 2):
        if not json.loads((L.at(f"results/stage{stage}/report.json")).read_text(encoding="utf-8"))["gate_pass"]:
            raise RuntimeError(f"Stage {stage} gate has not passed")
    cfg = load_config()
    out = L.at("03_dataset/data/mc").parent        # 03_dataset/data
    out.mkdir(parents=True, exist_ok=True)
    bg = load_run(L.at("01_simulator_checks/results/N0.npz"))
    with np.load(L.at("02_forward_model/results/lut_K.npz"), allow_pickle=False) as k:
        if str(k["config_hash"]) != config_hash(cfg):
            raise ValueError("K LUT config identity does not match")
        if str(k["readout_version"]) != READOUT_VERSION:
            raise ValueError("K LUT preprocessing version does not match")
        depths, native_x = k["depths_mm"].copy(), k["native_x_mm"].copy()
        tables = {0.508: (depths, native_x, k["native_profiles"].copy())}
        base_sources = json.loads(str(k["sources_json"]))
    checks = []
    sources = [{**s, "mua_mm_inv": 0.508, "depth_mm": float(d)} for s, d in zip(base_sources, depths)]
    for j, mua in enumerate((0.15, 0.30, 1.0)):
        curves = []
        for i, depth in enumerate(depths):
            path = out / "mc" / f"mua_{mua:.3f}_depth_{depth:.3f}.npz"
            if not args.analyze_only:
                run(cfg, path, float(depth), mua, seed=10000 + 100*j + i)
            ch = load_run(path)
            curves.append(native_profile(ch, bg))
            sources.append({"path": str(path.relative_to(ROOT)), "identity_hash": str(ch["identity_hash"]),
                            "mua_mm_inv": mua, "depth_mm": float(depth)})
        tables[mua] = (depths, native_x, np.asarray(curves))
        # Representative interpolation checks, not an exhaustive validation of every depth.
        check_depth = {0.15: 2.25, 0.30: 4.75, 1.0: 0.625}[mua]
        path = out / "mc" / f"heldout_mua_{mua:.3f}_depth_{check_depth:.3f}.npz"
        if not args.analyze_only:
            run(cfg, path, check_depth, mua, seed=12000 + j)
        x, observed = profile(load_run(path), bg)
        predicted_native = interp1d(depths, tables[mua][2], axis=0, bounds_error=True)(check_depth)
        from readout import bin_average
        predicted = bin_average(native_x, predicted_native, np.linspace(-16, 16, 129))
        comparison = compare(diagnostics(x, observed), diagnostics(x, predicted))
        checks.append({"kind": "interpolation", "mua_mm_inv": mua, "depth_mm": check_depth, **comparison})
        # Independent seed at two optical/depth extremes.
        repeat_depth = 5.0 if mua == 0.15 else 0.5 if mua == 1.0 else None
        if repeat_depth is not None:
            ref = load_run(out / "mc" / f"mua_{mua:.3f}_depth_{repeat_depth:.3f}.npz")
            path = out / "mc" / f"repeat_mua_{mua:.3f}_depth_{repeat_depth:.3f}.npz"
            if not args.analyze_only:
                run(cfg, path, repeat_depth, mua, seed=13000 + j)
            x, repeated = profile(load_run(path), bg)
            reference = profile(ref, bg)[1]
            checks.append({"kind": "independent_seed", "mua_mm_inv": mua, "depth_mm": repeat_depth,
                           **compare(diagnostics(x, reference), diagnostics(x, repeated))})
        (out / "progress.json").write_text(json.dumps({"completed_absorptions": sorted(tables), "checks": checks}, indent=2), encoding="utf-8")
    report = {"gate_pass": all(c["pass"] for c in checks), "checks": checks, "sources": sources,
              "config_hash": config_hash(cfg), "readout_version": READOUT_VERSION,
              "scope": "Representative numerical checks; not exhaustive optical validation or physical camera validation"}
    (out / "simulation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["gate_pass"]:
        raise RuntimeError("Expanded numerical checks failed; refusing to create final F/K data")
    levels = sorted(tables)
    np.savez_compressed(out / "lut_FK.npz", levels=levels, depths_mm=depths, native_x_mm=native_x,
                        native_profiles=np.asarray([tables[a][2] for a in levels]),
                        config_hash=config_hash(cfg), readout_version=READOUT_VERSION, sources_json=json.dumps(sources))
    data_cfg = json.loads((L.at("configs/learning.json")).read_text(encoding="utf-8"))["dataset"]
    path = generate(tables, out / "dataset_FK", seed=data_cfg["seed"],
                    sizes=tuple(data_cfg[k] for k in ("train", "validation", "test")), provenance={
        "lut_path": "results/stage4/lut_FK.npz",
        "readout_version": READOUT_VERSION,
        "lut_sha256": hashlib.sha256((out / "lut_FK.npz").read_bytes()).hexdigest()})
    print(f"DATASET_READY {path}", flush=True)


if __name__ == "__main__":
    main()
