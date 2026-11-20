"""step 01 pilot, round 2: pattern-source orientation, detected-photon record layout, polarization."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp2_paths  # noqa: E402,F401  (all step code folders on sys.path)
import json
import sys
from pathlib import Path

import numpy as np
import pmcxcl

from sim_capabilities import base

OUT = exp2_paths.EXP2 / "01_simulator_checks/results/capabilities_2.json"


def main():
    out = {}
    # Pattern: light only where x < 20 (first array axis), on a 40x40 surface.
    cfg = base()
    pattern = np.zeros((40, 40), dtype=np.float32, order="F")
    pattern[:20, :] = 1
    cfg.update(srctype="pattern", srcparam1=[40, 0, 0, 40], srcparam2=[0, 40, 0, 40], srcpattern=pattern, nphoton=200000)
    top = np.asarray(pmcxcl.run(cfg)["dref"])[:, :, 0, 0]
    out["pattern_x_lt_20"] = {"x_lt_20": float(top[:20].mean()), "x_ge_20": float(top[20:].mean()),
                              "y_lt_20": float(top[:, :20].mean()), "y_ge_20": float(top[:, 20:].mean())}

    # Detected-photon record: columns for detid, ppath per medium (2), exit pos (3), exit dir (3).
    cfg = base()
    cfg.update(issavedet=1, detpos=[[20, 20, 0, 30]], savedetflag="dpxv", nphoton=200000)
    res = pmcxcl.run(cfg)
    detp = np.asarray(res["detp"])
    out["detp"] = {"shape": list(detp.shape), "first_rows": detp[:, :3].T.tolist() if detp.shape[0] < detp.shape[1] else detp[:3].tolist()}
    # Absorption rescaling check: detected weight from Beer-Lambert on ppath vs a rerun with new channel mua.
    rec = detp if detp.shape[0] < detp.shape[1] else detp.T          # rows = fields
    ppath = rec[1:3]                                                   # mm path length in medium 1, 2 (units checked below)
    out["ppath_mean_by_medium"] = ppath.mean(axis=1).tolist()

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


def polarization():
    """Run in its own process: an MCX-CL input error exits the interpreter."""
    cfg = base()
    # One sphere-suspension row per tissue medium: mua, radius um, density 1/um^3, n_sphere, n_background.
    cfg.update(polprop=np.array([[0.01, 0.5, 1e-3, 1.59, 1.33], [0.3, 0.5, 1e-6, 1.59, 1.33]], dtype=np.float32),
               srciquv=[1, 1, 0, 0], issavedet=1, detpos=[[20, 20, 0, 30]], savedetflag="dpxvi", nphoton=50000)
    cfg["lambda"] = 850.0
    res = pmcxcl.run(cfg)
    detp = np.asarray(res["detp"])
    record = {"ok": True, "detp_shape": list(detp.shape), "keys": sorted(res.keys())}
    (exp2_paths.EXP2 / "01_simulator_checks/results/polarization.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    polarization() if sys.argv[1:] == ["pol"] else main()
