"""Fit the homogeneous B3 model to the homogeneous-background / scattering-vessel profiles."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp2_paths  # noqa: E402,F401  (all step code folders on sys.path)
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(exp2_paths.EXP1 / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from readout import load_run, profile  # noqa: E402
from b3_physics_fit import Forward, fit  # noqa: E402

R = exp2_paths.EXP2 / "02_forward_model/results/split"
n0 = load_run(R / "n0_hom.npz")
f = Forward()
out = []
for z in (1.6, 2.6, 3.6):
    y = profile(load_run(R / f"z{z:.1f}_hom_scatvessel.npz"), n0)[1].astype(float)
    zj, aj, sj, cj = fit(f, y, True)
    ref = f(z, 0.46, 0.0)     # same depth and mua, non-scattering vessel (stage-4 LUT)
    out.append({"depth_mm": z, "contrast_scattering_vessel": float(1 - y.min()),
                "contrast_nonscattering_lut_mua0.46": float(1 - ref.min()),
                "pred_mm": zj, "bias_mm": zj - z, "mua_fit": aj, "rms_residual": float(np.sqrt(2 * cj / len(y)))})
    print(json.dumps(out[-1]))
(R / "split_fit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
