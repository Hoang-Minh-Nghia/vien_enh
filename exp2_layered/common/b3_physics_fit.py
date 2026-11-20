"""Baseline B3: physics-based fit using only the stage-4 training lookup table.

Forward model: native ratio profiles from results/stage4/lut_FK.npz (4 mua x 12 depths x 600 x),
interpolated in (log mua, depth), then binned to the 128-point readout with a lateral shift,
exactly as dataset.py builds CNN inputs. Each V1 profile is fitted for (depth, mua, shift) by
least squares with multi-start. Two variants:
  B3-joint : depth, mua and shift free (mua within the trained range 0.15-1.0)
  B3-fixed : mua fixed at 0.508 (the physics analogue of K)
Scored on V1 (already opened; exploratory, not a locked test) with the V3-V6 grouping.
"""
import exp2_paths  # noqa: F401  (all step code folders on sys.path)
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares

ROOT = exp2_paths.EXP1
sys.path.insert(0, str(exp2_paths.exp1_at("common")))
from readout import bin_average  # noqa: E402  (locked module, used read-only)

OUT = exp2_paths.EXP2 / "04_training/results/b3_physics_fit"
EDGES = np.linspace(-16, 16, 129)
GROUPS = {"unseen_pooled": (0.22, 0.40), "unseen_0.22": (0.22,), "unseen_0.40": (0.40,),
          "seen_weak_pooled": (0.15, 0.30), "seen_weak_0.15": (0.15,), "seen_weak_0.30": (0.30,),
          "standard_0.508": (0.508,), "high_1.0": (1.0,), "all": (0.15, 0.22, 0.30, 0.40, 0.508, 1.0)}


class Forward:
    def __init__(self):
        with np.load(exp2_paths.exp1_at("results/stage4/lut_FK.npz"), allow_pickle=False) as z:
            self.levels, self.depths = z["levels"], z["depths_mm"]
            self.x, prof = z["native_x_mm"], z["native_profiles"]
        self.interp = RegularGridInterpolator((np.log(self.levels), self.depths), prof, method="cubic")

    def __call__(self, depth, mua, shift):
        native = self.interp([[np.log(mua), depth]])[0]
        return bin_average(self.x, native, EDGES - shift)


def fit(forward, y, joint):
    lo = [0.5, np.log(0.15), -2.0] if joint else [0.5, -2.0]
    hi = [5.0, np.log(1.0), 2.0] if joint else [5.0, 2.0]
    def resid(p):
        return forward(p[0], np.exp(p[1]), p[2]) - y if joint else forward(p[0], 0.508, p[1]) - y
    best = None
    for z0 in (0.8, 1.6, 2.6, 3.6, 4.6):
        starts = [[z0, np.log(a0), 0.0] for a0 in (0.2, 0.4, 0.8)] if joint else [[z0, 0.0]]
        for p0 in starts:
            r = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac")
            if best is None or r.cost < best.cost:
                best = r
    p = best.x
    return (p[0], float(np.exp(p[1])), p[2], float(best.cost)) if joint else (p[0], 0.508, p[1], float(best.cost))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    forward = Forward()
    # Sanity: the forward model reproduces a LUT node at shift 0 (SciPy's cubic solve leaves ~2e-5,
    # about 100x below the MC noise per bin).
    node = bin_average(forward.x, np.load(exp2_paths.exp1_at("results/stage4/lut_FK.npz"))["native_profiles"][1, 5], EDGES)
    assert np.abs(forward(forward.depths[5], forward.levels[1], 0.0) - node).max() < 1e-4

    with np.load(exp2_paths.exp1_at("results/validation_independent/predictions/profiles.npz")) as z:
        X, depth, mua, repeat = z["X"].astype(float), z["y_mm"].astype(float), z["mua_mm_inv"].astype(float), z["repeat"]
    rows = []
    for i in range(len(X)):
        zj, aj, sj, cj = fit(forward, X[i], True)
        zf, _, sf, cf = fit(forward, X[i], False)
        rows.append({"depth_mm": round(depth[i], 3), "mua": round(mua[i], 3), "repeat": int(repeat[i]),
                     "pred_joint_mm": zj, "mua_joint": aj, "shift_joint_mm": sj, "cost_joint": cj,
                     "pred_fixed_mm": zf, "shift_fixed_mm": sf, "cost_fixed": cf})
        print(f"{i+1}/180 z={depth[i]:.1f} mua={mua[i]:.3f} joint={zj:.3f} (mua {aj:.3f}) fixed={zf:.3f}", flush=True)
    with open(OUT / "b3_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # Condition-level MAE (mean over the 3 repeats), equal weight over conditions, as in V3-V6.
    cond = {}
    for r in rows:
        cond.setdefault((r["depth_mm"], r["mua"]), []).append(r)
    summary = {}
    for name, muas in GROUPS.items():
        keys = [k for k in cond if any(abs(k[1] - m) < 1e-9 for m in muas)]
        def mae(field):
            return float(np.mean([np.mean([abs(r[field] - r["depth_mm"]) for r in cond[k]]) for k in keys]))
        summary[name] = {"n_conditions": len(keys), "MAE_B3_joint_mm": mae("pred_joint_mm"), "MAE_B3_fixed_mm": mae("pred_fixed_mm")}
    worst = sorted(((np.mean([abs(r["pred_joint_mm"] - k[0]) for r in v]), k) for k, v in cond.items()), reverse=True)[:6]
    summary["worst_conditions_joint"] = [{"depth_mm": k[0], "mua": k[1], "MAE_mm": float(e)} for e, k in worst]
    summary["mua_estimate_mae"] = float(np.mean([abs(r["mua_joint"] - r["mua"]) for r in rows]))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
