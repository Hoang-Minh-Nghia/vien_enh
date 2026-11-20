"""phantom phantom design numbers (no MC run, read-only on existing results).

1. Error budget: depth-equivalent bias when a phantom optical/geometric parameter is mis-measured by 1 sd and the
   profile is explained by depth (+ lateral shift) only. Uses the smoothed step 05 Jacobians (40 validation_ext points).
   This bounds how well the "MC of the phantom vs measured phantom" check and a physics fit (B3-type) can work.
   It does NOT enter the network-vs-ground-truth comparison (the network never sees the phantom's properties).
2. Number of phantom blocks: per-skin MAE of the final hetero ensemble on locked_test (saved predictions.npz,
   sigma_tot = 0.005), bootstrap CI half-width of the mean over n blocks. Only reads the saved file; no re-scoring.

Writes phantom/results/design/budget.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "exp2_layered" / "common"))
import exp2_paths  # noqa: E402,F401
import crlb_analysis as A  # noqa: E402
import cnn_variants as V  # noqa: E402

RES = exp2_paths.EXP2
OUT = Path(__file__).resolve().parents[1] / "results" / "design"
BANDS = [(0, 2), (2, 3), (3, 4), (4, 9)]

# 1 sd measurement uncertainty, in the T6a parametrisation (log params -> log(1 + r))
SCEN = {
    "typical": {"derm_musp": np.log(1.10), "fat_musp": np.log(1.10), "musc_musp": np.log(1.10),
                "derm_mua": np.log(1.20), "fat_mua": np.log(1.20), "musc_mua": np.log(1.20),
                "mua": np.log(1.10), "mus": np.log(1.10),
                "dermis_mm": 0.10, "fat_mm": 0.5, "diameter_mm": 0.10},
    "strict": {"derm_musp": np.log(1.05), "fat_musp": np.log(1.05), "musc_musp": np.log(1.10),
               "derm_mua": np.log(1.10), "fat_mua": np.log(1.05), "musc_mua": np.log(1.20),
               "mua": np.log(1.05), "mus": np.log(1.10),
               "dermis_mm": 0.05, "fat_mm": 0.5, "diameter_mm": 0.05},
}


def budget():
    files = sorted(f for f in A.D.glob("pt_*.npz") if not f.name.endswith(".partial.npz"))
    b = V.Binner(np.load(files[0])["x_mm"])
    Bs = A.spline_basis(); H = Bs @ np.linalg.pinv(Bs)
    O = A.ORDER
    depth, unit = [], {k: [] for k in SCEN["typical"]}
    for f in files:
        z = np.load(f); p, J, _ = A.jacobians(z, b)
        Jm = H @ ((J["a"] + J["b"]) / 2)
        pinv = np.linalg.pinv(Jm[:, [O.index("depth_mm"), O.index("shift")]])
        depth.append(p["vessel"]["depth_mm"])
        for k in unit:                                   # depth shift per unit parameter change
            unit[k].append(abs((pinv @ Jm[:, O.index(k)])[0]))
    d = np.array(depth); unit = {k: np.array(v) for k, v in unit.items()}
    out = {"n_points": len(files)}
    for name, sc in SCEN.items():
        rows, tot = {}, np.zeros(len(d))
        for k, u in sc.items():
            r = unit[k] * u; tot += r ** 2
            rows[k] = {"sd": float(u), "median_mm": float(np.median(r)),
                       "bands_mm": [float(np.median(r[(d >= lo) & (d < hi)])) for lo, hi in BANDS]}
        tot = np.sqrt(tot)
        rows["total"] = {"median_mm": float(np.median(tot)),
                         "bands_mm": [float(np.median(tot[(d >= lo) & (d < hi)])) for lo, hi in BANDS]}
        out[name] = rows
    return out


def blocks():
    z = np.load(RES / "06_final_test/results/locked_test/predictions.npz", allow_pickle=True)
    y, c, p = z["y"], z["config"], z["tot_0.005__hetero"]
    k = len(p) // len(y)
    e = p - np.tile(y, k); cr = np.tile(c, k)
    sk = np.unique(c)
    mae = np.array([np.mean(np.abs(e[cr == s])) for s in sk])
    bias = np.array([np.mean(e[cr == s]) for s in sk])
    rng = np.random.default_rng(0)
    ci = {}
    for n in (1, 2, 3, 4, 6, 8, 10):
        m = np.array([rng.choice(mae, n).mean() for _ in range(4000)])
        ci[n] = float((np.percentile(m, 97.5) - np.percentile(m, 2.5)) / 2)
    return {"n_skins": int(len(sk)), "per_skin_mae_mean": float(mae.mean()), "per_skin_mae_sd": float(mae.std(ddof=1)),
            "per_skin_bias_sd": float(bias.std(ddof=1)), "ci95_halfwidth_by_n_blocks": ci}


def gt_inflation(rmse=0.26, mae=0.21):
    """Observed MAE when ground truth has independent Gaussian error sd s (errors ~ Gaussian)."""
    return {f"{s:.2f}": float(mae * np.sqrt(rmse ** 2 + s ** 2) / rmse) for s in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"budget": budget(), "blocks": blocks(), "gt_inflation": gt_inflation()}
    (OUT / "budget.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for name in SCEN:
        print(name)
        for k, r in res["budget"][name].items():
            print(f"  {k:12s} {r['median_mm']:.3f}  " + " ".join(f"{x:.3f}" for x in r["bands_mm"]))
    print(json.dumps(res["blocks"], indent=1)); print(res["gt_inflation"])
