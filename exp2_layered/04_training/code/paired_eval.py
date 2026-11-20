"""step 03 paired test evaluation: models trained on the same 120 skins at 12 M (A, *_n120) vs 96 M (B, *_hisnr) photons.

Scored on validation_ext (90 new skins, sigma_mc ~0.00355) at total per-bin noise 0.004/0.005/0.007/0.010 (primary)
plus 0.00355 (MC only) and 0.014 for the floor/k fit  RMSE^2 = floor^2 + (k sigma)^2.
Writes exp2_layered/04_training/results/paired/paired_eval.json.  locked_test is never read.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp2_paths  # noqa: E402,F401  (all step code folders on sys.path)
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cnn_variants as V  # noqa: E402

R = exp2_paths.EXP2
PRIMARY = (0.004, 0.005, 0.007, 0.010)
EXTRA = (0.0, 0.014)   # 0.0 -> MC noise only


def main():
    torch.set_num_threads(8)
    x, P, meta = V.load("validation_ext", "manifest_ext.json")
    b = V.Binner(x)
    xc = (V.EDGES[:-1] + V.EDGES[1:]) / 2
    sig_mc = float(np.median(b(P, np.zeros(len(P)))[:, np.abs(xc) > 12].std(1)))
    y = np.array([m["depth_mm"] for m in meta])
    groups = {}
    for f in sorted((R / "04_training/results/variants").glob("*.pt")):
        if "_n120_" not in f.stem and "_hisnr_" not in f.stem:
            continue
        ck = torch.load(f, map_location="cpu", weights_only=False)
        m = V.build(ck["variant"]); m.load_state_dict(ck["model"])
        groups.setdefault(f.stem.rsplit("_seed", 1)[0], []).append(m.eval())
    print({g: len(v) for g, v in groups.items()}, "sigma_mc", round(sig_mc, 5), flush=True)
    levels = sorted(set(PRIMARY) | {sig_mc, 0.014})
    rows = []
    for k, tot in enumerate(levels):
        add = float(np.sqrt(max(tot ** 2 - sig_mc ** 2, 0.0)))
        rng = np.random.default_rng(1700 + k)
        X = torch.from_numpy(np.concatenate([V.augment(b, P, rng, sigma=add) for _ in range(4)])[:, None, :])
        yy = np.tile(y, 4)
        row = {"sigma_tot": tot, "sigma_add": add, "primary": any(abs(tot - p) < 1e-9 for p in PRIMARY), "models": {}}
        with torch.no_grad():
            for g, ms in groups.items():
                outs = [m(X)[:, 0].numpy() for m in ms]
                e = np.mean(outs, 0) - yy
                row["models"][g] = {"ensemble_mae": float(np.abs(e).mean()), "ensemble_rmse": float(np.sqrt((e ** 2).mean())),
                                    "ensemble_bias": float(e.mean()),
                                    "mae_per_seed": [float(np.abs(o - yy).mean()) for o in outs],
                                    "mae_depth": {n: float(np.abs(e[(yy >= lo) & (yy < hi)]).mean())
                                                  for n, lo, hi in (("<2", 0, 2), ("2-3", 2, 3), ("3-4", 3, 4), (">=4", 4, 9))},
                                    "rmse_lt3": float(np.sqrt((e[yy < 3] ** 2).mean())),
                                    "rmse_ge3": float(np.sqrt((e[yy >= 3] ** 2).mean()))}
        rows.append(row)
        print(f"tot {tot:.4f}:", {g: round(r["ensemble_mae"], 4) for g, r in row["models"].items()}, flush=True)
    s2 = np.array([r["sigma_tot"] ** 2 for r in rows])
    fit = {}
    for g in groups:
        for key in ("ensemble_rmse", "rmse_lt3", "rmse_ge3"):
            e2 = np.array([r["models"][g][key] ** 2 for r in rows])
            (a, c), *_ = np.linalg.lstsq(np.stack([np.ones_like(s2), s2], 1), e2, rcond=None)
            fit[f"{g}:{key}"] = {"floor_mm": float(np.sqrt(max(a, 0))), "k_mm_per_unit_sigma": float(np.sqrt(max(c, 0)))}
    primary = {g: float(np.mean([r["models"][g]["ensemble_mae"] for r in rows if r["primary"]])) for g in groups}
    out = {"sigma_mc_validation_ext": sig_mc, "rows": rows, "fit": fit, "primary_mean_mae_4_levels": primary}
    print(json.dumps(primary, indent=1)); print(json.dumps(fit, indent=1))
    (R / "04_training/results/paired").mkdir(exist_ok=True)
    (R / "04_training/results/paired/paired_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
