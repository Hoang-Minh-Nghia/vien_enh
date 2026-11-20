"""Error-versus-noise decomposition on the (old) step 03 validation split.

RMSE^2(sigma_tot) = floor^2 + (k * sigma_tot)^2 : the k-term is the noise-limited part (scales like a CRLB),
the floor is noise-independent (model approximation or global non-identifiability).
sigma_tot^2 = sigma_mc^2 + sigma_add^2, sigma_mc measured on these profiles.  locked_test never read.
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

R = exp2_paths.EXP2 / "04_training/results/variants"


def main():
    torch.set_num_threads(8)
    x, P, meta = V.load("validation")
    b = V.Binner(x)
    xc = (V.EDGES[:-1] + V.EDGES[1:]) / 2
    sig_mc = float(np.median(b(P, np.zeros(len(P)))[:, np.abs(xc) > 12].std(1)))
    y = np.array([m["depth_mm"] for m in meta])
    fams = {"wide": [], "hetero": [], "base": []}
    for f in sorted(R.glob("*.pt")):
        ck = torch.load(f, map_location="cpu", weights_only=False)
        if ck["variant"] in fams and "_n" not in f.stem and not ck.get("ext"):
            m = V.build(ck["variant"]); m.load_state_dict(ck["model"]); fams[ck["variant"]].append(m.eval())
    rows = []
    for k, add in enumerate((0.0, 0.002, 0.004, 0.006, 0.008, 0.010, 0.014)):
        rng = np.random.default_rng(900 + k)
        X = np.concatenate([V.augment(b, P, rng, sigma=add) for _ in range(8)])
        yy = np.tile(y, 8)
        row = {"sigma_add": add, "sigma_tot": float(np.hypot(sig_mc, add))}
        with torch.no_grad():
            for fam, ms in fams.items():
                outs = []
                for m in ms:
                    o = m(torch.from_numpy(X[:, None, :]))
                    outs.append((o[:, 0] if o.dim() == 2 else o).numpy())
                e = np.mean(outs, 0) - yy
                row[fam] = {"rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean()),
                            "rmse_depth_lt3": float(np.sqrt((e[yy < 3] ** 2).mean())),
                            "rmse_depth_ge3": float(np.sqrt((e[yy >= 3] ** 2).mean()))}
        rows.append(row)
        print(json.dumps({k2: (round(v, 4) if isinstance(v, float) else {a: round(c, 3) for a, c in v.items()}) for k2, v in row.items()}), flush=True)
    fit = {}
    s2 = np.array([r["sigma_tot"] ** 2 for r in rows])
    for fam in fams:
        for key in ("rmse", "rmse_depth_lt3", "rmse_depth_ge3"):
            e2 = np.array([r[fam][key] ** 2 for r in rows])
            A = np.stack([np.ones_like(s2), s2], 1)
            (a, c), *_ = np.linalg.lstsq(A, e2, rcond=None)
            fit[f"{fam}_{key}"] = {"floor_mm": float(np.sqrt(max(a, 0))), "k_mm_per_unit_sigma": float(np.sqrt(max(c, 0))),
                                   "noise_term_at_0.003_mm": float(np.sqrt(max(c, 0)) * 0.003)}
    out = {"sigma_mc": sig_mc, "rows": rows, "fit_rmse2_eq_floor2_plus_k2sigma2": fit}
    print(json.dumps(fit, indent=1))
    (R / "noise_floor_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
