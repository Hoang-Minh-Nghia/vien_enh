"""step 05 analysis: 14-parameter Fisher / linearised Bayesian bound vs the trained networks (report_crlb_14_params.md).

Reads exp2_layered/05_error_analysis/results/crlb_14_params/pt_*.npz (any number done so far), writes exp2_layered/05_error_analysis/results/crlb_14_params/analysis.json.
locked_test is never read.
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
from crlb_runs import STEPS  # noqa: E402

RES = exp2_paths.EXP2
D = RES / "05_error_analysis/results/crlb_14_params"
SIGMAS = (0.0035, 0.004, 0.005, 0.007, 0.010, 0.014)
VESSEL_P = ["depth_mm", "diameter_mm", "mua", "mus"]
LAYER_P = [k for k, v in STEPS.items() if v[0] == "layer"]
ORDER = ["depth_mm", "shift"] + VESSEL_P[1:] + LAYER_P          # 14 parameters, depth first


def param_value(name, L, v, shift=0.0):
    if name == "shift":
        return shift
    val = v[name] if name in v else L[name]
    return float(np.log(val)) if STEPS[name][1] else float(val)


def prior_var():
    """Diagonal prior variances in the analysis parametrisation, from the validation_ext draws (T4 ranges)."""
    cf = json.loads((RES / "03_dataset/data/manifest_ext.json").read_text(encoding="utf-8"))["splits"]["validation_ext"]["configs"]
    rows = [[param_value(n, c["layers"], v) for n in ORDER] for c in cf for v in c["vessels"]]
    var = np.var(np.array(rows), 0)
    var[ORDER.index("shift")] = 4.0 ** 2 / 12                   # U(-2, 2) mm
    return var


def spline_basis(knot_mm=1.0):
    """Cubic B-spline basis on the 128 bin centres (knots every knot_mm over [-16, 16])."""
    from scipy.interpolate import BSpline
    xc = (V.EDGES[:-1] + V.EDGES[1:]) / 2
    t = np.r_[[-16.0] * 3, np.arange(-16, 16 + 1e-9, knot_mm), [16.0] * 3]
    n = len(t) - 4
    B = np.stack([BSpline(t, np.eye(n)[i], 3, extrapolate=False)(xc) for i in range(n)], 1)
    B = np.nan_to_num(B)
    return B[:, B.sum(0) > 1e-9]


def psd(F):
    w, U = np.linalg.eigh((F + F.T) / 2)
    return (U * np.clip(w, 0, None)) @ U.T, float(w.min() / w.max())


def jacobians(z, b):
    keys, P = list(z["keys"]), z["profiles"].astype(np.float64)
    n0 = dict(zip(z["n0_keys"], z["n0"]))
    p = json.loads(str(z["point_json"]))
    def binned(tag, h, shift=0.0):
        nk = tag if tag in n0 else "base"
        return b(P[keys.index(f"{tag}|{h}")][None] / n0[nk], [shift])[0]
    J, base = {}, {}
    for h in ("a", "b"):
        cols = []
        for name in ORDER:
            if name == "shift":
                cols.append((binned("base", h, 0.1) - binned("base", h, -0.1)) / 0.2)
            else:
                cols.append((binned(f"{name}+", h) - binned(f"{name}-", h)) / (2 * STEPS[name][2]))
        J[h] = np.stack(cols, 1)
        base[h] = binned("base", h)
    return p, J, base


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(8)
    files = sorted(D.glob("pt_*.npz"))
    files = [f for f in files if not f.name.endswith(".partial.npz")]
    z0 = np.load(files[0]); b = V.Binner(z0["x_mm"])
    pv = prior_var(); Pinv = np.diag(1 / pv)
    # networks: the main 300-skin models (no _n / _hisnr / _ext)
    nets = {"hetero": [], "wide": []}
    for f in sorted((RES / "04_training/results/variants").glob("*_seed4?.pt")):
        fam = f.stem.rsplit("_seed", 1)[0]
        if fam in nets:
            ck = torch.load(f, map_location="cpu", weights_only=False)
            m = V.build(ck["variant"]); m.load_state_dict(ck["model"]); nets[fam].append(m.eval())
    Bs = spline_basis(); H = Bs @ np.linalg.pinv(Bs)
    out_smooth = {"basis_functions": int(Bs.shape[1])}
    pts, sig_mc_half, resid = [], [], []
    for f in files:
        z = np.load(f); p, J, base = jacobians(z, b)
        noise_var = ((J["a"] - J["b"]) / 2) ** 2                  # per element variance of the half-mean J
        R = (np.eye(128) - H) @ ((J["a"] + J["b"]) / 2)
        resid.append((R ** 2).sum(0) / ((128 - Bs.shape[1]) * noise_var.mean(0)))   # ~1 if only noise removed
        J = {h: H @ J[h] for h in J}
        Jm = (J["a"] + J["b"]) / 2
        Nk = (((J["a"] - J["b"]) / 2) ** 2).sum(0)               # noise energy of each (projected) column of Jm
        Ek = (Jm ** 2).sum(0)
        J["js"] = Jm * np.clip(1 - Nk / Ek, 0, 1)[None, :]         # James-Stein shrinkage per column
        J["js_factor"] = np.clip(1 - Nk / Ek, 0, 1)
        sig_mc_half.append(float(np.std(base["a"] - base["b"]) / np.sqrt(2)))   # per half (24 M)
        pts.append((p, J, base))
    sig_mc48 = float(np.median(sig_mc_half) / np.sqrt(2))                     # mean of two halves (48 M)
    out_smooth["residual_over_noise_median"] = dict(zip(ORDER, np.median(np.array(resid), 0).tolist()))
    out = {"n_points": len(pts), "order": ORDER, "smoothing": out_smooth, "prior_sd": dict(zip(ORDER, np.sqrt(pv).tolist())),
           "sigma_mc_base_48M": sig_mc48, "points": []}
    rng = np.random.default_rng(2026)
    for p, J, base in pts:
        Ja, Jb = J["a"], J["b"]; Jm = (Ja + Jb) / 2
        F_ub_raw = (Ja.T @ Jb + Jb.T @ Ja) / 2
        F_ub, eig_ratio = psd(F_ub_raw)
        F_nv = Jm.T @ Jm
        F_js = J["js"].T @ J["js"]
        col_corr = {n: float(np.corrcoef(Ja[:, i], Jb[:, i])[0, 1]) for i, n in enumerate(ORDER)}
        ev = np.linalg.eigvalsh(F_ub_raw)
        rec = {"js_factor": dict(zip(ORDER, J["js_factor"].tolist())), "id": p["id"], "depth_mm": p["vessel"]["depth_mm"], "mel": p["layers"]["mel_fraction"],
               "jacobian_half_corr": col_corr, "F_ub_raw_min_over_max_eig": eig_ratio, "F_ub_raw_pd": bool(ev.min() > 0),
               "crlb_depth_only_per_sigma": float(1 / np.sqrt(F_ub[0, 0])), "levels": {}}
        try:
            rec["crlb_full_per_sigma"] = float(np.sqrt(np.linalg.inv(F_ub)[0, 0])) if ev.min() > 0 else None
        except np.linalg.LinAlgError:
            rec["crlb_full_per_sigma"] = None
        rec["crlb_full_per_sigma_naive"] = float(np.sqrt(np.linalg.inv(F_nv)[0, 0]))
        y = p["vessel"]["depth_mm"]
        clean = (base["a"] + base["b"]) / 2
        for s in SIGMAS:
            C = np.linalg.inv(F_ub / s ** 2 + Pinv)
            Cn = np.linalg.inv(F_nv / s ** 2 + Pinv)
            Cj = np.linalg.inv(F_js / s ** 2 + Pinv)
            corr = C[0] / np.sqrt(C[0, 0] * np.diag(C))
            add = np.sqrt(max(s ** 2 - sig_mc48 ** 2, 0))
            # network on the base profile: fresh shift + noise, 64 copies (profile re-binned with the shift)
            Pfine = None
            k = 64
            sh = rng.uniform(-2, 2, k)
            zf = np.load(D / f"{p['id']}.npz"); keys = list(zf["keys"]); n0b = dict(zip(zf["n0_keys"], zf["n0"]))["base"]
            Pfine = (zf["profiles"][keys.index("base|a")].astype(float) + zf["profiles"][keys.index("base|b")].astype(float)) / 2 / n0b
            X = b(np.repeat(Pfine[None], k, 0), sh) + rng.normal(0, 1, (k, 128)) * add
            Xt = torch.from_numpy(X.astype(np.float32)[:, None, :])
            lev = {"bayes_sd_mm": float(np.sqrt(C[0, 0])), "bayes_sd_naive_mm": float(np.sqrt(Cn[0, 0])),
                   "bayes_sd_js_mm": float(np.sqrt(Cj[0, 0])),
                   "posterior_corr_depth": {n: float(c) for n, c in zip(ORDER, Cj[0] / np.sqrt(Cj[0, 0] * np.diag(Cj)))}}
            with torch.no_grad():
                for fam, ms in nets.items():
                    pr = np.mean([m(Xt)[:, 0].numpy() for m in ms], 0)
                    e = pr - y
                    lev[f"{fam}_rmse"] = float(np.sqrt((e ** 2).mean()))
                    lev[f"{fam}_bias"] = float(e.mean())
                    lev[f"{fam}_sd"] = float(e.std())
            rec["levels"][f"{s}"] = lev
        out["points"].append(rec)
    # population summaries (Bayes risk of the linear-Gaussian model = mean posterior variance)
    summ = {}
    bins = (("<2", 0, 2), ("2-3", 2, 3), ("3-4", 3, 4), (">=4", 4, 9), ("all", 0, 9))
    for s in SIGMAS:
        row = {}
        for bn, lo, hi in bins:
            sel = [r for r in out["points"] if lo <= r["depth_mm"] < hi]
            if not sel:
                continue
            L = [r["levels"][f"{s}"] for r in sel]
            row[bn] = {"n": len(sel),
                       "bayes_rmse": float(np.sqrt(np.mean([l["bayes_sd_mm"] ** 2 for l in L]))),
                       "bayes_rmse_naive": float(np.sqrt(np.mean([l["bayes_sd_naive_mm"] ** 2 for l in L]))),
                       "bayes_rmse_js": float(np.sqrt(np.mean([l["bayes_sd_js_mm"] ** 2 for l in L]))),
                       "crlb_full_rmse": (float(s * np.sqrt(np.mean([r["crlb_full_per_sigma"] ** 2 for r in sel])))
                                          if all(r["crlb_full_per_sigma"] for r in sel) else None),
                       "crlb_depth_only_rmse": float(s * np.sqrt(np.mean([r["crlb_depth_only_per_sigma"] ** 2 for r in sel]))),
                       **{f"{fam}_rmse": float(np.sqrt(np.mean([l[f"{fam}_rmse"] ** 2 for l in L]))) for fam in nets},
                       **{f"{fam}_rms_bias": float(np.sqrt(np.mean([l[f"{fam}_bias"] ** 2 for l in L]))) for fam in nets}}
            for fam in nets:
                row[bn][f"ratio_bayes_over_{fam}"] = row[bn]["bayes_rmse"] / row[bn][f"{fam}_rmse"]
                row[bn][f"ratio_js_over_{fam}"] = row[bn]["bayes_rmse_js"] / row[bn][f"{fam}_rmse"]
                row[bn][f"ratio_naive_over_{fam}"] = row[bn]["bayes_rmse_naive"] / row[bn][f"{fam}_rmse"]
        summ[f"{s}"] = row
    # which nuisances are most correlated with depth in the posterior (median |corr| at sigma 0.005)
    out["median_abs_posterior_corr_depth_at_0.005"] = {
        n: float(np.median([abs(r["levels"]["0.005"]["posterior_corr_depth"][n]) for r in out["points"]])) for n in ORDER[1:]}
    out["summary"] = summ
    lo, hi = summ[f"{SIGMAS[0]}"]["all"], summ[f"{SIGMAS[-1]}"]["all"]
    out["bayes_ratio_sigma_hi_over_lo"] = hi["bayes_rmse"] / lo["bayes_rmse"]
    out["bayes_js_ratio_sigma_hi_over_lo"] = hi["bayes_rmse_js"] / lo["bayes_rmse_js"]
    out["bayes_naive_ratio_sigma_hi_over_lo"] = hi["bayes_rmse_naive"] / lo["bayes_rmse_naive"]
    out["net_ratio_sigma_hi_over_lo"] = {fam: hi[f"{fam}_rmse"] / lo[f"{fam}_rmse"] for fam in nets}
    (D / "analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("points", len(pts), "sigma_mc 48M", round(sig_mc48, 5), "basis", Bs.shape[1])
    print("projection residual / noise (median, ~1 = only noise removed):", {k: round(v, 2) for k, v in out_smooth["residual_over_noise_median"].items()})
    print("jacobian half-corr (median):", {n: round(float(np.median([r["jacobian_half_corr"][n] for r in out["points"]])), 3) for n in ORDER})
    for s, row in summ.items():
        a = row["all"]
        print(f"sigma {s}: js {a['bayes_rmse_js']:.3f} naive {a['bayes_rmse_naive']:.3f} ub {a['bayes_rmse']:.3f}  crlb_full {a['crlb_full_rmse']}  "
              f"hetero {a['hetero_rmse']:.3f} wide {a['wide_rmse']:.3f}  ratio js/hetero {a['ratio_js_over_hetero']:.2f} naive/hetero {a['ratio_naive_over_hetero']:.2f}")
    print("posterior |corr| with depth:", {k: round(v, 2) for k, v in out["median_abs_posterior_corr_depth_at_0.005"].items()})
    print("js factor (median):", {n: round(float(np.median([r["js_factor"][n] for r in out["points"]])), 2) for n in ORDER})
    print("sigma-ratio js", round(out["bayes_js_ratio_sigma_hi_over_lo"], 2), "naive", round(out["bayes_naive_ratio_sigma_hi_over_lo"], 2), "nets", {k: round(v, 2) for k, v in out["net_ratio_sigma_hi_over_lo"].items()})


if __name__ == "__main__":
    main()
