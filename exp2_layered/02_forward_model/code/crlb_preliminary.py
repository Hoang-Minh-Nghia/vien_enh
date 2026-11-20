"""step 02 preliminary: Fisher information and CRLB of depth from the V1 profiles (850 nm, uniform light).

Inputs are the 180 locked V1 profiles (10 depths x 6 channel mua x 3 MC repeats) and the
V1 F/K error summary. No CNN is run and nothing is fitted to CNN errors; the CNN errors are
only compared with the CRLB map afterwards.

Model: profile m(z, mua) = mean over the 3 MC repeats. Derivatives by second-order finite
differences on the non-uniform grid (np.gradient). Measurement noise: white Gaussian with
standard deviation sigma per 128-point bin (ratio units), several sigma levels.

Outputs:
- rho(z, mua): noise-whitened cosine between dm/dz and dm/dmua. With white noise the
  penalty for not knowing mua is 1/sqrt(1 - rho^2), independent of sigma.
- sqrt CRLB of z with mua known and unknown, per sigma.
- Derivative noise check: the same quantities from each single MC repeat, to see whether
  MC noise in the finite differences dominates.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp2_paths  # noqa: E402,F401  (all step code folders on sys.path)
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

V1 = exp2_paths.exp1_at("results/validation_independent")
OUT = exp2_paths.EXP2 / "02_forward_model/results/crlb_preliminary"
SIGMAS = (0.001, 0.003, 0.01)          # ratio units per bin; assumed, to be replaced by S2 camera data


def grid(profiles, depth, mua, repeat, which=None):
    zs, ms = np.unique(depth), np.unique(mua)
    m = np.zeros((len(zs), len(ms), profiles.shape[1]))
    for i, z in enumerate(zs):
        for j, a in enumerate(ms):
            sel = (np.isclose(depth, z)) & (np.isclose(mua, a))
            if which is not None:
                sel &= repeat == which
            m[i, j] = profiles[sel].mean(axis=0)
    return zs, ms, m


def fisher_maps(zs, ms, m):
    dz = np.gradient(m, zs, axis=0, edge_order=2)
    dm = np.gradient(m, ms, axis=1, edge_order=2)
    fzz = (dz * dz).sum(-1)
    fmm = (dm * dm).sum(-1)
    fzm = (dz * dm).sum(-1)
    rho = fzm / np.sqrt(fzz * fmm)
    return fzz, fmm, fzm, rho


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(V1 / "predictions/profiles.npz") as z:
        X, depth, mua, repeat = z["X"].astype(np.float64), z["y_mm"].astype(np.float64), z["mua_mm_inv"].astype(np.float64), z["repeat"]
    zs, ms, m = grid(X, depth, mua, repeat)
    fzz, fmm, fzm, rho = fisher_maps(zs, ms, m)
    penalty = 1 / np.sqrt(1 - rho ** 2)

    # Per-repeat estimates of rho to judge MC noise in the derivatives.
    rho_rep = np.stack([fisher_maps(*grid(X, depth, mua, repeat, r))[3] for r in (1, 2, 3)])

    # MC noise per bin (SD across repeats), for reference against the assumed camera sigma.
    mc_sd = np.stack([X[(np.isclose(depth, a)) & (np.isclose(mua, b))].std(axis=0, ddof=1)
                      for a in zs for b in ms]).mean()

    errs = {(float(r["depth_mm"]), float(r["mua"])): (float(r["mean_MAE_F_mm"]), float(r["mean_MAE_K_mm"]))
            for r in csv.DictReader(open(V1 / "report/v6_condition_summary.csv", encoding="utf-8"))}
    maeF = np.array([[errs[(round(a, 3), round(b, 3))][0] for b in ms] for a in zs])
    maeK = np.array([[errs[(round(a, 3), round(b, 3))][1] for b in ms] for a in zs])

    rows = []
    for i, a in enumerate(zs):
        for j, b in enumerate(ms):
            row = {"depth_mm": a, "mua": b, "rho": rho[i, j], "rho_rep_min": rho_rep[:, i, j].min(),
                   "rho_rep_max": rho_rep[:, i, j].max(), "penalty_unknown_mua": penalty[i, j],
                   "contrast": 1 - m[i, j].min(), "MAE_F": maeF[i, j], "MAE_K": maeK[i, j]}
            for s in SIGMAS:
                row[f"crlb_known_mm_s{s}"] = s / np.sqrt(fzz[i, j])
                row[f"crlb_unknown_mm_s{s}"] = s / np.sqrt(fzz[i, j]) * penalty[i, j]
            rows.append(row)
    with open(OUT / "crlb_map.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    s = 0.003
    unk = np.array([[r[f"crlb_unknown_mm_s{s}"] for r in rows if r["depth_mm"] == a] for a in zs])
    corr_F = spearmanr(unk.ravel(), maeF.ravel())
    corr_pen = spearmanr(penalty.ravel(), maeF.ravel())
    summary = {"sigma_levels": SIGMAS, "mc_noise_sd_per_bin": float(mc_sd),
               "rho_range": [float(rho.min()), float(rho.max())],
               "penalty_range": [float(penalty.min()), float(penalty.max())],
               "spearman_crlb_unknown_vs_MAE_F_sigma0.003": [float(corr_F.statistic), float(corr_F.pvalue)],
               "spearman_penalty_vs_MAE_F": [float(corr_pen.statistic), float(corr_pen.pvalue)],
               "max_abs_rho_repeat_spread": float((rho_rep.max(0) - rho_rep.min(0)).max()),
               "note": "Preliminary: coarse depth grid (0.3-0.5 mm), only mua varied, homogeneous background, assumed white noise."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), layout="constrained")
    panels = [(np.abs(rho), "|rho| between dm/dz and dm/dmua", "viridis", None),
              (penalty, "CRLB penalty when mua unknown", "magma_r", None),
              (unk, f"sqrt CRLB depth, mua unknown, sigma={s} (mm)", "magma_r", None),
              (maeF, "MAE F on V1 (mm)", "magma_r", None)]
    for ax, (data, title, cmap, _) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, aspect="auto", norm=matplotlib.colors.LogNorm() if "CRLB" in title or "penalty" in title else None)
        for i in range(len(zs)):
            for j in range(len(ms)):
                ax.text(j, i, f"{data[i, j]:.2g}", ha="center", va="center", fontsize=7,
                        bbox={"facecolor": "white", "alpha": 0.6, "pad": 0.5, "linewidth": 0})
        ax.set_xticks(range(len(ms)), [f"{b:g}" for b in ms])
        ax.set_yticks(range(len(zs)), [f"{a:g}" for a in zs])
        ax.set_xlabel("Channel mua (1/mm)")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
    axes[0].set_ylabel("Depth (mm)")
    fig.savefig(OUT / "crlb_maps.png", dpi=150)
    print(json.dumps(summary, indent=2))
    print("rho map (rows depth, cols mua):")
    print(np.array2string(rho, precision=3))
    print("penalty map:")
    print(np.array2string(penalty, precision=2))


if __name__ == "__main__":
    main()
