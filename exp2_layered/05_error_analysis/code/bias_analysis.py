"""step 05: where does the fixed per-skin bias of the selected hetero model come from?

Analysis only: no model is retrained or modified; locked_test is never read.
1. hetero ensemble (3 seeds) on train (1500 vessels) and validation_ext (450), MC noise only,
   8 lateral shifts per vessel -> per-vessel bias (mean error over shifts) and spread.
2. train vs validation_ext error (under- vs over-fitting).
3. dependence of bias on the 13 skin parameters: linear + gradient boosting, fitted on TRAIN
   residuals only, evaluated on validation_ext residuals (does a fixed function of the parameters
   explain the bias?).
4. within-skin correlation of bias (vessels of one skin share all layer parameters).
Output: exp2_layered/05_error_analysis/results/fixed_bias/analysis.json
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp2_paths  # noqa: E402,F401  (all step code folders on sys.path)
import json
from pathlib import Path

import numpy as np
import torch

import cnn_variants as V

R = exp2_paths.EXP2
OUT = R / "05_error_analysis/results/fixed_bias"; OUT.mkdir(exist_ok=True)
PAR = ["depth_mm", "diameter_mm", "mua", "mus", "mel_fraction", "dermis_mm", "derm_musp", "derm_mua",
       "fat_mm", "fat_musp", "fat_mua", "musc_musp", "musc_mua"]
NSH = 8


def models():
    L = []
    for s in (42, 43, 44):
        ck = torch.load(R / f"04_training/results/variants/hetero_seed{s}.pt", map_location="cpu", weights_only=False)
        m = V.build("hetero"); m.load_state_dict(ck["model"]); L.append(m.eval())
    return L


def predict(L, split, manifest=None):
    x, P, meta = V.load(split, manifest) if manifest else V.load(split)
    b = V.Binner(x); rng = np.random.default_rng(20261113)
    pr = []
    with torch.no_grad():
        for _ in range(NSH):
            X = torch.from_numpy(V.augment(b, P, rng, sigma=0.0)[:, None, :])
            pr.append(np.mean([m(X)[:, 0].numpy() for m in L], 0))
    pr = np.array(pr)  # NSH x n
    y = np.array([m["depth_mm"] for m in meta])
    th = np.array([[m[k] for k in PAR] for m in meta])
    skin = np.array([m["config"] for m in meta])
    return pr, y, th, skin, meta


def summ(pr, y):
    e = pr - y[None]
    bias = e.mean(0); spread = e.std(0, ddof=1)
    return {"rmse": float(np.sqrt((e ** 2).mean())), "rms_bias": float(np.sqrt((bias ** 2).mean())),
            "rms_shift_spread": float(np.sqrt((spread ** 2).mean())), "mean_bias": float(bias.mean())}, bias


def feats(th):
    f = th.copy()
    for j, k in enumerate(PAR):
        if k in ("mua", "mus", "mel_fraction", "derm_musp", "derm_mua", "fat_musp", "fat_mua", "musc_musp", "musc_mua"):
            f[:, j] = np.log(f[:, j])
    return f


def main():
    torch.set_num_threads(8)
    L = models()
    pt, yt, tht, skt, _ = predict(L, "train")
    pv, yv, thv, skv, _ = predict(L, "validation_ext", "manifest_ext.json")
    res = {"n_train": int(len(yt)), "n_val": int(len(yv)), "shifts": NSH}
    res["train"], bt = summ(pt, yt)
    res["validation_ext"], bv = summ(pv, yv)

    Ft, Fv = feats(tht), feats(thv)
    mu, sd = Ft.mean(0), Ft.std(0)
    Zt, Zv = (Ft - mu) / sd, (Fv - mu) / sd
    # per-parameter correlation of bias (validation_ext and train)
    res["corr_bias_param"] = {k: {"train": float(np.corrcoef(Zt[:, j], bt)[0, 1]),
                                  "val": float(np.corrcoef(Zv[:, j], bv)[0, 1])} for j, k in enumerate(PAR)}
    # binned bias vs depth (shrinkage signature)
    edges = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5.01]
    res["bias_by_depth_val"] = [{"lo": lo, "hi": hi, "mean_bias": float(bv[(yv >= lo) & (yv < hi)].mean()),
                                 "rms_bias": float(np.sqrt((bv[(yv >= lo) & (yv < hi)] ** 2).mean()))}
                                for lo, hi in zip(edges[:-1], edges[1:])]

    def r2(y_, p_):
        return float(1 - ((y_ - p_) ** 2).sum() / ((y_ - y_.mean()) ** 2).sum())

    # linear model fitted on train residuals
    A = np.c_[np.ones(len(Zt)), Zt]; w = np.linalg.lstsq(A, bt, rcond=None)[0]
    res["linear"] = {"r2_train": r2(bt, A @ w), "r2_val": r2(bv, np.c_[np.ones(len(Zv)), Zv] @ w),
                     "coef_mm_per_sd": {k: float(c) for k, c in zip(PAR, w[1:])}}
    # nonlinear: quadratic ridge (squares + pairwise interactions, depth cubed), fitted on TRAIN only;
    # ridge strength chosen by 5-fold CV over train skins
    def quad(Z):
        n, p = Z.shape; cols = [np.ones(n)] + [Z[:, j] for j in range(p)] + [Z[:, 0] ** 3]
        cols += [Z[:, i] * Z[:, j] for i in range(p) for j in range(i, p)]
        return np.stack(cols, 1)

    def ridge(Z, y_, lam):
        A_ = quad(Z); I = np.eye(A_.shape[1]); I[0, 0] = 0
        return np.linalg.solve(A_.T @ A_ + lam * I, A_.T @ y_)

    def fit_cv(Z, y_, groups):
        ug = np.unique(groups); rs = np.random.default_rng(0); fold = dict(zip(ug, rs.integers(0, 5, len(ug))))
        f = np.array([fold[g_] for g_ in groups]); best = None
        for lam in (0.1, 1, 3, 10, 30, 100, 300, 1000):
            pr_ = np.zeros_like(y_)
            for k_ in range(5):
                w_ = ridge(Z[f != k_], y_[f != k_], lam); pr_[f == k_] = quad(Z[f == k_]) @ w_
            sc = r2(y_, pr_)
            if best is None or sc > best[1]:
                best = (lam, sc)
        return ridge(Z, y_, best[0]), best

    w2, (lam2, cv2) = fit_cv(Zt, bt, skt)
    pv_g = quad(Zv) @ w2
    res["quad"] = {"lambda": lam2, "r2_train_cv": cv2, "r2_val": r2(bv, pv_g),
                   "rms_bias_val_after_removing_predicted": float(np.sqrt(((bv - pv_g) ** 2).mean()))}
    rs = np.random.default_rng(1); drop = {}
    for j, k in enumerate(PAR):
        d_ = []
        for _ in range(20):
            Zp = Zv.copy(); Zp[:, j] = rs.permutation(Zp[:, j]); d_.append(res["quad"]["r2_val"] - r2(bv, quad(Zp) @ w2))
        drop[k] = float(np.mean(d_))
    res["quad"]["perm_importance_val_r2_drop"] = drop
    # depth only, and depth + one other parameter
    w0, _ = fit_cv(Zt[:, [0]], bt, skt); res["quad_depth_only_r2_val"] = r2(bv, quad(Zv[:, [0]]) @ w0)
    pair = {}
    for j, k in enumerate(PAR[1:], 1):
        wj, _ = fit_cv(Zt[:, [0, j]], bt, skt); pair[k] = r2(bv, quad(Zv[:, [0, j]]) @ wj)
    res["quad_depth_plus_one_r2_val"] = pair
    # within-skin correlation (ICC) of bias on validation_ext
    u = np.unique(skv); grp = [bv[skv == s] for s in u]
    k_ = np.mean([len(x) for x in grp]); gm = bv.mean()
    msb = sum(len(x) * (x.mean() - gm) ** 2 for x in grp) / (len(grp) - 1)
    msw = sum(((x - x.mean()) ** 2).sum() for x in grp) / (len(bv) - len(grp))
    res["icc_bias_within_skin_val"] = {"n_skins": int(len(u)), "vessels_per_skin": float(k_),
                                       "icc": float((msb - msw) / (msb + (k_ - 1) * msw))}
    np.savez(OUT / "bias.npz", bt=bt, bv=bv, yt=yt, yv=yv, tht=tht, thv=thv, pv_g=pv_g, skv=skv)
    (OUT / "analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
