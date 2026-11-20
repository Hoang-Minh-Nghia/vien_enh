"""step 05 evaluation: ensemble of 3 seeds per variant vs the selected hetero model.
train + validation_ext at sigma 0 (MC only); validation_ext at 0.005/0.007/0.010. locked_test never read.
Output: exp2_layered/05_error_analysis/results/retraining/eval.json
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp2_paths  # noqa: E402,F401  (all step code folders on sys.path)
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

import cnn_variants as V
from retrain_fit import build

R = exp2_paths.EXP2
NSH, SEEDS, LEVELS = 8, (42, 43, 44), (0.0, 0.005, 0.007, 0.010)


def load_models(v):
    L = []
    for s in SEEDS:
        if v == "hetero":
            m = V.build("hetero"); m.load_state_dict(torch.load(R / f"04_training/results/variants/hetero_seed{s}.pt", weights_only=False)["model"])
        else:
            m = build(v); m.load_state_dict(torch.load(R / f"05_error_analysis/results/retraining/{v}_seed{s}.pt", weights_only=False)["model"])
        L.append(m.eval())
    return L


def run(v, L, b, P, sigma, seed):
    rng = np.random.default_rng(seed); mus, sds = [], []
    with torch.no_grad():
        for _ in range(NSH):
            X = torch.from_numpy(V.augment(b, P, rng, sigma=sigma)[:, None, :])
            ls = torch.full((len(P),), float(np.log(max(sigma, 0.001))))
            outs = [m(X) if v == "hetero" else m(X, ls) for m in L]
            mu = np.array([o[:, 0].numpy() for o in outs]); sg = np.exp(np.array([o[:, 1].clamp(-6, 3).numpy() for o in outs]))
            mus.append(mu.mean(0)); sds.append(np.sqrt((sg ** 2 + mu ** 2).mean(0) - mu.mean(0) ** 2))
    return np.array(mus), np.array(sds)


def metrics(mu, sd, y, dm):
    e = mu - y[None]; bias = e.mean(0); z = np.abs(e) / sd
    return {"rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean()), "rms_bias": float(np.sqrt((bias ** 2).mean())),
            "cov68": float((z < 1).mean()), "cov95": float((z < norm.ppf(0.975)).mean()),
            "corr_bias_log_derm_musp": float(np.corrcoef(bias, np.log(dm))[0, 1])}


def main():
    torch.set_num_threads(8)
    xt, Pt, mt = V.load("train"); xv, Pv, mv = V.load("validation_ext", "manifest_ext.json")
    bt, bv = V.Binner(xt), V.Binner(xv)
    yt = np.array([m["depth_mm"] for m in mt]); yv = np.array([m["depth_mm"] for m in mv])
    dt = np.array([m["derm_musp"] for m in mt]); dv = np.array([m["derm_musp"] for m in mv])
    res = {}
    for v in ("hetero", "L", "LN", "LB", "LS"):
        try:
            L = load_models(v)
        except FileNotFoundError:
            continue
        r = {"train_s0": metrics(*run(v, L, bt, Pt, 0.0, 7), yt, dt)}
        for s in LEVELS:
            r[f"val_s{s}"] = metrics(*run(v, L, bv, Pv, s, 11), yv, dv)
        r["val_mae_mean_camera"] = float(np.mean([r[f"val_s{s}"]["mae"] for s in LEVELS[1:]]))
        res[v] = r
        print(v, json.dumps({k: (round(x["rmse"], 4) if isinstance(x, dict) else round(x, 4)) for k, x in r.items()}), flush=True)
    (R / "05_error_analysis/results/retraining/eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
