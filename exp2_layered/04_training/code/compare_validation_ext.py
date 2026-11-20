"""step 04: compare all model variants on validation_ext (90 new skins; never used for checkpoint choice).

Noise: total per-bin sigma in {0.005, 0.007, 0.010}; added camera noise = sqrt(tot^2 - sigma_mc^2), with
sigma_mc measured from the validation_ext profiles themselves (far field, binned).  4 shifted copies per vessel.
Writes exp2_layered/04_training/results/variants/compare_validation_ext.json.   locked_test is never read.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp2_paths  # noqa: E402,F401  (all step code folders on sys.path)
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cnn_variants as V  # noqa: E402
from train import DepthCNN  # noqa: E402

R = exp2_paths.EXP2
TOTALS = (0.005, 0.007, 0.010)


def main():
    torch.set_num_threads(8)
    x, P, meta = V.load("validation_ext", "manifest_ext.json")
    binner = V.Binner(x)
    xc = (V.EDGES[:-1] + V.EDGES[1:]) / 2
    clean = binner(P, np.zeros(len(P)))
    sig_mc = float(np.median(clean[:, np.abs(xc) > 12].std(1)))
    y = np.array([m["depth_mm"] for m in meta])
    mel = np.array([m["mel_fraction"] for m in meta])
    models = {}
    for f in sorted((R / "04_training/results/variants").glob("*.pt")):
        ck = torch.load(f, map_location="cpu", weights_only=False)
        m = V.build(ck["variant"]); m.load_state_dict(ck["model"]); models[f.stem] = (ck["variant"], m.eval())
    for s in (42, 43, 44):
        for tag, path in ((f"t5_F_layered_seed{s}", R / "04_training/results/layered_cnn/models" / f"F_layered_seed{s}.pt"),
                          (f"F_locked_seed{s}", exp2_paths.exp1_at(f"results/stage4/models/cnn_F_seed{s}.pt"))):
            m = DepthCNN(); m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
            models[tag] = ("base", m.eval())
    out = {"sigma_mc_validation_ext": sig_mc, "n_vessels": len(y), "levels": {}}
    for k, tot in enumerate(TOTALS):
        add = float(np.sqrt(max(tot ** 2 - sig_mc ** 2, 0.0)))
        rng = np.random.default_rng(700 + k)
        X = np.concatenate([V.augment(binner, P, rng, sigma=add) for _ in range(4)])
        yy, mm = np.tile(y, 4), np.tile(mel, 4)
        preds, sigmas = {}, {}
        with torch.no_grad():
            for tag, (var, m) in models.items():
                o = m(torch.from_numpy(X[:, None, :]))
                o = o[:, None] if o.dim() == 1 else o
                preds[tag] = o[:, 0].numpy()
                if var == "hetero":
                    sigmas[tag] = np.exp(o[:, 1].clamp(-6, 3).numpy())
        groups = defaultdict(list)
        for tag in preds:
            groups[tag.rsplit("_seed", 1)[0]].append(tag)
        lev = {}
        for g, tags in sorted(groups.items()):
            per_seed = [float(np.abs(preds[t] - yy).mean()) for t in tags]
            ens = np.mean([preds[t] for t in tags], 0)
            e = ens - yy
            row = {"n_seeds": len(tags), "mae_per_seed": per_seed, "mae_mean_seed": float(np.mean(per_seed)),
                   "ensemble_mae": float(np.abs(e).mean()), "ensemble_bias": float(e.mean()),
                   "ensemble_rmse": float(np.sqrt((e ** 2).mean())),
                   "ensemble_mae_depth": {b: float(np.abs(e[(yy >= lo) & (yy < hi)]).mean())
                                          for b, lo, hi in (("<2", 0, 2), ("2-3", 2, 3), ("3-4", 3, 4), (">=4", 4, 9))},
                   "ensemble_mae_mel_lt0.05": float(np.abs(e[mm < 0.05]).mean()),
                   "ensemble_mae_mel_ge0.15": float(np.abs(e[mm >= 0.15]).mean())}
            hs = [t for t in tags if t in sigmas]
            if hs:   # calibration of the predicted uncertainty (mixture of seeds)
                mu = np.mean([preds[t] for t in hs], 0)
                var = np.mean([sigmas[t] ** 2 + preds[t] ** 2 for t in hs], 0) - mu ** 2
                sd = np.sqrt(var)
                z = np.abs(mu - yy) / sd
                row["uncertainty"] = {"coverage_68": float(np.mean(z < 1.0)), "coverage_90": float(np.mean(z < 1.645)),
                                      "coverage_95": float(np.mean(z < 1.96)), "median_sigma_mm": float(np.median(sd)),
                                      "spearman_sigma_abs_error": float(np.corrcoef(np.argsort(np.argsort(sd)),
                                                                                    np.argsort(np.argsort(np.abs(mu - yy))))[0, 1])}
            lev[g] = row
        out["levels"][f"tot_{tot:.3f}"] = {"sigma_add": add, "models": lev}
        print(f"tot {tot}:", {g: round(r["ensemble_mae"], 3) for g, r in lev.items()}, flush=True)
    summary = defaultdict(list)
    for L in out["levels"].values():
        for g, r in L["models"].items():
            summary[g].append(r["ensemble_mae"])
    out["primary_metric_mean_over_levels"] = {g: float(np.mean(v)) for g, v in sorted(summary.items(), key=lambda kv: np.mean(kv[1]))}
    print(json.dumps(out["primary_metric_mean_over_levels"], indent=1))
    (R / "04_training/results/variants/compare_validation_ext.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
