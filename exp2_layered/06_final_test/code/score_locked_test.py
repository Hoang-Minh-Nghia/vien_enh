"""step 06: score the locked test set ONCE (plan: report_locked_test.md, written before opening).

Final model: hetero ensemble (seeds 42/43/44), locked by SHA-256. Comparators: wide, F_layered (step 04),
F locked (stage 4, homogeneous), B3 homogeneous physics fit (1 shift per vessel), constant train mean.
Same scoring as compare_validation_ext.py: sigma_mc measured on the set itself, camera noise added to reach
total sigma 0.005 / 0.007 / 0.010 (+ MC-only level), 4 shifted copies per vessel.
Refuses to run if exp2_layered/06_final_test/results/locked_test/OPENED.json exists.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp2_paths  # noqa: E402,F401  (all step code folders on sys.path)
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cnn_variants as V  # noqa: E402
from train import DepthCNN  # noqa: E402

R = exp2_paths.EXP2
OUT = R / "06_final_test/results/locked_test"
ROOT = exp2_paths.EXP1
TOTALS = (0.005, 0.007, 0.010)
DRY = "--dry-run-on-validation-ext" in sys.argv   # full code path on an already-opened set, no flag
COPIES = 4
EXPECTED = {
    "04_training/results/variants/hetero_seed42.pt": "b0c908e98173e78b2b7240410001a9754fa7ce5e5d85277a21b30dd5a1d2b5e7",
    "04_training/results/variants/hetero_seed43.pt": "eaa7b263a6b6397224c53de36b33e8b0d2105c863945428746aae10e23e69e7c",
    "04_training/results/variants/hetero_seed44.pt": "d0f3086e88015f6a6d3e790baadd2433d5187043620ea310950c2a28490afcd9",
    "04_training/results/variants/wide_seed42.pt": "92da4cfd1c19b8e559c4d1d5fca939336a56d332268f69dd0fe7641ababf5650",
    "04_training/results/variants/wide_seed43.pt": "012a6949100b4cd3faac9941b3f3cf8dfc7b4f4daec2379b1e60c0278005c1a1",
    "04_training/results/variants/wide_seed44.pt": "2ec1049dcf26ba7b31fe025d3620b57920b873dd9015597d91ea9583e81b95e7",
    "04_training/results/layered_cnn/models/F_layered_seed42.pt": "e0db287318c04d1ebaba3873bc72a4d44194744e64016038afb316f5e1604fa5",
    "04_training/results/layered_cnn/models/F_layered_seed43.pt": "e3f3571badacfdce0b3b3b46c1496ec53dde5f713f64dd236b5d39ae8c46ef09",
    "04_training/results/layered_cnn/models/F_layered_seed44.pt": "97c0ae495021357072317ae4678730c310185c195449ef49791daf894349c1ac",
    "STAGE4/cnn_F_seed42.pt": "15f97faa2c6a2d07a52ee165434973037674b28eb49b5579805b391e46868a6b",
    "STAGE4/cnn_F_seed43.pt": "1a69c8c5b65dea38d14ffb734e8211ff0ae2aa82061081ee27b776c6ace434e1",
    "STAGE4/cnn_F_seed44.pt": "52cfa983414364bf8677bb111318070a34ec8f1f7278f6680ccde8152d18dc38",
    "STAGE4/lut_FK.npz": "71e7ec1597c6d553e1138c1883580ee3ad57378314e7712143ca9fcabb5a080e",
}


def path_of(k):
    return exp2_paths.exp1_at("results/stage4/" + ("models/" + k.split("/")[1] if k.endswith(".pt") else k.split("/")[1])) \
        if k.startswith("STAGE4/") else R / k


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_locked():
    m = json.loads((R / "03_dataset/data/manifest.json").read_text(encoding="utf-8"))
    P, meta = [], []
    cfgs = m["splits"]["locked_test"]["configs"]
    for cm in cfgs:
        f = R / "03_dataset/data/locked_test" / f"{cm['id']}.npz"
        z = np.load(f)
        c = json.loads(str(z["config_json"])); assert c == cm, cm["id"]
        x, s = z["x_mm"], float(z["n0_scalar"])
        for v, p in zip(c["vessels"], z["profiles_y_mean"]):
            p = p.astype(np.float64) / s
            assert np.all(np.isfinite(p)), cm["id"]
            P.append(p); meta.append({"config": c["id"], **v, **c["layers"]})
    return x, np.array(P), meta, len(cfgs)


def _b3(prof):
    from b3_physics_fit import Forward, fit
    global _FWD
    if "_FWD" not in globals():
        _FWD = Forward()
    return fit(_FWD, np.asarray(prof, float), True)[0]


def summarise(pred, y, cfg, rng, sd=None):
    e = pred - y
    out = {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()),
           "mae_depth": {b: float(np.abs(e[(y >= lo) & (y < hi)]).mean()) for b, lo, hi in
                         (("<2", 0, 2), ("2-3", 2, 3), ("3-4", 3, 4), (">=4", 4, 9))}}
    ids = np.unique(cfg)
    per = {c: np.abs(e[cfg == c]) for c in ids}
    boots = [np.concatenate([per[c] for c in rng.choice(ids, len(ids))]).mean() for _ in range(2000)]
    out["mae_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    if sd is not None:
        z = np.abs(e) / sd
        out["coverage"] = {"68": float((z < 1).mean()), "90": float((z < 1.645).mean()), "95": float((z < 1.96).mean())}
        out["median_sigma"] = float(np.median(sd))
    return out


def main():
    global OUT
    if DRY:
        OUT = R / "06_final_test/results/dryrun_validation_ext"
    OUT.mkdir(parents=True, exist_ok=True)
    flag = OUT / "OPENED.json"
    if flag.exists():
        sys.exit("locked_test was already opened: " + flag.read_text(encoding="utf-8"))
    hashes = {k: sha(path_of(k)) for k in EXPECTED}
    bad = [k for k in EXPECTED if hashes[k] != EXPECTED[k]]
    (OUT / "hashes.json").write_text(json.dumps(hashes, indent=1), encoding="utf-8")
    if bad:
        sys.exit(f"hash mismatch, not opening: {bad}")
    if not DRY:
        flag.write_text(json.dumps({"opened_at": time.strftime("%Y-%m-%d %H:%M:%S"), "plan": "T8_locked_test.md"}), encoding="utf-8")

    torch.set_num_threads(8)
    if DRY:
        x, P, meta = V.load("validation_ext", "manifest_ext.json"); n_cfg = len({m["config"] for m in meta})
    else:
        x, P, meta, n_cfg = load_locked()
    y = np.array([m["depth_mm"] for m in meta]); mel = np.array([m["mel_fraction"] for m in meta])
    cfg = np.array([m["config"] for m in meta])
    binner = V.Binner(x)
    xc = (V.EDGES[:-1] + V.EDGES[1:]) / 2
    sig_mc = float(np.median(binner(P, np.zeros(len(P)))[:, np.abs(xc) > 12].std(1)))
    ytrain = float(np.mean([m["depth_mm"] for m in V.load("train")[2]]))

    models = {}
    for s in (42, 43, 44):
        for v in ("hetero", "wide"):
            ck = torch.load(R / f"04_training/results/variants/{v}_seed{s}.pt", map_location="cpu", weights_only=False)
            m = V.build(ck["variant"]); m.load_state_dict(ck["model"]); models[f"{v}_seed{s}"] = (v, m.eval())
        for tag, p in ((f"F_layered_seed{s}", R / f"04_training/results/layered_cnn/models/F_layered_seed{s}.pt"),
                       (f"F_locked_seed{s}", exp2_paths.exp1_at(f"results/stage4/models/cnn_F_seed{s}.pt"))):
            m = DepthCNN(); m.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model"]); models[tag] = ("base", m.eval())

    res = {"n_vessels": int(len(y)), "n_skins": n_cfg, "sigma_mc_locked": sig_mc, "train_mean_depth": ytrain, "levels": {}}
    levels = [("mc_only", None)] + [(f"tot_{t:.3f}", t) for t in TOTALS]
    preds_all = {}
    for k, (name, tot) in enumerate(levels):
        add = 0.0 if tot is None else float(np.sqrt(max(tot ** 2 - sig_mc ** 2, 0.0)))
        rng = np.random.default_rng(900 + k)
        X = np.concatenate([V.augment(binner, P, rng, sigma=add) for _ in range(COPIES)])
        yy, cc, mm = np.tile(y, COPIES), np.tile(cfg, COPIES), np.tile(mel, COPIES)
        pr, sg = {}, {}
        with torch.no_grad():
            for tag, (var, m) in models.items():
                o = m(torch.from_numpy(X[:, None, :]))
                o = o[:, None] if o.dim() == 1 else o
                pr[tag] = o[:, 0].numpy().astype(float)
                if var == "hetero":
                    sg[tag] = np.exp(o[:, 1].clamp(-6, 3).numpy().astype(float))
        brng = np.random.default_rng(5)
        lev = {"sigma_add": add}
        for g in ("hetero", "wide", "F_layered", "F_locked"):
            tags = [t for t in pr if t.rsplit("_seed", 1)[0] == g]
            ens = np.mean([pr[t] for t in tags], 0)
            sd = None
            if g == "hetero":
                sd = np.sqrt(np.mean([sg[t] ** 2 + pr[t] ** 2 for t in tags], 0) - ens ** 2)
            row = summarise(ens, yy, cc, brng, sd)
            row["mae_per_seed"] = [float(np.abs(pr[t] - yy).mean()) for t in tags]
            row["mae_mel_lt0.05"] = float(np.abs(ens - yy)[mm < 0.05].mean())
            row["mae_mel_ge0.15"] = float(np.abs(ens - yy)[mm >= 0.15].mean())
            lev[g] = row
            preds_all[f"{name}__{g}"] = ens
            if sd is not None:
                preds_all[f"{name}__hetero_sd"] = sd
        lev["constant_train_mean"] = summarise(np.full(len(yy), ytrain), yy, cc, brng)
        if tot is not None:
            first = np.arange(len(X)) < len(y)              # first shifted copy of every vessel
            with Pool(12) as pool:
                b3 = np.array(pool.map(_b3, list(X[first]), chunksize=4), float)
            lev["B3_homog_joint"] = summarise(b3, y, cfg, brng)
            lev["B3_homog_joint"]["fraction_at_upper_bound_5mm"] = float(np.mean(b3 >= 4.999))
            preds_all[f"{name}__B3"] = b3
        res["levels"][name] = lev
        print(name, json.dumps({g: round(v["mae"], 4) for g, v in lev.items() if isinstance(v, dict) and "mae" in v}), flush=True)

    prim = {g: float(np.mean([res["levels"][f"tot_{t:.3f}"][g]["mae"] for t in TOTALS]))
            for g in ("hetero", "wide", "F_layered", "F_locked", "B3_homog_joint", "constant_train_mean")}
    res["primary_mean_over_3_levels"] = prim
    # bootstrap CI of the primary metric (resample skins, same draw for all three levels)
    ids = np.unique(cfg); brng = np.random.default_rng(11)
    errs = {t: np.abs(preds_all[f"tot_{t:.3f}__hetero"] - np.tile(y, COPIES)) for t in TOTALS}
    cc = np.tile(cfg, COPIES)
    per = {t: {c: errs[t][cc == c] for c in ids} for t in TOTALS}
    boots = []
    for _ in range(2000):
        pick = brng.choice(ids, len(ids))
        boots.append(np.mean([np.concatenate([per[t][c] for c in pick]).mean() for t in TOTALS]))
    res["primary_hetero_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    res["validation_ext_reference_hetero"] = 0.1955
    res["consistent_with_validation_ext"] = bool(abs(prim["hetero"] - 0.1955) <= 0.03)
    (OUT / "eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez_compressed(OUT / "predictions.npz", y=y, config=cfg, **preds_all)
    print(json.dumps({"primary": prim, "ci95": res["primary_hetero_ci95"], "consistent": res["consistent_with_validation_ext"]}, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
