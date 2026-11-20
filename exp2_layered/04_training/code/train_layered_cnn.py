"""step 04: train F_layered on the step 03 layered-skin data and compare it on VALIDATION only.

Usage (analysis venv, CPU torch):
  train_layered_cnn.py build            -> exp2_layered/04_training/results/layered_cnn/data.npz   (train / validation / evaluation sets)
  train_layered_cnn.py train <seed>     -> exp2_layered/04_training/results/layered_cnn/models/F_layered_seed<seed>.pt + history json
  train_layered_cnn.py eval             -> exp2_layered/04_training/results/layered_cnn/validation_eval.json (+ predictions npz)

Rules: locked_test is never read here (asserted).  Locked thesis checkpoints are loaded read-only.

Readout (same as stage 4 / dataset.py): native ratio profile = y-mean reflectance / scalar N0,
bin-averaged onto 128 bins over [-16, 16] mm after a lateral shift U(-2, 2) mm.

Noise: step 03 profiles already carry Monte Carlo noise, measured at ~0.0049 per 0.25-mm bin (12 M
photons).  Training adds camera noise sigma_add ~ log-uniform(0.001, 0.01) as planned.  Evaluation
uses TOTAL per-bin noise levels sigma_tot in {MC only, 0.007, 0.010}: sigma_add = sqrt(tot^2 - mc^2),
so the locked test (60 M photons, lower MC noise) can later be scored at the same total noise.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp2_paths  # noqa: E402,F401  (all step code folders on sys.path)
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = exp2_paths.EXP1
sys.path.insert(0, str(exp2_paths.exp1_at("common")))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from readout import bin_average  # noqa: E402  (locked module, read-only)

T4 = exp2_paths.EXP2 / "03_dataset/data"
OUT = exp2_paths.EXP2 / "04_training/results/layered_cnn"
EDGES = np.linspace(-16, 16, 129)
SIGMA_MC = 0.0049                      # measured per-bin MC noise of 12 M-photon T4 profiles
EVAL_TOTAL = {"mc_only": None, "tot_0.007": 0.007, "tot_0.010": 0.010}
COPIES = {"train": 24, "validation": 8, "eval": 4}
MAX_EPOCHS, PATIENCE, BATCH, LR = 400, 40, 64, 1e-3


def load_split(split):
    assert split in ("train", "validation"), "locked_test must not be opened in T5 development"
    m = json.loads((T4 / "manifest.json").read_text(encoding="utf-8"))
    prof, meta = [], []
    for cm in m["splits"][split]["configs"]:
        z = np.load(T4 / split / f"{cm['id']}.npz")
        c = json.loads(str(z["config_json"]))
        assert c == cm
        x, s = z["x_mm"], float(z["n0_scalar"])
        for j, (v, p) in enumerate(zip(c["vessels"], z["profiles_y_mean"])):
            prof.append(p.astype(float) / s)
            meta.append({"config": c["id"], "vessel": j, **v, **c["layers"]})
    return x, np.array(prof), meta


def augment(x, prof, copies, rng, sigma_mode):
    """sigma_mode: ('train',) log-uniform add; ('total', tot) fixed total; ('none',) MC only."""
    X, idx, shifts, sig = [], [], [], []
    for i, p in enumerate(prof):
        for _ in range(copies):
            sh = rng.uniform(-2.0, 2.0)
            b = bin_average(x, p, EDGES - sh)
            if sigma_mode[0] == "train":
                sa = float(np.exp(rng.uniform(np.log(0.001), np.log(0.01))))
            elif sigma_mode[0] == "total":
                sa = float(np.sqrt(sigma_mode[1] ** 2 - SIGMA_MC ** 2))
            else:
                sa = 0.0
            X.append(b + rng.normal(0.0, sa, b.shape) if sa else b)
            idx.append(i); shifts.append(sh); sig.append(sa)
    return np.asarray(X, np.float32), np.array(idx), np.array(shifts), np.array(sig)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for split, seed in (("train", 51), ("validation", 52)):
        x, prof, meta = load_split(split)
        X, idx, sh, sg = augment(x, prof, COPIES[split], np.random.default_rng(seed), ("train",))
        y = np.array([meta[i]["depth_mm"] for i in idx], np.float32)
        out.update({f"X_{split}": X, f"y_{split}": y, f"idx_{split}": idx, f"shift_{split}": sh, f"sigadd_{split}": sg})
        if split == "validation":
            (OUT / "validation_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            for k, (name, tot) in enumerate(EVAL_TOTAL.items()):
                mode = ("none",) if tot is None else ("total", tot)
                Xe, ie, se, ge = augment(x, prof, COPIES["eval"], np.random.default_rng(60 + k), mode)
                out.update({f"Xeval_{name}": Xe, f"idxeval_{name}": ie, f"shifteval_{name}": se})
    assert all(np.isfinite(v).all() for k, v in out.items() if k.startswith("X"))
    np.savez_compressed(OUT / "data.npz", **out)
    print({k: v.shape for k, v in out.items() if k.startswith("X")})


def train(seed):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from train import DepthCNN
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    d = np.load(OUT / "data.npz")
    xt, yt = torch.from_numpy(d["X_train"][:, None, :]), torch.from_numpy(d["y_train"])
    xv, yv = torch.from_numpy(d["X_validation"][:, None, :]), torch.from_numpy(d["y_validation"])
    loader = DataLoader(TensorDataset(xt, yt), batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed))
    model, loss_fn = DepthCNN(), nn.L1Loss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, state, wait, best_epoch, hist, t0 = float("inf"), None, 0, 0, [], time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); tot = 0.0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward(); opt.step(); tot += float(loss) * len(xb)
        model.eval()
        with torch.no_grad():
            v = float(loss_fn(model(xv), yv))
        hist.append({"epoch": epoch, "train_mae_mm": tot / len(yt), "validation_mae_mm": v})
        if v < best:
            best, state, wait, best_epoch = v, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            wait += 1
        print(f"seed {seed} epoch {epoch} train {tot/len(yt):.4f} val {v:.4f} best {best:.4f}@{best_epoch} {time.time()-t0:.0f}s", flush=True)
        if wait >= PATIENCE:
            break
    (OUT / "models").mkdir(exist_ok=True)
    torch.save({"model": state, "seed": seed, "best_epoch": best_epoch, "validation_mae_mm": best,
                "data": "exp2_layered/04_training/results/layered_cnn/data.npz", "lr": LR, "batch": BATCH, "patience": PATIENCE},
               OUT / "models" / f"F_layered_seed{seed}.pt")
    (OUT / "models" / f"history_seed{seed}.json").write_text(json.dumps(
        {"seed": seed, "best_epoch": best_epoch, "last_epoch": epoch, "validation_mae_mm": best,
         "runtime_s": time.time() - t0, "history": hist}, indent=1), encoding="utf-8")


def _b3_worker(args):
    from b3_physics_fit import Forward, fit
    global _FWD
    if "_FWD" not in globals():
        _FWD = Forward()
    return fit(_FWD, np.asarray(args, float), True)[0]


def metrics(pred, truth):
    e = np.asarray(pred) - np.asarray(truth)
    return {"n": int(len(e)), "mae_mm": float(np.abs(e).mean()), "bias_mm": float(e.mean()),
            "rmse_mm": float(np.sqrt((e ** 2).mean())), "p90_abs_mm": float(np.quantile(np.abs(e), 0.9))}


def evaluate():
    import torch
    from multiprocessing import Pool
    from train import DepthCNN
    torch.set_num_threads(8)
    d = np.load(OUT / "data.npz")
    meta = json.loads((OUT / "validation_meta.json").read_text(encoding="utf-8"))
    ytrain_mean = float(d["y_train"].mean())

    def load(path, key=None):
        m = DepthCNN(); obj = torch.load(path, map_location="cpu", weights_only=False)
        m.load_state_dict(obj["model"] if key else obj); return m.eval()
    models = {}
    for s in (42, 43, 44):
        models[f"F_layered_seed{s}"] = load(OUT / "models" / f"F_layered_seed{s}.pt", "model")
        models[f"F_locked_seed{s}"] = load(exp2_paths.exp1_at("results/stage4/models") / f"cnn_F_seed{s}.pt", "model")
        models[f"F_long_homog_seed{s}"] = load(exp2_paths.EXP2 / "04_training/results/longer_training" / f"cnn_F_long_seed{s}.pt")
    res, preds = {}, {}
    for name in EVAL_TOTAL:
        X, idx = d[f"Xeval_{name}"], d[f"idxeval_{name}"]
        y = np.array([meta[i]["depth_mm"] for i in idx])
        P = {}
        with torch.no_grad():
            for k, m in models.items():
                P[k] = m(torch.from_numpy(X[:, None, :])).numpy()
        P["constant_train_mean"] = np.full(len(y), ytrain_mean)
        first = np.arange(len(idx)) % COPIES["eval"] == 0         # B3: one shifted copy per vessel
        with Pool(12) as pool:
            b3 = np.array(pool.map(_b3_worker, list(X[first]), chunksize=4))
        P["B3_homog_joint"] = np.full(len(y), np.nan); P["B3_homog_joint"][first] = b3
        res[name] = {"B3_fraction_at_upper_bound_5mm": float(np.mean(b3 >= 4.999))}
        for k, p in P.items():
            ok = np.isfinite(p)
            res[name][k] = metrics(p[ok], y[ok])
        # grouped breakdown for ensemble-of-seeds summaries
        groups = {
            "depth<2": y < 2, "depth2-3": (y >= 2) & (y < 3), "depth3-4": (y >= 3) & (y < 4), "depth>=4": y >= 4,
            "mel<0.05": np.array([meta[i]["mel_fraction"] < 0.05 for i in idx]),
            "mel>=0.15": np.array([meta[i]["mel_fraction"] >= 0.15 for i in idx]),
        }
        res[name]["groups"] = {}
        for g, mask in groups.items():
            res[name]["groups"][g] = {k: metrics(P[k][mask & np.isfinite(P[k])], y[mask & np.isfinite(P[k])])["mae_mm"]
                                      for k in ("F_layered_seed42", "F_layered_seed43", "F_layered_seed44",
                                                "F_locked_seed42", "B3_homog_joint")}
        preds[name] = {"y": y, "idx": idx, **P}
        print(name, json.dumps({k: round(v["mae_mm"], 3) for k, v in res[name].items() if isinstance(v, dict) and "mae_mm" in v}), flush=True)
    (OUT / "validation_eval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "validation_predictions.npz",
                        **{f"{n}__{k}": v for n, P in preds.items() for k, v in P.items()})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "train":
        train(int(sys.argv[2]))
    elif cmd == "eval":
        evaluate()
