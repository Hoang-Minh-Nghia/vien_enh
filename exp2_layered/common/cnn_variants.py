"""step 04: model-improvement variants for F_layered (CPU; validation only; locked_test never read).

Data: step 03 'train' (+ 'train_ext' configs when present, via --ext) and step 03 'validation' for checkpoint
choice.  Comparison between variants is done on 'validation_ext' (90 new skins, never used for
checkpoint selection), see compare_validation_ext.py.

Augmentation is ON THE FLY: each epoch = 8 fresh copies (new shift, new camera noise) of every vessel.

Variants:
  base      DepthCNN (thesis), fixed LR 1e-3                     (reference, on-the-fly aug)
  sched     DepthCNN, LR 1e-3 with ReduceLROnPlateau(0.5, 15), patience 60, max 600
  wide      WideCNN: dilated convs, receptive field ~ 120 bins (30 mm), BatchNorm, same schedule
  multi     WideCNN + auxiliary targets (diameter, log mua_blood, log melanin), same schedule
  hetero    WideCNN + Gaussian NLL head (mean, log sigma) -> calibrated uncertainty (H3)

Usage: cnn_variants.py <variant> <seed> [--ext] [--nskin=N] [--hisnr]
"""
import exp2_paths  # noqa: F401  (all step code folders on sys.path)
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = exp2_paths.EXP1
sys.path.insert(0, str(exp2_paths.exp1_at("common")))
from train import DepthCNN  # noqa: E402

T4 = exp2_paths.EXP2 / "03_dataset/data"
OUT = exp2_paths.EXP2 / "04_training/results/variants"
EDGES = np.linspace(-16, 16, 129)
BATCH = 64
REPEAT = 8          # fresh augmented copies of every training vessel per epoch


# ---------------------------------------------------------------- data
def load(split, manifest="manifest.json"):
    assert split in ("train", "validation", "train_ext", "validation_ext", "train_hisnr"), "locked_test is not allowed here"
    m = json.loads((T4 / manifest).read_text(encoding="utf-8"))
    P, meta = [], []
    for cm in m["splits"][split]["configs"]:
        f = T4 / split / f"{cm['id']}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        c = json.loads(str(z["config_json"])); assert c == cm
        x, s = z["x_mm"], float(z["n0_scalar"])
        for v, p in zip(c["vessels"], z["profiles_y_mean"]):
            P.append(p.astype(np.float64) / s)
            meta.append({"config": c["id"], **v, **c["layers"]})
    return x, np.array(P), meta


def targets(meta):
    return np.array([[m["depth_mm"], m["diameter_mm"], np.log(m["mua"]), np.log(m["mel_fraction"])] for m in meta], np.float32)


class Binner:
    """Vectorised equivalent of readout.bin_average for many profiles and shifts."""
    def __init__(self, x):
        dx = float(np.median(np.diff(x)))
        self.src = np.r_[x - dx / 2, x[-1] + dx / 2]
        self.w = np.diff(EDGES)

    def __call__(self, P, shifts):
        cum = np.concatenate([np.zeros((len(P), 1)), np.cumsum(P * np.diff(self.src), 1)], 1)
        out = np.empty((len(P), len(EDGES) - 1))
        for i, sh in enumerate(shifts):
            F = np.interp(EDGES - sh, self.src, cum[i])
            out[i] = np.diff(F) / self.w
        return out


def augment(binner, P, rng, sigma=None):
    sh = rng.uniform(-2, 2, len(P))
    X = binner(P, sh)
    sa = np.exp(rng.uniform(np.log(0.001), np.log(0.01), len(P))) if sigma is None else np.full(len(P), sigma)
    X += rng.normal(0, 1, X.shape) * sa[:, None]
    return X.astype(np.float32)


# ---------------------------------------------------------------- models
class WideCNN(nn.Module):
    def __init__(self, n_out=1):
        super().__init__()
        ch, layers, c_in = 48, [], 1
        for d in (1, 2, 4, 8, 16):          # kernel 5, dilations -> receptive field 1 + 4*31 = 125 bins
            layers += [nn.Conv1d(c_in, ch, 5, padding=2 * d, dilation=d), nn.BatchNorm1d(ch), nn.ReLU()]
            c_in = ch
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(2 * ch, 64), nn.ReLU(), nn.Linear(64, n_out))

    def forward(self, x):
        h = self.features(x)
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        return self.head(h)


def build(variant):
    if variant in ("base", "sched"):
        return DepthCNN()
    return WideCNN({"wide": 1, "multi": 4, "hetero": 2}[variant])


def loss_and_pred(variant, out, y):
    if out.dim() == 1:                       # DepthCNN returns (B,)
        out = out[:, None]
    if variant == "multi":
        w = torch.tensor([1.0, 0.3, 0.3, 0.3])
        return (torch.abs(out - y) * w).mean(), out[:, 0]
    if variant == "hetero":
        mu, logs = out[:, 0], out[:, 1].clamp(-6, 3)
        return (logs + 0.5 * ((y[:, 0] - mu) / logs.exp()) ** 2).mean(), mu
    return torch.abs(out[:, 0] - y[:, 0]).mean(), out[:, 0]


# ---------------------------------------------------------------- training
def main():
    variant, seed, ext = sys.argv[1], int(sys.argv[2]), "--ext" in sys.argv
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(3)
    hisnr = "--hisnr" in sys.argv               # T4c: same first-120 train skins, 96 M photons (paired test)
    x, Ptr, mtr = load("train_hisnr", "manifest_hisnr.json") if hisnr else load("train")
    if ext:
        _, Pe, me = load("train_ext", "manifest_ext.json")
        Ptr, mtr = np.concatenate([Ptr, Pe]), mtr + me
    nskin = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--nskin=")), None)
    if nskin:                                   # learning curve: keep the first n skins (manifest order is random)
        keep_ids = list(dict.fromkeys(m["config"] for m in mtr))[:nskin]
        keep = np.array([m["config"] in keep_ids for m in mtr])
        Ptr, mtr = Ptr[keep], [m for m, k in zip(mtr, keep) if k]
    _, Pva, mva = load("validation")
    binner = Binner(x)
    ytr, yva = torch.from_numpy(targets(mtr)), torch.from_numpy(targets(mva))
    vrng = np.random.default_rng(52)
    Xva = torch.from_numpy(np.concatenate([augment(binner, Pva, vrng) for _ in range(8)])[:, None, :])
    yva8 = yva.repeat(8, 1)
    model = build(variant)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    fixed = variant == "base"
    sched = None if fixed else torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=15)
    max_ep, patience = (400, 40) if fixed else (600, 60)
    rng = np.random.default_rng(1000 + seed)
    best, state, wait, best_ep, hist, t0 = float("inf"), None, 0, 0, [], time.time()
    tag = f"{variant}{'_hisnr' if hisnr else ''}{'_ext' if ext else ''}{f'_n{nskin}' if nskin else ''}_seed{seed}"
    for ep in range(1, max_ep + 1):
        Xtr = torch.from_numpy(np.concatenate([augment(binner, Ptr, rng) for _ in range(REPEAT)])[:, None, :])
        ytr_ep = ytr.repeat(REPEAT, 1)
        perm = torch.from_numpy(rng.permutation(len(Xtr)))
        model.train(); tot = 0.0
        for k in range(0, len(perm), BATCH):
            b = perm[k:k + BATCH]
            if len(b) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss, _ = loss_and_pred(variant, model(Xtr[b]), ytr_ep[b])
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward(); opt.step(); tot += float(loss.detach()) * len(b)
        model.eval()
        with torch.no_grad():
            _, pv = loss_and_pred(variant, model(Xva), yva8)
            v = float(torch.abs(pv - yva8[:, 0]).mean())
        if sched:
            sched.step(v)
        hist.append({"epoch": ep, "train_loss": tot / len(Xtr), "val_mae_mm": v, "lr": opt.param_groups[0]["lr"]})
        if v < best:
            best, state, wait, best_ep = v, copy.deepcopy(model.state_dict()), 0, ep
        else:
            wait += 1
        print(f"{tag} ep {ep} loss {tot/len(Xtr):.4f} val {v:.4f} best {best:.4f}@{best_ep} lr {opt.param_groups[0]['lr']:.1e} {time.time()-t0:.0f}s", flush=True)
        if wait >= patience:
            break
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "variant": variant, "seed": seed, "ext": ext, "hisnr": hisnr, "best_epoch": best_ep,
                "val_mae_mm": best, "n_train_vessels": len(Ptr)}, OUT / f"{tag}.pt")
    (OUT / f"{tag}.json").write_text(json.dumps({"variant": variant, "seed": seed, "ext": ext, "best_epoch": best_ep,
                                                 "last_epoch": ep, "val_mae_mm": best, "n_train_vessels": len(Ptr),
                                                 "runtime_s": time.time() - t0, "history": hist}), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
