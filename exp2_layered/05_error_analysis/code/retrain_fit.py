"""step 05: retrain hetero variants to test whether the fixed per-skin bias is a fitting problem.

Variants (see report_retraining.md): L, LN, LB, LS. 400 fixed epochs, cosine LR 1e-3 -> 1e-5,
final weights kept (no checkpoint selection). Train split only; locked_test never read.
Usage: retrain_fit.py <variant> <seed>
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp2_paths  # noqa: E402,F401  (all step code folders on sys.path)
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import cnn_variants as V

OUT = exp2_paths.EXP2 / "05_error_analysis/results/retraining"
EPOCHS, BATCH, REPEAT = 400, 64, 8
NOISE = {"L": (0.001, 0.01), "LN": (0.001, 0.003), "LB": (0.001, 0.01), "LS": (0.001, 0.01)}


class WideS(nn.Module):
    """WideCNN with configurable width and optional log-sigma input to the head."""
    def __init__(self, ch=48, hid=64, sig_in=False):
        super().__init__()
        layers, c_in = [], 1
        for d in (1, 2, 4, 8, 16):
            layers += [nn.Conv1d(c_in, ch, 5, padding=2 * d, dilation=d), nn.BatchNorm1d(ch), nn.ReLU()]
            c_in = ch
        self.features = nn.Sequential(*layers)
        self.sig_in = sig_in
        self.head = nn.Sequential(nn.Linear(2 * ch + int(sig_in), hid), nn.ReLU(), nn.Linear(hid, 2))

    def forward(self, x, logsig=None):
        h = self.features(x)
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        if self.sig_in:
            h = torch.cat([h, (logsig[:, None] - np.log(0.003)) / 1.2], 1)
        return self.head(h)


def build(variant):
    return WideS(96, 128) if variant == "LB" else WideS(sig_in=variant == "LS")


def augment(binner, P, rng, lo, hi, sigma=None):
    sh = rng.uniform(-2, 2, len(P))
    X = binner(P, sh)
    sa = np.exp(rng.uniform(np.log(lo), np.log(hi), len(P))) if sigma is None else np.full(len(P), sigma)
    X += rng.normal(0, 1, X.shape) * sa[:, None]
    return X.astype(np.float32), sa.astype(np.float32)


def main():
    variant, seed = sys.argv[1], int(sys.argv[2])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(3)
    x, Ptr, mtr = V.load("train")
    binner = V.Binner(x)
    ytr = torch.tensor([m["depth_mm"] for m in mtr], dtype=torch.float32)
    lo, hi = NOISE[variant]
    model = build(variant)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    rng = np.random.default_rng(2000 + seed)
    tag = f"{variant}_seed{seed}"
    OUT.mkdir(parents=True, exist_ok=True)
    hist, t0 = [], time.time()
    for ep in range(1, EPOCHS + 1):
        parts = [augment(binner, Ptr, rng, lo, hi) for _ in range(REPEAT)]
        Xtr = torch.from_numpy(np.concatenate([p[0] for p in parts])[:, None, :])
        Str = torch.from_numpy(np.log(np.concatenate([p[1] for p in parts])))
        y = ytr.repeat(REPEAT)
        perm = torch.from_numpy(rng.permutation(len(Xtr)))
        model.train(); tot = 0.0; ae = 0.0
        for k in range(0, len(perm), BATCH):
            b = perm[k:k + BATCH]
            if len(b) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            out = model(Xtr[b], Str[b])
            mu, logs = out[:, 0], out[:, 1].clamp(-6, 3)
            loss = (logs + 0.5 * ((y[b] - mu) / logs.exp()) ** 2).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward(); opt.step()
            tot += float(loss.detach()) * len(b); ae += float((mu.detach() - y[b]).abs().sum())
        sched.step()
        hist.append({"epoch": ep, "train_nll": tot / len(Xtr), "train_mae_aug": ae / len(Xtr), "lr": opt.param_groups[0]["lr"]})
        if ep % 5 == 0 or ep == 1:
            print(f"{tag} ep {ep} nll {tot/len(Xtr):.4f} mae {ae/len(Xtr):.4f} lr {opt.param_groups[0]['lr']:.1e} {time.time()-t0:.0f}s", flush=True)
    torch.save({"model": model.state_dict(), "variant": variant, "seed": seed, "noise": [lo, hi], "epochs": EPOCHS}, OUT / f"{tag}.pt")
    (OUT / f"{tag}.json").write_text(json.dumps({"variant": variant, "seed": seed, "runtime_s": time.time() - t0,
                                                 "history": hist}), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
