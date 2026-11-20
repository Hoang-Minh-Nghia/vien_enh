"""Exploratory step 04 check: is F limited by training length?

Same stage-4 dataset, same DepthCNN, Adam 1e-3, batch 64, L1 loss and validation-only
checkpoint choice as the locked run; only max_epochs (100 -> 400) and patience (10 -> 40) change.
Scored on the V1 profiles (already opened, so this is exploratory, not a locked test).
Writes nothing into the thesis package.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp2_paths  # noqa: E402,F401  (all step code folders on sys.path)
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = exp2_paths.EXP1
sys.path.insert(0, str(exp2_paths.exp1_at("common")))
from train import DepthCNN  # noqa: E402

OUT = exp2_paths.EXP2 / "04_training/results/longer_training"
MAX_EPOCHS, PATIENCE = 400, 40
GROUPS = {"unseen_pooled": (0.22, 0.40), "seen_weak_pooled": (0.15, 0.30), "standard_0.508": (0.508,),
          "high_1.0": (1.0,), "all": (0.15, 0.22, 0.30, 0.40, 0.508, 1.0)}


def train(seed, x, y, split):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2)
    tr, va = split == "train", split == "validation"
    loader = DataLoader(TensorDataset(x[tr], y[tr]), batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
    model, loss_fn = DepthCNN(), nn.L1Loss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, state, wait, best_epoch = float("inf"), None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(loss_fn(model(x[va]), y[va]))
        if v < best:
            best, state, wait, best_epoch = v, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    model.load_state_dict(state)
    return model.eval(), best_epoch, best, epoch


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(exp2_paths.exp1_at("results/stage4/data/dataset.npz"), allow_pickle=False) as z:
        x = torch.from_numpy(z["X_F"][:, None, :].astype(np.float32))
        y = torch.from_numpy(z["y_mm"].astype(np.float32))
        split = z["split"]
    with np.load(exp2_paths.exp1_at("results/validation_independent/predictions/profiles.npz")) as z:
        V = torch.from_numpy(z["X"][:, None, :].astype(np.float32))
        depth, mua = z["y_mm"].astype(float), z["mua_mm_inv"].astype(float)
    result = {}
    for seed in (42, 43, 44):
        model, be, bv, last = train(seed, x, y, split)
        torch.save(model.state_dict(), OUT / f"cnn_F_long_seed{seed}.pt")
        with torch.no_grad():
            pred = model(V).numpy()
        groups = {}
        for name, muas in GROUPS.items():
            keys = sorted({(d, a) for d, a in zip(depth, mua) if any(abs(a - m) < 1e-6 for m in muas)})
            groups[name] = float(np.mean([np.mean(np.abs(pred[(np.isclose(depth, d)) & (np.isclose(mua, a))] - d)) for d, a in keys]))
        hard = [(3.3, 0.15), (3.8, 0.15)]
        groups["hard_z3.3-3.8_mua0.15"] = float(np.mean([np.mean(np.abs(pred[(np.isclose(depth, d)) & (np.isclose(mua, a))] - d)) for d, a in hard]))
        result[seed] = {"best_epoch": be, "last_epoch": last, "validation_mae_mm": bv, "V1_MAE_mm": groups}
        print(seed, json.dumps(result[seed]), flush=True)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
