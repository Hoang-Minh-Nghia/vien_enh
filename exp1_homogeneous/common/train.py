"""Fixed CNN training and paired evaluation on explicitly selected test inputs."""
import argparse
import copy
import csv
import json
import random
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from simulate import ROOT


class DepthCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(),
                                      nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
                                      nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
                                      nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


def metrics(pred, truth):
    error = np.asarray(pred) - np.asarray(truth)
    if not len(error):
        return None
    return {"n": len(error), "mae_mm": float(np.abs(error).mean()),
            "rmse_mm": float(np.sqrt(np.square(error).mean())), "bias_mm": float(error.mean())}


def train(data_path, output, branch="K", seed=42):
    config = json.loads((ROOT / "configs/learning.json").read_text(encoding="utf-8"))
    data_sha256 = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    cfg = config["training"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cfg["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    with np.load(data_path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    x = torch.from_numpy(data[f"X_{branch}"][:, None, :])
    y = torch.from_numpy(data["y_mm"])
    train_mask, val_mask = data["split"] == "train", data["split"] == "validation"
    loader = DataLoader(TensorDataset(x[train_mask], y[train_mask]), batch_size=cfg["batch_size"],
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    model = DepthCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.L1Loss()
    best, best_state, waiting, history, best_epoch = float("inf"), None, 0, [], 0
    start = time.perf_counter()
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(xb)
        model.eval()
        with torch.no_grad():
            val_mae = float(loss_fn(model(x[val_mask]), y[val_mask]))
        history.append({"epoch": epoch, "train_mae_mm": total / int(train_mask.sum()), "validation_mae_mm": val_mae})
        if val_mae < best:
            best, best_state, waiting, best_epoch = val_mae, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            waiting += 1
        print(f"TRAIN {branch} seed={seed} epoch={epoch} train={history[-1]['train_mae_mm']:.5f} val={val_mae:.5f}", flush=True)
        if waiting >= cfg["early_stopping_patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "branch": branch, "seed": seed, "config": config,
                "dataset_sha256": data_sha256,
                "best_epoch": best_epoch, "validation_mae_mm": best}, output / f"cnn_{branch}_seed{seed}.pt")
    report = {"branch": branch, "seed": seed, "best_epoch": best_epoch, "validation_mae_mm": best,
              "runtime_s": time.perf_counter() - start, "history": history,
              "dataset_sha256": data_sha256,
              "checkpoint_selection": "validation only", "dataset": str(Path(data_path).resolve())}
    (output / f"training_{branch}_seed{seed}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return model


def evaluate(model, data_path, branch_input="F"):
    with np.load(data_path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    test = data["split"] == "test"
    x = torch.from_numpy(data[f"X_{branch_input}"][test, None, :])
    with torch.no_grad():
        pred = model(x).numpy()
    return pred, {k: v[test] for k, v in data.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--branch", choices=["F", "K"], default="K")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-input", choices=["F", "K"], default="K")
    args = p.parse_args()
    model = train(args.dataset, args.output, args.branch, args.seed)
    pred, test = evaluate(model, args.dataset, args.test_input)
    with np.load(args.dataset, allow_pickle=False) as z:
        mean_train = float(z["y_mm"][z["split"] == "train"].mean())
    report = {"cnn": metrics(pred, test["y_mm"]),
              "constant_training_mean": metrics(np.full(len(pred), mean_train), test["y_mm"]),
              "scope": "Development CNN test on interpolated MC data; no physical validation"}
    Path(args.output, "trial_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(Path(args.output, "trial_predictions.npz"),
                        prediction_mm=pred, true_depth_mm=test["y_mm"], scene_id=test["scene_id"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
