"""Sanity check of step 03 files (train/validation only; never opens locked_test)."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp2_paths  # noqa: E402,F401  (all step code folders on sys.path)
import json
import sys
from pathlib import Path

import numpy as np

D = exp2_paths.EXP2 / "03_dataset/data"
files = sorted((D / "train").glob("*.npz")) + sorted((D / "validation").glob("*.npz"))
files = [f for f in files if not f.name.endswith(".partial.npz")][: int(sys.argv[1]) if len(sys.argv) > 1 else None]
for f in files:
    z = np.load(f); c = json.loads(str(z["config_json"]))
    x = z["x_mm"]; n0 = z["n0_scalar"]
    far = np.abs(x) > 20
    lat = z["n0_y_mean"] / n0
    print(f"{f.stem}: N0 {n0:.4f}, N0 lateral ripple (y-mean, std/mean) {lat.std():.4f}, mel {c['layers']['mel_fraction']:.3f}")
    for v, p in zip(c["vessels"], z["profiles_y_mean"]):
        r = p / n0
        print(f"   z {v['depth_mm']:.2f} D {v['diameter_mm']:.2f} mua {v['mua']:.3f}: contrast {1 - r.min():.3f}, "
              f"far-field ratio {r[far].mean():.4f}, argmin x {x[np.argmin(r)]:+.2f} mm")
