"""step 03: high-SNR re-simulation of step 03 train skins 120..299 (completes the 96 M-photon train set; seed 20261119).

Purpose: test whether the ~0.19 mm noise-independent error floor of F_layered comes from the noisy
training data (12 M photons -> sigma_mc ~0.0049 per bin).  96 M photons per vessel -> sigma_mc ~0.0017.
Paired design: same skins at 12 M (existing) vs 96 M (this run).
Manifest written BEFORE running; locked_test untouched; resumable; stop on invalid.

Usage: dataset_high_snr_a.py manifest | dataset_high_snr_a.py run
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp2_paths  # noqa: E402,F401  (all step code folders on sys.path)
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generate as G  # noqa: E402

OUT = G.OUT
LO, HI, PHOTONS, N0_PHOTONS, SEED = 120, 300, 96_000_000, 24_000_000, 20261119


def manifest():
    old = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))["splits"]["train"]["configs"][LO:HI]
    r = np.random.default_rng(SEED)
    cfgs = [{"id": f"train_hisnr2_{i:03d}", "source_id": c["id"], "layers": c["layers"], "vessels": c["vessels"],
             "mc_seed": int(r.integers(1, 2 ** 31 - 1))} for i, c in enumerate(old)]
    return {"version": "t4d/1", "source": "manifest.json train[120:300]", "splits": {"train_hisnr2": {
        "n_cfg": HI - LO, "vessels": 5, "photons": PHOTONS, "n0_photons": N0_PHOTONS, "seed": SEED, "configs": cfgs}}}


def run():
    s = json.loads((OUT / "manifest_hisnr2.json").read_text(encoding="utf-8"))["splits"]["train_hisnr2"]
    d = OUT / "train_hisnr2"; d.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); count = 0
    for c in s["configs"]:
        f = d / f"{c['id']}.npz"
        if f.exists():
            count += 1; continue
        try:
            n0, x = G.simulate(c["layers"], None, s["n0_photons"], c["mc_seed"])
            profs = [G.simulate(c["layers"], v, s["photons"], c["mc_seed"] + 1 + j)[0].mean(1) for j, v in enumerate(c["vessels"])]
        except RuntimeError as e:
            (d / f"{c['id']}.invalid.json").write_text(json.dumps({"error": str(e)}), encoding="utf-8")
            raise
        tmp = f.with_suffix(".partial.npz")
        np.savez_compressed(tmp, x_mm=x, profiles_y_mean=np.array(profs, dtype=np.float32), n0_scalar=float(n0.mean()),
                            n0_cv=float(n0.std() / n0.mean()), n0_y_mean=n0.mean(1).astype(np.float32), config_json=json.dumps(c))
        tmp.replace(f)
        count += 1
        print(f"CFG train_hisnr2 {c['id']} done; total {count}; elapsed {time.time()-t0:.0f}s", flush=True)
    print("T4D_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "manifest":
        p = OUT / "manifest_hisnr2.json"
        if p.exists():
            raise SystemExit("manifest_hisnr exists; refusing to overwrite")
        p.write_text(json.dumps(manifest(), indent=1), encoding="utf-8")
        print("written", p)
    else:
        run()
