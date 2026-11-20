"""step 03: extend the layered-skin dataset (more skins for training, a larger validation set).

Same physics, ranges, readout and simulate() as dataset_generate.py (imported, unchanged).
New splits use NEW seeds and a separate manifest written BEFORE any run:
  validation_ext : 90 configs x 5 vessels, 24 M photons per vessel, N0 12 M  (seed 20261117)  -> run first
  train_ext      : 900 configs x 5 vessels, 12 M photons per vessel, N0 6 M  (seed 20261116)
locked_test is untouched.  Resumable; on RuntimeError writes <id>.invalid.json and stops.

Usage: dataset_extend_ext.py manifest | dataset_extend_ext.py run [split ...]
       (e.g. `run validation_ext` simulates only validation_ext; train_ext is not part of the results)
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
SPLITS = {"validation_ext": dict(n_cfg=90, vessels=5, photons=24_000_000, n0_photons=12_000_000, seed=20261117),
          "train_ext": dict(n_cfg=900, vessels=5, photons=12_000_000, n0_photons=6_000_000, seed=20261116)}


def manifest():
    m = {"version": "t4b/1", "extends": "manifest.json (t4/1)", "voxel_mm": G.V, "g_tissue": G.G_TISSUE,
         "n_tissue": G.N_TISSUE, "g_blood": G.G_BLOOD, "readout": "y-mean profile / scalar N0", "splits": {}}
    for name, s in SPLITS.items():
        r = np.random.default_rng(s["seed"])
        cfgs = []
        for i in range(s["n_cfg"]):
            c = G.draw_config(r)
            cfgs.append({"id": f"{name}_{i:03d}", "layers": c, "mc_seed": int(r.integers(1, 2 ** 31 - 1)),
                         "vessels": [G.draw_vessel(r, c) for _ in range(s["vessels"])]})
        m["splits"][name] = {**s, "configs": cfgs}
    return m


def run(splits=("validation_ext", "train_ext")):
    m = json.loads((OUT / "manifest_ext.json").read_text(encoding="utf-8"))
    t0 = time.time(); count = 0
    for split in splits:
        s = m["splits"][split]
        d = OUT / split; d.mkdir(parents=True, exist_ok=True)
        for c in s["configs"]:
            f = d / f"{c['id']}.npz"
            if f.exists():
                count += 1; continue
            try:
                n0, x = G.simulate(c["layers"], None, s["n0_photons"], c["mc_seed"])
                profs = []
                for j, v in enumerate(c["vessels"]):
                    R, _ = G.simulate(c["layers"], v, s["photons"], c["mc_seed"] + 1 + j)
                    profs.append(R.mean(1))
            except RuntimeError as e:
                (d / f"{c['id']}.invalid.json").write_text(json.dumps({"error": str(e)}), encoding="utf-8")
                raise
            tmp = f.with_suffix(".partial.npz")
            np.savez_compressed(tmp, x_mm=x, profiles_y_mean=np.array(profs, dtype=np.float32),
                                n0_scalar=float(n0.mean()), n0_cv=float(n0.std() / n0.mean()),
                                n0_y_mean=n0.mean(1).astype(np.float32), config_json=json.dumps(c))
            tmp.replace(f)
            count += 1
            print(f"CFG {split} {c['id']} done; total {count}; elapsed {time.time()-t0:.0f}s", flush=True)
    print("T4B_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "manifest":
        p = OUT / "manifest_ext.json"
        if p.exists():
            raise SystemExit("manifest_ext exists; refusing to overwrite")
        m = manifest()
        # no id or config may collide with the original manifest
        old = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
        old_ids = {c["id"] for s in old["splits"].values() for c in s["configs"]}
        assert not old_ids & {c["id"] for s in m["splits"].values() for c in s["configs"]}
        p.write_text(json.dumps(m, indent=1), encoding="utf-8")
        print({k: len(v["configs"]) for k, v in m["splits"].items()})
    else:
        run(tuple(sys.argv[2:]) or ("validation_ext", "train_ext"))
