"""step 03: layered-skin training / validation / locked-test data at 850 nm (resumable, single GPU runner).

Design (ranges and sources: 00_parameters_850nm.md, report_layered_model.md, report_dataset_design.md):
  * One "configuration" = one set of skin-layer optics/thicknesses. Its N0 (no vessel) is laterally
    uniform (planar source over the whole cyclic domain, laterally uniform layers), so it is reduced
    to a single scalar: the mean reflectance over the whole surface (readout "t4/scalar_N0").
  * Each configuration carries several vessels (depth, diameter, blood mua, blood mus).
  * Saved per vessel: y-mean reflectance profile (600 x points, 0.1 mm), divided by the scalar N0
    at training time.  Shift, binning and camera noise are applied later, as in stage 4.
  * The manifest is drawn from fixed seeds and written BEFORE any run.  The locked test split uses
    a different seed, more photons, and is generated last; it must not be read before the final
    evaluation (same rule as V1).

Usage: dataset_generate.py manifest | dataset_generate.py run [max_configs]
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

OUT = exp2_paths.EXP2 / "03_dataset/data"
V, WIDTH, LENGTH, THICK, BATCH = 0.1, 60.0, 8.0, 20.0, 6_000_000
G_TISSUE, N_TISSUE, G_BLOOD = 0.9, 1.37, 0.9832
EPI_MM, MEL_MUA, EPI_BASE_MUA, EPI_MUSP = 0.06, 11.6, 0.025, 3.71
SPLITS = {"train": dict(n_cfg=300, vessels=5, photons=12_000_000, n0_photons=6_000_000, seed=20261113),
          "validation": dict(n_cfg=30, vessels=5, photons=12_000_000, n0_photons=6_000_000, seed=20261114),
          "locked_test": dict(n_cfg=40, vessels=3, photons=60_000_000, n0_photons=12_000_000, seed=20261115)}


def loguni(r, lo, hi):
    return float(np.exp(r.uniform(np.log(lo), np.log(hi))))


def draw_config(r):
    c = {"mel_fraction": loguni(r, 0.01, 0.30),        # Jacques 1998: 1.3-43 %; dark skin truncated at 30 %
         "dermis_mm": float(r.uniform(0.6, 1.5)),       # skin 0.65-1.55 mm (Van Mulder 2016) minus epidermis
         "derm_musp": float(r.uniform(1.5, 3.0)),       # Jacques 2013 table 1: 1.78-2.28 around skin rows
         "derm_mua": loguni(r, 0.01, 0.04),            # estimate (baseline 0.025 + B 0.2 %)
         "fat_mm": float(r.uniform(4.0, 10.0)),         # assumption (no forearm source read yet)
         "fat_musp": float(r.uniform(1.0, 2.1)),        # Simpson 1998 1.12 ... Salomatina 2006 2.08
         "fat_mua": loguni(r, 0.005, 0.02),            # estimate around Simpson 633 nm value 0.013
         "musc_musp": float(r.uniform(0.6, 1.0)),       # Jacques 2013 muscle row 0.80
         "musc_mua": float(r.uniform(0.02, 0.04))}     # estimate from forearm blood volume
    return c


def draw_vessel(r, c):
    skin = EPI_MM + c["dermis_mm"]
    for _ in range(1000):
        D = float(r.uniform(2.0, 4.0))
        z = float(r.uniform(skin + 0.1, 5.0))
        if z + D <= skin + c["fat_mm"] - 0.3:          # vessel lies inside the fat layer
            break
    else:
        raise RuntimeError("no admissible vessel")
    return {"depth_mm": z, "diameter_mm": D,
            "mua": loguni(r, 0.30, 0.60),               # Bosschaart 2014: 0.42-0.49 at Hct 45 %, widened for Hct/SO2
            "mus": float(r.uniform(60.0, 85.0))}        # Bosschaart 2014: 56-74 mm^-1


def manifest():
    m = {"version": "t4/1", "voxel_mm": V, "g_tissue": G_TISSUE, "n_tissue": N_TISSUE, "g_blood": G_BLOOD,
         "readout": "y-mean profile / scalar N0 (mean over whole surface)", "splits": {}}
    for name, s in SPLITS.items():
        r = np.random.default_rng(s["seed"])
        cfgs = []
        for i in range(s["n_cfg"]):
            c = draw_config(r)
            cfgs.append({"id": f"{name}_{i:03d}", "layers": c, "mc_seed": int(r.integers(1, 2 ** 31 - 1)),
                         "vessels": [draw_vessel(r, c) for _ in range(s["vessels"])]})
        m["splits"][name] = {**s, "configs": cfgs}
    return m


def volume(c, vessel=None):
    nx, ny, nz = int(round(WIDTH / V)), int(round(LENGTH / V)), int(round(THICK / V))
    zc = (np.arange(nz + 1) - 0.5) * V
    d2 = EPI_MM + c["dermis_mm"]; d3 = d2 + c["fat_mm"]
    lab = np.where(zc < d2, 2, np.where(zc < d3, 3, 4)).astype(np.uint8)
    lab[0], lab[1] = 0, 1
    vol = np.empty((nx, ny, nz + 1), dtype=np.uint8, order="F")
    vol[:] = lab[None, None, :]
    x = (np.arange(nx) + 0.5) * V - WIDTH / 2
    if vessel is not None:
        rr = vessel["diameter_mm"] / 2
        mask = x[:, None] ** 2 + (zc[None, :] - vessel["depth_mm"] - rr) ** 2 <= rr ** 2
        mask[:, :2] = False
        for ix in np.flatnonzero(mask.any(1)):
            vol[ix, :, mask[ix]] = 5
    return vol, x


def props(c, vessel=None):
    k, s = EPI_MM / V, 1 - G_TISSUE
    epi_mua = c["mel_fraction"] * MEL_MUA + (1 - c["mel_fraction"]) * EPI_BASE_MUA
    b = vessel or {"mua": 0.46, "mus": 74.2}
    return np.array([[0, 0, 1, 1],
                     [epi_mua * k, EPI_MUSP * k / s, G_TISSUE, N_TISSUE],
                     [c["derm_mua"], c["derm_musp"] / s, G_TISSUE, N_TISSUE],
                     [c["fat_mua"], c["fat_musp"] / s, G_TISSUE, N_TISSUE],
                     [c["musc_mua"], c["musc_musp"] / s, G_TISSUE, N_TISSUE],
                     [b["mua"], b["mus"], G_BLOOD, N_TISSUE]], dtype=np.float32)


def simulate(c, vessel, photons, seed):
    import pmcxcl
    vol, x = volume(c, vessel)
    nx, ny, _ = vol.shape
    acc = np.zeros((nx, ny)); done = 0
    rng = np.random.default_rng(seed)
    while done < photons:
        n = min(BATCH, photons - done)
        res = pmcxcl.run(dict(nphoton=n, vol=vol, prop=props(c, vessel), tstart=0, tend=5e-9, tstep=5e-9,
                              srctype="planar", srcpos=[0, 0, 1], srcdir=[0, 0, 1], srcparam1=[nx, 0, 0, 0],
                              srcparam2=[0, ny, 0, 0], unitinmm=V, issaveref=1, isreflect=1, bc="ccrcca",
                              issrcfrom0=1, isnormalized=1, outputtype="fluence", issavedet=0, autopilot=1,
                              gpuid=1, seed=int(rng.integers(1, 2 ** 31 - 1))))
        top = np.asarray(res["dref"], dtype=np.float64)[:, :, 0, :].sum(-1)
        if not np.isfinite(top).all() or top.min() < 0 or top.max() <= 0:
            raise RuntimeError("invalid reflectance")
        acc += top * n; done += n
    return acc / photons, x


def run(max_cfg=None):
    m = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    t0 = time.time(); count = 0
    for split in ("train", "validation", "locked_test"):
        s = m["splits"][split]
        d = OUT / split; d.mkdir(parents=True, exist_ok=True)
        for c in s["configs"]:
            if max_cfg is not None and count >= max_cfg:
                return
            f = d / f"{c['id']}.npz"
            if f.exists():
                count += 1; continue
            try:
                n0, x = simulate(c["layers"], None, s["n0_photons"], c["mc_seed"])
                profs = []
                for j, v in enumerate(c["vessels"]):
                    R, _ = simulate(c["layers"], v, s["photons"], c["mc_seed"] + 1 + j)
                    profs.append(R.mean(1))
            except RuntimeError as e:          # keep evidence, stop (project rule: do not repair)
                (d / f"{c['id']}.invalid.json").write_text(json.dumps({"error": str(e)}), encoding="utf-8")
                raise
            tmp = f.with_suffix(".partial.npz")
            np.savez_compressed(tmp, x_mm=x, profiles_y_mean=np.array(profs, dtype=np.float32),
                                n0_scalar=float(n0.mean()), n0_cv=float(n0.std() / n0.mean()),
                                n0_y_mean=n0.mean(1).astype(np.float32), config_json=json.dumps(c))
            tmp.replace(f)
            count += 1
            print(f"CFG {split} {c['id']} done; total {count}; elapsed {time.time()-t0:.0f}s", flush=True)
    print("T4_DONE", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "manifest":
        p = OUT / "manifest.json"
        if p.exists():
            raise SystemExit("manifest exists; refusing to overwrite")
        m = manifest()
        p.write_text(json.dumps(m, indent=1), encoding="utf-8")
        n = sum(len(c["vessels"]) for s in m["splits"].values() for c in s["configs"])
        print("configs", {k: len(v["configs"]) for k, v in m["splits"].items()}, "vessels", n)
    else:
        run(int(sys.argv[2]) if len(sys.argv) > 2 else None)
