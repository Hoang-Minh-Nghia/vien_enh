"""step 02 GPU job: layered skin model at 850 nm with a scattering blood vessel.

Separate runner (simulate.py stays locked). Same MCX settings as simulate.run: planar source over
the whole top, 60 x 8 x 20 mm, 0.1 mm voxels, bc 'ccrcca', dref at the air layer, batches of 6M
photons with batch seeds drawn from default_rng(seed). Saved files have the same keys, so
readout.profile can be used; vessel parameters are kept out of config so channel and N0 match.

Layers (labels): 0 air, 1 epidermis, 2 dermis, 3 subcutaneous fat, 4 muscle, 5 venous blood.
Sources are listed in 00_parameters_850nm.md (section "Bổ sung cho step 02"). The 60 um epidermis is one 0.1 mm voxel with mua and
mus scaled by 0.6 so the optical thickness is kept (approximation, noted in the report).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp2_paths  # noqa: E402,F401  (all step code folders on sys.path)
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

OUT = exp2_paths.EXP2 / "02_forward_model/results/layered"
SEED, PH, N0PH, BATCH = 20261112, 60_000_000, 120_000_000, 6_000_000
G_TISSUE = 0.9

BASE = {
    "geometry": {"width_mm": 60.0, "length_mm": 8.0, "thickness_mm": 20.0, "voxel_mm": 0.1,
                 "epidermis_mm": 0.06, "dermis_mm": 1.1, "fat_mm": 6.0},
    "optics": {"epi_mua": 0.03 * 11.6 + 0.97 * 0.025, "epi_musp": 3.71,
               "derm_mua": 0.02, "derm_musp": 2.28, "fat_mua": 0.010, "fat_musp": 1.29,
               "musc_mua": 0.026, "musc_musp": 0.80, "n": 1.37},
    "mc": {"boundary": "ccrcca", "tend_s": 5e-9, "gpuid": 1},
}
VESSEL = {"diameter_mm": 3.0, "mua": 0.46, "mus": 74.2, "g": 0.9832}


def volume(cfg, depth=None, diameter=None):
    g = cfg["geometry"]; v = g["voxel_mm"]
    nx, ny, nz = (int(round(g[k] / v)) for k in ("width_mm", "length_mm", "thickness_mm"))
    vol = np.full((nx, ny, nz + 1), 4, dtype=np.uint8, order="F")
    zc = (np.arange(nz + 1) - 0.5) * v          # voxel-centre depth below the surface
    vol[:, :, 0] = 0
    d1 = g["epidermis_mm"]; d2 = d1 + g["dermis_mm"]; d3 = d2 + g["fat_mm"]
    vol[:, :, 1] = 1                              # epidermis = first tissue voxel
    lab = np.where(zc < d2, 2, np.where(zc < d3, 3, 4)).astype(np.uint8)
    lab[0], lab[1] = 0, 1
    vol[:] = lab[None, None, :]
    x = (np.arange(nx) + 0.5) * v - g["width_mm"] / 2
    if depth is not None:
        r = diameter / 2
        mask = x[:, None] ** 2 + (zc[None, :] - depth - r) ** 2 <= r ** 2
        mask[:, :2] = False
        for ix in range(nx):
            vol[ix, :, mask[ix]] = 5
    return vol, x, zc


def props(cfg, vmua):
    o = cfg["optics"]; s = 1 - G_TISSUE
    k = cfg["geometry"]["epidermis_mm"] / cfg["geometry"]["voxel_mm"]
    return np.array([[0, 0, 1, 1],
                     [o["epi_mua"] * k, o["epi_musp"] * k / s, G_TISSUE, o["n"]],
                     [o["derm_mua"], o["derm_musp"] / s, G_TISSUE, o["n"]],
                     [o["fat_mua"], o["fat_musp"] / s, G_TISSUE, o["n"]],
                     [o["musc_mua"], o["musc_musp"] / s, G_TISSUE, o["n"]],
                     [vmua, VESSEL["mus"], VESSEL["g"], o["n"]]], dtype=np.float32)


def run(cfg, out, depth=None, vmua=VESSEL["mua"], diameter=VESSEL["diameter_mm"], photons=PH, seed=SEED):
    import pmcxcl
    out = Path(out)
    if out.exists():
        print("SKIP", out.name, flush=True); return
    vol, x, z = volume(cfg, depth, diameter)
    nx, ny, _ = vol.shape
    acc = np.zeros((nx, ny)); done = 0
    rng = np.random.default_rng(seed); t0 = time.time()
    while done < photons:
        n = min(BATCH, photons - done)
        res = pmcxcl.run(dict(nphoton=n, vol=vol, prop=props(cfg, vmua), tstart=0, tend=cfg["mc"]["tend_s"],
                              tstep=cfg["mc"]["tend_s"], srctype="planar", srcpos=[0, 0, 1], srcdir=[0, 0, 1],
                              srcparam1=[nx, 0, 0, 0], srcparam2=[0, ny, 0, 0], unitinmm=cfg["geometry"]["voxel_mm"],
                              issaveref=1, isreflect=1, bc=cfg["mc"]["boundary"], issrcfrom0=1, isnormalized=1,
                              outputtype="fluence", issavedet=0, autopilot=1, gpuid=cfg["mc"]["gpuid"],
                              seed=int(rng.integers(1, 2 ** 31 - 1))))
        top = np.asarray(res["dref"], dtype=np.float64)[:, :, 0, :].sum(-1)
        if not np.isfinite(top).all() or top.min() < 0 or top.max() <= 0:
            np.savez_compressed(out.with_suffix(".invalid.npz"), top=top)
            raise RuntimeError(f"invalid reflectance in {out.name}")
        acc += top * n; done += n
    R = acc / photons
    meta = {"config": cfg, "vessel": {"depth_mm": depth, "mua": vmua, "diameter_mm": diameter, **{k: VESSEL[k] for k in ("mus", "g")}},
            "photons": photons, "seed": seed, "runtime_s": time.time() - t0}
    np.savez_compressed(out, R_xy=R.astype(np.float32), R=R.mean(1).astype(np.float32), x_mm=x, z_mm=z,
                        labels_xz=vol[:, ny // 2, :], metadata_json=json.dumps(meta))
    print(f"SAVED {out.name} {meta['runtime_s']:.0f}s", flush=True)


# Background (layer) nuisances: relative +-20 % except thicknesses (absolute).
BG_STEPS = {"epi_mua": ("optics", 0.2), "derm_musp": ("optics", 0.2), "fat_musp": ("optics", 0.2),
            "fat_mua": ("optics", 0.2), "dermis_mm": ("geometry", 0.2)}
POINTS = (1.6, 2.6, 3.6)        # vessel depth (mm, surface to top edge); all below the 1.16 mm skin
DEPTH_STEP, MUA_STEP, D_STEP = 0.2, 0.1, 0.4


def bg_variant(name, sign):
    cfg = copy.deepcopy(BASE); sec, st = BG_STEPS[name]
    if name == "dermis_mm":
        cfg[sec][name] += sign * st
    else:
        cfg[sec][name] *= 1 + sign * st
    return cfg


def dark():
    cfg = copy.deepcopy(BASE); cfg["optics"]["epi_mua"] = 0.15 * 11.6 + 0.85 * 0.025
    return cfg


def jobs():
    J = [(BASE, "n0_base", None, {})]
    for k in BG_STEPS:
        for s, sg in (("+", 1), ("-", -1)):
            J.append((bg_variant(k, sg), f"n0_{k}{s}", None, {}))
    J.append((dark(), "n0_dark", None, {}))
    for z in POINTS:
        t = f"z{z:.1f}"
        J.append((BASE, f"{t}_base", z, {}))
        J.append((dark(), f"{t}_dark", z, {}))
        for s, sg in (("+", 1), ("-", -1)):
            J.append((BASE, f"{t}_depth{s}", z + sg * DEPTH_STEP, {}))
            J.append((BASE, f"{t}_vmua{s}", z, {"vmua": VESSEL["mua"] * (1 + sg * MUA_STEP)}))
            J.append((BASE, f"{t}_diam{s}", z, {"diameter": VESSEL["diameter_mm"] + sg * D_STEP}))
            for k in BG_STEPS:
                J.append((bg_variant(k, sg), f"{t}_{k}{s}", z, {}))
    return J


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "design.json").write_text(json.dumps({"base": BASE, "vessel": VESSEL, "bg_steps": BG_STEPS, "points": POINTS,
                                                  "depth_step": DEPTH_STEP, "mua_step": MUA_STEP, "d_step": D_STEP,
                                                  "photons": PH, "n0_photons": N0PH, "seed": SEED}, indent=2), encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        vol, x, z = volume(BASE, 1.6, 3.0)
        col = vol[300, 4, :]
        print("labels along centre column:", [(int(l), round(float(z[i]), 2)) for i, l in enumerate(col) if i == 0 or col[i] != col[i - 1]])
        print("props\n", props(BASE, 0.46)); print("jobs", len(jobs())); sys.exit()
    t0 = time.time(); J = jobs()
    for i, (cfg, name, z, kw) in enumerate(J, 1):
        run(cfg, OUT / f"{name}.npz", z, photons=N0PH if z is None else PH, **kw)
        print(f"JOB {i}/{len(J)} {name} elapsed {time.time()-t0:.0f}s", flush=True)
    print("T3_DONE", flush=True)
