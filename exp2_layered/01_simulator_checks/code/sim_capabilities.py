"""step 01 pilot: probe which pmcxcl 0.7.5 features work, on a tiny volume.

Each probe runs a few thousand photons and records success or the error text.
Nothing here is used as research data.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp2_paths  # noqa: E402,F401  (all step code folders on sys.path)
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pmcxcl

OUT = exp2_paths.EXP2 / "01_simulator_checks/results/capabilities.json"


def base(nx=40, ny=40, nz=30):
    vol = np.ones((nx, ny, nz), dtype=np.uint8, order="F")
    vol[:, :, 0] = 0
    vol[15:25, :, 8:14] = 2
    prop = np.array([[0, 0, 1, 1], [0.01, 2.045, 0.511, 1.35], [0.3, 0, 0, 1.35]], dtype=np.float32)
    return dict(nphoton=20000, vol=vol, prop=prop, tstart=0, tend=5e-9, tstep=5e-9,
                srctype="planar", srcpos=[0, 0, 1], srcdir=[0, 0, 1], srcparam1=[nx, 0, 0, 0],
                srcparam2=[0, ny, 0, 0], unitinmm=0.1, issaveref=1, isreflect=1, bc="ccrcca",
                issrcfrom0=1, isnormalized=1, outputtype="fluence", autopilot=1, gpuid=1, seed=1)


def probe(name, cfg, check):
    try:
        res = pmcxcl.run(cfg)
        info = check(res)
        return {"probe": name, "ok": True, "keys": sorted(res.keys()), **info}
    except Exception as exc:  # record the failure text; the probe is the experiment
        return {"probe": name, "ok": False, "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()[-600:]}


def main():
    results = []
    cfg = base()
    results.append(probe("baseline_dref", cfg, lambda r: {"dref_shape": list(np.shape(r["dref"]))}))

    # Detected photons over the whole top surface with exit position, direction and partial path lengths.
    cfg = base()
    cfg.update(issavedet=1, detpos=[[20, 20, 0, 30]], savedetflag="dpxv", maxdetphoton=200000)
    results.append(probe("savedet_ppath_exitpos_dir", cfg, lambda r: {
        "detp_keys": sorted(r["detp"].keys()) if isinstance(r.get("detp"), dict) else str(type(r.get("detp"))),
        "detp_shapes": {k: list(np.shape(v)) for k, v in r["detp"].items()} if isinstance(r.get("detp"), dict) else None}))

    # Patterned source (structured / segmented illumination).
    cfg = base()
    pattern = np.zeros((40, 40), dtype=np.float32)
    pattern[:, :20] = 1
    cfg.update(srctype="pattern", srcparam1=[40, 0, 0, 40], srcparam2=[0, 40, 0, 40], srcpattern=pattern)
    results.append(probe("pattern_source", cfg, lambda r: {"dref_top_left_mean": float(np.asarray(r["dref"])[:20, :, 0].mean()),
                                                           "dref_top_right_mean": float(np.asarray(r["dref"])[20:, :, 0].mean())}))

    # Oblique pencil beam (off-axis illumination).
    cfg = base()
    cfg.update(srctype="pencil", srcpos=[5, 20, 0], srcdir=[0.5, 0, 0.8660254])
    results.append(probe("oblique_pencil", cfg, lambda r: {"dref_shape": list(np.shape(r["dref"]))}))

    # Line source.
    cfg = base()
    cfg.update(srctype="line", srcpos=[5, 0, 0], srcparam1=[0, 40, 0, 0])
    results.append(probe("line_source", cfg, lambda r: {"dref_shape": list(np.shape(r["dref"]))}))

    # Polarization (MCX 2023+ in CUDA uses polprop + Stokes source); try the same keys here.
    cfg = base()
    cfg.update(polprop=np.array([[0.5e-3, 1.0, 1.33, 1.35]], dtype=np.float32), srciquv=[1, 1, 0, 0],
               issavedet=1, detpos=[[20, 20, 0, 30]], savedetflag="dpxvi")
    results.append(probe("polarization_polprop", cfg, lambda r: {
        "detp_keys": sorted(r["detp"].keys()) if isinstance(r.get("detp"), dict) else None}))

    # Disable volumetric fluence output to save memory on fine grids.
    cfg = base()
    cfg.update(issave2pt=0)
    results.append(probe("no_volume_output", cfg, lambda r: {"has_flux": "flux" in r, "dref_shape": list(np.shape(r.get("dref", [])))}))

    # Henyey-Greenstein high-g scattering inside the channel (blood-like).
    cfg = base()
    cfg["prop"][2] = [0.3, 50.0, 0.98, 1.4]
    results.append(probe("scattering_channel_g098", cfg, lambda r: {"dref_shape": list(np.shape(r["dref"]))}))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"pmcxcl": pmcxcl.__version__, "probes": results}, indent=2), encoding="utf-8")
    for r in results:
        print(r["probe"], "OK" if r["ok"] else "FAIL", {k: v for k, v in r.items() if k not in ("probe", "ok", "trace", "keys")})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
