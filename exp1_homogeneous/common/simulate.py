"""Run MCX-CL surface reflectance with explicit geometry and resumable output.

This module does not import or relabel earlier MC results. Its output is
hemispherical diffuse reflectance, not a camera with a finite collection NA.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    return json.loads(Path(path or ROOT / "configs/simulation.json").read_text(encoding="utf-8"))


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def make_volume(config, depth=None):
    g = config["geometry"]
    v = g["voxel_mm"]
    dims = [int(round(g[k] / v)) for k in ("width_mm", "length_mm", "thickness_mm")]
    if any(abs(n * v - g[k]) > 1e-8 for n, k in zip(dims, ("width_mm", "length_mm", "thickness_mm"))):
        raise ValueError("Domain dimensions must be multiples of voxel size")
    nx, ny, tissue_nz = dims
    # Index 0 in z is air; tissue surface is at z=1 in MCX voxel coordinates.
    vol = np.ones((nx, ny, tissue_nz + 1), dtype=np.uint8, order="F")
    vol[:, :, 0] = 0
    x = (np.arange(nx) + 0.5) * v - g["width_mm"] / 2
    z = (np.arange(tissue_nz + 1) - 0.5) * v
    if depth is not None:
        radius = g["diameter_mm"] / 2
        if depth <= 0 or depth + 2 * radius >= g["thickness_mm"]:
            raise ValueError("Channel must lie completely inside tissue")
        mask = x[:, None] ** 2 + (z[None, :] - depth - radius) ** 2 <= radius ** 2
        mask[:, 0] = False
        # Broadcast without allocating a second full-volume Boolean mask.
        for ix in range(nx):
            vol[ix, :, mask[ix]] = 2
    return vol, x, z


def properties(config, mua):
    o = config["optics"]
    return np.array([
        [0, 0, 1, 1],
        [o["background_mua_mm_inv"], o["background_musp_mm_inv"] / (1 - o["background_g"]),
         o["background_g"], o["background_n"]],
        [mua, o["channel_mus_mm_inv"], 0, o["channel_n"]],
    ], dtype=np.float32)


def geometry_summary(config, vol, z):
    v = config["geometry"]["voxel_mm"]
    mask = vol[:, vol.shape[1] // 2, :] == 2
    ii, kk = np.where(mask)
    result = {"shape_xyz": list(vol.shape), "top_air_layer": bool(np.all(vol[:, :, 0] == 0)),
              "voxel_mm": v, "channel_voxels_cross_section": int(mask.sum())}
    if len(ii):
        result.update(voxelized_top_mm=float(z[kk.min()] - v / 2),
                      voxelized_bottom_mm=float(z[kk.max()] + v / 2),
                      voxelized_width_mm=float((ii.max() - ii.min() + 1) * v))
    return result


def run(config, output, depth=None, mua=0.508, photons=None, seed=20261103):
    import pmcxcl

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    photon_count = int(photons or config["mc"]["background_photons" if depth is None else "channel_photons"])
    identity = {"config": config, "depth_mm": depth, "mua_mm_inv": mua,
                "photons": photon_count, "seed": seed, "pmcxcl_version": pmcxcl.__version__}
    identity_hash = config_hash(identity)
    if output.exists():
        with np.load(output, allow_pickle=False) as old:
            if str(old["identity_hash"]) != identity_hash:
                raise ValueError(f"Refusing to overwrite a different run: {output}")
        print(f"SKIP {output.name}", flush=True)
        return output
    vol, x, z = make_volume(config, depth)
    mc = config["mc"]
    nx, ny, _ = vol.shape
    metadata = {**identity, "identity_hash": identity_hash,
                "geometry_check": geometry_summary(config, vol, z),
                "python": platform.python_version(), "numpy": importlib.metadata.version("numpy"),
                "readout": "MCX dref at top air layer z=0, summed over time; all exiting directions",
                "started_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "batches": []}
    accumulator = np.zeros((nx, ny), dtype=np.float64)
    rng = np.random.default_rng(seed)
    completed = 0
    start = time.perf_counter()
    while completed < photon_count:
        count = min(mc["batch_photons"], photon_count - completed)
        batch_seed = int(rng.integers(1, 2**31 - 1))
        settings = dict(nphoton=int(count), vol=vol, prop=properties(config, mua),
                        tstart=0, tend=mc["tend_s"], tstep=mc["tend_s"],
                        srctype="planar", srcpos=[0, 0, 1], srcdir=[0, 0, 1],
                        srcparam1=[nx, 0, 0, 0], srcparam2=[0, ny, 0, 0],
                        unitinmm=config["geometry"]["voxel_mm"], issaveref=1,
                        isreflect=1, bc=mc["boundary"], issrcfrom0=1,
                        isnormalized=1, outputtype="fluence", issavedet=0,
                        autopilot=1, gpuid=mc["gpuid"], seed=batch_seed)
        result = pmcxcl.run(settings)
        dref = np.asarray(result["dref"], dtype=np.float64)
        if dref.ndim != 4 or dref.shape[:3] != vol.shape:
            raise RuntimeError(f"Unexpected dref shape: {dref.shape}")
        top = dref[:, :, 0, :].sum(axis=-1)
        stat = result["stat"]
        if not np.isfinite(top).all() or top.min() < 0 or top.max() <= 0:
            finite_top = top[np.isfinite(top)]
            details = {"finite": int(finite_top.size), "size": int(top.size),
                       "min": float(finite_top.min()) if finite_top.size else None,
                       "max": float(finite_top.max()) if finite_top.size else None,
                       "negative": int((top < 0).sum()), "stat": stat,
                       "seed": batch_seed, "identity": identity}
            np.savez_compressed(output.with_suffix(".invalid.npz"), top=top,
                                diagnostics_json=json.dumps(details))
            raise RuntimeError(f"Invalid top reflectance: finite={details['finite']}/{details['size']}, "
                               f"range=[{details['min']}, {details['max']}], negative={details['negative']}")
        actual = int(round(float(stat["nphoton"])))
        if actual != count:
            raise RuntimeError(f"Requested {count} photons but engine reported {actual}")
        energy = float(stat["energytot"])
        if not np.isfinite(energy) or abs(energy / count - 1) > 1e-5:
            raise RuntimeError(f"Unexpected initial energy {energy} for {count} unit-weight photons")
        accumulator += top * count
        completed += count
        batch_stat = {k: (v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, np.generic) else v)
                      for k, v in stat.items()}
        metadata["batches"].append({"seed": batch_seed, "requested_photons": count, "stat": batch_stat})
        print(f"PROGRESS {output.name}: {completed}/{photon_count} photons, {time.perf_counter()-start:.1f}s", flush=True)
    reflectance = accumulator / photon_count
    metadata["runtime_s"] = time.perf_counter() - start
    # Preserve x-y maps and cross section. No smoothing is applied to saved MC data.
    temp = output.with_suffix(".partial.npz")
    np.savez_compressed(temp, R_xy=reflectance.astype(np.float32),
                        R=reflectance.mean(axis=1).astype(np.float32), x_mm=x, z_mm=z,
                        labels_xz=vol[:, ny // 2, :], identity_hash=identity_hash,
                        metadata_json=json.dumps(metadata, ensure_ascii=False))
    temp.replace(output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SAVED {output} ({metadata['runtime_s']:.1f}s)", flush=True)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--output", required=True)
    p.add_argument("--depth", type=float)
    p.add_argument("--mua", type=float, default=0.508)
    p.add_argument("--photons", type=int)
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--voxel", type=float)
    p.add_argument("--width", type=float)
    p.add_argument("--tend", type=float)
    a = p.parse_args()
    config = copy.deepcopy(load_config(a.config))
    for name, value in (("voxel_mm", a.voxel), ("width_mm", a.width)):
        if value is not None:
            config["geometry"][name] = value
    if a.tend is not None:
        config["mc"]["tend_s"] = a.tend
    run(config, a.output, a.depth, a.mua, a.photons, a.seed)


if __name__ == "__main__":
    main()
