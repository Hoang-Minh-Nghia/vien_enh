"""step 01: independent CPU Monte Carlo (MCML-style, vectorised numpy), semi-infinite medium.

Deliberately different from MCX: implicit-capture absorption (w *= albedo at each interaction),
Henyey-Greenstein scattering, stochastic Fresnel reflection at the top surface, exit radius
scored continuously at the boundary crossing (no voxels), Russian roulette at w < 1e-4.
Pencil beam at normal incidence entering at the surface with weight 1 (no specular loss).
A reflected photon is mirrored about z = 0 for the rest of its step (exact for a flat surface).

Usage: cpu_mc_reference.py <mua> <mus> <g> <n_tissue> <photons> <seed> <out.npz>
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp2_paths  # noqa: E402,F401  (all step code folders on sys.path)
import sys
import time

import numpy as np

EDGES = np.arange(0.0, 15.01, 0.4)


def fresnel(n1, n2, cos_i):
    if n1 == n2:
        return np.zeros_like(cos_i)
    sin_t = n1 / n2 * np.sqrt(np.maximum(0.0, 1 - cos_i ** 2))
    R = np.ones_like(cos_i)
    ok = sin_t < 1
    ct, ci = np.sqrt(1 - sin_t[ok] ** 2), cos_i[ok]
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
    rp = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci)
    R[ok] = 0.5 * (rs ** 2 + rp ** 2)
    return R


def run(mua, mus, g, n_t, photons, seed, chunk=1_000_000):
    mut, alb = mua + mus, mus / (mua + mus)
    rng = np.random.default_rng(seed)
    score = np.zeros(len(EDGES) - 1)
    escaped = 0.0
    done = 0
    while done < photons:
        n = min(chunk, photons - done)
        x, y, z = np.zeros(n), np.zeros(n), np.zeros(n)
        ux, uy, uz = np.zeros(n), np.zeros(n), np.ones(n)
        w = np.ones(n)
        while x.size:
            s = -np.log(rng.random(x.size)) / mut
            x = x + s * ux; y = y + s * uy; zn = z + s * uz
            hit = zn < 0
            keep = np.ones(x.size, bool)
            if hit.any():
                idx = np.flatnonzero(hit)
                t = -z[idx] / uz[idx]                      # distance to the surface
                xb = x[idx] - (s[idx] - t) * ux[idx]
                yb = y[idx] - (s[idx] - t) * uy[idx]
                esc = rng.random(idx.size) >= fresnel(n_t, 1.0, -uz[idx])
                score += np.histogram(np.hypot(xb[esc], yb[esc]), EDGES, weights=w[idx[esc]])[0]
                escaped += w[idx[esc]].sum()
                keep[idx[esc]] = False
                r = idx[~esc]
                zn[r] = -zn[r]; uz[r] = -uz[r]
            z = zn
            x, y, z, ux, uy, uz, w = (a[keep] for a in (x, y, z, ux, uy, uz, w))
            w = w * alb
            # Henyey-Greenstein deflection
            xi = rng.random(x.size)
            if abs(g) < 1e-9:
                ct = 2 * xi - 1
            else:
                tt = (1 - g * g) / (1 - g + 2 * g * xi)
                ct = np.clip((1 + g * g - tt * tt) / (2 * g), -1, 1)
            phi = 2 * np.pi * rng.random(x.size)
            st, cp, sp = np.sqrt(1 - ct ** 2), np.cos(phi), np.sin(phi)
            vert = np.abs(uz) > 0.99999
            den = np.sqrt(np.maximum(1e-30, 1 - uz ** 2))
            nux = np.where(vert, st * cp, st * (ux * uz * cp - uy * sp) / den + ux * ct)
            nuy = np.where(vert, st * sp, st * (uy * uz * cp + ux * sp) / den + uy * ct)
            nuz = np.where(vert, np.sign(uz) * ct, -st * cp * den + uz * ct)
            ux, uy, uz = nux, nuy, nuz
            low = w < 1e-4
            if low.any():
                wl = w[low]
                surv = rng.random(wl.size) < 0.1
                wl[surv] *= 10; wl[~surv] = 0; w[low] = wl
                alive = w > 0
                x, y, z, ux, uy, uz, w = (a[alive] for a in (x, y, z, ux, uy, uz, w))
        done += n
        print(f"PROGRESS {done}/{photons}", flush=True)
    area = np.pi * (EDGES[1:] ** 2 - EDGES[:-1] ** 2)
    return score / (photons * area), escaped / photons


if __name__ == "__main__":
    mua, mus, g, n_t = map(float, sys.argv[1:5])
    photons, seed, out = int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
    t0 = time.time()
    R, Rd = run(mua, mus, g, n_t, photons, seed)
    np.savez_compressed(out, rho_mm=0.5 * (EDGES[:-1] + EDGES[1:]), R=R, Rd_total=Rd,
                        mua=mua, mus=mus, g=g, n=n_t, photons=photons, seed=seed)
    print(f"DONE mua={mua} mus={mus} g={g} n={n_t} Rd_total={Rd:.5f} runtime {time.time()-t0:.0f}s", flush=True)
