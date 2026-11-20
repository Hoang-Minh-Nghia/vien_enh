"""step 01 MCX side: same homogeneous pencil-beam run as step 01 part A, with a chosen tissue index n."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp2_paths  # noqa: E402,F401  (all step code folders on sys.path)
import sys

import numpy as np
import pmcxcl

n_t, out = float(sys.argv[1]), sys.argv[2]
SEED, v, n = 20261111, 0.2, (300, 300, 100)
vol = np.ones(n, dtype=np.uint8, order="F")
vol[:, :, 0] = 0
prop = np.array([[0, 0, 1, 1], [0.01, 1.0 / (1 - 0.511), 0.511, n_t]], dtype=np.float32)
total = np.zeros(n[:2])
for b in range(10):
    res = pmcxcl.run(dict(nphoton=6_000_000, vol=vol, prop=prop, tstart=0, tend=5e-9, tstep=5e-9,
                          srctype="pencil", srcpos=[150, 150, 0], srcdir=[0, 0, 1], unitinmm=v, issaveref=1,
                          isreflect=1, bc="ccrcca", issrcfrom0=1, isnormalized=1, outputtype="fluence",
                          autopilot=1, gpuid=1, seed=SEED + b))
    total += np.asarray(res["dref"])[:, :, 0, 0]
    print("BATCH", b, flush=True)
total /= 10
x = (np.arange(n[0]) + 0.5 - 150) * v
rr = np.hypot(*np.meshgrid(x, x, indexing="ij"))
edges = np.arange(0.0, 15.01, 0.4)
idx = np.digitize(rr.ravel(), edges) - 1
mc = np.array([total.ravel()[idx == k].mean() for k in range(len(edges) - 1)])
np.savez_compressed(out, rho_mm=0.5 * (edges[:-1] + edges[1:]), R_mc=mc, Rd_total=float(total.sum() * v * v), n=n_t)
print("DONE Rd_total", float(total.sum() * v * v), flush=True)
