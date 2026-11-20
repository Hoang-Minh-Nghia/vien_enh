"""real-hand (EXPLORATORY ONLY): apply the selected multi-layer-skin model (hetero, 3 seeds) to real 850 nm
reflectance photos of the back of the hand (My_DATASET). There is NO ground-truth depth: outputs are
model predictions, not measurements. Nothing is trained or tuned on these images. The dataset folder
is only read.

Pipeline per image
 1. vessel mask (model-predicted, from My_DATASET/masks) -> centre-line points = ridge of the distance
    transform, thinned to one point per CELL px.
 2. local vessel direction from the structure tensor of the smoothed mask.
 3. profile perpendicular to the vessel over -16..+16 mm, averaged over parallel lines +-AVG_MM along
    the vessel; dark offset subtracted.
 4. background: quadratic fit to the samples outside every vessel (masks dilated) and outside +-EXCL_MM of
    the centre; ratio = profile / background -> same normalisation as the simulation (background = 1).
    Other vessels inside the window are replaced by the background (= 1).
 5. bin to the 128 x 0.25 mm grid; hetero ensemble -> depth mu and sigma.
Assumed scale MM_PER_PX (no calibration available); sensitivity runs change it and the dark offset.

Usage: measure_depth.py [mm_per_px] [dark_offset] [tag] [excl_mm] [hand_mode otsu|thr]
Hand ROI : Otsu + largest component; the old fixed threshold (tag main, s017...) leaked into the wall.
"""
import csv
import json
import sys
from pathlib import Path

import _paths as RP  # noqa: E402  (adds exp2_layered/code to sys.path)

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

import cnn_variants as V

DS = RP.DATA
R = RP.EXP2                # trained models
OUT = RP.OUT / "depth_points"
EDGES = V.EDGES
MM_PER_PX = float(sys.argv[1]) if len(sys.argv) > 1 else 156.6 / 752   # measured field of view 156,6 x 116,6 mm (user, 16/11/2026)
DARK = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
TAG = sys.argv[3] if len(sys.argv) > 3 else "main"
CELL, AVG_MM, HAND_GREY = 14, 1.5, 70
EXCL_MM = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
HAND_MODE = sys.argv[5] if len(sys.argv) > 5 else "otsu"   # "thr" = old fixed threshold (leaked into the wall)
EDGE_MM = 1.0                                               # hand ROI eroded by this margin


def otsu(v):
    h, e = np.histogram(v, 256, (0, 256)); p = h / h.sum(); c = (e[:-1] + e[1:]) / 2
    w = np.cumsum(p); m = np.cumsum(p * c)
    return c[np.argmax((m[-1] * w - m) ** 2 / (w * (1 - w) + 1e-12))]


def hand_roi(img):
    """Hand only: Otsu on the smoothed image, largest component, holes filled, eroded by EDGE_MM."""
    if HAND_MODE == "thr":
        return ndi.binary_opening(ndi.gaussian_filter(img, 4) > HAND_GREY, iterations=5)
    g = ndi.gaussian_filter(img, 4)
    h = ndi.binary_opening(g > otsu(g), iterations=5)
    lab, n = ndi.label(h)
    if n == 0:
        return h
    h = lab == 1 + int(np.argmax(ndi.sum(h, lab, range(1, n + 1))))
    h = ndi.binary_fill_holes(h)
    return ndi.binary_erosion(h, iterations=max(1, int(round(EDGE_MM / MM_PER_PX))))


def models():
    L = []
    for s in (42, 43, 44):
        m = V.build("hetero"); m.load_state_dict(torch.load(R / f"04_training/results/variants/hetero_seed{s}.pt", weights_only=False)["model"])
        L.append(m.eval())
    return L


def centre_points(mask):
    dt = ndi.distance_transform_edt(mask)
    ridge = (dt >= ndi.maximum_filter(dt, 3)) & (dt >= 2)
    ys, xs = np.nonzero(ridge)
    keep, seen = [], set()
    for y, x in sorted(zip(ys, xs), key=lambda p: -dt[p]):
        c = (y // CELL, x // CELL)
        if c not in seen:
            seen.add(c); keep.append((y, x))
    return keep, dt


def directions(mask):
    m = ndi.gaussian_filter(mask.astype(float), 3)
    gy, gx = ndi.sobel(m, 0), ndi.sobel(m, 1)
    Jxx, Jyy, Jxy = (ndi.gaussian_filter(a, 6) for a in (gx * gx, gy * gy, gx * gy))
    ang = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)        # dominant gradient direction = across the vessel
    return np.cos(ang), np.sin(ang)


def profile(img, mask, hand, y, x, nx, ny, dt):
    t_mm = np.arange(-16, 16.001, 0.125)              # fine grid, binned to 0.25 mm afterwards
    s_mm = np.linspace(-AVG_MM, AVG_MM, 7)
    tx, ty = -ny, nx                                  # along-vessel unit vector
    T, S = np.meshgrid(t_mm / MM_PER_PX, s_mm / MM_PER_PX)
    X = x + T * nx + S * tx; Y = y + T * ny + S * ty
    if X.min() < 0 or Y.min() < 0 or X.max() > img.shape[1] - 1 or Y.max() > img.shape[0] - 1:
        return None
    hv = ndi.map_coordinates(hand.astype(float), [Y, X], order=0)
    if hv.mean() < 0.999:
        return None
    p = ndi.map_coordinates(img, [Y, X], order=1).mean(0) - DARK
    mv = ndi.map_coordinates(mask.astype(float), [Y[3], X[3]], order=0) > 0.5
    # central vessel run: contiguous mask samples through t = 0
    c = len(t_mm) // 2
    if not mv[c]:
        return None
    lo = c; hi = c
    while lo > 0 and mv[lo - 1]: lo -= 1
    while hi < len(mv) - 1 and mv[hi + 1]: hi += 1
    width_mm = (hi - lo + 1) * 0.125
    others = mv.copy(); others[lo:hi + 1] = False
    oth_d = ndi.binary_dilation(others, iterations=int(round(1.0 / 0.125)))   # +1 mm around other vessels
    fit = ~oth_d & (np.abs(t_mm) > EXCL_MM)
    if fit.sum() < 40 or (fit & (t_mm < 0)).sum() < 15 or (fit & (t_mm > 0)).sum() < 15:
        return None
    w = np.polyfit(t_mm[fit], p[fit], 2)
    base = np.polyval(w, t_mm)
    if base.min() <= 5:
        return None
    r = p / base
    r[oth_d & (np.abs(t_mm) > width_mm)] = 1.0
    # bin to the model grid (0.25 mm, -16..16)
    B = np.array([r[(t_mm >= a) & (t_mm < b)].mean() for a, b in zip(EDGES[:-1], EDGES[1:])])
    return B.astype(np.float32), width_mm, float(1 - r[lo:hi + 1].min())


def main():
    torch.set_num_threads(2)
    L = models()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, profs = [], []
    files = sorted((DS / "images").glob("*.png"))
    for f in files:
        mf = DS / "masks" / f.name
        if not mf.exists():
            continue
        img = np.asarray(Image.open(f), dtype=float)
        mask = np.asarray(Image.open(mf)) > 0
        hand = hand_roi(img)
        mask = mask & hand
        pts, dt = centre_points(mask)
        cx, cy = directions(mask)
        for (y, x) in pts:
            out = profile(img, mask, hand, y, x, cx[y, x], cy[y, x], dt)
            if out is None:
                continue
            B, wmm, con = out
            stem = f.stem; person = stem.rsplit("_", 2)[0]; side = stem.rsplit("_", 2)[1]
            rows.append({"image": stem, "person": person, "side": side, "x": int(x), "y": int(y),
                         "mask_width_mm": wmm, "contrast": con}); profs.append(B)
    X = torch.from_numpy(np.array(profs)[:, None, :])
    with torch.no_grad():
        outs = [m(X) for m in L]
    mu = np.array([o[:, 0].numpy() for o in outs]); sg = np.exp(np.array([o[:, 1].clamp(-6, 3).numpy() for o in outs]))
    d = mu.mean(0); sd = np.sqrt((sg ** 2 + mu ** 2).mean(0) - d ** 2)
    for r, a, b in zip(rows, d, sd):
        r["depth_mm"] = float(a); r["sigma_mm"] = float(b)
    with open(OUT / f"points_{TAG}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    np.save(OUT / f"profiles_{TAG}.npy", np.array(profs))
    meta = {"mm_per_px": MM_PER_PX, "dark_offset": DARK, "hand_mode": HAND_MODE, "excl_mm": EXCL_MM, "n_images": len(files), "n_points": len(rows),
            "n_images_with_points": len({r['image'] for r in rows}), "note": "exploratory; no ground truth"}
    (OUT / f"meta_{TAG}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta), "depth median", np.median(d), "IQR", np.percentile(d, [25, 75]), "sigma median", np.median(sd))


if __name__ == "__main__":
    main()
