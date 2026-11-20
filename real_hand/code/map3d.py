"""Connected 3D vein depth map on real hand images (EXPLORATORY: model prediction, no ground truth).

Steps per image
 1. vessel skeleton: Zhang-Suen thinning of the (hand-restricted) vessel mask -> 1-pixel centre lines,
    8-connected graph; components shorter than MIN_COMP px dropped.
 2. measured depth: hetero ensemble on the perpendicular profile at EVERY skeleton pixel where the
    profile is valid (same rules as measure_depth.profile).
 3. gaps: harmonic interpolation along the vessel graph (Laplace equation, measured pixels fixed);
    components with no measured pixel are left out.
 4. smoothing: graph diffusion (SMOOTH_IT Jacobi steps, ~2 mm along the vessel).
 5. a pixel is "interpolated" when its distance ALONG THE VESSEL to the nearest measured pixel > GAP_MM.
Outputs: real_hand/figs/map3d_<k>.png, map3d_<k>.html (plotly.js, rotatable), real_hand/results/depth_points/map3d_<k>.npz
Images are labelled only by an anonymous index.
Usage: map3d.py [n_images]
"""
import csv
import json
import sys
from pathlib import Path

import _paths as RP  # noqa: E402  (adds exp2_layered/code to sys.path)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import spsolve

N_IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 2
sys.argv = sys.argv[:1]          # measure_depth parses argv at import (argv[1] = mm/px)
import measure_depth as T  # noqa: E402

R = RP.OUT / "depth_points"
FIG = RP.FIG
MIN_COMP, SMOOTH_IT, GAP_MM = 15, 120, 1.0
VMIN, VMAX = 1.0, 4.0
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def thin(mask):
    """Zhang-Suen thinning (vectorised)."""
    im = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            P = im
            p2, p3, p4 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:]
            p5, p6, p7 = P[2:, 2:], P[2:, 1:-1], P[2:, :-2]
            p8, p9 = P[1:-1, :-2], P[:-2, :-2]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(n.astype(int) for n in nb)
            A = sum(((nb[i] == 0) & (nb[(i + 1) % 8] == 1)).astype(int) for i in range(8))
            c = P[1:-1, 1:-1] == 1
            if step == 0:
                m = c & (B >= 2) & (B <= 6) & (A == 1) & (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                m = c & (B >= 2) & (B <= 6) & (A == 1) & (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            if m.any():
                im[1:-1, 1:-1][m] = 0; changed = True
        if not changed:
            return im[1:-1, 1:-1].astype(bool)


def graph(sk):
    ys, xs = np.nonzero(sk)
    idx = -np.ones(sk.shape, int); idx[ys, xs] = np.arange(len(ys))
    rows, cols, w = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < sk.shape[0]) & (x2 >= 0) & (x2 < sk.shape[1])
        j = np.full(len(ys), -1); j[ok] = idx[y2[ok], x2[ok]]
        k = j >= 0
        rows += list(np.nonzero(k)[0]); cols += list(j[k]); w += [np.hypot(dy, dx)] * int(k.sum())
    n = len(ys)
    G = sparse.coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    return ys, xs, G + G.T


def depth_map(name, L):
    im = np.asarray(Image.open(T.DS / "images" / f"{name}.png"), float)
    mask = np.asarray(Image.open(T.DS / "masks" / f"{name}.png")) > 0
    hand = T.hand_roi(im); mk = mask & hand
    dt = ndi.distance_transform_edt(mk)
    sk = thin(mk)
    lab, nl = ndi.label(sk, structure=np.ones((3, 3)))
    sizes = ndi.sum(sk, lab, range(1, nl + 1))
    sk = np.isin(lab, 1 + np.nonzero(sizes >= MIN_COMP)[0])
    ys, xs, G = graph(sk)
    cx, cy = T.directions(mk)
    meas_i, prof = [], []
    for i, (y, x) in enumerate(zip(ys, xs)):
        o = T.profile(im, mk, hand, y, x, cx[y, x], cy[y, x], dt)
        if o is not None:
            meas_i.append(i); prof.append(o[0])
    meas_i = np.array(meas_i, int)
    with torch.no_grad():
        X = torch.from_numpy(np.array(prof)[:, None, :])
        mu = np.mean([m(X)[:, 0].numpy() for m in L], 0)
    n = len(ys)
    raw = np.full(n, np.nan); raw[meas_i] = mu
    # keep components that contain at least one measurement
    ncomp, comp = csgraph.connected_components(G, directed=False)
    good = np.isin(comp, np.unique(comp[meas_i]))
    # harmonic fill: L_uu x_u = -L_um x_m   (L = D - W, W = 1/length)
    W = G.copy(); W.data = 1.0 / W.data
    Lap = sparse.diags(np.asarray(W.sum(1)).ravel()) - W
    known = ~np.isnan(raw); unk = good & ~known
    val = raw.copy()
    if unk.any():
        Luu = Lap[unk][:, unk].tocsc(); Lum = Lap[unk][:, known]
        val[unk] = spsolve(Luu, -Lum @ raw[known])
    val[~good] = np.nan
    # diffusion smoothing along the vessel
    deg = np.asarray(W.sum(1)).ravel()
    sm = val.copy(); g = good
    for _ in range(SMOOTH_IT):
        nb = W @ np.where(g, sm, 0.0)
        sm = np.where(g, (sm + nb) / (1 + deg), np.nan)
    # distance along the vessel to the nearest measured pixel
    dist = csgraph.dijkstra(G, directed=False, indices=meas_i, min_only=True) * T.MM_PER_PX
    interp = dist > GAP_MM
    width = 2 * dt[ys, xs] * T.MM_PER_PX
    return dict(im=im, hand=hand, ys=ys, xs=xs, G=G, raw=raw, depth=sm, good=good, interp=interp, width=width,
                n_meas=len(meas_i), n=n)


def edges(G, keep):
    C = sparse.triu(G).tocoo()
    m = keep[C.row] & keep[C.col]
    return C.row[m], C.col[m]


def png(k, D):
    s = T.MM_PER_PX
    ys, xs, d, g, it = D["ys"], D["xs"], D["depth"], D["good"], D["interp"]
    i, j = edges(D["G"], g)
    X, Y = xs * s, ys * s
    cmap = plt.get_cmap("jet_r"); norm = plt.Normalize(VMIN, VMAX)
    fig = plt.figure(figsize=(16, 6.6))
    a = fig.add_subplot(1, 2, 1, projection="3d")
    for sel, alpha, lwk in ((~(it[i] | it[j]), 1.0, 1.0), (it[i] | it[j], 0.35, 0.5)):
        ii, jj = i[sel], j[sel]
        segs = np.stack([np.c_[X[ii], Y[ii], -d[ii]], np.c_[X[jj], Y[jj], -d[jj]]], 1)
        dm = (d[ii] + d[jj]) / 2; wm = (D["width"][ii] + D["width"][jj]) / 2
        lc = Line3DCollection(segs, colors=cmap(norm(dm)), linewidths=lwk * (0.8 + 1.6 * wm), alpha=alpha)
        a.add_collection(lc)
    hc = np.argwhere(D["hand"] ^ ndi.binary_erosion(D["hand"]))
    a.scatter(hc[::3, 1] * s, hc[::3, 0] * s, 0, s=0.6, color="0.55", depthshade=False)
    hx0, hx1, hy0, hy1 = hc[:, 1].min() * s, hc[:, 1].max() * s, hc[:, 0].min() * s, hc[:, 0].max() * s
    a.set_xlim(hx0, hx1); a.set_ylim(hy1, hy0); a.set_zlim(-5, 0)
    a.set_box_aspect((hx1 - hx0, hy1 - hy0, 30))
    a.set_xlabel("X (mm)"); a.set_ylabel("Y (mm)"); a.set_zlabel("Độ sâu (mm)", labelpad=6)
    a.view_init(elev=28, azim=-62); a.set_title("Bản đồ độ sâu tĩnh mạch 3D (dự đoán)")
    b = fig.add_subplot(1, 2, 2)
    im, hand = D["im"], D["hand"]
    b.imshow(np.where(hand, im, im * 0.3), cmap="gray", vmin=0, vmax=255)
    sel = g & ~it
    b.scatter(xs[g & it], ys[g & it], c="white", s=1.2, alpha=0.6)
    sc = b.scatter(xs[sel], ys[sel], c=d[sel], cmap="jet_r", vmin=VMIN, vmax=VMAX, s=2.5)
    hy, hx = np.nonzero(hand)
    b.set_xlim(hx.min() - 5, hx.max() + 5); b.set_ylim(hy.max() + 5, hy.min() - 5); b.axis("off")
    b.set_title("Nhìn từ trên xuống (trắng = đoạn nội suy, không đo được)")
    cb = fig.colorbar(sc, ax=[a, b], shrink=0.8, pad=0.02); cb.set_label("Độ sâu dưới da (mm)"); cb.ax.invert_yaxis()
    fig.savefig(FIG / f"map3d_{k}.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def html(k, D):
    s = T.MM_PER_PX
    ys, xs, d, g, it = D["ys"], D["xs"], D["depth"], D["good"], D["interp"]
    i, j = edges(D["G"], g)
    def trace(sel, width, opacity, name):
        xx, yy, zz, cc = [], [], [], []
        for a_, b_ in zip(i[sel], j[sel]):
            xx += [round(xs[a_] * s, 2), round(xs[b_] * s, 2), None]
            yy += [round(ys[a_] * s, 2), round(ys[b_] * s, 2), None]
            zz += [round(-d[a_], 3), round(-d[b_], 3), None]
            cc += [round(d[a_], 3), round(d[b_], 3), round(d[b_], 3)]
        return {"type": "scatter3d", "mode": "lines", "name": name, "x": xx, "y": yy, "z": zz, "opacity": opacity,
                "line": {"width": width, "color": cc, "colorscale": "Jet", "reversescale": True, "cmin": VMIN, "cmax": VMAX,
                         "showscale": name.startswith("đo"), "colorbar": {"title": {"text": "Độ sâu (mm)"}, "autocolorscale": False}},
                "hovertemplate": "độ sâu %{line.color:.2f} mm<extra></extra>"}
    meas = ~(it[i] | it[j])
    hc = np.argwhere(D["hand"] ^ ndi.binary_erosion(D["hand"]))[::4]
    order = np.argsort(np.arctan2(hc[:, 0] - hc[:, 0].mean(), hc[:, 1] - hc[:, 1].mean()))
    hc = hc[order]
    traces = [trace(meas, 7, 1.0, "đo được"), trace(~meas, 3, 0.35, "nội suy"),
              {"type": "scatter3d", "mode": "lines", "name": "viền bàn tay", "x": list(np.round(hc[:, 1] * s, 1)) + [None],
               "y": list(np.round(hc[:, 0] * s, 1)) + [None], "z": [0] * len(hc) + [None],
               "line": {"color": "#9AA5AD", "width": 2}, "hoverinfo": "skip"}]
    X, Y = xs * s, ys * s
    layout = {"margin": {"l": 0, "r": 0, "t": 10, "b": 0}, "paper_bgcolor": "rgba(0,0,0,0)",
              "legend": {"x": 0.01, "y": 0.98},
              "scene": {"xaxis": {"title": "X (mm)"}, "yaxis": {"title": "Y (mm)", "autorange": "reversed"},
                        "zaxis": {"title": "Độ sâu (mm)", "range": [-5, 0]}, "aspectmode": "manual",
                        "aspectratio": {"x": 1, "y": float(np.ptp(Y) / np.ptp(X)), "z": 0.35},
                        "camera": {"eye": {"x": -1.2, "y": 1.5, "z": 0.9}}}}
    stats = f"{int(D['good'].sum()):,} điểm trên trục mạch · {D['n_meas']:,} điểm đo được · trung vị {np.nanmedian(D['raw']):.2f} mm".replace(",", ".")
    page = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bản đồ tĩnh mạch 3D</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<style>
:root{{--bg:#FFFFFF;--fg:#1F2A33;--muted:#5B6770}}
@media (prefers-color-scheme: dark){{:root{{--bg:#12191E;--fg:#E6ECEF;--muted:#9AA5AD}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
header{{padding:14px 16px 4px}} h1{{font-size:20px;margin:0 0 4px}} p{{margin:2px 0;color:var(--muted);font-size:14px}}
#plot{{width:100%;height:calc(100vh - 110px);min-height:420px}}
</style></head><body>
<header><h1>Bản đồ độ sâu tĩnh mạch 3D — Ảnh {k + 1}</h1>
<p>{stats}. Kéo chuột để xoay, cuộn để phóng to.</p>
<p>Thăm dò: độ sâu là dự đoán của mô hình huấn luyện trên mô phỏng, chưa kiểm bằng độ sâu thật. Nét mờ = đoạn nội suy (không đo được).</p></header>
<div id="plot"></div>
<script>
const dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
const layout = {json.dumps(layout, ensure_ascii=False)};
layout.font = {{color: dark ? '#E6ECEF' : '#1F2A33'}};
Plotly.newPlot('plot', {json.dumps(traces, ensure_ascii=False)}, layout, {{responsive: true, displaylogo: false}});
</script></body></html>"""
    (FIG / f"map3d_{k}.html").write_text(page, encoding="utf-8")


def main():
    with open(R / "points_cal.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    cnt = {}
    for q in rows:
        cnt[q["image"]] = cnt.get(q["image"], 0) + 1
    names = sorted(cnt)
    picks = [max(names[::25], key=lambda n: cnt[n])] + names[::max(1, len(names) // 4)][1:]
    L = T.models()
    for k, name in enumerate(picks[:N_IMG]):
        D = depth_map(name, L)
        np.savez(R / f"map3d_{k}.npz", ys=D["ys"], xs=D["xs"], raw=D["raw"], depth=D["depth"], good=D["good"],
                 interp=D["interp"], width=D["width"])
        png(k, D); html(k, D)
        g = D["good"]
        print(k, "skeleton", D["n"], "kept", int(g.sum()), "measured", D["n_meas"], "interp frac", float(D["interp"][g].mean()),
              "raw median", float(np.nanmedian(D["raw"])), "smoothed median", float(np.nanmedian(D["depth"][g])), flush=True)


if __name__ == "__main__":
    main()
