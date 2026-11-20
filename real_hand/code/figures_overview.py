"""real-hand exploratory figures/summary from points_<tag>.csv (no ground truth)."""
import csv
import json
import sys
from pathlib import Path

import _paths as RP  # noqa: E402  (adds exp2_layered/code to sys.path)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import cnn_variants as V

R = RP.OUT / "depth_points"
DS = RP.DATA
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def load(tag):
    with open(R / f"points_{tag}.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("depth_mm", "sigma_mm", "contrast", "mask_width_mm"):
            r[k] = float(r[k])
        r["x"], r["y"] = int(r["x"]), int(r["y"])
    return rows


def main():
    rows = load("main")
    d = np.array([r["depth_mm"] for r in rows]); s = np.array([r["sigma_mm"] for r in rows])
    con = np.array([r["contrast"] for r in rows])
    P = np.load(R / "profiles_main.npy")
    # simulated reference (validation_ext, no shift)
    x, Ps, meta = V.load("validation_ext", "manifest_ext.json"); Bs = V.Binner(x)(Ps, np.zeros(len(Ps)))
    ys = np.array([m["depth_mm"] for m in meta])
    out = {}
    # per image, per person
    img = {}
    for r in rows:
        img.setdefault(r["image"], []).append(r)
    per_img = {k: float(np.median([q["depth_mm"] for q in v])) for k, v in img.items()}
    per_hand = {}
    for k, v in per_img.items():
        per_hand.setdefault(k.rsplit("_", 1)[0], []).append(v)
    rep_sd = np.array([np.std(v, ddof=1) for v in per_hand.values() if len(v) >= 3])
    hand_med = {k: float(np.median(v)) for k, v in per_hand.items()}
    persons = {}
    for k, v in hand_med.items():
        p, side = k.rsplit("_", 1); persons.setdefault(p, {})[side] = v
    lr = np.array([[v["L"], v["R"]] for v in persons.values() if "L" in v and "R" in v])
    between_sd = float(np.std(list(hand_med.values()), ddof=1))
    out.update({"n_points": len(rows), "n_images": len(img), "n_hands": len(per_hand), "n_persons": len(persons),
                "depth_median": float(np.median(d)), "depth_p5_p95": np.percentile(d, [5, 95]).tolist(),
                "sigma_median": float(np.median(s)), "frac_sigma_gt_0p4": float((s > 0.4).mean()),
                "frac_pred_outside_1_5": float(((d < 1) | (d > 5)).mean()),
                "contrast_real_median": float(np.median(con)), "contrast_sim_median": float(np.median(1 - Bs.min(1))),
                "repeat_sd_per_hand_median_mm": float(np.median(rep_sd)), "between_hand_sd_mm": between_sd,
                "LR_corr": float(np.corrcoef(lr[:, 0], lr[:, 1])[0, 1]), "n_LR_pairs": int(len(lr))})
    # sensitivity
    sens = {}
    for tag in ("s017", "s023", "dark10", "excl9"):
        f = R / f"points_{tag}.csv"
        if f.exists():
            q = load(tag); sens[tag] = {"depth_median": float(np.median([r["depth_mm"] for r in q])), "n": len(q)}
    out["sensitivity"] = sens
    (R / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))

    # fig 1: distributions + mean profiles real vs sim
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].hist(d, bins=60, range=(0.5, 5.5), color="#0F6E78"); ax[0].set_xlabel("Độ sâu dự đoán (mm)"); ax[0].set_ylabel("Số điểm")
    ax[0].set_title(f"Ảnh thật: {len(rows)} điểm trên {len(img)} ảnh\n(không có độ sâu thật để so)")
    ax[1].hist(con * 100, bins=60, range=(0, 25), color="#0F6E78", alpha=0.7, density=True, label="ảnh thật")
    ax[1].hist((1 - Bs.min(1)) * 100, bins=60, range=(0, 25), color="#C2562B", alpha=0.6, density=True, label="mô phỏng")
    ax[1].set_xlabel("Độ tương phản bóng mạch (%)"); ax[1].legend(frameon=False); ax[1].set_title("Bóng mạch: thật so với mô phỏng")
    t = (V.EDGES[:-1] + V.EDGES[1:]) / 2
    for lo, hi, col in ((1.5, 2.5, "#0F6E78"), (2.5, 3.5, "#C2562B")):
        sel = (d >= lo) & (d < hi); ss = (ys >= lo) & (ys < hi)
        ax[2].plot(t, np.median(P[sel], 0), color=col, lw=2, label=f"thật, dự đoán {lo}–{hi} mm")
        ax[2].plot(t, np.median(Bs[ss], 0), color=col, lw=1.5, ls="--", label=f"mô phỏng, thật {lo}–{hi} mm")
    ax[2].set_xlabel("Vị trí ngang (mm)"); ax[2].set_ylabel("Profile / nền"); ax[2].legend(frameon=False, fontsize=9)
    ax[2].set_title("Profile trung vị")
    fig.tight_layout(); fig.savefig(R / "overview.png", dpi=130); plt.close(fig)

    # fig 2: overlays for 4 images
    names = sorted(img)[::max(1, len(img) // 4)][:4]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.4))
    for a, n in zip(axs, names):
        im = np.asarray(Image.open(DS / "images" / f"{n}.png"))
        a.imshow(im, cmap="gray"); q = img[n]
        sc = a.scatter([r["x"] for r in q], [r["y"] for r in q], c=[r["depth_mm"] for r in q], s=10, cmap="viridis", vmin=1, vmax=4)
        a.set_title(f"ảnh {list(img).index(n) + 1}: trung vị {per_img[n]:.2f} mm"); a.axis("off")
    fig.colorbar(sc, ax=axs, shrink=0.8, label="Độ sâu dự đoán (mm)")
    fig.savefig(R / "overlays.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # fig 3: repeatability
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ks = sorted(per_hand, key=lambda k: np.median(per_hand[k]))
    for i, k in enumerate(ks):
        ax[0].plot([i] * len(per_hand[k]), per_hand[k], ".", color="#0F6E78", ms=4)
    ax[0].set_xlabel("Bàn tay (xếp theo trung vị)"); ax[0].set_ylabel("Trung vị độ sâu mỗi ảnh (mm)")
    ax[0].set_title(f"Lặp lại: SD trong cùng bàn tay {np.median(rep_sd):.2f} mm\nkhác biệt giữa các bàn tay SD {between_sd:.2f} mm")
    ax[1].plot(lr[:, 0], lr[:, 1], "o", color="#0F6E78"); m_ = [lr.min() - 0.1, lr.max() + 0.1]; ax[1].plot(m_, m_, color="gray", lw=1)
    ax[1].set_xlabel("Tay trái (mm)"); ax[1].set_ylabel("Tay phải (mm)"); ax[1].set_title(f"Trái so với phải: r = {out['LR_corr']:.2f}")
    fig.tight_layout(); fig.savefig(R / "repeat.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
