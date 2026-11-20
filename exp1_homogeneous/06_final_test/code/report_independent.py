"""V3-V6: metrics, stability, depth discrimination and limits from V2 predictions.

Follows analysis_plan in the V0 manifest: metrics per condition and model seed
across the three MC repeats, then equal weight over conditions; each seed, mean
and SD (ddof=1) are reported. The primary analysis set is the conditions accepted
by MC quality; all valid conditions are reported separately as a sensitivity check.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp1_paths as L  # noqa: E402  (step folders on sys.path; L.at() resolves stored paths)
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MUAS = (0.15, 0.22, 0.30, 0.40, 0.508, 1.0)
GROUPS = {"unseen_pooled": (0.22, 0.40), "unseen_0.22": (0.22,), "unseen_0.40": (0.40,),
          "seen_weak_pooled": (0.15, 0.30), "seen_weak_0.15": (0.15,), "seen_weak_0.30": (0.30,),
          "standard_0.508": (0.508,), "high_1.0": (1.0,), "all": MUAS}
SCOPE = "Direct MC, ideal surface reflectance, one straight 3 mm channel, fixed background; no camera or phantom validation."
BRANCHES = ("F", "K")
COLORS = {"F": "#167d9a", "K": "#d27a29"}


def read_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in ("depth_mm", "mua_mm_inv", "true_depth_mm", "pred_F_mm", "pred_K_mm", "error_F_mm", "error_K_mm",
                        "abs_error_K_minus_F_mm"):
                row[key] = float(row[key])
            row["repeat"], row["model_seed"] = int(row["repeat"]), int(row["model_seed"])
            rows.append(row)
    return rows


def condition_table(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition_id"], row["model_seed"]].append(row)
    table = []
    for (condition, seed), items in sorted(grouped.items()):
        if len(items) != 3:
            raise ValueError(f"{condition} seed {seed}: expected three MC repeats, found {len(items)}")
        entry = {"condition_id": condition, "seed": seed, "depth_mm": items[0]["depth_mm"], "mua": round(items[0]["mua_mm_inv"], 3),
                 "status": items[0]["quality_status"], "truth": items[0]["true_depth_mm"]}
        for branch in BRANCHES:
            error = np.array([i[f"error_{branch}_mm"] for i in items])
            entry[f"mae_{branch}"] = float(np.abs(error).mean())
            entry[f"rmse_{branch}"] = float(np.sqrt(np.mean(error ** 2)))
            entry[f"bias_{branch}"] = float(error.mean())
            entry[f"pred_{branch}"] = float(np.mean([i[f"pred_{branch}_mm"] for i in items]))
        table.append(entry)
    return table


def summarize(table, muas):
    chosen = [t for t in table if t["mua"] in muas]
    if not chosen:
        return None
    seeds = sorted({t["seed"] for t in chosen})
    by_seed = {}
    for seed in seeds:
        items = [t for t in chosen if t["seed"] == seed]
        values = {f"{m}_{b}_mm": float(np.mean([t[f"{m}_{b}"] for t in items])) for m in ("mae", "rmse", "bias") for b in BRANCHES}
        values["delta_MAE_K_minus_F_mm"] = values["mae_K_mm"] - values["mae_F_mm"]
        by_seed[seed] = values
    keys = list(next(iter(by_seed.values())))
    return {"n_conditions": len({t["condition_id"] for t in chosen}), "by_seed": by_seed,
            "mean": {k: float(np.mean([by_seed[s][k] for s in seeds])) for k in keys},
            "sd_ddof1": {k: float(np.std([by_seed[s][k] for s in seeds], ddof=1)) for k in keys},
            "F_lower_MAE_all_seeds": all(by_seed[s]["delta_MAE_K_minus_F_mm"] > 0 for s in seeds)}


def stability(table):
    out = []
    for depth in sorted({t["depth_mm"] for t in table}):
        for seed in sorted({t["seed"] for t in table}):
            items = [t for t in table if t["depth_mm"] == depth and t["seed"] == seed]
            row = {"depth_mm": depth, "seed": seed, "n_mua": len(items)}
            for branch in BRANCHES:
                pred = [t[f"pred_{branch}"] for t in items]
                row[f"R_{branch}_mm"] = float(max(pred) - min(pred))
                row[f"MAE_{branch}_mm"] = float(np.mean([t[f"mae_{branch}"] for t in items]))
            row["R_F_smaller_and_MAE_F_lower"] = row["R_F_mm"] < row["R_K_mm"] and row["MAE_F_mm"] < row["MAE_K_mm"]
            out.append(row)
    return out


def discrimination(table, baseline):
    out = []
    for mua in sorted({t["mua"] for t in table}):
        for seed in sorted({t["seed"] for t in table}):
            items = [t for t in table if t["mua"] == mua and t["seed"] == seed]
            truth = np.array([t["truth"] for t in items])
            row = {"mua": mua, "seed": seed, "n_conditions": len(items), "MAE_constant_baseline_mm": float(np.abs(baseline - truth).mean())}
            for branch in BRANCHES:
                slope, intercept = np.polyfit(truth, [t[f"pred_{branch}"] for t in items], 1) if len(items) > 1 else (float("nan"),) * 2
                row[f"slope_{branch}"], row[f"intercept_{branch}_mm"] = float(slope), float(intercept)
                row[f"MAE_{branch}_mm"] = float(np.mean([t[f"mae_{branch}"] for t in items]))
            out.append(row)
    return out


def condition_summary(table, rows, target):
    paired = defaultdict(list)
    for row in rows:
        paired[row["condition_id"]].append(row["abs_error_K_minus_F_mm"])
    out = []
    for condition in sorted({t["condition_id"] for t in table}):
        items = [t for t in table if t["condition_id"] == condition]
        row = {"condition_id": condition, "depth_mm": items[0]["depth_mm"], "mua": items[0]["mua"], "status": items[0]["status"]}
        for branch in BRANCHES:
            mae = float(np.mean([t[f"mae_{branch}"] for t in items]))
            row[f"mean_MAE_{branch}_mm"] = mae
            row[f"MAE_{branch}_le_target"] = mae <= target
        row["F_lower_MAE_seed_count"] = sum(t["mae_F"] < t["mae_K"] for t in items)
        row["mean_paired_abs_error_K_minus_F_mm"] = float(np.mean(paired[condition]))
        out.append(row)
    return out


def plot_stability(table, out):
    depths = sorted({t["depth_mm"] for t in table})
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), layout="constrained", sharex=True)
    for ax, depth in zip(axes.flat, depths):
        for branch in BRANCHES:
            lines = []
            for seed in sorted({t["seed"] for t in table}):
                items = sorted((t for t in table if t["depth_mm"] == depth and t["seed"] == seed), key=lambda t: t["mua"])
                xs, ys = [t["mua"] for t in items], [t[f"pred_{branch}"] for t in items]
                ax.plot(xs, ys, color=COLORS[branch], alpha=0.3, linewidth=1)
                lines.append(ys)
            ax.plot(xs, np.mean(lines, axis=0), color=COLORS[branch], marker="o", markersize=3, label=branch)
        ax.axhline(depth, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"z = {depth} mm")
        ax.set_xscale("log")
    axes[0, 0].set_xticks(MUAS, [str(m) for m in MUAS])
    axes[0, 0].set_xticks([], minor=True)
    for ax in axes[1]:
        ax.tick_params(axis="x", labelsize=8, labelrotation=45)
    axes[0, 0].legend()
    fig.supxlabel("Channel absorption (1/mm)")
    fig.supylabel("Predicted depth (mm)")
    fig.savefig(out / "v4_prediction_vs_absorption.png", dpi=160)
    plt.close(fig)


def plot_depth(table, out):
    muas = sorted({t["mua"] for t in table})
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), layout="constrained", sharex=True, sharey=True)
    for ax, mua in zip(axes.flat, muas):
        for branch in BRANCHES:
            lines = []
            for seed in sorted({t["seed"] for t in table}):
                items = sorted((t for t in table if t["mua"] == mua and t["seed"] == seed), key=lambda t: t["truth"])
                xs, ys = [t["truth"] for t in items], [t[f"pred_{branch}"] for t in items]
                ax.plot(xs, ys, color=COLORS[branch], alpha=0.3, linewidth=1)
                lines.append(ys)
            ax.plot(xs, np.mean(lines, axis=0), color=COLORS[branch], marker="o", markersize=3, label=branch)
        ax.plot([0.5, 5], [0.5, 5], color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"mua = {mua} 1/mm")
    axes[0, 0].legend()
    fig.supxlabel("True depth (mm)")
    fig.supylabel("Predicted depth (mm)")
    fig.savefig(out / "v5_prediction_vs_depth.png", dpi=160)
    plt.close(fig)


def plot_map(summary, out):
    depths = sorted({r["depth_mm"] for r in summary})
    muas = sorted({r["mua"] for r in summary})
    lookup = {(r["depth_mm"], r["mua"]): r for r in summary}
    panels = [("MAE F (mm)", "mean_MAE_F_mm", "viridis_r", 0, None), ("MAE K (mm)", "mean_MAE_K_mm", "viridis_r", 0, None),
              ("MAE K - MAE F (mm); positive: F lower", None, "RdBu", None, None)]
    grid = {key: np.array([[lookup[d, m][key] if (d, m) in lookup else np.nan for m in muas] for d in depths])
            for key in ("mean_MAE_F_mm", "mean_MAE_K_mm")}
    grid[None] = grid["mean_MAE_K_mm"] - grid["mean_MAE_F_mm"]
    top = float(np.nanmax([grid["mean_MAE_F_mm"], grid["mean_MAE_K_mm"]]))
    limit = float(np.nanmax(np.abs(grid[None])))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), layout="constrained")
    for ax, (title, key, cmap, _, _) in zip(axes, panels):
        vmin, vmax = (-limit, limit) if key is None else (0, top)
        image = ax.imshow(grid[key], cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i, depth in enumerate(depths):
            for j, mua in enumerate(muas):
                if np.isfinite(grid[key][i, j]):
                    flag = "" if lookup[depth, mua]["status"] == "accepted" else "*"
                    ax.text(j, i, f"{grid[key][i, j]:.2f}{flag}", ha="center", va="center", fontsize=8,
                            bbox={"facecolor": "white", "alpha": 0.65, "pad": 1, "linewidth": 0})
        ax.set_xticks(range(len(muas)), [str(m) for m in muas])
        ax.set_yticks(range(len(depths)), [str(d) for d in depths])
        ax.set_xlabel("Channel absorption (1/mm)")
        ax.set_ylabel("True depth (mm)")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(out / "v6_error_map.png", dpi=160)
    plt.close(fig)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def num(value):
    return f"{value:.3f}".replace(".", ",")


def seed_line(group):
    return " / ".join(num(v["delta_MAE_K_minus_F_mm"]) for v in group["by_seed"].values())


def conclusions(result):
    groups, lines = result["groups"], []
    unseen, weak, standard = groups["unseen_pooled"], groups["seen_weak_pooled"], groups["standard_0.508"]
    if unseen:
        delta = unseen["mean"]["delta_MAE_K_minus_F_mm"]
        if unseen["F_lower_MAE_all_seeds"]:
            lines.append("F giảm MAE ở nhóm μa chưa học 0,22 / 0,40 ở cả ba seed. Kết luận về khái quát hóa chỉ dùng được nếu mục khả năng phân biệt độ sâu cũng đạt, và chỉ trong mô hình vật lý đã khảo sát.")
        elif delta > 0:
            lines.append("ΔMAE trung bình ở nhóm chưa học dương nhưng không phải cả ba seed đều cải thiện: báo lợi ích trung bình kèm sự thiếu nhất quán.")
        else:
            lines.append("F không giảm MAE trung bình ở nhóm chưa học: giả thuyết chưa được hỗ trợ trong phép kiểm này.")
    if weak:
        lines.append(("F giảm MAE trên MC mới ở nhóm 0,15 / 0,30 (mức F đã học); chưa đủ để nói khái quát sang μa mới."
                      if weak["mean"]["delta_MAE_K_minus_F_mm"] > 0 else "F không giảm MAE trung bình trên MC mới ở nhóm 0,15 / 0,30."))
    if standard:
        difference = standard["mean"]["mae_F_mm"] - standard["mean"]["mae_K_mm"]
        lines.append(f"Đánh đổi tại mức chuẩn 0,508: MAE_F − MAE_K = {num(difference)} mm." + (" F kém K ở mức này." if difference > 0 else ""))
    return lines


def group_table(groups):
    out = ["| Nhóm | Số điều kiện | MAE F (mm) | MAE K (mm) | K − F, TB ± SD (mm) | K − F theo seed 42/43/44 | Cả 3 seed F thấp hơn |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, g in groups.items():
        if g:
            out.append(f"| {name} | {g['n_conditions']} | {num(g['mean']['mae_F_mm'])} | {num(g['mean']['mae_K_mm'])} | "
                       f"{num(g['mean']['delta_MAE_K_minus_F_mm'])} ± {num(g['sd_ddof1']['delta_MAE_K_minus_F_mm'])} | {seed_line(g)} | {'Có' if g['F_lower_MAE_all_seeds'] else 'Không'} |")
    return out


def markdown(result):
    reliable = result["analysis_sets"]["accepted"]
    lines = ["# Kết luận kiểm chứng mô phỏng V2–V6", "", f"Phạm vi: {SCOPE}", "",
             f"Điều kiện theo chất lượng MC: {result['n_conditions_by_status']}.", "",
             "Tệp này do chương trình sinh tự động theo quy tắc đã khóa; cần người đọc lại trước khi đưa vào luận văn.", "",
             "## Chỉ số chính và các nhóm (tập điều kiện đạt chất lượng MC)", ""] + group_table(reliable["groups"])
    lines += ["", "## Câu chữ kết luận theo mục 6 của kế hoạch", ""] + [f"- {x}" for x in conclusions(reliable)]
    disc = [r for r in reliable["depth_discrimination"] if r["mua"] in (0.22, 0.4)]
    lines += ["", "## Khả năng phân biệt độ sâu ở μa chưa học", "", "| μa | Seed | Hệ số góc F | Chặn F (mm) | MAE F (mm) | Hệ số góc K | MAE K (mm) | MAE đoán hằng (mm) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {r['mua']} | {r['seed']} | {num(r['slope_F'])} | {num(r['intercept_F_mm'])} | {num(r['MAE_F_mm'])} | {num(r['slope_K'])} | {num(r['MAE_K_mm'])} | {num(r['MAE_constant_baseline_mm'])} |" for r in disc]
    stable = reliable["absorption_stability"]
    lines += ["", "## Ổn định theo hấp thụ", "", f"Số cặp độ sâu–seed có R_F nhỏ hơn R_K và MAE_F cũng thấp hơn: {sum(r['R_F_smaller_and_MAE_F_lower'] for r in stable)}/{len(stable)}. R nhỏ chỉ có ý nghĩa khi dự đoán cũng gần độ sâu thật."]
    summary = reliable["condition_summary"]
    failing = [r for r in summary if not r["MAE_F_le_target"] or not r["MAE_K_le_target"]]
    lines += ["", "## Mục tiêu MAE ≤ 0,5 mm (trung bình ba seed, không phải chuẩn lâm sàng)", "",
              f"F đạt {sum(r['MAE_F_le_target'] for r in summary)}/{len(summary)} điều kiện; K đạt {sum(r['MAE_K_le_target'] for r in summary)}/{len(summary)} điều kiện.", "",
              "| Điều kiện chưa đạt ở ít nhất một nhánh | MAE F (mm) | MAE K (mm) |", "| --- | --- | --- |"]
    lines += [f"| z = {r['depth_mm']}, μa = {r['mua']} | {num(r['mean_MAE_F_mm'])} | {num(r['mean_MAE_K_mm'])} |" for r in failing]
    if result["sensitivity_differs"]:
        lines += ["", "## Kiểm độ nhạy: gồm cả điều kiện chưa hội tụ", ""] + group_table(result["analysis_sets"]["all_valid"]["groups"])
    lines += ["", "## Giới hạn", "", "- Ba MC lặp đo biến thiên số học, ba seed CNN đo biến thiên huấn luyện; không nhân thành cỡ mẫu vật lý.", "- Chỉ mô tả ghép cặp và độ nhất quán; không tuyên bố ý nghĩa thống kê quần thể.", "- Chưa có camera hay phantom (V7–V8)."]
    return "\n".join(lines) + "\n"


def build(rows, baseline, target, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table = condition_table(rows)
    accepted_ids = {t["condition_id"] for t in table if t["status"] == "accepted"}
    sets = {"accepted": [t for t in table if t["condition_id"] in accepted_ids], "all_valid": table}
    analysis = {}
    for name, items in sets.items():
        chosen = [r for r in rows if name == "all_valid" or r["condition_id"] in accepted_ids]
        analysis[name] = {"n_conditions": len({t["condition_id"] for t in items}),
                          "groups": {g: summarize(items, m) for g, m in GROUPS.items()},
                          "absorption_stability": stability(items), "depth_discrimination": discrimination(items, baseline),
                          "condition_summary": condition_summary(items, chosen, target)}
    statuses = defaultdict(set)
    for t in table:
        statuses[t["status"]].add(t["condition_id"])
    result = {"scope": SCOPE, "primary_endpoint": "delta_MAE_unseen = groups.unseen_pooled, set accepted", "target_mae_mm": target,
              "constant_baseline_mm": baseline, "n_conditions_by_status": {k: len(v) for k, v in statuses.items()},
              "sensitivity_differs": len(accepted_ids) != len({t["condition_id"] for t in table}), "analysis_sets": analysis}
    (out / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    primary = analysis["accepted"]
    write_csv(out / "v3_group_metrics.csv", [{"group": g, "seed": s, **v} for g, item in primary["groups"].items() if item for s, v in item["by_seed"].items()])
    write_csv(out / "v4_absorption_stability.csv", primary["absorption_stability"])
    write_csv(out / "v5_depth_discrimination.csv", primary["depth_discrimination"])
    write_csv(out / "v6_condition_summary.csv", analysis["all_valid"]["condition_summary"])
    plot_stability(sets["accepted"], out)
    plot_depth(sets["accepted"], out)
    plot_map(analysis["all_valid"]["condition_summary"], out)
    (out / "KET_LUAN_V2_V6.md").write_text(markdown(result), encoding="utf-8")
    return result


def main():
    from simulate import ROOT
    from run_independent_mc import OUTPUT, load_locked_manifest
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    plan = load_locked_manifest()["analysis_plan"]
    rows = read_rows(OUTPUT / "predictions/predictions.csv")
    build(rows, plan["constant_baseline"]["value_mm"], plan["target"]["mae_mm"], OUTPUT / "report")
    print(f"REPORT_WRITTEN {OUTPUT / 'report'}", flush=True)


if __name__ == "__main__":
    main()
