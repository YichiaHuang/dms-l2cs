"""Tables + accuracy-vs-compute chart for the lightweight study.

Reads the light runs (5-fold screening at 20 epochs, + any 15-fold tag), results/lightweight/bench_h200.json,
int8.json, qat.json, and the 15-fold ResNet-50 reference (improved_e50 @ep50 in --main-runs).
Writes results/lightweight/{screening.csv, summary.json, tables.md, acc_vs_gmacs.png}.

  python -I scripts/make_light_report.py --runs /work/u8714813/runs/l2cs_light --main-runs /work/u8714813/runs/l2cs
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIVE = [0, 3, 7, 11, 14]
# tag -> (label, backbone, img_size); colours follow this fixed order
CONFIGS = [("r50_448_e20", "ResNet-50 @448", "resnet50", 448),
           ("r50_224_e20", "ResNet-50 @224", "resnet50", 224),
           ("effb0_448_e20", "EfficientNet-B0 @448", "efficientnet_b0", 448),
           ("mnv3l_448_e20", "MobileNetV3-L @448", "mobilenet_v3_large", 448),
           ("mnv3l_448_kd_e20", "MobileNetV3-L @448 + KD", "mobilenet_v3_large", 448),
           ("effb0_448_kd_e20", "EfficientNet-B0 @448 + KD", "efficientnet_b0", 448)]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
# budget lines: GMACs/frame that fit at 25 fps if gaze gets 1/3 of the NPU at 40 % utilisation (assumption)
NPU_LINES = [("1.1 TOPS (OAX8000)", 1.1), ("2 TOPS (OAX4600 / AM62A)", 2.0), ("4 TOPS (TDA4VEN)", 4.0), ("8 TOPS (TDA4VE)", 8.0)]
UTIL, SHARE, FPS = 0.4, 1 / 3, 25


def last_err(path):
    rows = list(csv.DictReader(open(os.path.join(path, "metrics.csv"))))
    return float(rows[-1]["test_angular_err"]), int(rows[-1]["epoch"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/work/u8714813/runs/l2cs_light")
    ap.add_argument("--main-runs", default="/work/u8714813/runs/l2cs")
    ap.add_argument("--out", default="results/lightweight")
    a = ap.parse_args()
    bench = {(r["backbone"], r["img_size"]): r for r in json.load(open(os.path.join(a.out, "bench_h200.json")))["rows"]}
    int8 = json.load(open(os.path.join(a.out, "int8.json")))
    qat = json.load(open(os.path.join(a.out, "qat.json"))) if os.path.exists(os.path.join(a.out, "qat.json")) else {}
    summ, rows_csv = {}, []
    for tag, label, bb, size in CONFIGS:
        errs = [last_err(os.path.join(a.runs, f"fold{f}_{tag}"))[0] for f in FIVE]
        b = bench[(bb, size)]
        s = dict(label=label, gmacs=b["gmacs"], params_m=b["params_m"], cpu_1t_ms=b["cpu_1t"]["mean_ms"], cpu_4t_ms=b["cpu_4t"]["mean_ms"],
                 fps_ceiling_2tops=b["npu_fps_ceiling"]["OAX4600 / AM62A (2 TOPS)"], per_person=errs, fp32_mean=float(np.mean(errs)))
        if all(f"fold{f}_{tag}" in int8 for f in FIVE):
            s["int8_ptq_mean"] = float(np.mean([int8[f"fold{f}_{tag}"]["int8_err"] for f in FIVE]))
            s["fp32_cpu_mean"] = float(np.mean([int8[f"fold{f}_{tag}"]["fp32_err"] for f in FIVE]))
        if all(f"fold{f}_{tag}" in qat for f in FIVE):
            s["int8_qat_mean"] = float(np.mean([qat[f"fold{f}_{tag}"]["int8_qat_err"] for f in FIVE]))
        summ[tag] = s
        rows_csv.append([tag, label, f"{s['gmacs']:.3f}", f"{s['params_m']:.2f}"] + [f"{e:.4f}" for e in errs] +
                        [f"{s['fp32_mean']:.4f}", f"{s.get('int8_ptq_mean', float('nan')):.4f}", f"{s.get('int8_qat_mean', float('nan')):.4f}"])
    with open(os.path.join(a.out, "screening.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "label", "gmacs", "params_m"] + [f"p{f:02d}" for f in FIVE] + ["fp32_mean", "int8_ptq_mean", "int8_qat_mean"])
        w.writerows(rows_csv)

    # 15-fold: every light tag that has all 15 folds, vs ResNet-50 improved_e50 @ep50
    ref = [last_err(os.path.join(a.main_runs, f"fold{f}_improved_e50"))[0] for f in range(15)]
    full = {}
    for tag, label, *_ in CONFIGS:
        if all(os.path.exists(os.path.join(a.runs, f"fold{f}_{tag}", "done.json")) for f in range(15)):
            e = [last_err(os.path.join(a.runs, f"fold{f}_{tag}"))[0] for f in range(15)]
            d = np.array(e) - np.array(ref)
            try:
                from scipy import stats
                pval = float(stats.ttest_rel(e, ref).pvalue)
            except Exception:  # noqa: BLE001
                pval = float("nan")
            full[tag] = dict(label=label, per_person=e, mean=float(np.mean(e)), std=float(np.std(e, ddof=1)),
                             diff_mean=float(d.mean()), diff_std=float(d.std(ddof=1)), wins=int((d < 0).sum()), p_paired_t=pval)
    summary = dict(screening=summ, fifteen_fold=full, ref_15fold=dict(label="ResNet-50 @448 improved, 50 epoch (exp. B)", per_person=ref,
                   mean=float(np.mean(ref)), std=float(np.std(ref, ddof=1))),
                   npu_budget_assumption=dict(utilisation=UTIL, gaze_share=SHARE, fps=FPS))
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)

    # ---- tables.md
    md = ["### 篩選：5 人（p00/p03/p07/p11/p14）@ep20，誤差（度）", "",
          "| 設定 | GMACs | 參數 (M) | Xeon 1 核 (ms) | 2 TOPS 理論上限 (fps) | fp32 平均 | int8 PTQ | int8 QAT |", "|---|---|---|---|---|---|---|---|"]
    fm = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"  # noqa: E731
    for tag, s in summ.items():
        md.append(f"| {s['label']} | {s['gmacs']:.2f} | {s['params_m']:.1f} | {s['cpu_1t_ms']:.0f} | {s['fps_ceiling_2tops']:.0f} | "
                  f"{s['fp32_mean']:.2f} | {fm(s.get('int8_ptq_mean'))} | {fm(s.get('int8_qat_mean'))} |")
    md += ["", "### 逐人（fp32 @ep20）", "", "| 設定 | " + " | ".join(f"p{f:02d}" for f in FIVE) + " | 平均 |", "|---|" + "---|" * 6]
    for s in summ.values():
        md.append(f"| {s['label']} | " + " | ".join(f"{e:.2f}" for e in s["per_person"]) + f" | {s['fp32_mean']:.2f} |")
    if full:
        md += ["", "### 15 fold（fp32，最後一個 epoch）", "", "| 模型 | GMACs | 平均 ± 標準差 | 與 ResNet-50 差（成對） | 較好人數 | 配對 t 檢定 p |", "|---|---|---|---|---|---|",
               f"| ResNet-50 @448（實驗 B，50 epoch） | 16.35 | {np.mean(ref):.2f} ± {np.std(ref, ddof=1):.2f} | — | — | — |"]
        for tag, f in full.items():
            md.append(f"| {f['label']}（20 epoch） | {summ[tag]['gmacs']:.2f} | {f['mean']:.2f} ± {f['std']:.2f} | {f['diff_mean']:+.2f} ± {f['diff_std']:.2f} | {f['wins']}/15 | {f['p_paired_t']:.2f} |")
    open(os.path.join(a.out, "tables.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    # ---- chart: 5-person mean error vs GMACs (log), colour = model, marker = precision
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURF)
    ax.set_facecolor(SURF)
    for (tag, *_), c in zip(CONFIGS, COLORS):
        s = summ[tag]; x = s["gmacs"] * (1.08 if tag.endswith("_kd_e20") else 1.0)  # KD variants share GMACs: nudge right
        ax.scatter([x], [s["fp32_mean"]], s=70, color=c, zorder=3, edgecolor=SURF, linewidth=1.5)
        top = max(v for v in (s["fp32_mean"], s.get("int8_ptq_mean", 0), s.get("int8_qat_mean", 0)))
        dy = 6 if tag != "effb0_448_e20" else -12
        ax.annotate(s["label"], (x, top if tag != "effb0_448_e20" else s["fp32_mean"]), xytext=(8, dy), textcoords="offset points",
                    fontsize=8, color=INK2)
        if "int8_ptq_mean" in s:
            ax.scatter([x], [s["int8_ptq_mean"]], s=70, facecolor="none", edgecolor=c, linewidth=1.8, zorder=3, marker="s")
            ax.plot([x, x], [s["fp32_mean"], s["int8_ptq_mean"]], color=c, lw=1, ls=":", zorder=2)
        if "int8_qat_mean" in s:
            ax.scatter([x], [s["int8_qat_mean"]], s=80, color=c, zorder=4, marker="X", edgecolor=SURF, linewidth=1)
    for name, tops in NPU_LINES:
        g = tops * 1e12 * UTIL * SHARE / (2 * FPS) / 1e9
        ax.axvline(g, color=BASE, lw=1, ls="--", zorder=1)
        ax.text(g * 1.03, ax.get_ylim()[1] if False else 7.9, f"{name}\n≤{g:.1f} GMACs", fontsize=7, color=MUTED, va="top")
    ax.set_xscale("log"); ax.set_ylim(3.6, 8.0); ax.set_xlim(0.6, 30)
    ax.set_xlabel("GMACs per face (log scale)", color=INK2); ax.set_ylabel("mean angular error, 5 subjects (°)", color=INK2)
    ax.set_title("Accuracy vs compute — ● fp32   □ int8 post-training quant   ✕ int8 QAT", color=INK, fontsize=11, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.text(0.01, 0.01, f"Dashed lines: per-frame budget at {FPS} fps if gaze gets 1/3 of the NPU at {int(UTIL*100)} % utilisation (assumption, not measured).",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(a.out, "acc_vs_gmacs.png"), dpi=150, facecolor=SURF); plt.close(fig)


if __name__ == "__main__":
    main()
