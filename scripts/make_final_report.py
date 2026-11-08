"""Final, paper-style result tables + figures for the whole project, computed from per-image predictions.

Every number comes from <run>/predictions/*.csv (one row per test image of the held-out person):
  fp32   -> epoch_<last>.csv        (end of training: the honest number, no test-set selection)
  best   -> epoch_<argmin>.csv      (oracle epoch chosen ON the test person: optimistic, reported for reference only)
  flip   -> final_fp32_flip.csv     (horizontal-flip TTA, scripts/eval_tta.py)
  ptq    -> final_int8_ptq.csv      (int8 post-training quantization, scripts/eval_int8.py)
  qat    -> final_int8_qat.csv      (int8 QAT 2 ep, lr 1e-5, scripts/qat_finetune.py)
  qat5   -> final_int8_qat_q5.csv   (int8 QAT 5 ep, lr 5e-5)
Missing files are skipped, so the script can be re-run as jobs finish.

Metrics (per model x variant):
  person-level: mean angular error per held-out person -> mean, std (ddof=1), 95 % t-CI over persons (the standard
                MPIIFaceGaze number is this mean); horizontal / vertical MAE (person-averaged)
  image-level (pooled over all test images; every image is tested exactly once in leave-one-person-out):
                median, p90, Acc@2/3/5/10 deg (fraction of images with error <= theta)
  gaze-zone classification (derived, MPIIFaceGaze itself is a regression benchmark): the 2-D gaze plane is split
                into a 3x3 grid at the tertiles of the ground truth over the whole dataset (fixed thresholds); the
                prediction's zone vs the ground-truth zone -> accuracy and macro-F1 over the 9 zones.
  paired comparison vs a reference model on the same persons: mean diff, wins, paired t-test, Wilcoxon signed-rank.

  python -I scripts/make_final_report.py
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

M = "/work/u8714813/runs/l2cs"
L = "/work/u8714813/runs/l2cs_light"
ALL15 = list(range(15))
FIVE = [0, 3, 7, 11, 14]
VAR_FILE = dict(flip="final_fp32_flip.csv", ptq="final_int8_ptq.csv", qat="final_int8_qat.csv", qat5="final_int8_qat_q5.csv")
VAR_NAME = dict(fp32="fp32", best="fp32 最佳 epoch（看測試者，偏樂觀）", flip="fp32 + 翻轉 TTA", ptq="int8 PTQ", qat="int8 QAT", qat5="int8 QAT（5 ep）")
# key -> (label, root, tag, backbone, img_size)
MODELS = {
    "A": ("ResNet-50 @448，論文配方（50 ep）", M, "paper_e50", "resnet50", 448),
    "B": ("ResNet-50 @448，改良配方（50 ep）", M, "improved_e50", "resnet50", 448),
    "C1": ("ResNet-50，只 cosine（50 ep）", M, "abl_cosine_e50", "resnet50", 448),
    "C2": ("ResNet-50，只增強（50 ep）", M, "abl_aug_e50", "resnet50", 448),
    "r50": ("ResNet-50 @448（20 ep）", L, "r50_448_e20", "resnet50", 448),
    "r50_224": ("ResNet-50 @224（20 ep）", L, "r50_224_e20", "resnet50", 224),
    "effb0": ("EfficientNet-B0 @448", L, "effb0_448_e20", "efficientnet_b0", 448),
    "effb0_kd": ("EfficientNet-B0 @448 + KD（T=2）", L, "effb0_448_kd_e20", "efficientnet_b0", 448),
    "mnv3l": ("MobileNetV3-L @448", L, "mnv3l_448_e20", "mobilenet_v3_large", 448),
    "mnv3l_kd": ("MobileNetV3-L @448 + KD（T=2）", L, "mnv3l_448_kd_e20", "mobilenet_v3_large", 448),
    "mnv3l_kdT4": ("MobileNetV3-L @448 + KD（T=4）", L, "mnv3l_448_kdT4_e20", "mobilenet_v3_large", 448),
}
# literature, MPIIFaceGaze leave-one-person-out mean angular error (as reported by the L2CS-Net paper, Table 2,
# and GazeTR's paper). Not re-run by us.
LIT = [("FullFace (Zhang et al., 2017)", 4.93), ("RT-Gene (Fischer et al., 2018)", 4.66), ("Dilated-Net (Chen & Shi, 2018)", 4.42),
       ("CA-Net (Cheng et al., 2020)", 4.27), ("Gaze360 (Kellnhofer et al., 2019)", 4.06), ("GazeTR-Hybrid (Cheng & Lu, 2022)", 4.00),
       ("L2CS-Net (Abdelrahman et al., 2022)", 3.92)]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
THETAS = [2, 3, 5, 10]


def run_dir(key, f):
    _, root, tag, *_ = MODELS[key]
    return os.path.join(root, f"fold{f}_{tag}")


def pred_file(key, f, var):
    d = run_dir(key, f)
    if not os.path.exists(os.path.join(d, "done.json")):
        return None
    if var in ("fp32", "best"):
        rows = list(csv.DictReader(open(os.path.join(d, "metrics.csv"))))
        e = int(rows[-1]["epoch"]) if var == "fp32" else int(min(rows[1:], key=lambda r: float(r["test_angular_err"]))["epoch"])
        p = os.path.join(d, "predictions", f"epoch_{e}.csv")
    else:
        p = os.path.join(d, "predictions", VAR_FILE[var])
    return p if os.path.exists(p) else None


def load(p):
    a = np.genfromtxt(p, delimiter=",", skip_header=1, usecols=(2, 3, 4, 5, 6))
    return dict(gh=a[:, 0], gv=a[:, 1], ph=a[:, 2], pv=a[:, 3], err=a[:, 4])


def zone(h, v, th):
    return np.digitize(h, th["h"]) * 3 + np.digitize(v, th["v"])


def macro_f1(y, yhat, k=9):
    f1 = []
    for c in range(k):
        tp = np.sum((yhat == c) & (y == c)); fp = np.sum((yhat == c) & (y != c)); fn = np.sum((yhat != c) & (y == c))
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1))


def metrics(key, var, folds, th):
    per = {}
    for f in folds:
        p = pred_file(key, f, var)
        if p is None:
            return None
        per[f] = load(p)
    pm = np.array([per[f]["err"].mean() for f in folds])
    cat = {k: np.concatenate([per[f][k] for f in folds]) for k in ("gh", "gv", "ph", "pv", "err")}
    n = len(folds)
    ci = stats.t.ppf(0.975, n - 1) * pm.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    zg, zp = zone(cat["gh"], cat["gv"], th), zone(cat["ph"], cat["pv"], th)
    return dict(per_person=pm.tolist(), mean=float(pm.mean()), std=float(pm.std(ddof=1)), ci95=float(ci),
                mae_h=float(np.mean([np.abs(per[f]["ph"] - per[f]["gh"]).mean() for f in folds])),
                mae_v=float(np.mean([np.abs(per[f]["pv"] - per[f]["gv"]).mean() for f in folds])),
                median=float(np.median(cat["err"])), p90=float(np.percentile(cat["err"], 90)),
                acc={t: float((cat["err"] <= t).mean()) for t in THETAS},
                zone_acc=float((zg == zp).mean()), zone_f1=macro_f1(zg, zp), n_img=int(len(cat["err"])), n_person=n,
                _err=cat["err"])


def paired(a, b):
    a, b = np.array(a), np.array(b); d = a - b
    w = stats.wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
    return dict(diff=float(d.mean()), diff_std=float(d.std(ddof=1)), wins=int((d < 0).sum()), n=len(d),
                p_t=float(stats.ttest_rel(a, b).pvalue), p_w=float(w))


def style(ax):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/final")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    # fixed zone thresholds: tertiles of the ground truth over all 15 persons
    gt = [load(pred_file("B", f, "fp32")) for f in ALL15]
    gh, gv = np.concatenate([g["gh"] for g in gt]), np.concatenate([g["gv"] for g in gt])
    th = dict(h=np.percentile(gh, [100 / 3, 200 / 3]).tolist(), v=np.percentile(gv, [100 / 3, 200 / 3]).tolist())

    res = {}  # (key, var, scope) -> metrics
    for key in MODELS:
        for var in ("fp32", "best", "flip", "ptq", "qat", "qat5"):
            for scope, folds in (("15", ALL15), ("5", FIVE)):
                m = metrics(key, var, folds, th)
                if m is not None:
                    res[(key, var, scope)] = m
    bench = {(r["backbone"], r["img_size"]): r for r in json.load(open("results/lightweight/bench_h200.json"))["rows"]}
    b8p = os.path.join(a.out, "bench_int8.json")
    b8 = {(r["backbone"], r["img_size"]): r for r in json.load(open(b8p))["rows"]} if os.path.exists(b8p) else {}
    fig2 = json.load(open("results/slides_charts/paper_fig2_per_subject.json"))

    def lab(key, var):
        return MODELS[key][0] + ("" if var == "fp32" else f"，{VAR_NAME[var]}")

    def full_row(key, var, scope, ref=None):
        m = res[(key, var, scope)]; bb, size = MODELS[key][3], MODELS[key][4]
        cells = [lab(key, var), f"{bench[(bb, size)]['gmacs']:.2f}" + ("×2" if var == "flip" else ""),
                 f"**{m['mean']:.2f}** ± {m['std']:.2f}", f"[{m['mean'] - m['ci95']:.2f}, {m['mean'] + m['ci95']:.2f}]",
                 f"{m['median']:.2f}", f"{m['p90']:.2f}", f"{m['mae_h']:.2f}", f"{m['mae_v']:.2f}"]
        cells += [f"{100 * m['acc'][t]:.1f}" for t in THETAS] + [f"{100 * m['zone_acc']:.1f}", f"{m['zone_f1']:.3f}"]
        if ref is not None and ref in res and (key, var, scope) != ref:
            pr = paired(m["per_person"], res[ref]["per_person"])
            cells += [f"{pr['diff']:+.2f}", f"{pr['wins']}/{pr['n']}", f"{pr['p_t']:.3f}", f"{pr['p_w']:.3f}"]
        else:
            cells += ["—"] * 4
        return "| " + " | ".join(cells) + " |"

    HDR = ("| 模型 | GMACs | 平均 ± SD（度） | 95% CI | 中位數 | P90 | 水平 MAE | 垂直 MAE | " + " | ".join(f"Acc@{t}°" for t in THETAS) +
           " | 區域 Acc | 區域 macro-F1 | 與參考差 | 較好人數 | t 檢定 p | Wilcoxon p |")
    SEP = "|" + "---|" * 18
    md = [f"<!-- generated by scripts/make_final_report.py; zone thresholds (deg): horizontal {th['h'][0]:.2f}/{th['h'][1]:.2f}, "
          f"vertical {th['v'][0]:.2f}/{th['v'][1]:.2f} -->", ""]

    # T1 literature
    md += ["### 表 1：與文獻比較（MPIIFaceGaze，leave-one-person-out，平均角度誤差）", "", "| 方法 | 平均誤差（度） | 來源 |", "|---|---|---|"]
    md += [f"| {n} | {v:.2f} | 文獻報告值 |" for n, v in LIT]
    for key, var in (("A", "fp32"), ("A", "best"), ("B", "fp32"), ("B", "best"), ("B", "flip"), ("effb0", "fp32"), ("effb0", "flip"),
                     ("mnv3l_kd", "fp32"), ("mnv3l_kd", "flip"), ("mnv3l_kd", "qat"), ("mnv3l_kd", "qat5")):
        if (key, var, "15") in res:
            m = res[(key, var, "15")]
            md.append(f"| **本研究**：{lab(key, var)} | {m['mean']:.2f} ± {m['std']:.2f} | 本研究 15 fold |")

    # T2 full metrics, 15 fold
    md += ["", "### 表 2：完整指標（15 fold；參考 = ResNet-50 改良配方 fp32）", "", HDR, SEP]
    for key, var in (("A", "fp32"), ("A", "flip"), ("B", "fp32"), ("B", "flip"), ("B", "ptq"), ("effb0", "fp32"), ("effb0", "flip"),
                     ("mnv3l_kd", "fp32"), ("mnv3l_kd", "flip"), ("mnv3l_kd", "ptq"), ("mnv3l_kd", "qat"), ("mnv3l_kd", "qat5"),
                     ("A", "best"), ("B", "best")):
        if (key, var, "15") in res:
            md.append(full_row(key, var, "15", ("B", "fp32", "15")))

    # T3 per subject
    keys15 = [(k, v) for k, v in (("A", "fp32"), ("B", "fp32"), ("B", "flip"), ("effb0", "fp32"), ("mnv3l_kd", "fp32"), ("mnv3l_kd", "qat"),
                                  ("mnv3l_kd", "qat5")) if (k, v, "15") in res]
    md += ["", "### 表 3：逐人誤差（度，15 fold）", "", "| 測試者 | 論文 Fig. 2 | " + " | ".join(lab(k, v) for k, v in keys15) + " |",
           "|---|---|" + "---|" * len(keys15)]
    for i in ALL15:
        md.append(f"| p{i:02d} | {fig2[f'p{i:02d}']:.2f} | " + " | ".join(f"{res[(k, v, '15')]['per_person'][i]:.2f}" for k, v in keys15) + " |")
    md.append(f"| **平均** | **{np.mean([fig2[f'p{i:02d}'] for i in ALL15]):.2f}** | " +
              " | ".join(f"**{res[(k, v, '15')]['mean']:.2f}**" for k, v in keys15) + " |")

    # T4 recipe ablation, 5 persons
    md += ["", "### 表 4：訓練配方消融（ResNet-50 @448，50 epoch，5 人 p00/p03/p07/p11/p14；參考 = 論文配方）", "",
           "| 配方 | cosine | 增強 | 平均 @ep50 ± SD | 最佳 epoch 平均（偏樂觀） | Acc@5° | 區域 F1 | 與論文配方差 | 較好人數 | t 檢定 p |", "|---|---|---|---|---|---|---|---|---|---|"]
    for key, cos, aug in (("A", "", ""), ("C1", "✓", ""), ("C2", "", "✓"), ("B", "✓", "✓")):
        if (key, "fp32", "5") in res:
            m, mb = res[(key, "fp32", "5")], res[(key, "best", "5")]
            pr = paired(m["per_person"], res[("A", "fp32", "5")]["per_person"]) if key != "A" else None
            md.append(f"| {MODELS[key][0]} | {cos} | {aug} | {m['mean']:.2f} ± {m['std']:.2f} | {mb['mean']:.2f} | {100 * m['acc'][5]:.1f} | {m['zone_f1']:.3f} | " +
                      (f"{pr['diff']:+.2f} | {pr['wins']}/5 | {pr['p_t']:.3f} |" if pr else "— | — | — |"))

    # T5 lightweight screening, 5 persons
    md += ["", "### 表 5：輕量化篩選（20 epoch 改良配方，5 人；參考 = ResNet-50 @448 20 ep fp32）", "", HDR, SEP]
    for key in ("r50", "r50_224", "effb0", "effb0_kd", "mnv3l", "mnv3l_kd", "mnv3l_kdT4"):
        for var in ("fp32", "flip", "ptq", "qat", "qat5"):
            if (key, var, "5") in res:
                md.append(full_row(key, var, "5", ("r50", "fp32", "5")))

    # T6 efficiency
    md += ["", "### 表 6：運算量與推論速度（只含視線模型，不含人臉偵測；batch 1）", "",
           "| 模型 | 參數 (M) | GMACs | ONNX | H200 fp32 (ms) | H200 bf16 (ms) | H200 bf16 batch 64 (img/s) | Xeon fp32 1 核 (ms) | Xeon int8 1 核 (ms) | Xeon fp32 4 核 (ms) | Xeon int8 4 核 (ms) | 2 TOPS NPU 理論上限 (fps) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for bb, size in [(r["backbone"], r["img_size"]) for r in json.load(open("results/lightweight/bench_h200.json"))["rows"]]:
        r = bench[(bb, size)]; q = b8.get((bb, size), {})
        g = lambda d, k: f"{d[k]['mean_ms']:.1f}" if k in d else "—"  # noqa: E731
        tput = f"{q['gpu_bf16_b64']['img_per_s']:.0f}" if "gpu_bf16_b64" in q else "—"
        # CPU fp32: the int8 bench (same process as int8, same node) when available, else bench_light
        c1 = g(q, "cpu_fp32_1t") if q else g(r, "cpu_1t"); c4 = g(q, "cpu_fp32_4t") if q else g(r, "cpu_4t")
        md.append(f"| {bb} @{size} | {r['params_m']:.1f} | {r['gmacs']:.2f} | {r['onnx_export']} | {r['gpu_fp32']['mean_ms']:.2f} | "
                  f"{r['gpu_bf16']['mean_ms']:.2f} | {tput} | {c1} | {g(q, 'cpu_int8_1t')} | {c4} | {g(q, 'cpu_int8_4t')} | "
                  f"{r['npu_fps_ceiling']['OAX4600 / AM62A (2 TOPS)']:.0f} |")

    open(os.path.join(a.out, "tables.md"), "w").write("\n".join(md) + "\n")
    summ = {f"{k}|{v}|{s}": {kk: vv for kk, vv in m.items() if not kk.startswith("_")} for (k, v, s), m in res.items()}
    json.dump(dict(zone_thresholds_deg=th, models={k: v[0] for k, v in MODELS.items()}, results=summ), open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    with open(os.path.join(a.out, "per_person.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model", "variant", "scope", "person", "mean_err_deg"])
        for (k, v, s), m in res.items():
            for f, e in zip(ALL15 if s == "15" else FIVE, m["per_person"]):
                w.writerow([MODELS[k][2], v, s, f"p{f:02d}", f"{e:.4f}"])
    print("\n".join(md))

    # ---- Fig: per-subject bars (15 persons)
    series = [("paper", "L2CS-Net paper (Fig. 2)", [fig2[f"p{i:02d}"] for i in ALL15])]
    for k, v, name in (("B", "fp32", "ResNet-50 (ours, improved)"), ("effb0", "fp32", "EfficientNet-B0"), ("mnv3l_kd", "qat", "MobileNetV3-L + KD, int8 QAT")):
        if (k, v, "15") in res:
            series.append((k, name, res[(k, v, "15")]["per_person"]))
    fig, ax = plt.subplots(figsize=(12, 4.4), facecolor=SURF); style(ax)
    w = 0.8 / len(series); x = np.arange(15)
    for j, (_, name, vals) in enumerate(series):
        ax.bar(x + (j - (len(series) - 1) / 2) * w, vals, w * 0.92, color=COLORS[j], label=f"{name} — mean {np.mean(vals):.2f}°",
               hatch="///" if j == 0 else None, edgecolor=SURF if j else COLORS[0], fill=j != 0, linewidth=0.8 if j == 0 else 0)
    ax.set_xticks(x); ax.set_xticklabels([f"p{i:02d}" for i in ALL15]); ax.set_ylabel("mean angular error (°)", color=INK2)
    ax.set_title("Per-subject error, leave-one-person-out (end of training, no test-set selection)", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left", labelcolor=INK2); ax.set_ylim(0, 8.5)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_per_subject.png"), dpi=150, facecolor=SURF); plt.close(fig)

    # ---- Fig: cumulative error distribution
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURF); style(ax); ax.xaxis.grid(True, color=GRID, lw=0.8)
    grid = np.linspace(0, 15, 301)
    j = 0
    for k, v, name in (("A", "fp32", "ResNet-50, paper recipe"), ("B", "fp32", "ResNet-50, improved"), ("B", "flip", "ResNet-50, improved + flip TTA"),
                       ("effb0", "fp32", "EfficientNet-B0"), ("mnv3l_kd", "qat", "MobileNetV3-L + KD, int8 QAT")):
        if (k, v, "15") in res:
            e = np.sort(res[(k, v, "15")]["_err"])
            ax.plot(grid, np.searchsorted(e, grid, side="right") / len(e) * 100, color=COLORS[j], lw=2, label=name); j += 1
    ax.set_xlim(0, 15); ax.set_ylim(0, 100); ax.set_xlabel("angular error threshold θ (°)", color=INK2); ax.set_ylabel("images with error ≤ θ (%)", color=INK2)
    ax.set_title("Cumulative error distribution, all 15 held-out persons", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK2)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_error_cdf.png"), dpi=150, facecolor=SURF); plt.close(fig)

    # ---- Fig: recipe ablation curves (5 persons, test error per epoch)
    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor=SURF); style(ax)
    j = 0
    for k, name in (("A", "paper recipe (const lr, no aug)"), ("C1", "cosine only"), ("C2", "augmentation only"), ("B", "cosine + augmentation")):
        if all(pred_file(k, f, "fp32") for f in FIVE):
            curves = np.array([[float(r["test_angular_err"]) for r in csv.DictReader(open(os.path.join(run_dir(k, f), "metrics.csv")))][1:] for f in FIVE])
            mc = curves.mean(0)
            ax.plot(np.arange(1, len(mc) + 1), mc, color=COLORS[j], lw=2, label=f"{name} — ep50 {mc[-1]:.2f}°"); j += 1
    ax.set_xlabel("epoch", color=INK2); ax.set_ylabel("mean test error, 5 persons (°)", color=INK2); ax.set_xlim(1, 50)
    ax.set_title("Recipe ablation: test error during training (ResNet-50 @448)", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_ablation_curves.png"), dpi=150, facecolor=SURF); plt.close(fig)


if __name__ == "__main__":
    main()
