"""Collect ~/runs/l2cs results into DMS/results: CSV aggregates, learning-curve plots, summary tables (markdown).

Outputs in --out:
  all_epochs.csv        every epoch of every run, one row each (run, fold, person, recipe, epoch, metrics...)
  all_runs.csv          one row per run (best / last epoch, config)
  all_iters.csv         iteration-level training losses of every run that logged them
  curves_<person>.png   test error vs epoch, paper recipe vs improved, per held-out person
  curves_all.png        all runs on one axis
  zeroshot_per_person.png, RESULTS.md

  python make_report.py --runs ~/runs/l2cs --out ../results
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER_MPII = 3.92
PAPER_GAZE360 = 10.41
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
RECIPE_COLOR = {"paper": "#2a78d6", "improved": "#eb6834"}
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2); ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load_run(path):
    """Return (config, rows) where rows are dicts with unified keys across old/new metrics.csv formats."""
    cfg = json.load(open(os.path.join(path, "config.json")))
    cfg.setdefault("recipe", "paper")
    rows = []
    for r in csv.DictReader(open(os.path.join(path, "metrics.csv"))):
        if "train_loss_total" in r:
            loss = f(r["train_loss_total"])
        else:
            loss = f(r.get("train_loss_pitch")) + f(r.get("train_loss_yaw"))
        rows.append(dict(epoch=int(r["epoch"]), train_loss=loss, test_err=f(r["test_angular_err"]),
                         test_median=f(r.get("test_err_median")), test_std=f(r.get("test_err_std")), test_p90=f(r.get("test_err_p90")),
                         pitch_mae=f(r.get("test_pitch_mae")), yaw_mae=f(r.get("test_yaw_mae")),
                         ce_pitch=f(r.get("train_ce_pitch")), mse_pitch=f(r.get("train_mse_pitch")),
                         ce_yaw=f(r.get("train_ce_yaw")), mse_yaw=f(r.get("train_mse_yaw")),
                         lr_end=f(r.get("lr_end")), epoch_sec=f(r.get("epoch_sec")), img_per_sec=f(r.get("img_per_sec")),
                         gpu_mem_mb=f(r.get("gpu_mem_mb")), timestamp=r.get("timestamp", "")))
    return cfg, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.expanduser("~/runs/l2cs"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    md = ["# L2CS-Net on MPIIFaceGaze — 結果整理（`scripts/make_report.py` 自動產生）\n"]

    runs = {}
    for d in sorted(os.listdir(args.runs)):
        p = os.path.join(args.runs, d)
        if d.startswith("fold") and os.path.exists(os.path.join(p, "metrics.csv")) and os.path.exists(os.path.join(p, "config.json")):
            cfg, rows = load_run(p)
            if len(rows) >= 2:
                runs[d] = (cfg, rows)

    # ---- aggregated CSVs
    ep_keys = ["epoch", "train_loss", "ce_pitch", "mse_pitch", "ce_yaw", "mse_yaw", "test_err", "test_median", "test_std", "test_p90",
               "pitch_mae", "yaw_mae", "lr_end", "epoch_sec", "img_per_sec", "gpu_mem_mb", "timestamp"]
    with open(os.path.join(args.out, "all_epochs.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["run", "fold", "person", "recipe", "sched", "aug", "amp"] + ep_keys)
        for d, (cfg, rows) in runs.items():
            for r in rows:
                w.writerow([d, cfg["fold"], cfg["held_out"], cfg["recipe"], cfg.get("sched", "const"), cfg.get("aug", False), cfg.get("amp", False)] + [r[k] for k in ep_keys])
    with open(os.path.join(args.out, "all_runs.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "fold", "person", "recipe", "epochs_done", "best_epoch", "best_err", "last_err", "err_at_ep7", "mean_err_ep1_7",
                    "n_train", "n_test", "lr", "sched", "aug", "amp", "batch_size", "min_per_epoch", "finished"])
        for d, (cfg, rows) in runs.items():
            tr = [r for r in rows if r["epoch"] > 0]
            k = int(np.argmin([r["test_err"] for r in tr]))
            e7 = [r["test_err"] for r in tr if r["epoch"] == 7]
            w.writerow([d, cfg["fold"], cfg["held_out"], cfg["recipe"], tr[-1]["epoch"], tr[k]["epoch"], f"{tr[k]['test_err']:.4f}", f"{tr[-1]['test_err']:.4f}",
                        f"{e7[0]:.4f}" if e7 else "", f"{np.mean([r['test_err'] for r in tr if r['epoch'] <= 7]):.4f}",
                        cfg["n_train"], cfg["n_test"], cfg["lr"], cfg.get("sched", "const"), cfg.get("aug", False), cfg.get("amp", False),
                        cfg["batch_size"], f"{np.nanmean([r['epoch_sec'] for r in tr])/60:.1f}", os.path.exists(os.path.join(args.runs, d, "done.json"))])
    with open(os.path.join(args.out, "all_iters.csv"), "w", newline="") as fh:
        w = csv.writer(fh); first = True
        for d, (cfg, rows) in runs.items():
            ip = os.path.join(args.runs, d, "iters.csv")
            if not os.path.exists(ip):
                continue
            for r in csv.DictReader(open(ip)):
                if first:
                    w.writerow(["run", "fold", "person", "recipe"] + list(r.keys())); first = False
                w.writerow([d, cfg["fold"], cfg["held_out"], cfg["recipe"]] + list(r.values()))

    # ---- summary table
    if runs:
        md.append("## 自己訓練（leave-one-person-out）\n")
        md.append("論文：15 fold 平均 **3.92°**（50 epoch）。`paper` = 論文配方（Adam 1e-5 固定、無增強）；`improved` = cosine 衰減 1e-5→1e-7 + 色彩抖動／平移／水平翻轉（翻轉時水平角取負）。\n")
        md.append("| run | 測試者 | 配方 | epoch 數 | 最佳 epoch | 最佳誤差 (°) | 最後 epoch 誤差 (°) | 1–7 epoch 平均 (°) | 每 epoch 分鐘 |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for d, (cfg, rows) in runs.items():
            tr = [r for r in rows if r["epoch"] > 0]
            k = int(np.argmin([r["test_err"] for r in tr]))
            md.append(f"| {d} | {cfg['held_out']} | {cfg['recipe']} | {tr[-1]['epoch']} | {tr[k]['epoch']} | {tr[k]['test_err']:.2f} | {tr[-1]['test_err']:.2f} | "
                      f"{np.mean([r['test_err'] for r in tr if r['epoch'] <= 7]):.2f} | {np.nanmean([r['epoch_sec'] for r in tr])/60:.1f} |")
        # paired comparison per person
        persons = sorted({cfg["held_out"] for cfg, _ in runs.values()})
        pairs = []
        for person in persons:
            by = {cfg["recipe"]: rows for cfg, rows in runs.values() if cfg["held_out"] == person}
            if "paper" in by and "improved" in by:
                bp = min(r["test_err"] for r in by["paper"] if 0 < r["epoch"] <= 7)
                bi = min(r["test_err"] for r in by["improved"] if 0 < r["epoch"] <= 7)
                lp = [r["test_err"] for r in by["paper"] if r["epoch"] == 7]; li = [r["test_err"] for r in by["improved"] if r["epoch"] == 7]
                pairs.append((person, bp, bi, lp[0] if lp else np.nan, li[0] if li else np.nan))
        if pairs:
            md.append("\n### 成對比較（同一個測試者，前 7 個 epoch）\n")
            md.append("| 測試者 | paper 最佳 | improved 最佳 | paper @ep7 | improved @ep7 |\n|---|---|---|---|---|")
            for p, bp, bi, lp, li in pairs:
                md.append(f"| {p} | {bp:.2f} | {bi:.2f} | {lp:.2f} | {li:.2f} |")
            a = np.array([(bp, bi, lp, li) for _, bp, bi, lp, li in pairs], dtype=float)
            md.append(f"| **平均（{len(pairs)} 人）** | **{np.nanmean(a[:,0]):.2f}** | **{np.nanmean(a[:,1]):.2f}** | **{np.nanmean(a[:,2]):.2f}** | **{np.nanmean(a[:,3]):.2f}** |\n")
        for d, (cfg, rows) in runs.items():
            md.append(f"<details><summary>{d} 每個 epoch</summary>\n\n| epoch | train loss | test err (°) | median | pitch MAE | yaw MAE | lr |\n|---|---|---|---|---|---|---|")
            for r in rows:
                def fmt(key, spec):
                    return "" if np.isnan(r[key]) else format(r[key], spec)
                md.append(f"| {r['epoch']} | {fmt('train_loss', '.2f')} | {r['test_err']:.3f} | {fmt('test_median', '.2f')} | "
                          f"{fmt('pitch_mae', '.2f')} | {fmt('yaw_mae', '.2f')} | {fmt('lr_end', '.1e')} |")
            md.append("\n</details>\n")

        # ---- plots: per person (paper vs improved) and all runs
        if persons:
            n = len(persons); cols = min(3, n); rws = int(np.ceil(n / cols))
            fig, axes = plt.subplots(rws, cols, figsize=(5 * cols, 3.6 * rws), facecolor="#fcfcfb", squeeze=False)
            for ax, person in zip(axes.flat, persons):
                for d, (cfg, rows) in runs.items():
                    if cfg["held_out"] != person:
                        continue
                    ax.plot([r["epoch"] for r in rows], [r["test_err"] for r in rows], color=RECIPE_COLOR.get(cfg["recipe"], INK2), lw=2, marker="o", ms=4, label=f"{cfg['recipe']} ({d})")
                ax.axhline(PAPER_MPII, color=INK2, lw=1, ls="--")
                ax.set_title(f"held-out {person}", fontsize=10); ax.set_xlabel("epoch"); ax.set_ylabel("angular error (deg)")
                ax.set_ylim(0, None); style(ax); ax.legend(frameon=False, fontsize=7, labelcolor=INK2)
            for ax in list(axes.flat)[n:]:
                ax.axis("off")
            fig.suptitle("Test angular error per epoch, paper recipe vs improved (dashed = paper's 15-fold mean 3.92°)", color=INK, fontsize=11)
            fig.tight_layout(); fig.savefig(os.path.join(args.out, "curves_per_person.png"), dpi=150); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor="#fcfcfb")
        for i, (d, (cfg, rows)) in enumerate(runs.items()):
            ls = "-" if cfg["recipe"] == "improved" else "--"
            c = SERIES[i % len(SERIES)]
            axes[0].plot([r["epoch"] for r in rows], [r["test_err"] for r in rows], color=c, lw=1.8, ls=ls, marker="o", ms=3, label=d)
            axes[1].plot([r["epoch"] for r in rows if r["epoch"] > 0], [r["train_loss"] for r in rows if r["epoch"] > 0], color=c, lw=1.8, ls=ls, marker="o", ms=3, label=d)
        axes[0].axhline(PAPER_MPII, color=INK2, lw=1, ls=":"); axes[0].set_ylim(0, None)
        axes[0].set_title("Test angular error (solid = improved, dashed = paper recipe)"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("deg")
        axes[1].set_title("Training loss (CE + MSE, pitch + yaw)"); axes[1].set_xlabel("epoch"); axes[1].set_yscale("log")
        for ax in axes:
            style(ax); ax.legend(frameon=False, fontsize=7, labelcolor=INK2, ncol=2)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "curves_all.png"), dpi=150); plt.close(fig)

    # ---- zero-shot
    zs = os.path.join(args.runs, "gaze360_zeroshot", "per_person.csv")
    if os.path.exists(zs):
        rows = list(csv.DictReader(open(zs)))
        persons_z = [r["person"] for r in rows]; e = [float(r["angular_err_deg"]) for r in rows]; nn_ = [int(r["n"]) for r in rows]
        mean_p = float(np.mean(e)); mean_img = float(np.average(e, weights=nn_))
        md.append("## Zero-shot：官方 Gaze360 權重直接測 MPIIFaceGaze（跨資料集，未在 MPII 上訓練）\n")
        md.append(f"- 15 人平均：**{mean_p:.2f}°**（逐張平均 {mean_img:.2f}°）；論文 MPIIFaceGaze 內訓練 3.92°、Gaze360 10.41°。\n")
        md.append("| 人 | 張數 | 角度誤差 (°) | label0 MAE (°) | label1 MAE (°) |\n|---|---|---|---|---|")
        for r in rows:
            md.append(f"| {r['person']} | {r['n']} | {float(r['angular_err_deg']):.2f} | {float(r['mae_label0_deg']):.2f} | {float(r['mae_label1_deg']):.2f} |")
        md.append("")
        fig, ax = plt.subplots(figsize=(9, 3.6), facecolor="#fcfcfb")
        ax.bar(persons_z, e, color=SERIES[0], width=0.6)
        ax.axhline(mean_p, color=INK2, lw=1, ls="--"); ax.text(len(persons_z) - 0.5, mean_p, f" mean {mean_p:.2f}°", va="bottom", ha="right", fontsize=8, color=INK2)
        ax.axhline(PAPER_MPII, color=SERIES[1], lw=1, ls=":"); ax.text(-0.4, PAPER_MPII, f"paper (trained on MPII) {PAPER_MPII}°", va="bottom", fontsize=8, color=SERIES[1])
        ax.set_ylabel("angular error (deg)"); ax.set_title("Gaze360-pretrained L2CS-Net, zero-shot on MPIIFaceGaze, per person")
        style(ax); fig.tight_layout(); fig.savefig(os.path.join(args.out, "zeroshot_per_person.png"), dpi=150); plt.close(fig)

    open(os.path.join(args.out, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print(f"{len(runs)} runs -> {args.out}")


if __name__ == "__main__":
    main()
