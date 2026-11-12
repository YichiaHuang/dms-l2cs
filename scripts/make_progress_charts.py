"""Charts for the final report §5 and the "06 Our Progress" slides (same look as the existing Canva deck:
white panels, deck blue + amber, gray for references). All numbers come from results/ (tables in RESULTS_FINAL.md,
results/crossdata/*.json, results/crossdata/rtgene_ft/*/metrics.csv).

  /work/<user>/envs/l2cs/bin/python -I scripts/make_progress_charts.py   -> results/final/figs/*.png
"""
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "final", "figs")
BLUE, AMBER, GRAY, INK, MUTED = "#0B4AA8", "#F0A30A", "#A7B1BC", "#1F2933", "#5F6B7A"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": "#C9D1DA", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#E6EAEE", "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white"})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); print("wrote", name)


def hbars(ax, labels, vals, colors, fmt="{:.2f}°", hatches=None):
    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, vals, color=colors, height=0.62, edgecolor="white", linewidth=2)
    for b, v, h in zip(bars, vals, hatches or [None] * len(vals)):
        if h:
            b.set_hatch(h); b.set_edgecolor("white")
        ax.text(v + max(vals) * 0.01, b.get_y() + b.get_height() / 2, fmt.format(v), va="center", color=INK, fontsize=12)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.grid(axis="y", visible=False)


def fig_reproduction():
    fig, ax = plt.subplots(figsize=(9, 4.6))
    labels = ["Paper (reported)", "Ours: paper recipe, last epoch", "Ours: paper recipe, best epoch*", "Ours: improved recipe",
              "MobileNetV3-L + KD (fp32)", "MobileNetV3-L + KD (int8 QAT)"]
    vals = [3.92, 4.70, 3.53, 4.15, 4.01, 4.15]
    hbars(ax, labels, vals, [GRAY, BLUE, "#9DB4DD", BLUE, AMBER, AMBER], hatches=[None, None, "//", None, None, None])
    ax.set_xlabel("Mean angular error (°), MPIIFaceGaze, 15-fold leave-one-person-out"); ax.set_xlim(0, 5.4)
    fig.text(0.01, -0.04, "* picked on the test subject: optimistic, not a fair result", color=MUTED, fontsize=10.5)
    save(fig, "p1_reproduction.png")


def fig_ablation():
    fig, ax = plt.subplots(figsize=(7.5, 4))
    hbars(ax, ["Paper recipe", "Cosine LR only", "Augmentation only", "Cosine + augmentation"], [4.33, 4.55, 3.99, 4.09],
          [GRAY, GRAY, BLUE, BLUE])
    ax.set_xlabel("Error at epoch 50 (°), 5 subjects"); ax.set_xlim(0, 5.1)
    save(fig, "p2_ablation.png")


def fig_backbones():
    models = ["ResNet-50\n16.35 GMACs", "EfficientNet-B0\n1.54 GMACs", "MobileNetV3-L + KD\n0.85 GMACs"]
    fp32, ptq, qat = [4.11, 4.01, 4.18], [3.86, 7.70, 4.49], [np.nan, 4.98, 4.13]
    x = np.arange(3); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, (v, c, lab) in enumerate([(fp32, BLUE, "fp32"), (ptq, GRAY, "int8 PTQ"), (qat, AMBER, "int8 QAT")]):
        bs = ax.bar(x + (i - 1) * w, v, w * 0.92, color=c, label=lab)
        for b, val in zip(bs, v):
            if not np.isnan(val):
                ax.text(b.get_x() + b.get_width() / 2, val + 0.1, f"{val:.2f}", ha="center", fontsize=11, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel("Mean error (°), 5 subjects"); ax.set_ylim(0, 9.5)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12)); ax.grid(axis="x", visible=False)
    save(fig, "p3_backbones_int8.png")


def fig_cost():
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for g, lab in [(2.9, "1.1 TOPS chip"), (5.3, "2 TOPS chip"), (21, "8 TOPS chip")]:
        ax.axvline(g, color=GRAY, ls="--", lw=1); ax.text(g * 1.04, 4.42, lab, color=MUTED, fontsize=10.5, rotation=90, va="top")
    pts = [("ResNet-50", 16.35, 4.15, BLUE), ("EfficientNet-B0", 1.54, 4.09, GRAY), ("MobileNetV3-L + KD", 0.85, 4.01, AMBER),
           ("MobileNetV3-L + KD int8", 0.85, 4.15, AMBER)]
    for lab, g, e, c in pts:
        ax.scatter(g, e, s=110, color=c, edgecolor="white", linewidth=2, zorder=3, marker="s" if "int8" in lab else "o")
        off = {"ResNet-50": (8, 6), "EfficientNet-B0": (8, -16), "MobileNetV3-L + KD": (8, -16), "MobileNetV3-L + KD int8": (8, 6)}[lab]
        ax.annotate(lab, (g, e), xytext=off, textcoords="offset points", fontsize=11, color=INK)
    ax.set_xscale("log"); ax.set_xlim(0.5, 40); ax.set_ylim(3.8, 4.45)
    ax.set_xlabel("Compute per face (GMACs, log)   · dashed = budget per frame at 25 fps"); ax.set_ylabel("Error (°), 15 folds")
    save(fig, "p4_cost.png")


def fig_crossdata():
    rows = ["MPII → ResNet-50", "MPII → MobileNetV3-L + KD", "Gaze360 → ResNet-50", "Gaze360 + MPII → MobileNetV3-L (J1)"]
    cols = ["MPIIFaceGaze", "Gaze360", "RT-GENE"]
    v = np.array([[4.15, 29.3, 15.0], [4.01, 31.0, 15.0], [8.38, 11.04, 28.6], [5.27, 11.57, 28.9]])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.imshow(v, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#EAF0FA", "#5B83C9", BLUE]), vmin=0, vmax=32, aspect="auto")
    for i in range(4):
        for j in range(3):
            ax.text(j, i, f"{v[i, j]:.1f}°", ha="center", va="center", fontsize=14, color="white" if v[i, j] > 16 else INK, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(cols); ax.set_yticks(range(4)); ax.set_yticklabels(rows)
    ax.xaxis.tick_top(); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Test set  (MPII: unseen persons; others: test split / all images)", color=MUTED, fontsize=10.5)
    save(fig, "p5_crossdata.png")


def last(run):
    rows = list(csv.DictReader(open(run + "/metrics.csv")))
    return {k: float(rows[-1][k]) for k in ("rtgene_mean", "mpii_person_mean", "gaze360_mean")}


def fig_fewshot():
    base = os.path.join(ROOT, "results", "crossdata", "rtgene_ft")
    # (a) amount of target data, fold 0
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    n = [110, 1100, 79819]; r = [last(f"{base}/J1_f0_n{k}")["rtgene_mean"] for k in (10, 100, 0)]
    ax.plot([20] + n, [29.71] + r, color=BLUE, lw=2, marker="o", ms=9, mec="white", mew=2)
    ax.axhline(15.71, color=GRAY, ls="--", lw=1.2); ax.text(2500, 16.4, "current car model (MPII only): 15.7°", color=MUTED, fontsize=10.5)
    for x, y in zip([20] + n, [29.71] + r):
        ax.annotate(f"{y:.1f}°", (x, y), xytext=(0, -20), textcoords="offset points", ha="center", fontsize=11)
    ax.set_xscale("log"); ax.set_xticks([20] + n); ax.set_xticklabels(["0", "110\n(10/person)", "1,100\n(100/person)", "79,819\n(all)"])
    ax.set_ylabel("RT-GENE error, unseen people (°)"); ax.set_ylim(0, 33); ax.set_xlabel("RT-GENE images used for fine-tuning (subject fold 0)")
    save(fig, "p6_fewshot_amount.png")
    # (b) trade-off, mean over the 3 subject folds
    sets = {"n10": "10/person", "n10_replay50": "10/person + 50% replay", "n10_replay25": "10/person + 25% replay",
            "n100_replay50": "100/person + 50% replay", "n100_replay25": "100/person + 25% replay"}
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.scatter(28.59, 11.57, s=120, color=GRAY, edgecolor="white", lw=2, zorder=3); ax.annotate("J1 before", (28.59, 11.57), xytext=(0, -26), textcoords="offset points", ha="center")
    ax.scatter(14.76, 31.0, s=120, color=GRAY, marker="s", edgecolor="white", lw=2, zorder=3); ax.annotate("current car model", (14.76, 31.0), xytext=(8, -4), textcoords="offset points")
    for k, lab in sets.items():
        rs = [f"{base}/J1_f{f}_{k}" for f in range(3)]
        if not all(os.path.exists(r + "/done.json") for r in rs):
            continue
        m = np.mean([[last(r)["rtgene_mean"], last(r)["gaze360_mean"]] for r in rs], 0)
        c = AMBER if "replay" in k else BLUE
        ax.scatter(*m, s=120, color=c, edgecolor="white", lw=2, zorder=3)
        # replay points sit close together: fan labels above/below with leader lines
        pos = {"n10": (8, 4, "left"), "n100_replay25": (10.3, 17.0, "left"), "n100_replay50": (10.3, 6.0, "left"),
               "n10_replay25": (18.0, 17.0, "left"), "n10_replay50": (18.0, 6.0, "left")}
        tx, ty, ha = pos[k]
        txt = f"{lab}\n({m[0]:.1f}°, {m[1]:.1f}°)"
        if k == "n10":
            ax.annotate(txt, m, xytext=(tx, ty), textcoords="offset points", fontsize=10.5)
        else:
            ax.annotate(txt, m, xytext=(tx, ty), ha=ha, va="center", fontsize=10.5, bbox=dict(fc="white", ec="none", pad=1),
                        arrowprops=dict(arrowstyle="-", color=GRAY, lw=1))
    ax.axvline(13, color=GRAY, ls="--", lw=1); ax.text(13.1, 33, "target ≤ 13°", fontsize=10, color="#5b6573", va="top")
    ax.axhline(11.57 + 0.5, color=GRAY, ls=":", lw=1); ax.text(24.0, 10.6, "no-forgetting limit (+0.5°)", fontsize=10, color="#5b6573", ha="right", va="top")
    ax.set_xlabel("RT-GENE error, unseen people (°)  → lower is better"); ax.set_ylabel("Gaze360 error (°)  ↓ forgetting")
    ax.set_xlim(10, 31); ax.set_ylim(3, 34)
    save(fig, "p7_fewshot_tradeoff.png")


def fig_widerange():
    ds = ["MPII (5 unseen persons)", "Gaze360 test", "RT-GENE"]
    vals = {"Current car model (MPII only)": ([4.18, 31.0, 15.0], GRAY), "G2: Gaze360 only": ([8.41, 11.71, 28.9], "#9DB4DD"),
            "J1: Gaze360 + 10 MPII persons": ([5.27, 11.57, 28.9], BLUE)}
    x = np.arange(3); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, (lab, (v, c)) in enumerate(vals.items()):
        bs = ax.bar(x + (i - 1) * w, v, w * 0.92, color=c, label=lab)
        for b, val in zip(bs, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.5, f"{val:.1f}", ha="center", fontsize=11, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(ds); ax.set_ylabel("Mean error (°)"); ax.set_ylim(0, 36)
    ax.legend(frameon=False, loc="upper left", fontsize=11); ax.grid(axis="x", visible=False)
    save(fig, "p8_widerange.png")


def fold_mean(base, key):
    rs = [f"{base}/J1_f{f}_{key}" for f in range(3)]
    if not all(os.path.exists(r + "/done.json") for r in rs):
        return None
    return {k: float(np.mean([last(r)[k] for r in rs])) for k in ("rtgene_mean", "gaze360_mean", "mpii_person_mean")}


def fig_replay():
    base = os.path.join(ROOT, "results", "crossdata", "rtgene_ft")
    rows = [("J1 before fine-tuning", {"rtgene_mean": 28.59, "gaze360_mean": 11.57, "mpii_person_mean": 5.27}, GRAY),
            ("Car model (MPII only)", {"rtgene_mean": 14.76, "gaze360_mean": 31.0, "mpii_person_mean": float("nan")}, GRAY)]
    for key, lab in [("n10", "10/person, no replay"), ("n10_replay25", "10/person + 25% replay"), ("n10_replay50", "10/person + 50% replay"),
                     ("n100_replay25", "100/person + 25% replay"), ("n100_replay50", "100/person + 50% replay")]:
        m = fold_mean(base, key)
        if m:
            rows.append((lab, m, BLUE if "no replay" in lab else AMBER))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
    for ax, (k, title, lim) in zip(axes, [("rtgene_mean", "RT-GENE, unseen people (target)", 33),
                                          ("gaze360_mean", "Gaze360 test (forgetting?)", 36),
                                          ("mpii_person_mean", "MPII, 5 unseen persons (forgetting?)", 8)]):
        y = np.arange(len(rows))[::-1]
        v = [r[1][k] for r in rows]
        ax.barh(y, np.nan_to_num(v), color=[r[2] for r in rows], height=0.62, edgecolor="white", linewidth=2)
        for yy, vv in zip(y, v):
            ax.text((0 if np.isnan(vv) else vv) + lim * 0.01, yy, "n/a" if np.isnan(vv) else f"{vv:.1f}°", va="center", fontsize=11, color=INK)
        ax.set_title(title, fontsize=13, color=INK, loc="left"); ax.set_xlim(0, lim); ax.grid(axis="y", visible=False)
        ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    fig.text(0.01, -0.03, "Mean of 3 subject folds, error at the last iteration (°). Lower is better.", color=MUTED, fontsize=11)
    fig.tight_layout()
    save(fig, "p9_replay.png")


def fig_convergence():
    """Training convergence, MPII 15 folds x 50 epochs: test error and training loss, 15-person mean per epoch."""
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "remote", "all_epochs.csv"))))
    rows = [r for r in rows if r["run"].endswith("_e50")]
    def curve(recipe, key):
        ep = sorted({int(r["epoch"]) for r in rows if r["recipe"] == recipe and int(r["epoch"]) >= 1})
        m = [np.mean([float(r[key]) for r in rows if r["recipe"] == recipe and int(r["epoch"]) == e]) for e in ep]
        n = {len([r for r in rows if r["recipe"] == recipe and int(r["epoch"]) == e]) for e in ep}
        assert n == {15}, (recipe, key, n)
        return np.array(ep), np.array(m)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.4))
    for recipe, c, lab in [("paper", AMBER, "Paper recipe"), ("improved", BLUE, "Improved recipe")]:
        x, y = curve(recipe, "test_err")
        a1.plot(x, y, color=c, lw=2.2, label=lab)
        a1.scatter([x[-1]], [y[-1]], s=60, color=c, edgecolor="white", lw=2, zorder=3)
        a1.annotate(f"{y[-1]:.2f}°", (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=12, color=INK)
        if recipe == "paper":
            i = int(np.argmin(y))
            a1.scatter([x[i]], [y[i]], s=60, color=c, edgecolor="white", lw=2, zorder=3)
            a1.annotate(f"lowest: epoch {x[i]}, {y[i]:.2f}°", (x[i], y[i]), xytext=(13, 3.78), textcoords="data", fontsize=11, color=INK,
                        arrowprops=dict(arrowstyle="-", color=GRAY, lw=1))
        x, y = curve(recipe, "train_loss")
        a2.plot(x, y, color=c, lw=2.2, label=lab)
        a2.annotate(f"{y[-1]:.1f}", (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=12, color=INK)
    a1.axhline(3.92, color=GRAY, ls="--", lw=1.2); a1.text(53, 3.89, "paper reported: 3.92°", color=MUTED, fontsize=10.5, va="top", ha="right")
    a1.set_ylim(3.7, 4.9); a1.set_xlim(0, 54); a1.set_xlabel("epoch"); a1.set_ylabel("test angular error (°)")
    a1.set_title("Test error on the held-out person", loc="left", fontsize=13, color=INK); a1.legend(frameon=False, loc="upper left")
    a2.set_yscale("log"); a2.set_xlim(0, 54); a2.set_xlabel("epoch"); a2.set_ylabel("training loss (CE + MSE, log scale)")
    a2.set_title("Training loss", loc="left", fontsize=13, color=INK); a2.legend(frameon=False, loc="upper right")
    fig.text(0.01, -0.03, "MPIIFaceGaze, ResNet-50, mean of 15 leave-one-person-out folds per epoch. Epoch 0 (before training, 11.1°) not shown.",
             color=MUTED, fontsize=11)
    fig.tight_layout()
    save(fig, "p10_convergence.png")


if __name__ == "__main__":
    fig_reproduction(); fig_ablation(); fig_backbones(); fig_cost(); fig_crossdata(); fig_widerange(); fig_fewshot(); fig_replay(); fig_convergence()
