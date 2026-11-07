"""Charts for the inference-speed slide, from results/speed/*.csv|json."""
import csv, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

S, OUT, FD = sys.argv[1], sys.argv[2], sys.argv[3]
fm.fontManager.addfont(f"{FD}/arimo.ttf")
plt.rcParams.update({"font.family": "Arimo", "font.size": 15})
BLUE, MID, AMBER, INK, MUTED, GRID, CARD = "#004aad", "#5b8fd9", "#eca406", "#1a1a1a", "#6b7280", "#d4d4d4", "#f5f5f5"

def style(ax, grid="x"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#9ca3af")
    ax.tick_params(colors=MUTED, labelsize=14, length=0); ax.set_facecolor(CARD)
    ax.grid(axis=grid, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)

o = json.load(open(f"{S}/summary_optimized.json"))
# --- per-frame breakdown, stacked horizontal bars (2px surface gap via edgecolor) ---
f = plt.figure(figsize=(9, 3.9), dpi=200, facecolor=CARD); ax = f.add_subplot(); style(ax)
rows = [("Original", o["original"]), ("GPU preprocessing", o["gpu_prep"])]
parts = [("detect_ms", "Face detection", BLUE), ("preprocess_ms", "Preprocessing", AMBER), ("gaze_ms", "Gaze model", MID)]
y = np.arange(len(rows))[::-1]
for (name, r), yy in zip(rows, y):
    left = 0
    for k, lab, c in parts:
        ax.barh(yy, r[k], 0.55, left=left, color=c, edgecolor=CARD, linewidth=2, label=lab if yy == y[0] else None)
        if r[k] > 6: ax.text(left + r[k] / 2, yy, f"{r[k]:.0f}", ha="center", va="center", fontsize=14, color="#f5f5f5" if c != AMBER else INK)
        left += r[k]
    ax.text(left + 1.5, yy, f"{r['total_ms']:.0f} ms · {r['fps']:.1f} FPS", va="center", fontsize=15, color=INK, fontweight="bold")
ax.set_yticks(y, [n for n, _ in rows], fontsize=15, color=INK); ax.set_xlim(0, 125)
ax.set_xlabel("Time per 1280×720 frame, 1 face (ms)", fontsize=14)
ax.legend(frameon=False, fontsize=14, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.2), labelcolor=INK)
f.savefig(f"{OUT}/s1_pipeline_breakdown.png", facecolor=CARD, bbox_inches="tight", pad_inches=0.15); plt.close(f)

# --- model only throughput ---
m = list(csv.DictReader(open(f"{S}/model_only.csv")))
f = plt.figure(figsize=(7, 3.9), dpi=200, facecolor=CARD); ax = f.add_subplot(); style(ax, grid="y")
bs = [1, 4, 16]; x = np.arange(3); w = 0.36
for i, (prec, c) in enumerate((("fp32", MID), ("bf16", BLUE))):
    v = [float(next(r["img_per_s"] for r in m if r["device"] == "GPU" and r["precision"] == prec and int(r["batch"]) == b)) for b in bs]
    bars = ax.bar(x + (i - 0.5) * w, v, w * 0.92, color=c, label=f"GPU {prec}")
    for b_, val in zip(bars, v): ax.text(b_.get_x() + b_.get_width() / 2, val + 4, f"{val:.0f}", ha="center", fontsize=13, color=INK)
ax.set_xticks(x, [f"batch {b}" for b in bs], fontsize=14, color=INK); ax.set_ylabel("Faces per second", fontsize=14); ax.set_ylim(0, 380)
ax.legend(frameon=False, fontsize=14, loc="upper left", labelcolor=INK, ncol=2)
f.savefig(f"{OUT}/s2_model_throughput.png", facecolor=CARD, bbox_inches="tight", pad_inches=0.15); plt.close(f)
print("ok")
