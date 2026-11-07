import csv, json, os, sys, math, random
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image, ImageDraw, ImageOps, ImageEnhance

OUT = sys.argv[1]; FD = sys.argv[2]
R = os.path.expanduser("~/runs/l2cs"); D = os.path.expanduser("~/datasets/MPIIFaceGaze")
fm.fontManager.addfont(f"{FD}/arimo.ttf")
plt.rcParams.update({"font.family": "Arimo", "font.size": 15, "axes.titlesize": 16})
BLUE, MID, AMBER, INK, MUTED, GRID, CARD = "#004aad", "#5b8fd9", "#eca406", "#1a1a1a", "#6b7280", "#d4d4d4", "#f5f5f5"
SUBJ = ["p00", "p03", "p07", "p11", "p14"]
FOLD = {"p00": 0, "p03": 3, "p07": 7, "p11": 11, "p14": 14}
paper_fig2 = json.load(open(os.path.join(os.path.dirname(OUT), "..", "paper_fig2_l2cs.json")))

def style(ax, grid="y"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#9ca3af")
    ax.tick_params(colors=MUTED, labelsize=14, length=0)
    if grid: ax.grid(axis=grid, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_facecolor(CARD)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
def fig(w, h): f = plt.figure(figsize=(w, h), dpi=200, facecolor=CARD); return f
def save(f, name): f.savefig(f"{OUT}/{name}.png", facecolor=CARD, bbox_inches="tight", pad_inches=0.15); plt.close(f)

def metrics(run):
    rows = list(csv.DictReader(open(f"{R}/{run}/metrics.csv")))
    out = {}
    for r in rows:
        e = int(r["epoch"])
        loss = float(r["train_loss_total"]) if r.get("train_loss_total") else (float(r["train_loss_pitch"]) + float(r["train_loss_yaw"]) if r.get("train_loss_pitch") else np.nan)
        out[e] = dict(err=float(r["test_angular_err"]), loss=loss)
    return out
PAPER_RUN = {"p00": "fold0_e15", "p03": "fold3_paper_e7", "p07": "fold7_e10", "p11": "fold11_paper_e7", "p14": "fold14_paper_e7"}
IMP_RUN = {s: f"fold{FOLD[s]}_improved_e7" for s in SUBJ}
mp = {s: metrics(PAPER_RUN[s]) for s in SUBJ}; mi = {s: metrics(IMP_RUN[s]) for s in SUBJ}
nums = {}
nums["paper_fig2_5"] = float(np.mean([paper_fig2[s] for s in SUBJ]))
nums["ours_paper_ep7"] = {s: mp[s][7]["err"] for s in SUBJ}; nums["ours_imp_ep7"] = {s: mi[s][7]["err"] for s in SUBJ}
nums["ours_paper_best"] = {s: min(v["err"] for e, v in mp[s].items() if 0 < e <= 7) for s in SUBJ}
nums["ours_imp_best"] = {s: min(v["err"] for e, v in mi[s].items() if 0 < e <= 7) for s in SUBJ}
for k in ["ours_paper_ep7", "ours_imp_ep7", "ours_paper_best", "ours_imp_best"]: nums[k + "_mean"] = float(np.mean(list(nums[k].values())))
nums["std_ep3_7_paper"] = {s: float(np.std([mp[s][e]["err"] for e in range(3, 8)])) for s in SUBJ}
nums["std_ep3_7_imp"] = {s: float(np.std([mi[s][e]["err"] for e in range(3, 8)])) for s in SUBJ}
nums["gap_last_best_paper"] = float(np.mean([nums["ours_paper_ep7"][s] - nums["ours_paper_best"][s] for s in SUBJ]))
nums["gap_last_best_imp"] = float(np.mean([nums["ours_imp_ep7"][s] - nums["ours_imp_best"][s] for s in SUBJ]))
nums["loss_ep7_paper"] = float(np.mean([mp[s][7]["loss"] for s in SUBJ])); nums["loss_ep7_imp"] = float(np.mean([mi[s][7]["loss"] for s in SUBJ]))

# ---- C1 per-subject vs paper (grouped bars) ----
f = fig(11, 5.2); ax = f.add_subplot(); style(ax)
cats = SUBJ + ["Mean (5)"]
vals = [[paper_fig2[s] for s in SUBJ] + [nums["paper_fig2_5"]],
        [nums["ours_paper_ep7"][s] for s in SUBJ] + [nums["ours_paper_ep7_mean"]],
        [nums["ours_imp_ep7"][s] for s in SUBJ] + [nums["ours_imp_ep7_mean"]]]
labels = ["Paper (Fig. 2, 50 ep)", "Ours: paper recipe (7 ep)", "Ours: improved (7 ep)"]
cols = [AMBER, MID, BLUE]; x = np.arange(len(cats)); w = 0.26
for i, (v, c, l) in enumerate(zip(vals, cols, labels)):
    bars = ax.bar(x + (i - 1) * w, v, w * 0.92, color=c, label=l)
    for b, val in zip(bars, v): ax.text(b.get_x() + b.get_width() / 2, val + 0.08, f"{val:.2f}", ha="center", va="bottom", fontsize=12, color=INK)
ax.axvspan(4.55, 5.45, color="#e5e7eb", zorder=0)
ax.set_xticks(x, cats, fontsize=15, color=INK); ax.set_ylabel("Mean angular error (°)", fontsize=15); ax.set_ylim(0, 7.6)
ax.legend(frameon=False, fontsize=14, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.1), labelcolor=INK)
save(f, "c1_per_subject_vs_paper")

# ---- C2 paper fig2 15 subjects, highlight ours ----
f = fig(11, 4.6); ax = f.add_subplot(); style(ax)
allp = [f"p{i:02d}" for i in range(15)]
v = [paper_fig2[s] for s in allp]
c = [BLUE if s in SUBJ else "#b9c3d3" for s in allp]
bars = ax.bar(allp, v, 0.7, color=c)
for b, s, val in zip(bars, allp, v):
    if s in SUBJ: ax.text(b.get_x() + b.get_width() / 2, val + 0.08, f"{val:.2f}", ha="center", fontsize=13, color=INK)
ax.axhline(3.92, color=AMBER, lw=2.2, ls="--", label="paper mean, all 15 subjects: 3.92°")
ax.axhline(nums["paper_fig2_5"], color=BLUE, lw=2.2, ls=":", label=f"paper mean, our 5 subjects: {nums['paper_fig2_5']:.2f}°")
ax.legend(frameon=False, fontsize=14, loc="upper left", labelcolor=INK)
ax.set_ylabel("Paper's error (°)", fontsize=15); ax.set_ylim(0, 7.4); ax.tick_params(axis="x", colors=INK, labelsize=14)
save(f, "c2_paper_subjects_highlight")

# ---- C3 paper recipe learning curves (5 subjects) ----
f = fig(10, 5.2); ax = f.add_subplot(); style(ax)
SC = {"p00": "#004aad", "p03": "#eca406", "p07": "#1baf7a", "p11": "#e87ba4", "p14": "#6250d6"}
for s in SUBJ:
    es = sorted(e for e in mp[s] if e <= 7)
    ax.plot(es, [mp[s][e]["err"] for e in es], color=SC[s], lw=2.4, marker="o", ms=6)
    ax.text(7.15, mp[s][7]["err"] + {"p00": -0.22, "p03": 0.18, "p07": -0.05, "p14": -0.3, "p11": 0.32}[s], s, va="center", fontsize=14, color=INK)
ax.axhline(3.92, color=MUTED, lw=1.5, ls="--", label="paper, 15-subject mean 3.92°"); ax.legend(frameon=False, fontsize=14, loc="upper right", labelcolor=INK)
ax.set_xlabel("Epoch (0 = ImageNet init, before training)", fontsize=15); ax.set_ylabel("Test angular error (°)", fontsize=15)
ax.set_xlim(-0.2, 7.7); ax.set_ylim(0, 12.5); ax.set_xticks(range(8))
save(f, "c3_paper_recipe_curves")

# ---- C3b fold0 15 epochs ----
f = fig(10, 4.4); ax = f.add_subplot(); style(ax)
es = sorted(mp["p00"]); ax.plot(es[1:], [mp["p00"][e]["err"] for e in es[1:]], color=BLUE, lw=2.4, marker="o", ms=6)
ax.axhline(2.38, color=AMBER, lw=2, ls="--"); ax.text(15, 2.32, "paper p00: 2.38°", ha="right", va="top", fontsize=13, color=INK)
k = min(es[1:], key=lambda e: mp["p00"][e]["err"])
ax.text(k, 2.495, f"best {mp['p00'][k]['err']:.2f}° (ep {k})", ha="center", va="center", fontsize=13, color=INK)
ax.set_xlabel("Epoch", fontsize=15); ax.set_ylabel("p00 test error (°)", fontsize=15); ax.set_ylim(2.0, 3.8); ax.set_xticks(range(1, 16))
save(f, "c3b_p00_15ep")

# ---- C4 LR schedule ----
def iters(run):
    rows = list(csv.DictReader(open(f"{R}/{run}/iters.csv"))); return [int(r["global_step"]) for r in rows], [float(r["lr"]) for r in rows]
f = fig(7, 4.2); ax = f.add_subplot(); style(ax)
sx, sy = iters("fold3_paper_e7"); ax.plot(np.array(sx) / 2171, np.array(sy) * 1e5, color=MID, lw=2.6, label="paper: constant 1e-5")
sx, sy = iters("fold3_improved_e7"); ax.plot(np.array(sx) / 2171, np.array(sy) * 1e5, color=BLUE, lw=2.6, label="improved: cosine → 1e-7")
ax.set_xlabel("Epoch", fontsize=15); ax.set_ylabel("Learning rate (×1e-5)", fontsize=15); ax.set_ylim(0, 1.15)
ax.legend(frameon=False, fontsize=14, loc="lower left", labelcolor=INK)
save(f, "c4_lr")

# ---- C5 augmentation strip ----
lines = open(f"{D}/Label/p01.label").read().splitlines()[1:]
ln = lines[1200].split(); img = Image.open(f"{D}/Image/{ln[0]}").convert("RGB"); yaw, pitch = map(float, ln[7].split(","))
def arrow(im, yaw, pitch):
    im = im.resize((448, 448)); d = ImageDraw.Draw(im); c = (224, 224)
    dx = -math.sin(yaw) * math.cos(pitch) * 150; dy = -math.sin(pitch) * 150
    end = (c[0] + dx, c[1] + dy); d.line([c, end], fill=(236, 164, 6), width=9)
    ang = math.atan2(dy, dx)
    for s in (2.6, -2.6): d.line([end, (end[0] + 30 * math.cos(ang + s), end[1] + 30 * math.sin(ang + s))], fill=(236, 164, 6), width=9)
    return im
random.seed(3)
jit = ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(1.3)).enhance(0.75)
shift = Image.new("RGB", img.size, (0, 0, 0)); sh = img.resize((int(224 * 1.05),) * 2); shift.paste(sh, (-14, 4))
flip = ImageOps.mirror(img)
panels = [arrow(img, yaw, pitch), arrow(jit, yaw, pitch), arrow(shift, yaw, pitch), arrow(flip, -yaw, pitch)]
strip = Image.new("RGB", (448 * 4 + 24 * 3, 448), CARD)
for i, p in enumerate(panels): strip.paste(p, (i * (448 + 24), 0))
strip.save(f"{OUT}/c5_aug_strip.png")
nums["aug_example"] = dict(face=ln[0], yaw=math.degrees(yaw), pitch=math.degrees(pitch))

# ---- C6 small multiples paper vs improved ----
f = plt.figure(figsize=(15, 4.6), dpi=200, facecolor=CARD)
for i, s in enumerate(SUBJ):
    ax = f.add_subplot(1, 5, i + 1); style(ax)
    for m, c, l in ((mp[s], MID, "paper recipe"), (mi[s], BLUE, "improved")):
        es = sorted(e for e in m if 1 <= e <= 7); ax.plot(es, [m[e]["err"] for e in es], color=c, lw=2.6, marker="o", ms=5, label=l)
    ax.axhline(paper_fig2[s], color=AMBER, lw=2, ls="--", label="paper (Fig. 2)")
    ax.set_title(s, fontsize=17, color=INK, pad=6); ax.set_xticks([1, 3, 5, 7]); ax.set_ylim(1.5, 7.2)
    if i == 0: ax.set_ylabel("Test error (°)", fontsize=15)
    ax.set_xlabel("Epoch", fontsize=14)
h, l = ax.get_legend_handles_labels(); f.legend(h, l, frameon=False, ncol=3, fontsize=15, loc="upper center", bbox_to_anchor=(0.5, 1.07), labelcolor=INK)
f.tight_layout(); save(f, "c6_small_multiples")

# ---- C7 paired difference (improved - paper) at ep7, diverging ----
f = fig(8, 4.6); ax = f.add_subplot(); style(ax, grid="x")
diff = [nums["ours_imp_ep7"][s] - nums["ours_paper_ep7"][s] for s in SUBJ]; mean_d = float(np.mean(diff))
names = SUBJ + ["Mean"]; dv = diff + [mean_d]
colors = [BLUE if d < 0 else AMBER for d in dv]
y = np.arange(len(names))[::-1]
ax.barh(y, dv, 0.62, color=colors)
for yy, d in zip(y, dv): ax.text(d + (0.06 if d >= 0 else -0.06), yy, f"{d:+.2f}°  {'better' if d < 0 else 'worse'}", va="center", ha="left" if d >= 0 else "right", fontsize=14, color=INK)
ax.axvline(0, color="#6b7280", lw=1.2); ax.set_yticks(y, names, fontsize=15, color=INK); ax.set_xlim(-1.6, 1.6)
ax.set_xlabel("Improved − paper recipe, error at epoch 7 (°)", fontsize=14)
save(f, "c7_paired_diff"); nums["diff_ep7"] = dict(zip(SUBJ, diff)); nums["diff_ep7_mean"] = mean_d

# ---- C8 train loss mean paper vs improved ----
f = fig(7, 4.4); ax = f.add_subplot(); style(ax)
for m, c, l in ((mp, MID, "paper recipe"), (mi, BLUE, "improved (with augmentation)")):
    ys = [np.mean([m[s][e]["loss"] for s in SUBJ]) for e in range(1, 8)]
    ax.plot(range(1, 8), ys, color=c, lw=2.6, marker="o", ms=6, label=l); ax.text(7.15, ys[-1], f"{ys[-1]:.1f}", va="center", fontsize=14, color=INK)
ax.set_xlabel("Epoch", fontsize=15); ax.set_ylabel("Train loss (CE + MSE)", fontsize=15); ax.set_xlim(0.7, 7.8)
ax.legend(frameon=False, fontsize=14, labelcolor=INK)
save(f, "c8_train_loss")

# ---- C9 zero-shot vs paper per subject (15) ----
zs = {r["person"]: float(r["angular_err_deg"]) for r in csv.DictReader(open(f"{R}/gaze360_zeroshot/per_person.csv"))}
zsr = {r["person"]: (float(r["mae_label0_deg"]), float(r["mae_label1_deg"])) for r in csv.DictReader(open(f"{R}/gaze360_zeroshot/per_person.csv"))}
f = fig(12, 4.8); ax = f.add_subplot(); style(ax)
x = np.arange(15); w = 0.38
ax.bar(x - w / 2, [zs[s] for s in allp], w * 0.92, color="#9aa5b8", label="Gaze360 weights, zero-shot")
ax.bar(x + w / 2, [paper_fig2[s] for s in allp], w * 0.92, color=AMBER, label="Trained on MPIIFaceGaze (paper)")
zm = float(np.mean(list(zs.values())))
ax.axhline(zm, color=MUTED, lw=1.6, ls="--"); ax.text(14.6, zm + 0.12, f"zero-shot mean {zm:.2f}°", ha="right", fontsize=14, color=INK)
ax.set_xticks(x, allp, fontsize=14, color=INK); ax.set_ylabel("Angular error (°)", fontsize=15); ax.set_ylim(0, 13)
ax.legend(frameon=False, fontsize=14, ncol=2, loc="upper left", labelcolor=INK)
save(f, "c9_zeroshot"); nums["zs_mean"] = zm
nums["zs_h_mae"] = float(np.mean([v[0] for v in zsr.values()])); nums["zs_v_mae"] = float(np.mean([v[1] for v in zsr.values()]))

# ---- C10 pred vs GT scatter (improved ep7, 5 subjects) ----
gx, gy, px, py, errs = [], [], [], [], []
for s in SUBJ:
    for r in csv.DictReader(open(f"{R}/{IMP_RUN[s]}/predictions/epoch_7.csv")):
        gx.append(float(r["gt_pitch_deg"])); gy.append(float(r["gt_yaw_deg"])); px.append(float(r["pred_pitch_deg"])); py.append(float(r["pred_yaw_deg"])); errs.append(float(r["angular_err_deg"]))
gx, gy, px, py, errs = map(np.array, (gx, gy, px, py, errs))
f = plt.figure(figsize=(11, 4.9), dpi=200, facecolor=CARD)
for i, (g, p, name) in enumerate(((gx, px, "Horizontal (yaw)"), (gy, py, "Vertical (pitch)"))):
    ax = f.add_subplot(1, 2, i + 1); style(ax, grid=None)
    ax.hexbin(g, p, gridsize=45, cmap="Blues", mincnt=1, bins="log", extent=(-35, 35, -35, 35))
    ax.plot([-35, 35], [-35, 35], color=AMBER, lw=1.8, ls="--")
    ax.set_xlim(-35, 35); ax.set_ylim(-35, 35); ax.set_aspect("equal")
    ax.set_title(f"{name}   MAE {np.mean(np.abs(p - g)):.2f}°", fontsize=17, color=INK)
    ax.set_xlabel("Ground truth (°)", fontsize=14); ax.set_ylabel("Prediction (°)", fontsize=14)
f.tight_layout(); save(f, "c10_scatter")
nums["h_mae"] = float(np.mean(np.abs(px - gx))); nums["v_mae"] = float(np.mean(np.abs(py - gy)))
nums["h_bias"] = float(np.mean(px - gx)); nums["v_bias"] = float(np.mean(py - gy))
nums["err_median"] = float(np.median(errs)); nums["err_p90"] = float(np.percentile(errs, 90)); nums["n_pred"] = int(len(errs))
nums["corr_h"] = float(np.corrcoef(gx, px)[0, 1]); nums["corr_v"] = float(np.corrcoef(gy, py)[0, 1])

# ---- C11 error histogram ----
f = fig(7, 4.4); ax = f.add_subplot(); style(ax)
ax.hist(errs, bins=np.arange(0, 16, 0.5), color=BLUE, rwidth=0.9)
ax.axvline(np.median(errs), color=AMBER, lw=2.2, ls="--"); ax.text(np.median(errs) + 0.3, ax.get_ylim()[1] * 0.9, f"median {np.median(errs):.2f}°", fontsize=14, color=INK)
ax.axvline(np.percentile(errs, 90), color=MUTED, lw=1.6, ls=":"); ax.text(np.percentile(errs, 90) + 0.3, ax.get_ylim()[1] * 0.7, f"90% below {np.percentile(errs, 90):.1f}°", fontsize=14, color=INK)
ax.set_xlabel("Per-image angular error (°)", fontsize=15); ax.set_ylabel("Images", fontsize=15)
save(f, "c11_err_hist")

# ---- C12 label distribution ----
G = []
for fn in sorted(os.listdir(f"{D}/Label")):
    for l in open(f"{D}/Label/{fn}").read().splitlines()[1:]:
        G.append([math.degrees(float(v)) for v in l.split()[7].split(",")])
G = np.array(G)
f = fig(6.4, 5); ax = f.add_subplot(); style(ax, grid=None)
hb = ax.hexbin(G[:, 0], G[:, 1], gridsize=40, cmap="Blues", mincnt=1, extent=(-42, 42, -42, 42))
ax.add_patch(plt.Rectangle((-42, -42), 84, 84, fill=False, ec=AMBER, lw=2.2, ls="--"))
ax.text(41, 39, "bin range ±42°", ha="right", va="top", fontsize=14, color=INK)
ax.set_xlim(-45, 45); ax.set_ylim(-45, 45); ax.set_aspect("equal")
ax.set_xlabel("Horizontal gaze, yaw (°)", fontsize=15); ax.set_ylabel("Vertical gaze, pitch (°)", fontsize=15)
save(f, "c12_label_dist")
nums["label"] = dict(n=len(G), yaw_min=G[:, 0].min(), yaw_max=G[:, 0].max(), yaw_std=G[:, 0].std(), pitch_min=G[:, 1].min(), pitch_max=G[:, 1].max(), pitch_std=G[:, 1].std(), pitch_mean=G[:, 1].mean())

# ---- sample mosaic (2 x 6) ----
tiles = []
for i, s in enumerate(["p00", "p02", "p04", "p06", "p08", "p10", "p01", "p03", "p05", "p09", "p12", "p13"]):
    ls_ = open(f"{D}/Label/{s}.label").read().splitlines()[1:]; l = ls_[len(ls_) // 3].split()
    y_, p_ = map(float, l[7].split(",")); tiles.append(arrow(Image.open(f"{D}/Image/{l[0]}").convert("RGB"), y_, p_).resize((224, 224)))
mos = Image.new("RGB", (6 * 224 + 5 * 10, 2 * 224 + 10), CARD)
for i, t in enumerate(tiles): mos.paste(t, ((i % 6) * 234, (i // 6) * 234))
mos.save(f"{OUT}/c13_samples.png")

json.dump(nums, open(f"{OUT}/numbers.json", "w"), indent=1, default=float)
print(json.dumps(nums, indent=1, default=float))
