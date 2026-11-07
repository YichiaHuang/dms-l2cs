"""Charts + markdown tables for the remote (Nano4) 15-fold runs.

Reads  results/remote/all_runs.csv (from make_report.py), results/all_runs.csv (local runs),
       results/slides_charts/paper_fig2_per_subject.json (paper Fig. 2 per-subject errors).
Writes results/remote/per_subject_bars.png, paired_diff.png, tables.md, summary.json

  python -I scripts/make_remote_charts.py [--remote results/remote] [--local results]
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
C_PAPER, C_IMPR, C_FIG2 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
PERSONS = [f"p{i:02d}" for i in range(15)]
TAGS = {"paper_e50": "paper", "improved_e50": "improved", "abl_cosine_e50": "cosine-only", "abl_aug_e50": "aug-only"}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURF)


def load_runs(path):
    """{tag: {person: row}} keyed by the run-name suffix after fold<N>_."""
    out = {}
    for r in csv.DictReader(open(path)):
        tag = r["run"].split("_", 1)[1]
        out.setdefault(tag, {})[r["person"]] = r
    return out


def ms(xs):
    xs = np.asarray([x for x in xs if x is not None and not np.isnan(x)], float)
    return (xs.mean(), xs.std(ddof=1) if len(xs) > 1 else 0.0, len(xs)) if len(xs) else (np.nan, np.nan, 0)


def g(row, key):
    return float(row[key]) if row and row.get(key) not in (None, "") else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="results/remote")
    ap.add_argument("--local", default="results")
    a = ap.parse_args()
    R = load_runs(os.path.join(a.remote, "all_runs.csv"))
    L = load_runs(os.path.join(a.local, "all_runs.csv"))
    fig2 = json.load(open(os.path.join(a.local, "slides_charts", "paper_fig2_per_subject.json")))
    P, I = R.get("paper_e50", {}), R.get("improved_e50", {})
    md, summary = [], {}

    # ---- Table 1: per subject
    md += ["## 表 1：15 人逐人誤差（度）", "",
           "| 測試者 | paper @ep50 | paper 最佳 (epoch) | improved @ep50 | improved 最佳 (epoch) | 論文 Fig. 2 |",
           "|---|---|---|---|---|---|"]
    cols = {"p_last": [], "p_best": [], "i_last": [], "i_best": [], "fig2": []}
    for p in PERSONS:
        pr, ir = P.get(p), I.get(p)
        vals = dict(p_last=g(pr, "last_err"), p_best=g(pr, "best_err"), i_last=g(ir, "last_err"), i_best=g(ir, "best_err"), fig2=fig2[p])
        for k, v in vals.items():
            cols[k].append(v)
        be = lambda r: f" ({r['best_epoch']})" if r else ""
        fm = lambda v: "—" if np.isnan(v) else f"{v:.2f}"
        md.append(f"| {p} | {fm(vals['p_last'])} | {fm(vals['p_best'])}{be(pr)} | {fm(vals['i_last'])} | {fm(vals['i_best'])}{be(ir)} | {vals['fig2']:.2f} |")
    stats = {k: ms(v) for k, v in cols.items()}
    md.append("| **平均 ± 標準差** | " + " | ".join(f"**{stats[k][0]:.2f} ± {stats[k][1]:.2f}** (n={stats[k][2]})" for k in ["p_last", "p_best", "i_last", "i_best", "fig2"]) + " |")
    summary["means"] = {k: dict(mean=round(m, 4), std=round(s, 4), n=n) for k, (m, s, n) in stats.items()}

    # ---- Table 2: vs paper
    md += ["", "## 表 2：和論文 15-fold 平均（3.92°）比較", "", "| 數字 | 平均（度） | 與 3.92° 差 | 百分比 |", "|---|---|---|---|",
           f"| 論文 L2CS-Net（MPIIFaceGaze） | {PAPER_MPII:.2f} | — | — |"]
    for k, name in [("p_last", "我們 paper @ep50（誠實數字）"), ("p_best", "我們 paper 最佳 epoch（偏樂觀）"),
                    ("i_last", "我們 improved @ep50"), ("i_best", "我們 improved 最佳 epoch（偏樂觀）")]:
        m = stats[k][0]
        md.append(f"| {name} | {m:.2f} | {m - PAPER_MPII:+.2f} | {100 * (m - PAPER_MPII) / PAPER_MPII:+.1f}% |")

    # ---- Table 3: reproducibility @ep7 (local 5 persons vs remote same recipe)
    md += ["", "## 表 3：可重現性（@ep7，同配方）", "", "| 測試者 | 配方 | 本機 run | 本機 @ep7 | 遠端 @ep7 | 差（遠端 − 本機） | 可比？ |", "|---|---|---|---|---|---|---|"]
    repro = []
    for tag, rows in sorted(L.items()):
        for p, lr in sorted(rows.items()):
            rec = lr["recipe"]
            rr = (P if rec == "paper" else I).get(p)
            loc, rem = g(lr, "err_at_ep7"), np.nan
            if rr is not None:
                rem = remote_err_at(a.remote, rr["run"], 7)
            d = rem - loc
            repro.append(dict(person=p, recipe=rec, local_run=lr["run"], local=loc, remote=rem, diff=d))
            # paper recipe = constant lr -> @ep7 is the same training state locally and remotely; the cosine
            # schedule is stretched over --epochs, so a local 7-epoch improved run is at lr 1e-7 by ep7 while the
            # remote 50-epoch run is still near 1e-5 -> not comparable.
            comparable = rec == "paper"
            flag = " ⚠" if comparable and not np.isnan(d) and abs(d) > 0.3 else ""
            note = "是" if comparable else "否（cosine 排程長度不同：本機 T=7，遠端 T=50）"
            repro[-1]["comparable"] = comparable
            md.append(f"| {p} | {rec} | {lr['run']} | {loc:.2f} | {'—' if np.isnan(rem) else f'{rem:.2f}'} | {'—' if np.isnan(d) else f'{d:+.2f}'}{flag} | {note} |")
    summary["repro"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in repro]

    # ---- Ablation
    if R.get("abl_cosine_e50") or R.get("abl_aug_e50"):
        five = ["p00", "p03", "p07", "p11", "p14"]
        md += ["", "## 消融：5 人 @ep50（度）", "", "| 測試者 | paper | cosine-only | aug-only | improved |", "|---|---|---|---|---|"]
        acc = {t: [] for t in ["paper_e50", "abl_cosine_e50", "abl_aug_e50", "improved_e50"]}
        for p in five:
            cells = []
            for t in acc:
                v = g(R.get(t, {}).get(p), "last_err"); acc[t].append(v)
                cells.append("—" if np.isnan(v) else f"{v:.2f}")
            md.append(f"| {p} | " + " | ".join(cells) + " |")
        md.append("| **平均** | " + " | ".join(f"**{ms(v)[0]:.2f}** (n={ms(v)[2]})" for v in acc.values()) + " |")
        summary["ablation_mean_ep50"] = {TAGS[t]: round(ms(v)[0], 4) for t, v in acc.items()}

    open(os.path.join(a.remote, "tables.md"), "w").write("\n".join(md) + "\n")
    json.dump(summary, open(os.path.join(a.remote, "summary.json"), "w"), indent=1, ensure_ascii=False)

    # ---- Fig (a): per-subject bars
    x = np.arange(len(PERSONS)); w = 0.27
    fig, ax = plt.subplots(figsize=(12, 4.2), facecolor=SURF)
    for off, key, c, lab, hatch in [(-w, "p_last", C_PAPER, "ours, paper recipe @ep50", None),
                                    (0, "i_last", C_IMPR, "ours, improved recipe @ep50", None),
                                    (w, "fig2", C_FIG2, "paper Fig. 2", "///")]:
        ax.bar(x + off, cols[key], w - 0.03, color=c, label=f"{lab} (mean {stats[key][0]:.2f}°)", hatch=hatch,
               edgecolor=SURF if hatch is None else "white", linewidth=0.6 if hatch else 0)
    ax.axhline(PAPER_MPII, color=MUTED, lw=1, ls="--"); ax.text(-0.9, PAPER_MPII + 0.08, "paper mean 3.92°", color=INK2, fontsize=8, ha="left")
    ax.set_xticks(x); ax.set_xticklabels(PERSONS); ax.set_xlim(-1, len(PERSONS)); ax.set_ylabel("angular error (°)", color=INK2)
    ax.set_title("MPIIFaceGaze leave-one-person-out, per held-out subject", color=INK, fontsize=11, loc="left")
    style(ax); ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left", labelcolor=INK2)
    ax.set_ylim(0, max(np.nanmax(cols["p_last"] + cols["i_last"] + cols["fig2"]) * 1.18, 1))
    fig.tight_layout(); fig.savefig(os.path.join(a.remote, "per_subject_bars.png"), dpi=150, facecolor=SURF); plt.close(fig)

    # ---- Fig (c): paired diff improved - paper @ep50
    d = np.array(cols["i_last"]) - np.array(cols["p_last"])
    fig, ax = plt.subplots(figsize=(10, 3.6), facecolor=SURF)
    ax.bar(x, d, 0.6, color=[C_IMPR if v < 0 else C_PAPER for v in np.nan_to_num(d)])
    ax.axhline(0, color=BASE, lw=1)
    md_ = np.nanmean(d)
    ax.axhline(md_, color=INK2, lw=1, ls="--"); 
    for xi, v in zip(x, d):
        if not np.isnan(v):
            ax.text(xi, v + (0.03 if v >= 0 else -0.03), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(PERSONS); ax.set_xlim(-1, len(PERSONS)); ax.set_ylabel("improved − paper @ep50 (°)", color=INK2)
    ax.set_title(f"improved − paper @ep50 per subject; dashed = mean {md_:+.2f}° (orange = improved better, blue = paper better)", color=INK, fontsize=11, loc="left")
    style(ax); fig.tight_layout(); fig.savefig(os.path.join(a.remote, "paired_diff.png"), dpi=150, facecolor=SURF); plt.close(fig)
    summary["paired_diff_mean"] = float(md_)
    json.dump(summary, open(os.path.join(a.remote, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print("\n".join(md))


def remote_err_at(remote_dir, run, epoch):
    for r in csv.DictReader(open(os.path.join(remote_dir, "all_epochs.csv"))):
        if r["run"] == run and int(r["epoch"]) == epoch:
            return float(r["test_err"])
    return np.nan


if __name__ == "__main__":
    main()
