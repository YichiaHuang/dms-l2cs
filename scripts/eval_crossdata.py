"""Cross-dataset evaluation: models trained on MPIIFaceGaze -> other datasets (no re-training).

A dataset is a label CSV (face, gaze_h_deg, gaze_v_deg, ...) + image root (scripts/prep_gaze360.py, prep_rtgene.py).
Each image is decoded once; every batch goes through all models (GPU, bf16 autocast like training eval).
Models:
  * official : the released L2CS-Net Gaze360 checkpoint (90 bins x 4 deg, +-180 deg) - sanity check of our
               preprocessing / sign convention: on Gaze360 test it must land near the paper's 10.41 deg
  * <tag>    : all 15 leave-one-person-out models of a run tag (e.g. improved_e50) -> mean +- sd over the 15
Signs: --check-signs evaluates the official model with every sign / axis-order variant of the ground truth and
prints them; the convention is then fixed in the prep script (we never pick signs per model).

Writes <out>/<dataset>.json (per-model metrics) and <out>/<dataset>_preds.npz (per-image predictions).

  python -I scripts/eval_crossdata.py --dataset gaze360_test --csv .../Label/test.csv --root .../Image \
      --tags paper_e50 improved_e50 effb0_448_e20 mnv3l_448_kd_e20 --official .../l2cs_gaze360_resnet50.safetensors
"""
import argparse
import csv
import importlib.util
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("train_mpii", os.path.join(HERE, "train_mpii.py"))
tm = importlib.util.module_from_spec(_s); _s.loader.exec_module(tm)
ROOTS = {"paper_e50": "/work/u8714813/runs/l2cs", "improved_e50": "/work/u8714813/runs/l2cs",
         "effb0_448_e20": "/work/u8714813/runs/l2cs_light", "mnv3l_448_kd_e20": "/work/u8714813/runs/l2cs_light"}
THETAS = [5, 10, 20]


class CsvFaces(Dataset):
    def __init__(self, csv_path, root, size):
        self.rows = list(csv.DictReader(open(csv_path)))
        self.root, self.tf = root, tm.make_normalize(size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(os.path.join(self.root, r["face"])).convert("RGB").resize((448, 448), Image.BILINEAR)
        return self.tf(img), torch.tensor([float(r["gaze_h_deg"]), float(r["gaze_v_deg"])])


def to3d(h, v):  # same mapping as train_mpii.gazeto3d, vectorised (degrees in)
    h, v = np.radians(h), np.radians(v)
    return np.stack([-np.cos(v) * np.sin(h), -np.sin(v), -np.cos(v) * np.cos(h)], -1)


def ang_err(ph, pv, gh, gv):
    a, b = to3d(ph, pv), to3d(gh, gv)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def summarise(err, gh, gv):
    inr = (np.abs(gh) <= 42) & (np.abs(gv) <= 42)  # inside the MPIIFaceGaze-trained models' output range
    return dict(mean=float(err.mean()), median=float(np.median(err)), p90=float(np.percentile(err, 90)),
                acc={t: float((err <= t).mean()) for t in THETAS}, n=int(len(err)),
                in_range_mean=float(err[inr].mean()), in_range_n=int(inr.sum()), in_range_frac=float(inr.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--official", default=None)
    ap.add_argument("--g360-runs", nargs="*", default=[], help="run dirs of scripts/train_gaze360.py (90x4 deg bins, last.pkl)")
    ap.add_argument("--check-signs", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/crossdata")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--from-preds", default=None, help="re-score a saved <dataset>_preds.npz against --csv labels (no GPU)")
    ap.add_argument("--pred-dir", default="/work/u8714813/runs/crossdata", help="per-image predictions (not committed)")
    a = ap.parse_args()
    ds = CsvFaces(a.csv, a.root, 448)
    if a.limit:
        ds.rows = ds.rows[:: max(1, len(ds.rows) // a.limit)][: a.limit]
    if a.from_preds:  # images unchanged, only the labels were re-derived -> re-score the stored predictions
        z = np.load(a.from_preds); pos = {n: i for i, n in enumerate(z["names"])}
        order = np.array([pos[r["face"]] for r in ds.rows])
        preds = {k.replace("__", "/"): z[k][order] for k in z.files if k not in ("gt", "names")}
        gt = np.array([[float(r["gaze_h_deg"]), float(r["gaze_v_deg"])] for r in ds.rows], dtype=np.float32)
    else:
        preds, gt = infer(a, ds)
    score(a, ds, preds, gt)


def infer(a, ds):
    dl = DataLoader(ds, batch_size=128, num_workers=a.workers, pin_memory=True)
    models = {}  # name -> (model, bins, bin_w, angle)
    if a.official:
        from safetensors.torch import load_file
        m = tm.L2CS(__import__("torchvision").models.resnet.Bottleneck, [3, 4, 6, 3], 90)
        m.load_state_dict(load_file(a.official))
        models["official_gaze360"] = (m, 90, 4, 180)
    for run in a.g360_runs:
        cfg = json.load(open(os.path.join(run, "config.json")))
        m = tm.L2CS(__import__("torchvision").models.resnet.Bottleneck, [3, 4, 6, 3], cfg["bins"])
        m.load_state_dict(torch.load(os.path.join(run, "last.pkl"), map_location="cpu"))
        models["g360_" + os.path.basename(run.rstrip("/"))] = (m, cfg["bins"], cfg["bin_w"], cfg["angle"])
    for tag in a.tags:
        for f in range(15):
            run = os.path.join(ROOTS[tag], f"fold{f}_{tag}")
            cfg = json.load(open(os.path.join(run, "config.json")))
            m = tm.build_model(pretrained=False, backbone=cfg.get("backbone", "resnet50"))
            m.load_state_dict(torch.load(os.path.join(run, "last.pkl"), map_location="cpu"))
            models[f"{tag}/fold{f}"] = (m, tm.BINS, tm.BIN_W, tm.ANGLE)
    for k, (m, *_) in models.items():
        models[k] = (m.cuda().eval().to(memory_format=torch.channels_last), *models[k][1:])
    preds = {k: [] for k in models}; gts = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, g in dl:
            x = x.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last); gts.append(g.numpy())
            for k, (m, bins, bw, ang) in models.items():
                idx = torch.arange(bins, device="cuda", dtype=torch.float32)
                o0, o1 = m(x)
                p0 = (torch.softmax(o0.float(), 1) * idx).sum(1) * bw - ang
                p1 = (torch.softmax(o1.float(), 1) * idx).sum(1) * bw - ang
                preds[k].append(torch.stack([p0, p1], 1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in preds.items()}, np.concatenate(gts)


def score(a, ds, preds, gt):
    gh, gv = gt[:, 0], gt[:, 1]
    os.makedirs(a.out, exist_ok=True)
    res = dict(dataset=a.dataset, csv=a.csv, n=int(len(gt)), models={})
    if a.check_signs and "official_gaze360" in preds:
        p = preds["official_gaze360"]; chk = {}
        for sh in (1, -1):
            for sv in (1, -1):
                chk[f"h{'+' if sh > 0 else '-'} v{'+' if sv > 0 else '-'}"] = float(ang_err(p[:, 0], p[:, 1], sh * gh, sv * gv).mean())
                chk[f"swapped h{'+' if sh > 0 else '-'} v{'+' if sv > 0 else '-'}"] = float(ang_err(p[:, 1], p[:, 0], sh * gh, sv * gv).mean())
        res["sign_check_official"] = chk
        print("sign check (official model, mean err):", json.dumps(chk, indent=1), flush=True)
    for k, p in preds.items():
        res["models"][k] = summarise(ang_err(p[:, 0], p[:, 1], gh, gv), gh, gv)
        res["models"][k]["corr_h"] = float(np.corrcoef(p[:, 0], gh)[0, 1]); res["models"][k]["corr_v"] = float(np.corrcoef(p[:, 1], gv)[0, 1])
    for tag in a.tags:  # aggregate the 15 fold models
        ks = [k for k in preds if k.startswith(tag + "/")]
        ms = np.array([res["models"][k]["mean"] for k in ks]); ir = np.array([res["models"][k]["in_range_mean"] for k in ks])
        res["models"][tag] = dict(mean_over_folds=float(ms.mean()), sd_over_folds=float(ms.std(ddof=1)),
                                  in_range_mean_over_folds=float(ir.mean()), in_range_sd=float(ir.std(ddof=1)),
                                  acc_over_folds={t: float(np.mean([res["models"][k]["acc"][t] for k in ks])) for t in THETAS},
                                  median_over_folds=float(np.mean([res["models"][k]["median"] for k in ks])))
        print(tag, json.dumps(res["models"][tag]), flush=True)
    for k in res["models"]:
        if k == "official_gaze360" or k.startswith("g360_"):
            print(k, json.dumps(res["models"][k]), flush=True)
    json.dump(res, open(os.path.join(a.out, f"{a.dataset}.json"), "w"), indent=1)
    os.makedirs(a.pred_dir, exist_ok=True)
    np.savez_compressed(os.path.join(a.pred_dir, f"{a.dataset}_preds.npz"), gt=gt, names=np.array([r["face"] for r in ds.rows]),
                        **{k.replace("/", "__"): v for k, v in preds.items()})


if __name__ == "__main__":
    main()
