"""Horizontal-flip test-time augmentation (TTA) for trained L2CS runs (GPU, bf16 autocast like training eval).

For each run: load last.pkl (end-of-training weights), evaluate the held-out person (a) plainly - must reproduce
the last row of metrics.csv, a sanity check - and (b) averaging the prediction with that of the mirrored image
(mirroring negates label[0], exactly as the training augmentation does). Writes <run>/predictions/final_fp32_flip.csv
and one JSON row per run to --out. Costs 2x compute at inference.

  python -I scripts/eval_tta.py --runs /work/u8714813/runs/l2cs/fold*_improved_e50 --data ... --out results/final/tta.json
"""
import argparse
import csv
import importlib.util
import json
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("train_mpii", os.path.join(HERE, "train_mpii.py"))
tm = importlib.util.module_from_spec(_s); _s.loader.exec_module(tm)
_s = importlib.util.spec_from_file_location("eval_int8", os.path.join(HERE, "eval_int8.py"))
ei = importlib.util.module_from_spec(_s); _s.loader.exec_module(ei)


@torch.no_grad()
def run_eval(model, loader, flip):
    idx = torch.arange(tm.BINS, dtype=torch.float32, device="cuda")
    dec = lambda o: (torch.softmax(o.float(), 1) * idx).sum(1) * tm.BIN_W - tm.ANGLE  # noqa: E731
    rows, errs = [], []
    for images, _, cont, names in loader:
        images = images.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a, b = model(images)
            p, y = dec(a), dec(b)
            if flip:
                fa, fb = model(torch.flip(images, dims=[3]))
                p, y = (p - dec(fa)) / 2, (y + dec(fb)) / 2
        for nm, pa, pb, ga, gb in zip(names, p.tolist(), y.tolist(), cont[:, 0].tolist(), cont[:, 1].tolist()):
            e = tm.angular(tm.gazeto3d([math.radians(pa), math.radians(pb)]), tm.gazeto3d([math.radians(ga), math.radians(gb)]))
            errs.append(e)
            rows.append([nm.split("|")[0], nm.split("|")[1], f"{ga:.3f}", f"{gb:.3f}", f"{pa:.3f}", f"{pb:.3f}", f"{e:.4f}"])
    return float(np.mean(errs)), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    labels = [os.path.join(a.data, "Label", f) for f in sorted(os.listdir(os.path.join(a.data, "Label")))]
    img_dir = os.path.join(a.data, "Image")
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for run in a.runs:
        name = os.path.basename(run.rstrip("/"))
        if name in out:
            print(name, "already done"); continue
        cfg = json.load(open(os.path.join(run, "config.json")))
        bb, size, fold = cfg.get("backbone", "resnet50"), cfg.get("img_size", 448), cfg["fold"]
        model = tm.build_model(pretrained=False, backbone=bb)
        model.load_state_dict(torch.load(os.path.join(run, "last.pkl"), map_location="cpu"))
        model = model.cuda().eval().to(memory_format=torch.channels_last)
        test = tm.MpiiFold(labels, img_dir, False, fold, aug=False, img_size=size)
        tl = DataLoader(test, batch_size=64, num_workers=a.workers, pin_memory=True)
        plain, _ = run_eval(model, tl, False)
        flip, rows = run_eval(model, tl, True)
        ei.write_preds(os.path.join(run, "predictions", "final_fp32_flip.csv"), rows)
        last = list(csv.DictReader(open(os.path.join(run, "metrics.csv"))))[-1]
        out[name] = dict(backbone=bb, img_size=size, fold=fold, plain_err=plain, logged_last_err=float(last["test_angular_err"]),
                         flip_tta_err=flip, delta=flip - plain, n_test=len(rows))
        print(json.dumps({name: out[name]}), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
