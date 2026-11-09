"""Train L2CS-Net on Gaze360 (GazeHub protocol, face subset) for a direct comparison with the paper's 10.41 deg.

Same model / loss / optimiser as scripts/train_mpii.py, re-using its helpers, but with the L2CS Gaze360 binning
(90 bins x 4 deg, +-180 deg) and the CSV labels from scripts/prep_gaze360.py (label[0] = horizontal, label[1] = vertical,
train_mpii.gazeto3d convention; validated with the official weights: 10.56 deg on test).
Every epoch is evaluated on val AND test; we report test at (a) the last epoch, (b) the epoch with the best VAL
error (honest model selection) and (c) the best test epoch (optimistic, what a "best checkpoint" table usually shows).

  python -I scripts/train_gaze360.py --data /work/u8714813/datasets/Gaze360 --recipe paper --epochs 50 --amp \
      --out /work/u8714813/runs/gaze360/r50_paper_e50
Resumable (--resume, state.pt), skips when done.json exists.
"""
import argparse
import csv
import importlib.util
import json
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("train_mpii", os.path.join(HERE, "train_mpii.py"))
tm = importlib.util.module_from_spec(_s); _s.loader.exec_module(tm)
tm.BINS, tm.BIN_W, tm.ANGLE = 90, 4, 180  # L2CS Gaze360 setting; helpers in train_mpii read these globals at call time
BINS, BIN_W, ANGLE = tm.BINS, tm.BIN_W, tm.ANGLE


class Gaze360Csv(Dataset):
    def __init__(self, root, split, aug=False, img_size=448):
        self.rows = list(csv.DictReader(open(os.path.join(root, "Label", f"{split}.csv"))))
        self.img_root = os.path.join(root, "Image")
        self.tf, self.aug = tm.make_normalize(img_size), aug
        self.aug_tf = transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        ])
        self.bins = np.array(range(-ANGLE, ANGLE, BIN_W))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(os.path.join(self.img_root, r["face"])).convert("RGB")
        h, v = float(r["gaze_h_deg"]), float(r["gaze_v_deg"])
        if self.aug:
            img = self.aug_tf(img)
            if random.random() < 0.5:
                img = ImageOps.mirror(img); h = -h
        img = self.tf(img)
        binned = np.clip(np.digitize([h, v], self.bins) - 1, 0, BINS - 1)
        return img, torch.as_tensor(binned), torch.FloatTensor([h, v]), f"{r['face']}|{r['recording']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--recipe", choices=list(tm.RECIPES), default="paper")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr-min", type=float, default=1e-7)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--backbone", choices=tm.BACKBONES, default="resnet50")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-iters", type=int, default=0, help="debug: iters per epoch")
    ap.add_argument("--eval-limit", type=int, default=0, help="debug: evaluate on a subset")
    a = ap.parse_args()
    rc = tm.RECIPES[a.recipe]
    if os.path.exists(os.path.join(a.out, "done.json")):
        print(a.out, "already finished"); return
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    os.makedirs(os.path.join(a.out, "predictions"), exist_ok=True)
    dev = torch.device("cuda"); torch.backends.cudnn.benchmark = True
    amp = torch.bfloat16 if a.amp else None
    tr = Gaze360Csv(a.data, "train", aug=rc["aug"])
    va, te = Gaze360Csv(a.data, "val"), Gaze360Csv(a.data, "test")
    if a.eval_limit:
        va.rows, te.rows = va.rows[:a.eval_limit], te.rows[:a.eval_limit]
    tl = DataLoader(tr, batch_size=a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True,
                    persistent_workers=True)
    vl = DataLoader(va, batch_size=128, num_workers=a.workers, pin_memory=True)
    el = DataLoader(te, batch_size=128, num_workers=a.workers, pin_memory=True)
    model = tm.build_model(backbone=a.backbone).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(tm.param_groups(model, a.lr), a.lr)
    ipe = min(a.max_iters, len(tl)) if a.max_iters else len(tl); total = a.epochs * ipe
    floor = a.lr_min / a.lr
    lr_fn = (lambda it: floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(it, total) / total))) if rc["sched"] == "cosine" else (lambda it: 1.0)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    idx = torch.arange(BINS, dtype=torch.float32, device=dev)
    sp = os.path.join(a.out, "state.pt"); start, best_val, step = 1, float("inf"), 0
    if a.resume and os.path.exists(sp):
        st = torch.load(sp, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"]); sch.load_state_dict(st["scheduler"])
        start, best_val, step = st["epoch"] + 1, st["best_val"], st["step"]
        torch.set_rng_state(st["rng_cpu"]); random.setstate(st["rng_py"]); np.random.set_state(st["rng_np"])
        print(f"resumed at epoch {start}", flush=True)
    json.dump(vars(a) | rc | {"bins": BINS, "bin_w": BIN_W, "angle": ANGLE, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                              "started": datetime.now().isoformat(timespec="seconds")},
              open(os.path.join(a.out, "config.json"), "w"), indent=2)
    mp = os.path.join(a.out, "metrics.csv")
    if start == 1 or not os.path.exists(mp):
        with open(mp, "w", newline="") as fh:
            csv.writer(fh).writerow(["epoch", "train_loss", "val_err", "val_median", "test_err", "test_median", "test_p90",
                                     "test_h_mae", "test_v_mae", "lr_end", "epoch_sec", "img_per_sec", "timestamp"])
    else:  # drop rows of a partially finished epoch
        lines = open(mp).read().splitlines()
        open(mp, "w").write("\n".join([lines[0]] + [l for l in lines[1:] if l and int(l.split(",")[0]) < start]) + "\n")
    print(f"recipe {a.recipe} {rc}: train {len(tr)} val {len(va)} test {len(te)}, {ipe} it/epoch", flush=True)
    for ep in range(start, a.epochs + 1):
        tm.set_train_mode(model, False); t0 = time.time(); tot = 0.0; n = 0
        for i, (x, lab, cont, _) in enumerate(tl):
            x = x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)
            l0, l1 = lab[:, 0].to(dev), lab[:, 1].to(dev); c0, c1 = cont[:, 0].to(dev), cont[:, 1].to(dev)
            with torch.autocast("cuda", dtype=amp or torch.bfloat16, enabled=amp is not None):
                o0, o1 = model(x)
            o0, o1 = o0.float(), o1.float()
            p0 = (torch.softmax(o0, 1) * idx).sum(1) * BIN_W - ANGLE; p1 = (torch.softmax(o1, 1) * idx).sum(1) * BIN_W - ANGLE
            loss = ce(o0, l0) + a.alpha * mse(p0, c0) + ce(o1, l1) + a.alpha * mse(p1, c1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); step += 1
            tot += loss.item(); n += 1
            if (i + 1) % 500 == 0:
                print(f"epoch {ep} iter {i+1}/{ipe} loss {tot/n:.3f} {(i+1)*a.batch_size/(time.time()-t0):.0f} img/s", flush=True)
            if a.max_iters and n >= a.max_iters:
                break
        dt = time.time() - t0
        ev = tm.evaluate(model, vl, dev, amp)
        et = tm.evaluate(model, el, dev, amp, os.path.join(a.out, "predictions", f"test_epoch_{ep}.csv"))
        print(f"== epoch {ep}/{a.epochs}: loss {tot/n:.3f} | val {ev['mean']:.3f} | test {et['mean']:.3f} "
              f"(median {et['median']:.2f}) | {dt/60:.1f} min", flush=True)
        with open(mp, "a", newline="") as fh:
            csv.writer(fh).writerow([ep, f"{tot/n:.5f}", f"{ev['mean']:.4f}", f"{ev['median']:.4f}", f"{et['mean']:.4f}", f"{et['median']:.4f}",
                                     f"{et['p90']:.4f}", f"{et['pitch_mae']:.4f}", f"{et['yaw_mae']:.4f}", f"{sch.get_last_lr()[-1]:.3e}",
                                     f"{dt:.1f}", f"{n*a.batch_size/dt:.1f}", datetime.now().isoformat(timespec="seconds")])
        torch.save(model.state_dict(), os.path.join(a.out, "last.pkl"))
        if ev["mean"] < best_val:
            best_val = ev["mean"]; torch.save(model.state_dict(), os.path.join(a.out, "best_val.pkl"))
            json.dump({"epoch": ep, "val_err": ev["mean"], "test_err": et["mean"]}, open(os.path.join(a.out, "best_val.json"), "w"))
        torch.save({"epoch": ep, "step": step, "best_val": best_val, "model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sch.state_dict(), "rng_cpu": torch.get_rng_state(), "rng_py": random.getstate(),
                    "rng_np": np.random.get_state()}, sp + ".tmp")
        os.replace(sp + ".tmp", sp)
    rows = list(csv.DictReader(open(mp)))
    bt = min(rows, key=lambda r: float(r["test_err"])); bv = min(rows, key=lambda r: float(r["val_err"]))
    summ = {"finished": datetime.now().isoformat(timespec="seconds"), "test_last": float(rows[-1]["test_err"]),
            "test_at_best_val": float(bv["test_err"]), "best_val_epoch": int(bv["epoch"]),
            "test_best": float(bt["test_err"]), "best_test_epoch": int(bt["epoch"])}
    json.dump(summ, open(os.path.join(a.out, "done.json"), "w"), indent=1)
    print("done", summ, flush=True)


if __name__ == "__main__":
    main()
