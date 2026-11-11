"""Next experiment (R1): does a SMALL amount of target-domain data fix the RT-GENE gap?

Fine-tune an existing model (default: J1 = MobileNetV3-L+KD trained on Gaze360 + 10 MPII persons) on a few images
from some RT-GENE subjects, test on OTHER RT-GENE subjects (never seen), and check that it does not forget the
original data (MPII held-out persons, Gaze360 test).

Subject split (fixed before running, see README "下一步實驗"): the 17 RT-GENE subjects sorted by id, subject index
i goes to test fold i % 3. --fold k -> test = subjects with i % 3 == k, adapt = the rest.
--per-subject N: N images per adapt subject, taken evenly over the recording (0 = all images).
--replay R: fraction of every batch drawn from the base model's ORIGINAL training data (Gaze360 + its MPII persons),
            to reduce forgetting (0 = pure fine-tuning).
Same loss / augmentation as training (CE + MSE on both heads, improved recipe aug), Adam, cosine to lr/100 over
--iters iterations. The base model's binning is kept (90x4 deg for Gaze360-style runs, 28x3 deg for MPII-only runs).
Evaluates before fine-tuning (iteration 0) and every --eval-every iterations: RT-GENE test subjects (pooled and
subject-mean), MPII --mpii-test persons (person-mean), Gaze360 test (pooled).

  python -I scripts/finetune_rtgene.py --init /work/u8714813/runs/gaze360/J1_mnv3l_kd_joint_e20 --fold 0 \
      --per-subject 100 --out /work/u8714813/runs/rtgene_ft/J1_f0_n100
Writes metrics.csv, config.json, done.json, last.pkl (weights are never committed).
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


def _load(name):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


tg = _load("train_gaze360")  # row helpers (gaze360_rows, mpii_rows); tm = its train_mpii module
tm = tg.tm
RT = "/work/u8714813/datasets/RTGENE"


def rtgene_rows(root, subjects, per_subject=0):
    by = {}
    for r in csv.DictReader(open(os.path.join(root, "Label", "all.csv"))):
        if r["subject"] in subjects:
            by.setdefault(r["subject"], []).append(dict(path=os.path.join(root, "Image", r["face"]), h=float(r["gaze_h_deg"]),
                                                       v=float(r["gaze_v_deg"]), src="rtgene", group=r["subject"]))
    rows = []
    for s in sorted(by):
        rs = sorted(by[s], key=lambda r: r["path"])
        if per_subject and per_subject < len(rs):  # evenly over the recording, not the first N frames
            rs = [rs[int(j)] for j in np.linspace(0, len(rs) - 1, per_subject)]
        rows += rs
    return rows


class Faces(Dataset):
    def __init__(self, rows, aug, bins, bin_w, angle):
        self.rows, self.aug, self.nb = rows, aug, bins
        self.tf = tm.make_normalize(448)
        self.aug_tf = transforms.Compose([transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                                          transforms.ColorJitter(0.3, 0.3, 0.2, 0.02)])
        self.edges = np.array(range(-angle, angle, bin_w))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["path"]).convert("RGB").resize((448, 448), Image.BILINEAR)
        h, v = r["h"], r["v"]
        if self.aug:
            img = self.aug_tf(img)
            if random.random() < 0.5:
                img = ImageOps.mirror(img); h = -h
        b = np.clip(np.digitize([h, v], self.edges) - 1, 0, self.nb - 1)
        return self.tf(img), torch.as_tensor(b), torch.FloatTensor([h, v]), f"{r['path']}|{r['group']}|{r['src']}"


def endless(loader):
    while True:
        yield from loader


def group_mean(pred_csv):
    per = {}
    for r in csv.DictReader(open(pred_csv)):
        per.setdefault(r["origin"], []).append(float(r["angular_err_deg"]))
    return float(np.mean([np.mean(v) for v in per.values()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="run dir with config.json + last.pkl (train_gaze360 or train_mpii)")
    ap.add_argument("--fold", type=int, required=True, choices=[0, 1, 2])
    ap.add_argument("--per-subject", type=int, default=0)
    ap.add_argument("--replay", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--rtgene", default=RT)
    ap.add_argument("--gaze360", default="/work/u8714813/datasets/Gaze360")
    ap.add_argument("--mpii-data", default="/work/u8714813/datasets/MPIIFaceGaze")
    ap.add_argument("--mpii-test", default="p00,p03,p07,p11,p14")
    ap.add_argument("--no-g360-eval", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--eval-limit", type=int, default=0, help="debug")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, "done.json")):
        print(a.out, "already finished"); return
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    os.makedirs(os.path.join(a.out, "predictions"), exist_ok=True)
    cfg = json.load(open(os.path.join(a.init, "config.json")))
    tm.BINS, tm.BIN_W, tm.ANGLE = cfg.get("bins", 28), cfg.get("bin_w", 3), cfg.get("angle", 42)  # evaluate() reads these
    bins = (tm.BINS, tm.BIN_W, tm.ANGLE)
    subs = sorted({r["subject"] for r in csv.DictReader(open(os.path.join(a.rtgene, "Label", "all.csv")))})
    test_s = [s for i, s in enumerate(subs) if i % 3 == a.fold]; adapt_s = [s for s in subs if s not in test_s]
    mte = [p for p in a.mpii_test.split(",") if p]
    mtr = [p for p in cfg.get("mpii_train", "").split(",") if p]
    assert not set(mte) & set(mtr), "MPII test persons were in the base model's training set"
    tr = Faces(rtgene_rows(a.rtgene, adapt_s, a.per_subject), True, *bins)
    evals = {"rtgene": Faces(rtgene_rows(a.rtgene, test_s), False, *bins)}
    if mte:
        evals["mpii"] = Faces(tg.mpii_rows(a.mpii_data, mte), False, *bins)
    if not a.no_g360_eval:
        evals["gaze360"] = Faces(tg.gaze360_rows(a.gaze360, "test"), False, *bins)
    if a.eval_limit:
        for d in evals.values():
            d.rows = d.rows[::max(1, len(d.rows) // a.eval_limit)][:a.eval_limit]
    dev = torch.device("cuda"); torch.backends.cudnn.benchmark = True
    model = tm.build_model(pretrained=False, backbone=cfg.get("backbone", "resnet50"))
    model.load_state_dict(torch.load(os.path.join(a.init, "last.pkl"), map_location="cpu"))
    model = model.to(dev).to(memory_format=torch.channels_last)
    n_rep = int(round(a.batch_size * a.replay)); n_new = a.batch_size - n_rep
    it_new = endless(DataLoader(tr, batch_size=n_new, shuffle=True, num_workers=a.workers // 2 if n_rep else a.workers,
                                drop_last=len(tr) >= n_new, pin_memory=True, persistent_workers=True))
    it_rep = None
    if n_rep:
        src = ("data" in cfg and "bins" in cfg)  # train_gaze360 run: Gaze360 train + its MPII persons; else MPII persons
        rep_rows = (tg.gaze360_rows(cfg["data"], "train") if src else []) + tg.mpii_rows(a.mpii_data, mtr or [])
        assert rep_rows, "replay needs the base model's training data"
        it_rep = endless(DataLoader(Faces(rep_rows, True, *bins), batch_size=n_rep, shuffle=True, num_workers=a.workers // 2,
                                    drop_last=True, pin_memory=True, persistent_workers=True))
    loaders = {k: DataLoader(d, batch_size=128, num_workers=a.workers, pin_memory=True) for k, d in evals.items()}
    opt = torch.optim.Adam(tm.param_groups(model, a.lr), a.lr)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(i, a.iters) / a.iters)))
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    idx = torch.arange(tm.BINS, dtype=torch.float32, device=dev)
    json.dump(vars(a) | {"bins": tm.BINS, "bin_w": tm.BIN_W, "angle": tm.ANGLE, "base_backbone": cfg.get("backbone", "resnet50"),
                         "adapt_subjects": adapt_s, "test_subjects": test_s, "n_adapt": len(tr),
                         **{f"n_eval_{k}": len(d) for k, d in evals.items()}, "replay_per_batch": n_rep,
                         "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                         "started": datetime.now().isoformat(timespec="seconds")},
              open(os.path.join(a.out, "config.json"), "w"), indent=2)
    cols = ["iter", "train_loss", "rtgene_mean", "rtgene_subject_mean", "rtgene_median", "mpii_person_mean", "gaze360_mean", "minutes"]
    mp = os.path.join(a.out, "metrics.csv"); csv.writer(open(mp, "w", newline="")).writerow(cols)
    t0 = time.time(); tot = n = 0

    def run_eval(it):
        res = {}
        for k, dl in loaders.items():
            pc = os.path.join(a.out, "predictions", f"{k}_iter{it}.csv")
            m = tm.evaluate(model, dl, dev, torch.bfloat16, pc)
            res[k] = m["mean"]; res[k + "_median"] = m["median"]; res[k + "_group"] = group_mean(pc)
            if it not in (0, a.iters):
                os.remove(pc)  # keep only before / after per-image predictions
        row = [it, f"{tot / max(n, 1):.4f}", f"{res['rtgene']:.4f}", f"{res['rtgene_group']:.4f}", f"{res['rtgene_median']:.4f}",
               f"{res.get('mpii_group', float('nan')):.4f}", f"{res.get('gaze360', float('nan')):.4f}", f"{(time.time() - t0) / 60:.1f}"]
        csv.writer(open(mp, "a", newline="")).writerow(row)
        print("== " + " | ".join(f"{c} {v}" for c, v in zip(cols, row)), flush=True)
        tm.set_train_mode(model, False)
        return res

    before = run_eval(0)
    tm.set_train_mode(model, False)
    for it in range(1, a.iters + 1):
        x, lab, cont, _ = next(it_new)
        if it_rep is not None:
            x2, lab2, cont2, _ = next(it_rep)
            x, lab, cont = torch.cat([x, x2]), torch.cat([lab, lab2]), torch.cat([cont, cont2])
        x = x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last); lab, cont = lab.to(dev), cont.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o0, o1 = model(x)
        o0, o1 = o0.float(), o1.float()
        p0 = (torch.softmax(o0, 1) * idx).sum(1) * tm.BIN_W - tm.ANGLE; p1 = (torch.softmax(o1, 1) * idx).sum(1) * tm.BIN_W - tm.ANGLE
        loss = ce(o0, lab[:, 0]) + mse(p0, cont[:, 0]) + ce(o1, lab[:, 1]) + mse(p1, cont[:, 1])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); tot += loss.item(); n += 1
        if it % a.eval_every == 0 or it == a.iters:
            after = run_eval(it); tot = n = 0
    torch.save(model.state_dict(), os.path.join(a.out, "last.pkl"))
    summ = {"finished": datetime.now().isoformat(timespec="seconds"), "before": before, "after": after,
            "minutes": (time.time() - t0) / 60}
    json.dump(summ, open(os.path.join(a.out, "done.json"), "w"), indent=1)
    print("done", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
