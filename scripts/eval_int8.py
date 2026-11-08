"""How much accuracy does int8 post-training quantization cost? (CPU, PyTorch FX graph mode, x86 backend)

For each run dir: rebuild the model from config.json, load last.pkl (the honest end-of-training weights),
evaluate fp32 on the held-out person, then calibrate static int8 quantization on N images drawn from that fold's
TRAINING persons (never the test person) and evaluate again. Writes one JSON row per run.

  python -I scripts/eval_int8.py --runs /work/u8714813/runs/l2cs_light/fold0_mnv3l_448_e20 ... \
      --data /work/u8714813/datasets/MPIIFaceGaze --out results/lightweight/int8.json
"""
import argparse
import copy
import importlib.util
import json
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_spec = importlib.util.spec_from_file_location("train_mpii", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_mpii.py"))
tm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(tm)


@torch.no_grad()
def evaluate_cpu(model, loader):
    idx = torch.arange(tm.BINS, dtype=torch.float32)
    errs = []
    for images, _, cont, _ in loader:
        a, b = model(images)
        p = (torch.softmax(a.float(), 1) * idx).sum(1) * tm.BIN_W - tm.ANGLE
        y = (torch.softmax(b.float(), 1) * idx).sum(1) * tm.BIN_W - tm.ANGLE
        for pa, pb, ga, gb in zip(p.tolist(), y.tolist(), cont[:, 0].tolist(), cont[:, 1].tolist()):
            errs.append(tm.angular(tm.gazeto3d([math.radians(pa), math.radians(pb)]), tm.gazeto3d([math.radians(ga), math.radians(gb)])))
    return float(np.mean(errs)), len(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--weights", default="last.pkl")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
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
        model.load_state_dict(torch.load(os.path.join(run, a.weights), map_location="cpu"))
        model.eval()
        test = tm.MpiiFold(labels, img_dir, False, fold, aug=False, img_size=size)
        train = tm.MpiiFold(labels, img_dir, True, fold, aug=False, img_size=size)
        random.seed(0)
        calib = DataLoader(Subset(train, random.sample(range(len(train)), a.calib)), batch_size=32, num_workers=4)
        tl = DataLoader(test, batch_size=64, num_workers=4)
        fp32, n = evaluate_cpu(model, tl)
        ex = (torch.randn(1, 3, size, size),)
        prepared = prepare_fx(copy.deepcopy(model), get_default_qconfig_mapping("x86"), ex)
        with torch.no_grad():
            for images, *_ in calib:
                prepared(images)
        q = convert_fx(prepared)
        int8, _ = evaluate_cpu(q, tl)
        out[name] = dict(backbone=bb, img_size=size, fold=fold, person=cfg.get("held_out"), weights=a.weights,
                         fp32_err=fp32, int8_err=int8, delta=int8 - fp32, n_test=n, calib=a.calib)
        print(json.dumps({name: out[name]}), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
