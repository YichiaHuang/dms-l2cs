"""Quantization-aware fine-tuning (QAT) of a trained L2CS variant, then int8 evaluation on the held-out person.

Starts from <run>/last.pkl (end-of-training weights, not the test-selected best.pkl), inserts fake-quant with
PyTorch FX (x86 QAT qconfig: per-channel int8 weights, uint8 activations), fine-tunes on the fold's TRAINING
persons with the same loss (CE + MSE, improved-recipe augmentation), Adam at a small constant lr. Epoch 1 trains
everything; from epoch 2 the observers and BN statistics are frozen (standard QAT schedule). Then converts to a
real int8 model and evaluates it on CPU. Output: one JSON row per run (appended to --out) + <out_dir>/<run>_qat.pt.

  python -I scripts/qat_finetune.py --run /work/u8714813/runs/l2cs_light/fold0_effb0_448_e20 \
      --data /work/u8714813/datasets/MPIIFaceGaze --out results/lightweight/qat.json --save-dir /work/.../qat
"""
import argparse
import copy
import fcntl
import importlib.util
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("train_mpii", os.path.join(HERE, "train_mpii.py"))
tm = importlib.util.module_from_spec(_s); _s.loader.exec_module(tm)
_s = importlib.util.spec_from_file_location("eval_int8", os.path.join(HERE, "eval_int8.py"))
ei = importlib.util.module_from_spec(_s); _s.loader.exec_module(ei)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--suffix", default="", help="variant name: JSON key <run><suffix>, files *_qat<suffix>*")
    ap.add_argument("--eval-only", action="store_true", help="re-evaluate the saved int8 model (writes per-image predictions)")
    a = ap.parse_args()
    from torch.ao.quantization import get_default_qat_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx
    torch.manual_seed(0)
    name = os.path.basename(a.run.rstrip("/"))
    key = name + a.suffix
    int8_path = os.path.join(a.save_dir, f"{name}_qat_int8{a.suffix}.pt")
    pred_csv = os.path.join(a.run, "predictions", f"final_int8_qat{a.suffix}.csv")
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    bb, size, fold = cfg.get("backbone", "resnet50"), cfg.get("img_size", 448), cfg["fold"]
    labels = [os.path.join(a.data, "Label", f) for f in sorted(os.listdir(os.path.join(a.data, "Label")))]
    img_dir = os.path.join(a.data, "Image")
    train = tm.MpiiFold(labels, img_dir, True, fold, aug=True, img_size=size)
    test = tm.MpiiFold(labels, img_dir, False, fold, aug=False, img_size=size)
    tl = None if a.eval_only else DataLoader(train, batch_size=16, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
    vl = DataLoader(test, batch_size=64, num_workers=a.workers)

    model = tm.build_model(pretrained=False, backbone=bb)
    model.load_state_dict(torch.load(os.path.join(a.run, "last.pkl"), map_location="cpu"))
    model.train()
    qat = prepare_qat_fx(model, get_default_qat_qconfig_mapping("x86"), (torch.randn(1, 3, size, size),))
    if a.eval_only:  # same graph -> convert -> load the saved int8 weights; reproduces the stored number exactly
        torch.set_num_threads(a.threads)
        int8 = convert_fx(qat.eval())
        int8.load_state_dict(torch.load(int8_path, map_location="cpu"))
        err, ntest = ei.evaluate_cpu(int8, vl, pred_csv=pred_csv)
        print(json.dumps({key: dict(int8_qat_err_reeval=err, n_test=ntest)}), flush=True)
        return
    qat = qat.cuda()
    opt = torch.optim.Adam(qat.parameters(), a.lr)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    idx = torch.arange(tm.BINS, dtype=torch.float32, device="cuda")
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        qat.train()
        if ep >= 2:  # freeze observers + BN running stats for the last epoch(s)
            qat.apply(torch.ao.quantization.disable_observer)
            qat.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
            for m in qat.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        tot = n = 0
        for images, lb, cont, _ in tl:
            images, lb, cont = images.cuda(non_blocking=True), lb.cuda(), cont.cuda()
            p, y = qat(images)
            pp = (torch.softmax(p, 1) * idx).sum(1) * tm.BIN_W - tm.ANGLE
            yp = (torch.softmax(y, 1) * idx).sum(1) * tm.BIN_W - tm.ANGLE
            loss = ce(p, lb[:, 0]) + mse(pp, cont[:, 0]) + ce(y, lb[:, 1]) + mse(yp, cont[:, 1])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        print(f"{name} qat epoch {ep}: loss {tot/n:.3f} ({(time.time()-t0)/60:.1f} min)", flush=True)
    qat = qat.cpu().eval()
    torch.set_num_threads(a.threads)
    int8 = convert_fx(copy.deepcopy(qat))
    err, ntest = ei.evaluate_cpu(int8, vl, pred_csv=pred_csv)
    os.makedirs(a.save_dir, exist_ok=True)
    torch.save(int8.state_dict(), int8_path)
    row = dict(backbone=bb, img_size=size, fold=fold, person=cfg.get("held_out"), qat_epochs=a.epochs, qat_lr=a.lr,
               int8_qat_err=err, n_test=ntest, minutes=(time.time() - t0) / 60)
    print(json.dumps({key: row}), flush=True)
    with open(a.out + ".lock", "w") as lk:  # several processes append to the same JSON
        fcntl.flock(lk, fcntl.LOCK_EX)
        out = json.load(open(a.out)) if os.path.exists(a.out) else {}
        out[key] = row
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
