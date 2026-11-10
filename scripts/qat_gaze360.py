"""int8 QAT for a model trained by scripts/train_gaze360.py, then CPU int8 evaluation on the cross-dataset CSVs.

Same QAT schedule as scripts/qat_finetune.py (FX, x86 qconfig, from last.pkl, Adam 1e-5 constant, 2 epochs, observers +
BN stats frozen from epoch 2), on the SAME training data as the run (Gaze360 train + the run's --mpii-train persons,
improved-recipe augmentation). No KD during QAT (as on MPII). Then convert to int8 and evaluate on CPU, fp32 alongside
(CPU, same loader) so the int8 drop is measured on identical inputs.

  python -I scripts/qat_gaze360.py --run /work/u8714813/runs/gaze360/J1_mnv3l_kd_joint_e20 \
      --eval gaze360_test=/work/u8714813/datasets/Gaze360/Label/test.csv:/work/u8714813/datasets/Gaze360/Image ...
Writes <run>/qat_int8.pt, <out_dir>/<run>_int8.json, per-image preds to <run>/predictions/int8_<dataset>.npz.
"""
import argparse
import copy
import importlib.util
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


tg = _load("train_gaze360")  # its tm has BINS/BIN_W/ANGLE = 90/4/180
xd = _load("eval_crossdata")
tm = tg.tm


@torch.no_grad()
def predict(model, loader, threads):
    torch.set_num_threads(threads)
    idx = torch.arange(tm.BINS, dtype=torch.float32)
    out = []
    for x, _ in loader:
        o0, o1 = model(x)
        out.append(torch.stack([(torch.softmax(o0.float(), 1) * idx).sum(1) * tm.BIN_W - tm.ANGLE,
                                (torch.softmax(o1.float(), 1) * idx).sum(1) * tm.BIN_W - tm.ANGLE], 1).numpy())
    return np.concatenate(out)


def metrics(p, ds):
    gt = np.array([[float(r["gaze_h_deg"]), float(r["gaze_v_deg"])] for r in ds.rows])
    e = xd.ang_err(p[:, 0], p[:, 1], gt[:, 0], gt[:, 1])
    m = xd.summarise(e, gt[:, 0], gt[:, 1])
    gcol = next((c for c in ("person", "subject") if c in ds.rows[0]), None)
    if gcol:
        g = np.array([r[gcol] for r in ds.rows])
        m["group_mean"] = float(np.mean([e[g == k].mean() for k in np.unique(g)]))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval", nargs="+", required=True, help="name=csv:image_root")
    ap.add_argument("--out-dir", default="results/crossdata")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--max-iters", type=int, default=0, help="debug")
    ap.add_argument("--eval-limit", type=int, default=0, help="debug")
    a = ap.parse_args()
    from torch.ao.quantization import get_default_qat_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx
    torch.manual_seed(0)
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    assert cfg["bins"] == tm.BINS
    name = os.path.basename(a.run.rstrip("/"))
    mtr = [p for p in cfg.get("mpii_train", "").split(",") if p]
    train = tg.Gaze360Csv(tg.gaze360_rows(cfg["data"], "train") + tg.mpii_rows(cfg.get("mpii_data", ""), mtr), aug=True)
    tl = DataLoader(train, batch_size=16, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
    fp32 = tm.build_model(pretrained=False, backbone=cfg["backbone"])
    fp32.load_state_dict(torch.load(os.path.join(a.run, "last.pkl"), map_location="cpu"))
    qat = prepare_qat_fx(copy.deepcopy(fp32).train(), get_default_qat_qconfig_mapping("x86"), (torch.randn(1, 3, 448, 448),)).cuda()
    opt = torch.optim.Adam(qat.parameters(), a.lr)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    idx = torch.arange(tm.BINS, dtype=torch.float32, device="cuda")
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        qat.train()
        if ep >= 2:
            qat.apply(torch.ao.quantization.disable_observer)
            qat.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
            for m in qat.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        tot = n = 0
        for x, lb, cont, _ in tl:
            x, lb, cont = x.cuda(non_blocking=True), lb.cuda(), cont.cuda()
            o0, o1 = qat(x)
            p0 = (torch.softmax(o0, 1) * idx).sum(1) * tm.BIN_W - tm.ANGLE; p1 = (torch.softmax(o1, 1) * idx).sum(1) * tm.BIN_W - tm.ANGLE
            loss = ce(o0, lb[:, 0]) + mse(p0, cont[:, 0]) + ce(o1, lb[:, 1]) + mse(p1, cont[:, 1])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot += loss.item(); n += 1
            if n % 1000 == 0:
                print(f"qat epoch {ep} iter {n} loss {tot/n:.2f} ({(time.time()-t0)/60:.1f} min)", flush=True)
            if a.max_iters and n >= a.max_iters:
                break
        print(f"{name} qat epoch {ep}: loss {tot/n:.3f} ({(time.time()-t0)/60:.1f} min)", flush=True)
    int8 = convert_fx(copy.deepcopy(qat).cpu().eval())
    torch.save(int8.state_dict(), os.path.join(a.run, "qat_int8.pt"))
    fp32.eval()
    res = dict(run=name, qat_epochs=a.epochs, qat_lr=a.lr, qat_minutes=(time.time() - t0) / 60, datasets={})
    for spec in a.eval:
        dname, rest = spec.split("=", 1); csv_path, root = rest.split(":", 1)
        ds = xd.CsvFaces(csv_path, root, 448)
        if a.eval_limit:
            ds.rows = ds.rows[:: max(1, len(ds.rows) // a.eval_limit)][: a.eval_limit]
        dl = DataLoader(ds, batch_size=64, num_workers=a.workers)
        t1 = time.time(); p8 = predict(int8, dl, a.threads); t8 = time.time() - t1
        t1 = time.time(); p32 = predict(fp32, dl, a.threads); t32 = time.time() - t1
        np.savez_compressed(os.path.join(a.run, "predictions", f"int8_{dname}.npz"), int8=p8, fp32_cpu=p32,
                            names=np.array([r["face"] for r in ds.rows]))
        res["datasets"][dname] = dict(int8=metrics(p8, ds), fp32_cpu=metrics(p32, ds), n=len(ds.rows),
                                      cpu_sec_int8=t8, cpu_sec_fp32=t32)
        r = res["datasets"][dname]
        print(f"{dname}: fp32 {r['fp32_cpu']['mean']:.3f} int8 {r['int8']['mean']:.3f}"
              + (f" | person-mean fp32 {r['fp32_cpu']['group_mean']:.3f} int8 {r['int8']['group_mean']:.3f}" if "group_mean" in r["int8"] else ""),
              flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(a.out_dir, f"{name}_int8.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
