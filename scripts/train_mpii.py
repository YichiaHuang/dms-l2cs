"""L2CS-Net on MPIIFaceGaze, one leave-one-person-out fold per run, evaluated on the held-out person every epoch.

Why not the repo's train.py / test.py: their mpiigaze branches do not run as shipped
(get_ignored_params called with 2 args, DataParallel wrapping breaks model.conv1, Mpiigaze(...)
gets fold passed as `angle`, test.py reads args.bins/args.angle that are never defined, and
leave_one_out_eval.py has os.path.exist + a hard-coded epoch).

Recipes
  --recipe paper     (default) ImageNet ResNet-50, 28 bins x 3 deg over [-42, 42), CE + alpha*MSE on the
                     softmax expectation, Adam lr 1e-5 constant, batch 16, conv1/bn1 frozen, no augmentation.
  --recipe improved  same model/loss, plus (1) cosine LR decay 1e-5 -> 1e-7 over the run and
                     (2) light augmentation: colour jitter, +-5 % translation / 5 % scale, and horizontal
                     flip with the horizontal gaze angle negated.
  Individual flags (--sched, --aug, --lr, ...) override the recipe.

--amp picks bf16 autocast where the GPU supports it natively (Ampere+), otherwise fp16 autocast with a
GradScaler (e.g. V100 on TWCC); force one with --amp-dtype. The dtype used is written to config.json.

Quirk kept from the original: train.py puts every BatchNorm inside layer1-4 into eval() mode
(ImageNet running stats are frozen) while the stem bn1 stays in train mode. --bn-train turns that off.

Everything is logged to CSV in --out:
  metrics.csv            one row per epoch (train losses split CE/MSE, test angular error mean/median/std,
                         per-axis MAE, lr, time, throughput, GPU memory, timestamp)
  iters.csv              one row per --log-every iterations (window-averaged losses, lr, img/s)
  predictions/epoch_N.csv per-image prediction on the held-out person for every epoch
  config.json, best.json, done.json, best.pkl, last.pkl [, epoch_N.pkl]

  python train_mpii.py --data ~/datasets/MPIIFaceGaze --fold 0 --epochs 8 --amp --recipe improved --out ~/runs/l2cs/fold0_improved
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "L2CS-Net"))
try:  # l2cs/__init__ imports pipeline.py -> face_detection (RetinaFace), which training never uses.
    import face_detection  # noqa: F401
except ImportError:  # allow training on machines where face_detection can't be installed
    import types
    _fd = types.ModuleType("face_detection")
    def _no_retinaface(*a, **k):
        raise ImportError("face_detection is not installed (only needed for the inference Pipeline)")
    _fd.RetinaFace = _no_retinaface
    sys.modules["face_detection"] = _fd
from l2cs import L2CS, Mpiigaze, gazeto3d, angular  # noqa: E402

BINS, BIN_W, ANGLE = 28, 3, 42
RESNET50_URL = "https://download.pytorch.org/models/resnet50-19c8e357.pth"
NORMALIZE = transforms.Compose([
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
RECIPES = {
    "paper": dict(sched="const", aug=False),
    "improved": dict(sched="cosine", aug=True),
}


class MpiiFold(Mpiigaze):
    """Same file list / filtering as l2cs.Mpiigaze, plus optional augmentation with label-consistent flip."""

    def __init__(self, labels, root, train, fold, aug=False):
        super().__init__(labels, root, NORMALIZE, train, ANGLE, fold)
        self.aug = aug
        self.aug_tf = transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        ])
        self.bins = np.array(range(-ANGLE, ANGLE, BIN_W))

    def __getitem__(self, idx):
        line = self.lines[idx].strip().split(" ")
        img = Image.open(os.path.join(self.root, line[0])).convert("RGB")
        label = np.array(line[7].split(","), dtype=np.float64)
        # L2CS naming: label[0] = "pitch" (actually the horizontal angle), label[1] = "yaw" (vertical)
        pitch, yaw = float(np.degrees(label[0])), float(np.degrees(label[1]))
        flipped = 0
        if self.aug:
            img = self.aug_tf(img)
            if random.random() < 0.5:
                img = ImageOps.mirror(img)
                pitch = -pitch
                flipped = 1
        img = self.transform(img)
        binned = np.digitize([pitch, yaw], self.bins) - 1
        return img, torch.as_tensor(binned), torch.FloatTensor([pitch, yaw]), f"{line[0]}|{line[3]}|{flipped}"


def build_model(pretrained=True):
    model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], BINS)
    if pretrained:
        sd = torch.hub.load_state_dict_from_url(RESNET50_URL, progress=False)
        own = model.state_dict()
        own.update({k: v for k, v in sd.items() if k in own and v.shape == own[k].shape})
        model.load_state_dict(own)
    return model


def set_train_mode(model, bn_train):
    model.train()
    if not bn_train:
        for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
            for name, m in layer.named_modules():
                if "bn" in name:
                    m.eval()


def param_groups(model, lr):
    ignored = [p for m in (model.conv1, model.bn1, model.fc_finetune) for p in m.parameters()]
    body = [p for m in (model.layer1, model.layer2, model.layer3, model.layer4) for p in m.parameters()]
    fc = [p for m in (model.fc_yaw_gaze, model.fc_pitch_gaze) for p in m.parameters()]
    return [{"params": ignored, "lr": 0}, {"params": body, "lr": lr}, {"params": fc, "lr": lr}]


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, pred_csv=None):
    model.eval()
    idx = torch.arange(BINS, dtype=torch.float32, device=device)
    rows, errs = [], []
    for images, _, cont, names in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
            pitch, yaw = model(images)
        p = (torch.softmax(pitch.float(), 1) * idx).sum(1).cpu().numpy() * BIN_W - ANGLE
        y = (torch.softmax(yaw.float(), 1) * idx).sum(1).cpu().numpy() * BIN_W - ANGLE
        pl, yl = cont[:, 0].numpy(), cont[:, 1].numpy()
        for nm, a, b, c, d in zip(names, p, y, pl, yl):
            e = angular(gazeto3d([math.radians(a), math.radians(b)]), gazeto3d([math.radians(c), math.radians(d)]))
            errs.append(e)
            rows.append([nm.split("|")[0], nm.split("|")[1], f"{c:.3f}", f"{d:.3f}", f"{a:.3f}", f"{b:.3f}", f"{e:.4f}"])
    errs = np.array(errs)
    p_mae = np.mean([abs(float(r[4]) - float(r[2])) for r in rows])
    y_mae = np.mean([abs(float(r[5]) - float(r[3])) for r in rows])
    if pred_csv:
        with open(pred_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["face", "origin", "gt_pitch_deg", "gt_yaw_deg", "pred_pitch_deg", "pred_yaw_deg", "angular_err_deg"])
            w.writerows(rows)
    return dict(mean=float(errs.mean()), median=float(np.median(errs)), std=float(errs.std()),
                p90=float(np.percentile(errs, 90)), pitch_mae=float(p_mae), yaw_mae=float(y_mae), n=len(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with Image/ and Label/")
    ap.add_argument("--fold", type=int, required=True, help="held-out person index 0..14")
    ap.add_argument("--recipe", choices=list(RECIPES), default="paper")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr-min", type=float, default=1e-7, help="cosine floor")
    ap.add_argument("--sched", choices=["const", "cosine"], default=None)
    ap.add_argument("--aug", dest="aug", action="store_true", default=None)
    ap.add_argument("--no-aug", dest="aug", action="store_false")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--amp", action="store_true", help="mixed precision autocast (see --amp-dtype)")
    ap.add_argument("--amp-dtype", choices=["auto", "bf16", "fp16"], default="auto",
                    help="auto = bf16 if the GPU supports it natively, else fp16 + GradScaler")
    ap.add_argument("--bn-train", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--max-iters", type=int, default=0, help="debug: stop each epoch after N iters")
    ap.add_argument("--save-every-epoch", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/state.pt if present (same --epochs)")
    ap.add_argument("--stop-after-epoch", type=int, default=0, help="debug: exit after this epoch without done.json")
    args = ap.parse_args()
    for k, v in RECIPES[args.recipe].items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    if os.path.exists(os.path.join(args.out, "done.json")):
        print(f"{args.out} already finished, skipping"); return
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    os.makedirs(os.path.join(args.out, "predictions"), exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    amp_dtype = None
    if args.amp:
        if args.amp_dtype == "auto":
            try:
                native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
            except TypeError:  # older torch: no keyword, fall back to compute capability
                native_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
            args.amp_dtype = "bf16" if native_bf16 else "fp16"
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.amp_dtype]
    else:
        args.amp_dtype = "none"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)
    print(f"device {torch.cuda.get_device_name(0)}, amp={args.amp_dtype}", flush=True)

    label_dir = os.path.join(args.data, "Label")
    labels = [os.path.join(label_dir, f) for f in sorted(os.listdir(label_dir))]
    assert len(labels) == 15, labels
    img_dir = os.path.join(args.data, "Image")
    train_set = MpiiFold(labels, img_dir, True, args.fold, aug=args.aug)
    test_set = MpiiFold(labels, img_dir, False, args.fold, aug=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=args.workers, pin_memory=True)
    held_out = os.path.basename(labels[args.fold]).split(".")[0]
    print(f"fold {args.fold} ({held_out}) recipe={args.recipe} sched={args.sched} aug={args.aug}: "
          f"train {len(train_set)} imgs, test {len(test_set)} imgs", flush=True)

    model = build_model().to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(param_groups(model, args.lr), args.lr)
    iters_per_epoch = len(train_loader) if not args.max_iters else min(args.max_iters, len(train_loader))
    total_iters = args.epochs * iters_per_epoch
    if args.sched == "cosine":
        floor = args.lr_min / args.lr
        lr_fn = lambda it: floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(it, total_iters) / total_iters))  # noqa: E731
    else:
        lr_fn = lambda it: 1.0  # noqa: E731
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    idx = torch.arange(BINS, dtype=torch.float32, device=device)

    state_path = os.path.join(args.out, "state.pt")
    start_epoch, best, step = 1, float("inf"), 0
    resumed = False
    if args.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        assert st["epochs"] == args.epochs and st["iters_per_epoch"] == iters_per_epoch, "resume needs the same --epochs / data"
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"]); scheduler.load_state_dict(st["scheduler"])
        if st.get("scaler") is not None and scaler.is_enabled():
            scaler.load_state_dict(st["scaler"])
        start_epoch, best, step = st["epoch"] + 1, st["best"], st["step"]
        torch.set_rng_state(st["rng_cpu"]); random.setstate(st["rng_py"]); np.random.set_state(st["rng_np"])
        resumed = True
        print(f"resumed from {state_path}: {st['epoch']} epochs done, best {best:.3f}, continuing at epoch {start_epoch}", flush=True)

    cfg = vars(args) | {"held_out": held_out, "n_train": len(train_set), "n_test": len(test_set),
                        "iters_per_epoch": iters_per_epoch, "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0), "started": datetime.now().isoformat(timespec="seconds"), "resumed_at_epoch": start_epoch if resumed else None}
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    if resumed:  # drop any rows from a partially finished epoch, keep the rest
        for name, col in (("metrics.csv", 0), ("iters.csv", 0)):
            fp = os.path.join(args.out, name)
            if os.path.exists(fp):
                lines = open(fp).read().splitlines()
                keep = [lines[0]] + [l for l in lines[1:] if l and int(l.split(",")[col]) < start_epoch]
                open(fp, "w").write("\n".join(keep) + "\n")
    mfile = open(os.path.join(args.out, "metrics.csv"), "a", newline="")
    mw = csv.writer(mfile)
    if not resumed:
        mw.writerow(["epoch", "train_loss_total", "train_ce_pitch", "train_mse_pitch", "train_ce_yaw", "train_mse_yaw",
                 "test_angular_err", "test_err_median", "test_err_std", "test_err_p90", "test_pitch_mae", "test_yaw_mae",
                 "n_test", "lr_end", "epoch_sec", "img_per_sec", "gpu_mem_mb", "timestamp"])
    ifile = open(os.path.join(args.out, "iters.csv"), "a", newline="")
    iw = csv.writer(ifile)
    if not resumed:
        iw.writerow(["epoch", "iter", "global_step", "loss_total", "ce_pitch", "mse_pitch", "ce_yaw", "mse_yaw", "lr", "img_per_sec", "timestamp"])
        ev = evaluate(model, test_loader, device, amp_dtype, os.path.join(args.out, "predictions", "epoch_0.csv"))
        print(f"epoch 0 (ImageNet init, untrained heads): test err {ev['mean']:.3f} deg on {ev['n']}", flush=True)
        mw.writerow([0, "", "", "", "", "", f"{ev['mean']:.4f}", f"{ev['median']:.4f}", f"{ev['std']:.4f}", f"{ev['p90']:.4f}",
                     f"{ev['pitch_mae']:.4f}", f"{ev['yaw_mae']:.4f}", ev["n"], args.lr, "", "", "", datetime.now().isoformat(timespec="seconds")])
        mfile.flush()

    for epoch in range(start_epoch, args.epochs + 1):
        set_train_mode(model, args.bn_train)
        t0 = time.time()
        acc = np.zeros(5); win = np.zeros(5); n_it = n_win = 0; t_win = time.time()
        for i, (images, labels_b, cont, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            lp, ly = labels_b[:, 0].to(device), labels_b[:, 1].to(device)
            cp, cy = cont[:, 0].to(device), cont[:, 1].to(device)
            with torch.autocast("cuda", dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
                pitch, yaw = model(images)
            pitch, yaw = pitch.float(), yaw.float()
            pp = (torch.softmax(pitch, 1) * idx).sum(1) * BIN_W - ANGLE
            yp = (torch.softmax(yaw, 1) * idx).sum(1) * BIN_W - ANGLE
            ce_p, mse_p, ce_y, mse_y = ce(pitch, lp), mse(pp, cp), ce(yaw, ly), mse(yp, cy)
            loss = ce_p + args.alpha * mse_p + ce_y + args.alpha * mse_y
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            parts = np.array([loss.item(), ce_p.item(), mse_p.item(), ce_y.item(), mse_y.item()])
            acc += parts; win += parts; n_it += 1; n_win += 1
            if n_win == args.log_every:
                ips = n_win * args.batch_size / (time.time() - t_win)
                iw.writerow([epoch, i + 1, step] + [f"{v:.5f}" for v in win / n_win] + [f"{scheduler.get_last_lr()[1]:.3e}", f"{ips:.1f}", datetime.now().isoformat(timespec="seconds")])
                win[:] = 0; n_win = 0; t_win = time.time()
            if (i + 1) % 200 == 0:
                ifile.flush()
                print(f"epoch {epoch} iter {i+1}/{iters_per_epoch} loss {acc[0]/n_it:.4f} lr {scheduler.get_last_lr()[1]:.2e} "
                      f"{(i+1)*args.batch_size/(time.time()-t0):.1f} img/s", flush=True)
            if args.max_iters and n_it >= args.max_iters:
                break
        dt = time.time() - t0
        ips = n_it * args.batch_size / dt
        ev = evaluate(model, test_loader, device, amp_dtype, os.path.join(args.out, "predictions", f"epoch_{epoch}.csv"))
        m = acc / n_it
        print(f"== epoch {epoch}/{args.epochs}: train loss {m[0]:.4f} | test err {ev['mean']:.3f} deg "
              f"(median {ev['median']:.2f}, pitch MAE {ev['pitch_mae']:.2f}, yaw MAE {ev['yaw_mae']:.2f}) | {dt/60:.1f} min, {ips:.1f} img/s", flush=True)
        mw.writerow([epoch] + [f"{v:.5f}" for v in m] + [f"{ev['mean']:.4f}", f"{ev['median']:.4f}", f"{ev['std']:.4f}", f"{ev['p90']:.4f}",
                     f"{ev['pitch_mae']:.4f}", f"{ev['yaw_mae']:.4f}", ev["n"], f"{scheduler.get_last_lr()[1]:.3e}", f"{dt:.1f}", f"{ips:.1f}",
                     f"{torch.cuda.max_memory_allocated()/2**20:.0f}", datetime.now().isoformat(timespec="seconds")])
        mfile.flush(); ifile.flush()
        torch.save(model.state_dict(), os.path.join(args.out, "last.pkl"))
        if args.save_every_epoch:
            torch.save(model.state_dict(), os.path.join(args.out, f"epoch_{epoch}.pkl"))
        if ev["mean"] < best:
            best = ev["mean"]
            torch.save(model.state_dict(), os.path.join(args.out, "best.pkl"))
            json.dump({"best_epoch": epoch, "best_err": best}, open(os.path.join(args.out, "best.json"), "w"))
        torch.save({"epoch": epoch, "step": step, "best": best, "epochs": args.epochs, "iters_per_epoch": iters_per_epoch,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "rng_cpu": torch.get_rng_state(), "rng_py": random.getstate(), "rng_np": np.random.get_state()}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
        if args.stop_after_epoch and epoch == args.stop_after_epoch:
            print(f"stopping after epoch {epoch} (debug)", flush=True); return
    json.dump({"finished": datetime.now().isoformat(timespec="seconds"), "best_err": best, "epochs": args.epochs},
              open(os.path.join(args.out, "done.json"), "w"))
    print(f"done. best test err {best:.3f} deg", flush=True)


if __name__ == "__main__":
    main()
