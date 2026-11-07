"""Evaluate an L2CS checkpoint on MPIIFaceGaze persons (angular error, degrees).

Works for the 28-bin MPIIGaze heads trained by train_mpii.py and for the released 90-bin
Gaze360 weights (--bins 90 --bin-width 4 --angle 180), i.e. a cross-dataset zero-shot number.

  python eval_mpii.py --data ~/datasets/MPIIFaceGaze --weights ../L2CS-Net/models/L2CSNet_gaze360.pkl \
      --bins 90 --bin-width 4 --angle 180 --folds all --out ~/runs/l2cs/gaze360_zeroshot.csv
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "L2CS-Net"))
from l2cs import L2CS, Mpiigaze, gazeto3d, angular  # noqa: E402

TRANSFORM = transforms.Compose([
    transforms.Resize(448), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--bins", type=int, default=28)
    ap.add_argument("--bin-width", type=float, default=3)
    ap.add_argument("--angle", type=float, default=42)
    ap.add_argument("--folds", default="all", help="'all' or comma list, e.g. 0,3")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", required=True, help="csv path")
    args = ap.parse_args()

    device = torch.device("cuda")
    model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], args.bins)
    sd = torch.load(args.weights, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    idx = torch.arange(args.bins, dtype=torch.float32, device=device)

    label_dir = os.path.join(args.data, "Label")
    labels = [os.path.join(label_dir, f) for f in sorted(os.listdir(label_dir))]
    folds = range(15) if args.folds == "all" else [int(x) for x in args.folds.split(",")]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    for fold in folds:
        ds = Mpiigaze(labels, os.path.join(args.data, "Image"), TRANSFORM, False, 42, fold)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=args.workers, pin_memory=True)
        errs, pe, ye = [], [], []
        with torch.no_grad():
            for images, _, cont, _ in loader:
                images = images.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                    pitch, yaw = model(images)
                p = (torch.softmax(pitch.float(), 1) * idx).sum(1).cpu().numpy() * args.bin_width - args.angle
                y = (torch.softmax(yaw.float(), 1) * idx).sum(1).cpu().numpy() * args.bin_width - args.angle
                pl, yl = cont[:, 0].numpy(), cont[:, 1].numpy()
                pe.extend(np.abs(p - pl)); ye.extend(np.abs(y - yl))
                for a, b, c, d in zip(np.radians(p), np.radians(y), np.radians(pl), np.radians(yl)):
                    errs.append(angular(gazeto3d([a, b]), gazeto3d([c, d])))
        person = os.path.basename(labels[fold]).split(".")[0]
        row = [person, len(errs), f"{np.mean(errs):.4f}", f"{np.mean(pe):.4f}", f"{np.mean(ye):.4f}"]
        rows.append(row)
        print(f"{person}: n={len(errs)} angular {np.mean(errs):.3f} deg (|d label0| {np.mean(pe):.2f}, |d label1| {np.mean(ye):.2f})", flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["person", "n", "angular_err_deg", "mae_label0_deg", "mae_label1_deg"])
        w.writerows(rows)
    if len(rows) > 1:
        print(f"mean over {len(rows)} persons: {np.mean([float(r[2]) for r in rows]):.3f} deg")


if __name__ == "__main__":
    main()
