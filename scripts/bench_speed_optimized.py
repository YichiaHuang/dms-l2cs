"""Optimized L2CS pipeline vs original, same frames: preprocessing moved to GPU (+ optional bf16).
Also checks that the optimized path gives the same gaze angles as the original one.

  python bench_speed_optimized.py --out ../results/speed
"""
import argparse, csv, glob, json, os, pathlib, statistics, sys, time
import cv2, numpy as np, torch, torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".." / "L2CS-Net"))
from l2cs import Pipeline  # noqa: E402
from l2cs.utils import prep_input_numpy  # noqa: E402

WEIGHTS = HERE / ".." / "L2CS-Net" / "models" / "L2CSNet_gaze360.pkl"
dev = torch.device("cuda:0")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def sync(): torch.cuda.synchronize()


def crops_of(pipe, f, faces):
    out = []
    for box, lm, score in faces or []:
        if score < pipe.confidence_threshold:
            continue
        x0, y0, x1, y1 = max(int(box[0]), 0), max(int(box[1]), 0), int(box[2]), int(box[3])
        out.append(cv2.resize(cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2RGB), (224, 224)))
    return out


def prep_gpu(crops):
    x = torch.from_numpy(np.stack(crops)).to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    x = F.interpolate(x, size=(448, 448), mode="bilinear", align_corners=False, antialias=True)
    return ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)


def angles(pipe, logits):
    p, y = logits
    p = (torch.softmax(p.float(), 1) * pipe.idx_tensor).sum(1) * 4 - 180
    y = (torch.softmax(y.float(), 1) * pipe.idx_tensor).sum(1) * 4 - 180
    return torch.stack([p, y], 1).cpu().numpy()


def run(pipe, frames, mode, warmup=10):
    rows, preds = [], []
    with torch.no_grad():
        for i, (name, f) in enumerate(frames):
            sync(); t0 = time.perf_counter()
            faces = pipe.detector(f)
            sync(); t1 = time.perf_counter()
            crops = crops_of(pipe, f, faces)
            if not crops:
                continue
            x = prep_input_numpy(np.stack(crops), dev) if mode == "original" else prep_gpu(crops)
            sync(); t2 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(mode == "gpu_prep_bf16")):
                a = angles(pipe, pipe.model(x))
            sync(); t3 = time.perf_counter()
            preds.append(a)
            if i >= warmup:
                rows.append(dict(mode=mode, frame=name, faces=len(crops), detect_ms=(t1 - t0) * 1e3, preprocess_ms=(t2 - t1) * 1e3,
                                 gaze_ms=(t3 - t2) * 1e3, total_ms=(t3 - t0) * 1e3))
    return rows, np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    paths = sorted(glob.glob(os.path.expanduser("~/datasets/raw/MPIIFaceGaze/MPIIFaceGaze/p0[0-4]/day01/*.jpg")))[:150]
    frames = [(os.path.relpath(p, os.path.expanduser("~/datasets/raw")), cv2.imread(p)) for p in paths]
    pipe = Pipeline(weights=WEIGHTS, arch="ResNet50", device=dev)
    torch.backends.cudnn.benchmark = False
    pipe.model = pipe.model.to(memory_format=torch.channels_last)
    allrows, P, summ = [], {m: [] for m in ("original", "gpu_prep", "gpu_prep_bf16")}, {}
    modes = ["original", "gpu_prep", "gpu_prep_bf16"]
    per = {m: [] for m in modes}
    for rep in range(2):
        for i, fr in enumerate(frames):
            order = modes[i % 3:] + modes[:i % 3]
            for m in order:
                rows, pr = run(pipe, [fr], m, warmup=0)
                if rep == 1 and i >= 5:
                    per[m] += rows
                if rep == 1:
                    P[m].append(pr)
    for m in modes:
        P[m] = np.concatenate(P[m])
    for mode in modes:
        rows = per[mode]
        allrows += rows
        r1 = [r for r in rows if r["faces"] == 1]
        agg = {k: statistics.mean(r[k] for r in r1) for k in ("detect_ms", "preprocess_ms", "gaze_ms", "total_ms")}
        agg["p95_total_ms"] = sorted(r["total_ms"] for r in r1)[int(len(r1) * 0.95) - 1]; agg["fps"] = 1000 / agg["total_ms"]; agg["n"] = len(r1)
        summ[mode] = agg
        print(mode, {k: round(v, 2) for k, v in agg.items()}, flush=True)
    for mode in ("gpu_prep", "gpu_prep_bf16"):
        d = np.abs(P[mode] - P["original"])
        summ[mode]["angle_diff_vs_original_deg"] = dict(mean=float(d.mean()), p99=float(np.percentile(d, 99)), max=float(d.max()))
        print(mode, "angle diff vs original (deg): mean %.3f p99 %.3f max %.3f" % (d.mean(), np.percentile(d, 99), d.max()))
    with open(os.path.join(args.out, "pipeline_optimized.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
    json.dump(summ, open(os.path.join(args.out, "summary_optimized.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
