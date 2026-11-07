"""Inference speed of L2CS-Net on this machine.

A) model only: ResNet-50 L2CS head, 448x448 input, GPU batch 1/4/16 x fp32/bf16, plus CPU batch 1
B) full pipeline on 1280x720 frames: RetinaFace detection -> crop/resize/normalize -> L2CS -> angles,
   each stage timed with cuda.synchronize, cudnn.benchmark off (as in run_l2cs.py) and on.
GPU clocks/power are polled in the background so a throttled run is visible.

  python bench_speed.py --out ../results/speed
"""
import argparse, csv, glob, json, os, pathlib, statistics, subprocess, sys, time

import cv2
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".." / "L2CS-Net"))
from l2cs import Pipeline  # noqa: E402
from l2cs.utils import prep_input_numpy, getArch  # noqa: E402

WEIGHTS = HERE / ".." / "L2CS-Net" / "models" / "L2CSNet_gaze360.pkl"


def stats(ms):
    ms = sorted(ms)
    return dict(n=len(ms), mean_ms=statistics.mean(ms), p50_ms=ms[len(ms) // 2], p95_ms=ms[int(len(ms) * 0.95) - 1],
                fps=1000 / statistics.mean(ms))


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_model(device, batch, amp, iters, warmup):
    torch.backends.cudnn.benchmark = True
    m = getArch("ResNet50", 90).to(device).eval()
    if device.type == "cuda":
        m = m.to(memory_format=torch.channels_last)
    x = torch.randn(batch, 3, 448, 448, device=device)
    if device.type == "cuda":
        x = x.contiguous(memory_format=torch.channels_last)
    out = []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        for i in range(warmup + iters):
            sync(); t = time.perf_counter(); m(x); sync()
            if i >= warmup:
                out.append((time.perf_counter() - t) * 1000)
    s = stats(out)
    s["img_per_s"] = batch * s["fps"]
    return s


def bench_pipeline(frames, bench_cudnn, warmup):
    dev = torch.device("cuda:0")
    pipe = Pipeline(weights=WEIGHTS, arch="ResNet50", device=dev)
    torch.backends.cudnn.benchmark = bench_cudnn
    rows = []
    with torch.no_grad():
        for i, (name, f) in enumerate(frames):
            sync(); t0 = time.perf_counter()
            faces = pipe.detector(f)
            sync(); t1 = time.perf_counter()
            crops = []
            for box, lm, score in faces or []:
                if score < pipe.confidence_threshold:
                    continue
                x0, y0, x1, y1 = max(int(box[0]), 0), max(int(box[1]), 0), int(box[2]), int(box[3])
                crops.append(cv2.resize(cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2RGB), (224, 224)))
            if crops:
                img = prep_input_numpy(np.stack(crops), dev)
                sync(); t2 = time.perf_counter()
                p, y = pipe.model(img)
                p = (torch.softmax(p, 1) * pipe.idx_tensor).sum(1).cpu().numpy()
                sync(); t3 = time.perf_counter()
            else:
                t2 = t3 = time.perf_counter()
            if i >= warmup:
                rows.append(dict(frame=name, faces=len(crops), detect_ms=(t1 - t0) * 1000, preprocess_ms=(t2 - t1) * 1000,
                                 gaze_ms=(t3 - t2) * 1000, total_ms=(t3 - t0) * 1000))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    poll = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,pstate,clocks.sm,clocks.max.sm,power.draw,utilization.gpu,temperature.gpu",
                             "--format=csv", "-lms", "500"], stdout=open(os.path.join(args.out, "gpu_poll.csv"), "w"))
    summary = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cpu_threads": torch.get_num_threads()}
    try:
        # A) model only
        model_rows = []
        dev = torch.device("cuda:0")
        for amp in (False, True):
            for b in (1, 4, 16):
                s = bench_model(dev, b, amp, iters=200 if b < 16 else 60, warmup=30)
                model_rows.append(dict(device="GPU", precision="bf16" if amp else "fp32", batch=b, **s))
                print("model", model_rows[-1], flush=True)
        s = bench_model(torch.device("cpu"), 1, False, iters=15, warmup=3)
        model_rows.append(dict(device="CPU", precision="fp32", batch=1, **s)); print("model", model_rows[-1], flush=True)
        with open(os.path.join(args.out, "model_only.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(model_rows[0].keys())); w.writeheader(); w.writerows(model_rows)
        summary["model_only"] = model_rows

        # B) full pipeline on 1280x720 frames
        mpii = sorted(glob.glob(os.path.expanduser("~/datasets/raw/MPIIFaceGaze/MPIIFaceGaze/p0[0-4]/day01/*.jpg")))[:150]
        frames = [(os.path.relpath(p, os.path.expanduser("~/datasets/raw")), cv2.imread(p)) for p in mpii]
        frames = [(n, f) for n, f in frames if f is not None]
        zid = cv2.imread(str(HERE / ".." / ".." / "Week2" / "yolov7" / "inference" / "images" / "zidane.jpg"))
        summary["frame_size"] = list(frames[0][1].shape[:2][::-1])
        for bench_cudnn in (False, True):
            tag = "cudnn_benchmark_on" if bench_cudnn else "cudnn_benchmark_off"
            rows = bench_pipeline(frames, bench_cudnn, warmup=10)
            rows2 = bench_pipeline([("zidane.jpg", zid)] * 40, bench_cudnn, warmup=10)
            with open(os.path.join(args.out, f"pipeline_{tag}.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows + rows2)
            for name, rr in (("mpii_1face", [r for r in rows if r["faces"] == 1]), ("zidane_2faces", rows2)):
                agg = {k: statistics.mean(r[k] for r in rr) for k in ("detect_ms", "preprocess_ms", "gaze_ms", "total_ms")}
                agg["p95_total_ms"] = sorted(r["total_ms"] for r in rr)[int(len(rr) * 0.95) - 1]
                agg["fps"] = 1000 / agg["total_ms"]; agg["n"] = len(rr)
                summary[f"pipeline_{tag}_{name}"] = agg
                print(tag, name, {k: round(v, 2) for k, v in agg.items()}, flush=True)
    finally:
        poll.terminate()
    g = list(csv.reader(open(os.path.join(args.out, "gpu_poll.csv"))))[1:]
    act = [r for r in g if len(r) > 5 and int(r[5].strip().split()[0]) > 20]
    summary["gpu_while_busy"] = dict(samples=len(act), pstates=sorted({r[1].strip() for r in act}),
                                     sm_clock_mhz_median=statistics.median(int(r[2].split()[0]) for r in act) if act else None,
                                     power_w_median=statistics.median(float(r[4].split()[0]) for r in act) if act else None,
                                     max_clock=g[0][3].strip() if g else None)
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print("GPU while busy:", summary["gpu_while_busy"])


if __name__ == "__main__":
    main()
