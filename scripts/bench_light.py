"""Deployment cost of the L2CS variants (model only, no face detector).

For every (backbone, input size): params, GMACs (torch FlopCounterMode, 1 MAC = 2 FLOPs), GPU batch-1 latency in
fp32 / bf16 autocast, CPU batch-1 latency with 1 and 4 threads, and whether ONNX export (opset 17) works.
Also prints the theoretical frame-rate ceiling on a few automotive NPUs (peak int8 TOPS / (2 * GMACs)); real
utilisation is typically well below peak, so treat it as an upper bound only.

  python -I scripts/bench_light.py --out results/lightweight/bench_<host>.json
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("train_mpii", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_mpii.py"))
tm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(tm)

CONFIGS = [("resnet50", 448), ("resnet50", 224), ("resnet18", 448), ("mobilenet_v3_large", 448),
           ("mobilenet_v3_large", 224), ("efficientnet_b0", 448), ("efficientnet_b0", 224)]
NPUS = {"OAX8000 (~1.1 TOPS)": 1.1, "OAX4600 / AM62A (2 TOPS)": 2.0, "i.MX 8M Plus (2.3 TOPS)": 2.3,
        "TDA4VEN (4 TOPS)": 4.0, "TDA4VE (8 TOPS)": 8.0, "TDA4VH (32 TOPS)": 32.0}


def gmacs(model, size):
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.randn(1, 3, size, size)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(x)
    return fc.get_total_flops() / 2 / 1e9


def timeit(fn, warmup, iters, sync=lambda: None):
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return dict(mean_ms=statistics.mean(ts), p50_ms=ts[len(ts) // 2], p95_ms=ts[int(len(ts) * 0.95) - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-iters", type=int, default=300)
    ap.add_argument("--cpu-iters", type=int, default=20)
    ap.add_argument("--skip-cpu", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    res = dict(host=platform.node(), torch=torch.__version__, cpu=platform.processor() or platform.machine(),
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, rows=[])
    for bb, size in CONFIGS:
        m = tm.build_model(pretrained=False, backbone=bb).eval()
        row = dict(backbone=bb, img_size=size, params_m=sum(p.numel() for p in m.parameters()) / 1e6, gmacs=gmacs(m, size))
        row["npu_fps_ceiling"] = {k: v * 1e12 / (2 * row["gmacs"] * 1e9) for k, v in NPUS.items()}
        x = torch.randn(1, 3, size, size)
        try:
            import io
            torch.onnx.export(m, x, io.BytesIO(), opset_version=17, input_names=["face"], output_names=["h", "v"], dynamo=False)
            row["onnx_export"] = "ok"
        except Exception as e:  # noqa: BLE001
            row["onnx_export"] = f"fail: {type(e).__name__}: {str(e)[:120]}"
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            mg = m.cuda().to(memory_format=torch.channels_last); xg = x.cuda().contiguous(memory_format=torch.channels_last)
            with torch.no_grad():
                row["gpu_fp32"] = timeit(lambda: mg(xg), 30, a.gpu_iters, torch.cuda.synchronize)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    row["gpu_bf16"] = timeit(lambda: mg(xg), 30, a.gpu_iters, torch.cuda.synchronize)
            m = m.cpu()
        if not a.skip_cpu:
            for th in (1, 4):
                torch.set_num_threads(th)
                with torch.no_grad():
                    row[f"cpu_{th}t"] = timeit(lambda: m(x), 3, a.cpu_iters)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "npu_fps_ceiling"}), flush=True)
        res["rows"].append(row)
    json.dump(res, open(a.out, "w"), indent=1)
    print("->", a.out)


if __name__ == "__main__":
    main()
