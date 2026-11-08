"""Latency of the deployable variants: CPU batch-1, fp32 vs int8 (PyTorch FX x86 static quantization, the same
graph eval_int8.py / qat_finetune.py produce), 1 and 4 threads; plus GPU (H200) batch-64 throughput in bf16.
Weights are random (latency does not depend on their values); int8 calibration uses random input.

  python -I scripts/bench_int8.py --out results/final/bench_int8.json
"""
import argparse
import copy
import importlib.util
import json
import os
import platform

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("train_mpii", os.path.join(HERE, "train_mpii.py"))
tm = importlib.util.module_from_spec(_s); _s.loader.exec_module(tm)
_s = importlib.util.spec_from_file_location("bench_light", os.path.join(HERE, "bench_light.py"))
bl = importlib.util.module_from_spec(_s); _s.loader.exec_module(bl)
CONFIGS = [("resnet50", 448), ("efficientnet_b0", 448), ("mobilenet_v3_large", 448)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu-iters", type=int, default=30)
    a = ap.parse_args()
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    cpu = open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": \t")
    res = dict(host=platform.node(), cpu=cpu, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               torch=torch.__version__, rows=[])
    for bb, size in CONFIGS:
        m = tm.build_model(pretrained=False, backbone=bb).eval()
        x = torch.randn(1, 3, size, size)
        p = prepare_fx(copy.deepcopy(m), get_default_qconfig_mapping("x86"), (x,))
        with torch.no_grad():
            for _ in range(4):
                p(torch.randn(8, 3, size, size))
        q = convert_fx(p)
        row = dict(backbone=bb, img_size=size)
        for th in (1, 4):
            torch.set_num_threads(th)
            with torch.no_grad():
                row[f"cpu_fp32_{th}t"] = bl.timeit(lambda: m(x), 3, a.cpu_iters)
                row[f"cpu_int8_{th}t"] = bl.timeit(lambda: q(x), 3, a.cpu_iters)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            mg = m.cuda().to(memory_format=torch.channels_last)
            xb = torch.randn(64, 3, size, size, device="cuda").contiguous(memory_format=torch.channels_last)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                t = bl.timeit(lambda: mg(xb), 20, 100, torch.cuda.synchronize)
            row["gpu_bf16_b64"] = dict(t, img_per_s=64 / (t["mean_ms"] / 1000))
        print(json.dumps(row), flush=True)
        res["rows"].append(row)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
