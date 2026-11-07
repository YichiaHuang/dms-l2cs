"""Headless L2CS-Net runner: image / video file / camera -> annotated output + per-frame CSV.

The official demo.py needs a webcam and cv2.imshow, neither of which works out of
the box in WSL2, and Pipeline.step() crashes on frames with no face (np.stack of an
empty list). This wrapper avoids both so the model can be checked on this machine.

  python run_l2cs.py -i ../Week2/yolov7/inference/images/zidane.jpg -o out/zidane.jpg
  python run_l2cs.py -i drive.mp4 -o out/drive.mp4 --csv out/drive.csv
"""
import argparse
import csv
import pathlib
import time

import cv2
import numpy as np
import torch

from l2cs import Pipeline, render

HERE = pathlib.Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def step_safe(pipeline, frame):
    """Pipeline.step() raises ValueError when no face passes the threshold."""
    try:
        return pipeline.step(frame)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="image, video file, or camera index")
    ap.add_argument("-o", "--output", default=None, help="annotated image/video path")
    ap.add_argument("--csv", default=None, help="per-frame pitch/yaw log (largest face)")
    ap.add_argument("--weights", default=str(HERE / "L2CS-Net" / "models" / "L2CSNet_gaze360.pkl"))
    ap.add_argument("--arch", default="ResNet50")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("-n", "--frames", type=int, default=0, help="max frames (0 = all)")
    args = ap.parse_args()

    device = torch.device(args.device)
    weights = pathlib.Path(args.weights)
    if not weights.exists():
        # The official Google Drive link is dead; the py-feat re-host ships the same
        # Gaze360 weights as safetensors. Pipeline only knows torch.load, so convert once.
        from safetensors.torch import load_file
        src = weights.with_name("l2cs_gaze360_resnet50.safetensors")
        torch.save(load_file(str(src)), str(weights))
        print(f"converted {src.name} -> {weights.name}")
    pipeline = Pipeline(weights=weights, arch=args.arch, device=device)
    # face_detection's load_net() switches cudnn.benchmark on. On this Blackwell GPU
    # (sm_120) the autotuner needs 6-8 s per distinct conv shape, i.e. minutes before
    # the first frame comes back. Measured: 0.9 s first conv with it off.
    torch.backends.cudnn.benchmark = False
    print(f"model loaded on {device}")

    if args.output:
        pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if pathlib.Path(args.input).suffix.lower() in IMAGE_EXTS:
        frame = cv2.imread(args.input)
        with torch.no_grad():
            results = step_safe(pipeline, frame)
        if results is None:
            print("no face detected")
            return
        for p, y, s in zip(results.pitch, results.yaw, results.scores):
            print(f"face score={s:.2f} pitch={np.degrees(p):6.1f} deg yaw={np.degrees(y):6.1f} deg")
        if args.output:
            cv2.imwrite(args.output, render(frame, results))
            print("saved", args.output)
        return

    cap = cv2.VideoCapture(int(args.input) if args.input.isdigit() else args.input)
    if not cap.isOpened():
        raise IOError(f"cannot open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer, log, rows = None, None, 0
    if args.csv:
        log = csv.writer(open(args.csv, "w", newline=""))
        log.writerow(["frame", "t_sec", "face", "pitch_deg", "yaw_deg", "det_score"])

    idx, infer_sec = 0, 0.0
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok or (args.frames and idx >= args.frames):
                break
            t0 = time.time()
            results = step_safe(pipeline, frame)
            infer_sec += time.time() - t0

            if results is not None:
                areas = (results.bboxes[:, 2] - results.bboxes[:, 0]) * (results.bboxes[:, 3] - results.bboxes[:, 1])
                k = int(np.argmax(areas))  # driver = largest face
                row = [idx, idx / fps, 1, np.degrees(results.pitch[k]), np.degrees(results.yaw[k]), results.scores[k]]
                frame = render(frame, results)
            else:
                row = [idx, idx / fps, 0, "", "", ""]
            if log:
                log.writerow(row)
                rows += 1

            if args.output:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(frame)
            idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    print(f"{idx} frames, {idx / max(infer_sec, 1e-6):.1f} FPS (inference only)")


if __name__ == "__main__":
    main()
