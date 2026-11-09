"""Gaze360 (official release: metadata.mat + imgs/rec_*/head/<person>/<frame>.jpg) -> face crops + label CSV.

Follows the common GazeHub protocol (Cheng et al., "Appearance-based gaze estimation with deep learning: a review
and benchmark"), which is what L2CS-Net's Gaze360 numbers use:
  * keep the official split ('train' / 'val' / 'test'); 'unused' is dropped
  * keep only frames with a detected face (person_face_bbox[0] != -1) -> the "front 180 deg" subset
  * crop the face box out of the provided head crop, resize to --size (224, like GazeHub)
  * 2D gaze from the 3D gaze direction gaze_dir (camera "eye" coordinates):
        h = atan2(x, -z)   (horizontal)        v = asin(y)   (vertical)
    i.e. the convention of train_mpii.gazeto3d / MPIIFaceGaze. Checked empirically (eval_crossdata.py --check-signs):
    with it the official L2CS Gaze360 weights give 10.56 deg on test (paper: 10.41) and our MPIIFaceGaze models
    have positive h/v correlation; the first guess atan2(-x,-z), asin(-y) had both signs flipped (NOTES.md).
Output: <out>/Image/<split>/<rec>_<person>_<frame>.jpg and <out>/Label/<split>.csv
        (face, gaze_h_deg, gaze_v_deg, recording, person, frame)

  python -I scripts/prep_gaze360.py --src /work/u8714813/datasets/raw/gaze360/extract --out /work/u8714813/datasets/Gaze360
"""
import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.io as sio
from PIL import Image


def work(job):
    src, dst, fb, hb, size = job
    if os.path.exists(dst):
        return True
    try:
        im = Image.open(src).convert("RGB")
    except Exception:  # noqa: BLE001
        return False
    W, H = im.size
    # face box (normalised to the full frame) -> pixel coords inside the head crop
    x0 = (fb[0] - hb[0]) / hb[2] * W; y0 = (fb[1] - hb[1]) / hb[3] * H
    x1 = x0 + fb[2] / hb[2] * W; y1 = y0 + fb[3] / hb[3] * H
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return False
    im.crop((int(x0), int(y0), int(round(x1)), int(round(y1)))).resize((size, size), Image.BILINEAR).save(dst, quality=95)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    m = sio.loadmat(os.path.join(a.src, "metadata.mat"))
    recs = [str(r[0]) for r in m["recordings"][0]]
    splits = [str(s[0]) for s in m["splits"][0]]
    rec, frame, pid, split = m["recording"][0], m["frame"][0], m["person_identity"][0], m["split"][0]
    g = m["gaze_dir"]
    h = np.degrees(np.arctan2(g[:, 0], -g[:, 2])); v = np.degrees(np.arcsin(np.clip(g[:, 1], -1, 1)))
    for sp in ("train", "val", "test"):
        idx = np.where((split == splits.index(sp)) & (m["person_face_bbox"][:, 0] != -1))[0]
        os.makedirs(os.path.join(a.out, "Image", sp), exist_ok=True); os.makedirs(os.path.join(a.out, "Label"), exist_ok=True)
        jobs, rows = [], []
        for i in idx:
            r = recs[rec[i]]
            src = os.path.join(a.src, "imgs", r, "head", f"{pid[i]:06d}", f"{frame[i]:06d}.jpg")
            name = f"{sp}/{r}_{pid[i]:06d}_{frame[i]:06d}.jpg"
            jobs.append((src, os.path.join(a.out, "Image", name), m["person_face_bbox"][i], m["person_head_bbox"][i], a.size))
            rows.append([name, f"{h[i]:.4f}", f"{v[i]:.4f}", r, int(pid[i]), int(frame[i])])
        with ProcessPoolExecutor(a.workers) as ex:
            ok = list(ex.map(work, jobs, chunksize=256))
        rows = [r for r, o in zip(rows, ok) if o]
        with open(os.path.join(a.out, "Label", f"{sp}.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["face", "gaze_h_deg", "gaze_v_deg", "recording", "person", "frame"]); w.writerows(rows)
        print(f"{sp}: {len(idx)} frames with a face, {len(rows)} written, {len(idx) - len(rows)} failed", flush=True)


if __name__ == "__main__":
    main()
