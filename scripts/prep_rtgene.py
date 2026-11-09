"""RT-GENE (Zenodo record 2529036) -> face crops + label CSV for scripts/eval_crossdata.py.

Only the s0XX_glasses archives carry gaze labels (label_combined.txt); s0XX_noglasses are unlabelled natural
images and are not used. We use the official inpainted face crops (eye-tracking glasses removed by the authors).
label_combined.txt line:  <id>, [head_a, head_b], [gaze_a, gaze_b], <timestamp>   (radians)
  Empirically (eval_crossdata.py --check-signs, NOTES.md) gaze_a is HORIZONTAL and gaze_b VERTICAL, both with the
  train_mpii.gazeto3d sign convention: the official L2CS Gaze360 model and 3 of our MPII models all agree
  (swapped order: corr ~0.55-0.67; our first guess a=vertical gave corr ~0). -> gaze_h_deg = deg(a), gaze_v_deg = deg(b).
  Head pose is assumed to use the same order (not verified; head pose is only stored, never used).
Image: <subject>/inpainted/face/face_<id:06d>_rgb.png
Output: <out>/Image/<subject>/face_<id>.png (copied as-is) and <out>/Label/all.csv
        (face, gaze_h_deg, gaze_v_deg, subject, id, head_h_deg, head_v_deg)

  python -I scripts/prep_rtgene.py --src /work/u8714813/datasets/raw/rtgene --out /work/u8714813/datasets/RTGENE
"""
import argparse
import csv
import glob
import math
import os
import re
import tarfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "Label"), exist_ok=True)
    rows = []
    for tp in sorted(glob.glob(os.path.join(a.src, "s*_glasses.tar"))):
        if not os.path.exists(tp + ".ok"):
            print("skip (not verified):", tp); continue
        subj = os.path.basename(tp)[: -len(".tar")]
        dst = os.path.join(a.out, "Image", subj); os.makedirs(dst, exist_ok=True)
        lab, n_img = None, 0
        with tarfile.open(tp) as tf:
            for mem in tf:
                if mem.name == f"{subj}/label_combined.txt":
                    lab = tf.extractfile(mem).read().decode()
                m = re.fullmatch(rf"{subj}/inpainted/face/face_(\d+)_rgb\.png", mem.name)
                if m and mem.isfile():
                    out = os.path.join(dst, f"face_{int(m.group(1)):06d}.png")
                    if not os.path.exists(out):
                        with open(out, "wb") as fh:
                            fh.write(tf.extractfile(mem).read())
                    n_img += 1
        have = {int(f[5:11]) for f in os.listdir(dst)}
        n_lab = n_kept = 0
        for line in lab.strip().splitlines():
            nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", line)]
            i, hth, hph, gth, gph = int(nums[0]), *nums[1:5]  # (a, b) order as in the file
            n_lab += 1
            if i not in have:
                continue
            n_kept += 1
            rows.append([f"{subj}/face_{i:06d}.png", f"{math.degrees(gth):.4f}", f"{math.degrees(gph):.4f}", subj, i,
                         f"{math.degrees(hth):.4f}", f"{math.degrees(hph):.4f}"])
        print(f"{subj}: {n_img} faces, {n_lab} labels, {n_kept} matched", flush=True)
    with open(os.path.join(a.out, "Label", "all.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["face", "gaze_h_deg", "gaze_v_deg", "subject", "id", "head_h_deg", "head_v_deg"]); w.writerows(rows)
    print("total", len(rows))


if __name__ == "__main__":
    main()
