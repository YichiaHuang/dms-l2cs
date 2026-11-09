"""MPIIFaceGaze (prep_mpiifacegaze.py output: Label/pXX.label + Image/) -> one CSV for scripts/eval_crossdata.py.
2DGaze (column 8, radians) = (horizontal, vertical) in the train_mpii.gazeto3d convention -> degrees. All 15 x 3000 faces.

  python -I scripts/mpii_to_csv.py --data /work/u8714813/datasets/MPIIFaceGaze --out /work/u8714813/datasets/MPIIFaceGaze/Label_csv/all.csv
"""
import argparse
import csv
import math
import os

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
rows = []
for f in sorted(os.listdir(os.path.join(a.data, "Label"))):
    for line in open(os.path.join(a.data, "Label", f)).read().splitlines()[1:]:
        p = line.split(" "); h, v = (math.degrees(float(x)) for x in p[7].split(","))
        rows.append([p[0], f"{h:.4f}", f"{v:.4f}", f.split(".")[0]])
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["face", "gaze_h_deg", "gaze_v_deg", "person"]); w.writerows(rows)
print(len(rows), "rows")
