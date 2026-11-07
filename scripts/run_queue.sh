#!/bin/bash
# Sequential experiment queue (see ../PLAN.md). Each stage drops a .done marker in ~/runs/l2cs/queue.
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate l2cs
S="$(cd "$(dirname "$0")" && pwd)"
D=~/datasets/MPIIFaceGaze
R=~/runs/l2cs
W="$S/../L2CS-Net/models/L2CSNet_gaze360.pkl"
mkdir -p "$R/queue"
echo "queue start $(date)" > "$R/queue/status.txt"

echo "stage1 zeroshot $(date)" >> "$R/queue/status.txt"
python -I "$S/eval_mpii.py" --data "$D" --weights "$W" --bins 90 --bin-width 4 --angle 180 --folds all \
  --out "$R/gaze360_zeroshot/per_person.csv" > "$R/gaze360_zeroshot.log" 2>&1
touch "$R/queue/stage1_zeroshot.done"

echo "stage2 fold0 $(date)" >> "$R/queue/status.txt"
python -I "$S/train_mpii.py" --data "$D" --fold 0 --epochs 15 --amp --save-every-epoch --out "$R/fold0_e15" > "$R/fold0_e15.log" 2>&1
touch "$R/queue/stage2_fold0.done"

for f in 7 14; do
  echo "stage3 fold$f $(date)" >> "$R/queue/status.txt"
  python -I "$S/train_mpii.py" --data "$D" --fold $f --epochs 10 --amp --save-every-epoch --out "$R/fold${f}_e10" > "$R/fold${f}_e10.log" 2>&1
  touch "$R/queue/stage3_fold${f}.done"
done
echo "queue done $(date)" >> "$R/queue/status.txt"
touch "$R/queue/all.done"
