#!/bin/bash
# Overnight queue 2026-10-19: paper recipe vs improved recipe, paired per fold, 7 epochs each.
# Runs that already have done.json are skipped, so re-running this script resumes the queue.
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate l2cs
S="$(cd "$(dirname "$0")" && pwd)"
D=~/datasets/MPIIFaceGaze
R=~/runs/l2cs
EP=7
mkdir -p "$R/queue"
echo "overnight start $(date)" >> "$R/queue/overnight_status.txt"
run() {  # fold recipe
  local name="fold$1_$2_e$EP"
  echo "$name start $(date)" >> "$R/queue/overnight_status.txt"
  python -I "$S/train_mpii.py" --data "$D" --fold "$1" --recipe "$2" --epochs $EP --amp --save-every-epoch --resume \
    --out "$R/$name" >> "$R/$name.log" 2>&1
  echo "$name end   $(date) exit=$?" >> "$R/queue/overnight_status.txt"
}
run 0 improved
run 7 improved
for f in 3 11 14; do
  run $f paper
  run $f improved
done
echo "overnight done $(date)" >> "$R/queue/overnight_status.txt"
touch "$R/queue/overnight.done"
