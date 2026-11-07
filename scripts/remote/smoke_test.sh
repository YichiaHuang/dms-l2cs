#!/bin/bash
# 5-minute end-to-end check on the training machine: data readable, model trains, AMP path works,
# resume works, and it prints the measured img/s so the schedule in REMOTE_PLAN.md can be sized.
#   bash scripts/remote/smoke_test.sh            (DATA / RUNS env overrides as in run_folds.sh)
set -eu
S="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-$HOME/datasets/MPIIFaceGaze}"
RUNS="${RUNS:-$HOME/runs/l2cs}"
OUT="$RUNS/smoke"
rm -rf "$OUT"
echo "== 1 epoch of 60 iters, then resume for a 2nd epoch"
python -I "$S/train_mpii.py" --data "$DATA" --fold 0 --epochs 2 --max-iters 60 --amp --workers "${WORKERS:-8}" --stop-after-epoch 1 --out "$OUT"
python -I "$S/train_mpii.py" --data "$DATA" --fold 0 --epochs 2 --max-iters 60 --amp --workers "${WORKERS:-8}" --resume --out "$OUT"
test -f "$OUT/done.json" && echo "resume OK"
echo "== throughput (last logged window, img/s) and dtype:"
tail -n 2 "$OUT/iters.csv" | cut -d, -f10
grep -o '"amp_dtype": "[a-z0-9]*"' "$OUT/config.json"
grep -o '"gpu": "[^"]*"' "$OUT/config.json"
echo "smoke test OK -> ~/runs/l2cs/smoke (safe to delete)"
