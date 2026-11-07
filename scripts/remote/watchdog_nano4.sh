#!/bin/bash
# Resubmit-to-resume watchdog for Nano4 (run inside tmux: tmux new -s l2cs 'bash scripts/remote/watchdog_nano4.sh').
# Every 10 min: for each tag that has no $RUNS/queue/<tag>.done and no queued/running job named l2cs-<X>,
# resubmit the same sbatch (train_mpii.py skips done folds and resumes from state.pt). Max 1 resubmit per tag (quota).
set -u
cd "$(dirname "$0")/../.."
RUNS=/work/u8714813/runs/l2cs; SB=scripts/remote/slurm_folds_nano4.sbatch; LOG=$RUNS/slurm/watchdog.log
declare -A N
# name|tag|sbatch args
JOBS=(
  "l2cs-A|paper_e50|--export=ALL,TAG=paper_e50,RECIPE=paper,EPOCHS=50"
  "l2cs-B|improved_e50|--export=ALL,TAG=improved_e50,RECIPE=improved,EPOCHS=50"
)
while true; do
  all=1
  for j in "${JOBS[@]}"; do
    IFS='|' read -r name tag args <<< "$j"
    [[ -f $RUNS/queue/$tag.done ]] && continue
    all=0
    st=$(squeue -h -u "$USER" -n "$name" -o '%T %r' 2>/dev/null | head -1)
    if [[ "$st" == *DependencyNeverSatisfied* ]]; then   # upstream failed: drop the stale dependency
      scancel -n "$name" -u "$USER"; st=""
    fi
    if [[ -z "$st" ]]; then
      n=${N[$tag]:-0}
      if (( n >= 1 )); then echo "[$(date '+%F %T')] $tag: not done, resubmit limit reached" >> "$LOG"; continue; fi
      id=$(sbatch --parsable -J "$name" $args $SB)
      N[$tag]=$((n+1)); echo "[$(date '+%F %T')] $tag: resubmitted as $id (#$((n+1)))" >> "$LOG"
    fi
  done
  (( all )) && { echo "[$(date '+%F %T')] all tags done" >> "$LOG"; break; }
  sleep 600
done
