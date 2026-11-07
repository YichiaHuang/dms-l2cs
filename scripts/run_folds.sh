#!/bin/bash
# Run leave-one-person-out folds in parallel, one training process per GPU, each GPU taking its
# share of the fold list sequentially. Re-running the same command resumes: finished runs (done.json)
# are skipped by train_mpii.py and unfinished ones continue from state.pt (--resume).
#
#   bash scripts/run_folds.sh --tag paper_e50 --recipe paper --epochs 50 --gpus 0,1,2,3
#   bash scripts/run_folds.sh --tag improved_e50 --recipe improved --epochs 50 --folds 0,3,7,11,14 --gpus 0,1
#   bash scripts/run_folds.sh --tag abl_cos_e50 --recipe improved --epochs 50 --extra "--no-aug" --gpus 0
#
# Run name = fold<N>_<tag>  (make_report.py picks up every ~/runs/l2cs/fold*/ with metrics.csv + config.json)
# Env overrides: DATA (default ~/datasets/MPIIFaceGaze), RUNS (default ~/runs/l2cs), PYTHON (default python)
set -u
S="$(cd "$(dirname "$0")" && pwd)"
DATA="${DATA:-$HOME/datasets/MPIIFaceGaze}"
RUNS="${RUNS:-$HOME/runs/l2cs}"
PYTHON="${PYTHON:-python}"
TAG=""; RECIPE="paper"; EPOCHS=50; FOLDS="0-14"; GPUS="0"; WORKERS=8; EXTRA=""; AMP="--amp"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2;;
    --recipe) RECIPE="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --folds) FOLDS="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --workers) WORKERS="$2"; shift 2;;
    --extra) EXTRA="$2"; shift 2;;
    --no-amp) AMP=""; shift;;
    *) echo "unknown arg $1"; exit 2;;
  esac
done
[[ -z "$TAG" ]] && { echo "--tag is required (run name = fold<N>_<tag>)"; exit 2; }
# expand "0-14" / "0,3,7" / "0-4,14"; "/" or spaces also accepted as separators (easier to pass through sbatch --export)
FOLDS="${FOLDS//[\/ ]/,}"; GPUS="${GPUS//[\/ ]/,}"
FOLD_LIST=()
IFS=',' read -ra parts <<< "$FOLDS"
for p in "${parts[@]}"; do
  if [[ "$p" == *-* ]]; then for ((i=${p%-*}; i<=${p#*-}; i++)); do FOLD_LIST+=("$i"); done; else FOLD_LIST+=("$p"); fi
done
IFS=',' read -ra GPU_LIST <<< "$GPUS"
NG=${#GPU_LIST[@]}
mkdir -p "$RUNS/queue"
STATUS="$RUNS/queue/${TAG}_status.txt"
echo "[$(date '+%F %T')] start tag=$TAG recipe=$RECIPE epochs=$EPOCHS folds=${FOLD_LIST[*]} gpus=$GPUS workers=$WORKERS extra='$EXTRA'" >> "$STATUS"

worker() {  # gpu  fold...
  local gpu="$1"; shift
  for f in "$@"; do
    local name="fold${f}_${TAG}"
    echo "[$(date '+%F %T')] gpu$gpu $name start" >> "$STATUS"
    CUDA_VISIBLE_DEVICES="$gpu" $PYTHON -I "$S/train_mpii.py" --data "$DATA" --fold "$f" --recipe "$RECIPE" --epochs "$EPOCHS" \
      $AMP --workers "$WORKERS" --resume $EXTRA --out "$RUNS/$name" >> "$RUNS/$name.log" 2>&1
    echo "[$(date '+%F %T')] gpu$gpu $name end exit=$?" >> "$STATUS"
  done
}
# round-robin folds over GPUs
declare -a ASSIGN
for ((k=0; k<${#FOLD_LIST[@]}; k++)); do g=$((k % NG)); ASSIGN[$g]="${ASSIGN[$g]:-} ${FOLD_LIST[$k]}"; done
PIDS=()
for ((g=0; g<NG; g++)); do
  [[ -z "${ASSIGN[$g]:-}" ]] && continue
  worker "${GPU_LIST[$g]}" ${ASSIGN[$g]} &
  PIDS+=($!)
done
wait "${PIDS[@]}"
echo "[$(date '+%F %T')] all done tag=$TAG" >> "$STATUS"
touch "$RUNS/queue/${TAG}.done"
