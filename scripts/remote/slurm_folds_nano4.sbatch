#!/bin/bash
# Nano4 (NCHC, hostname 25a-*) version of slurm_folds.sbatch: one job = one tag, 8x H200 on one node,
# folds round-robined over the GPUs by run_folds.sh. Re-submitting the same tag resumes
# (done.json -> skip, state.pt -> --resume), so a job killed by walltime is simply resubmitted.
#   sbatch --export=ALL,TAG=paper_e50,RECIPE=paper,EPOCHS=50 scripts/remote/slurm_folds_nano4.sbatch
#   sbatch --export=ALL,TAG=abl_cosine_e50,RECIPE=improved,EPOCHS=50,FOLDS=0/3/7/11/14,EXTRA=--no-aug ...
#   sbatch --export=ALL,TAG=mnv3l_e20,RECIPE=improved,EPOCHS=20,EXTRA=--backbone+mobilenet_v3_large+--lr+1e-4 ...
# Notes (Nano4): account acd114003 is not allowed on Nano4 -> acd110018; 12 CPU per GPU max (8*12=96);
# partition 8gpus has a 2-day walltime. Submit from the repo root (cd ~/DMS).
#SBATCH -J l2cs
#SBATCH -A acd110018
#SBATCH -p 8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=96
#SBATCH -t 2-00:00:00
#SBATCH -o /work/u8714813/runs/l2cs/slurm/%x-%j.out
set -u
TAG="${TAG:?set TAG}"; RECIPE="${RECIPE:-paper}"; EPOCHS="${EPOCHS:-50}"; FOLDS="${FOLDS:-0-14}"
EXTRA="${EXTRA:-}"; EXTRA="${EXTRA//+/ }"     # "--sched+const" -> "--sched const" (sbatch --export can't carry spaces)
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"; WORKERS="${WORKERS:-4}"; RUNS="${RUNS:-/work/u8714813/runs/l2cs}"
. /work/u8714813/envs/l2cs/bin/activate
export TMPDIR=/work/u8714813/tmp
cd "$SLURM_SUBMIT_DIR"
export DATA="${DATA:-/work/u8714813/datasets/MPIIFaceGaze}" RUNS="${RUNS:-/work/u8714813/runs/l2cs}"
echo "job $SLURM_JOB_ID on $(hostname) tag=$TAG recipe=$RECIPE epochs=$EPOCHS folds=$FOLDS extra='$EXTRA'"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
bash scripts/run_folds.sh --tag "$TAG" --recipe "$RECIPE" --epochs "$EPOCHS" --folds "$FOLDS" --gpus "$GPUS" \
  --workers "$WORKERS" --extra "$EXTRA"
