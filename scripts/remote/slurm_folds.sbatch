#!/bin/bash
# Taiwania 2 (台灣杉二號, Slurm) template: one job = one tag, N GPUs on one node, folds spread over the GPUs.
# Fill in the project ID, then:  sbatch --export=TAG=paper_e50,RECIPE=paper,EPOCHS=50 scripts/remote/slurm_folds.sbatch
# Re-submitting the same tag resumes (done.json skip + state.pt). Check partitions/limits with: sinfo ; sacctmgr show assoc user=$USER
#SBATCH -J l2cs
#SBATCH -A MST1XXXXX            # <-- project ID (計畫代號)
#SBATCH -p gp4d                 # gtest (short test) / gp1d / gp2d / gp4d  -> max wall time 1/2/4 days (verify with sinfo)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4            # GPUs on this node; keep in sync with GPUS below
#SBATCH --cpus-per-task=16      # 4 CPU per GPU for the dataloaders
#SBATCH -t 4-00:00:00
#SBATCH -o %x-%j.out
set -u
TAG="${TAG:?set TAG}"; RECIPE="${RECIPE:-paper}"; EPOCHS="${EPOCHS:-50}"; FOLDS="${FOLDS:-0-14}"; EXTRA="${EXTRA:-}"
GPUS="${GPUS:-0,1,2,3}"
module load miniconda3 2>/dev/null || true
eval "$(conda shell.bash hook)" && conda activate l2cs
cd "$SLURM_SUBMIT_DIR"
export DATA="${DATA:-$HOME/datasets/MPIIFaceGaze}" RUNS="${RUNS:-$HOME/runs/l2cs}"
nvidia-smi --query-gpu=name,memory.total --format=csv
bash scripts/run_folds.sh --tag "$TAG" --recipe "$RECIPE" --epochs "$EPOCHS" --folds "$FOLDS" --gpus "$GPUS" --workers 4 --extra "$EXTRA"
