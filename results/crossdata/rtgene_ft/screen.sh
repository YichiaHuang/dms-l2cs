#!/bin/bash
#SBATCH -A acd110018
#SBATCH -p 8gpus
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=12
#SBATCH -t 03:00:00
#SBATCH -o /work/u8714813/runs/rtgene_ft/%x-%j.out
# Phase 1 screening (README 下一步實驗): J1, RT-GENE fold 0, 4 settings. Re-submit = skips finished runs.
. /work/u8714813/envs/l2cs/bin/activate; export TMPDIR=/work/u8714813/tmp; cd ~/DMS
J1=/work/u8714813/runs/gaze360/J1_mnv3l_kd_joint_e20; O=/work/u8714813/runs/rtgene_ft
for n in 10 100 0; do
  python -I scripts/finetune_rtgene.py --init $J1 --fold 0 --per-subject $n --out $O/J1_f0_n$n
done
python -I scripts/finetune_rtgene.py --init $J1 --fold 0 --per-subject 0 --replay 0.5 --out $O/J1_f0_n0_replay50
