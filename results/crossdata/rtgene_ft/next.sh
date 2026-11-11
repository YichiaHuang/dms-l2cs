#!/bin/bash
#SBATCH -A acd110018
#SBATCH -p 8gpus
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=12
#SBATCH -t 01:30:00
#SBATCH -o /work/u8714813/runs/rtgene_ft/%x-%j.out
# R2 (README §4): balance target data vs replay. FOLD env. Re-submit = skips finished runs.
. /work/u8714813/envs/l2cs/bin/activate; export TMPDIR=/work/u8714813/tmp; cd ~/DMS
J1=/work/u8714813/runs/gaze360/J1_mnv3l_kd_joint_e20; O=/work/u8714813/runs/rtgene_ft; f=$FOLD
python -I scripts/finetune_rtgene.py --init $J1 --fold $f --per-subject 10 --replay 0.25 --out $O/J1_f${f}_n10_replay25
python -I scripts/finetune_rtgene.py --init $J1 --fold $f --per-subject 100 --replay 0.5 --out $O/J1_f${f}_n100_replay50
python -I scripts/finetune_rtgene.py --init $J1 --fold $f --per-subject 100 --replay 0.25 --out $O/J1_f${f}_n100_replay25
