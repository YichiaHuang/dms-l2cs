#!/bin/bash
#SBATCH -A acd110018
#SBATCH -p dev
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=12
#SBATCH -t 00:20:00
#SBATCH -o /work/u8714813/runs/rtgene_ft/%x-%j.out
. /work/u8714813/envs/l2cs/bin/activate; export TMPDIR=/work/u8714813/tmp; cd ~/DMS
J1=/work/u8714813/runs/gaze360/J1_mnv3l_kd_joint_e20
python -I scripts/finetune_rtgene.py --init $J1 --fold 0 --per-subject 10 --iters 20 --eval-every 10 --eval-limit 300 --out /work/u8714813/runs/rtgene_ft/smoke_a
python -I scripts/finetune_rtgene.py --init $J1 --fold 0 --per-subject 0 --replay 0.5 --iters 20 --eval-every 10 --eval-limit 300 --out /work/u8714813/runs/rtgene_ft/smoke_b
python -I scripts/finetune_rtgene.py --init /work/u8714813/runs/l2cs_light/fold0_mnv3l_448_kd_e20 --fold 0 --per-subject 100 --mpii-test p00 --iters 20 --eval-every 10 --eval-limit 300 --out /work/u8714813/runs/rtgene_ft/smoke_c
