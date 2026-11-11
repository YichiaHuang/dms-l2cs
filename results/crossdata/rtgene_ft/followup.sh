#!/bin/bash
#SBATCH -A acd110018
#SBATCH -p 8gpus
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=12
#SBATCH -t 02:30:00
#SBATCH -o /work/u8714813/runs/rtgene_ft/%x-%j.out
# Follow-up (README §4): n10 + 50% replay on fold 0; n10 and n10 + replay on folds 1, 2. Re-submit = skips finished runs.
. /work/u8714813/envs/l2cs/bin/activate; export TMPDIR=/work/u8714813/tmp; cd ~/DMS
J1=/work/u8714813/runs/gaze360/J1_mnv3l_kd_joint_e20; O=/work/u8714813/runs/rtgene_ft
python -I scripts/finetune_rtgene.py --init $J1 --fold 0 --per-subject 10 --replay 0.5 --out $O/J1_f0_n10_replay50
for f in 1 2; do
  python -I scripts/finetune_rtgene.py --init $J1 --fold $f --per-subject 10 --out $O/J1_f${f}_n10
  python -I scripts/finetune_rtgene.py --init $J1 --fold $f --per-subject 10 --replay 0.5 --out $O/J1_f${f}_n10_replay50
done
