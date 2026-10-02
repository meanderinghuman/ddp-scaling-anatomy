#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

PROJECT_DIR="/home/maneth14/pytorch-ddp-scaling-lab"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2

torchrun \
    --standalone \
    --nproc-per-node=4 \
    scripts/19_profile_syncbn.py \
    --config configs/resnet18_v100.yaml \
    --tag resnet_syncbn_world4 \
    --num-workers 2 \
    --wait-steps 5 \
    --warmup-steps 3 \
    --active-steps 10
