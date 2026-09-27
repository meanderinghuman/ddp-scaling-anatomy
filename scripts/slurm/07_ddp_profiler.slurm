#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail


: "${WORLD_SIZE:?WORLD_SIZE is required}"
: "${CONFIG:?CONFIG is required}"
: "${TAG:?TAG is required}"


PROJECT_DIR="/home/maneth14/pytorch-ddp-scaling-lab"

cd "${PROJECT_DIR}"


export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

export OMP_NUM_THREADS="${OMP_THREADS:-2}"


mkdir -p \
    results/rosi_v100/profiling/logs \
    results/rosi_v100/profiling/runs


echo "================================================================================"
echo "PYTORCH DDP / NCCL PROFILER"
echo "================================================================================"
echo "Date                : $(date)"
echo "Host                : $(hostname)"
echo "Tag                 : ${TAG}"
echo "World size          : ${WORLD_SIZE}"
echo "Config              : ${CONFIG}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_JOB_GPUS      : ${SLURM_JOB_GPUS:-}"
echo


nvidia-smi \
    --query-gpu=index,name,memory.total,temperature.gpu,power.draw \
    --format=csv,noheader


echo


torchrun \
    --standalone \
    --nproc-per-node="${WORLD_SIZE}" \
    scripts/15_profile_ddp.py \
    --config "${CONFIG}" \
    --tag "${TAG}" \
    --num-workers "${NUM_WORKERS:-2}" \
    --wait-steps 5 \
    --warmup-steps 3 \
    --active-steps 10


echo
echo "================================================================================"
echo "PROFILER JOB COMPLETE"
echo "================================================================================"
