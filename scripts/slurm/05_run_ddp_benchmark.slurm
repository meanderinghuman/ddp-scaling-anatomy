#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

PROJECT_DIR="/home/maneth14/pytorch-ddp-scaling-lab"

cd "${PROJECT_DIR}"

mkdir -p logs
mkdir -p results/raw
mkdir -p checkpoints

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_THREADS:-2}"

echo "======================================================================"
echo "FULL DDP BENCHMARK"
echo "======================================================================"
echo "Date               : $(date)"
echo "Host               : $(hostname)"
echo "World size         : ${WORLD_SIZE}"
echo "Config             : ${CONFIG}"
echo "Workers per rank   : ${NUM_WORKERS}"
echo "OMP threads        : ${OMP_NUM_THREADS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo

nvidia-smi \
    --query-gpu=index,name,memory.total \
    --format=csv,noheader

echo

torchrun \
    --standalone \
    --nproc-per-node="${WORLD_SIZE}" \
    scripts/10_train_ddp.py \
    --config "${CONFIG}" \
    --device cuda \
    --backend nccl \
    --num-workers "${NUM_WORKERS}"

echo
echo "======================================================================"
echo "FULL DDP BENCHMARK COMPLETE"
echo "======================================================================"
