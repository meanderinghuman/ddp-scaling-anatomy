#!/bin/bash
#SBATCH --job-name=ddp_w2_smoke
#SBATCH --partition=gpu-v100-int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=logs/ddp_w2_smoke_%j.out
#SBATCH --error=logs/ddp_w2_smoke_%j.err

set -euo pipefail


# =====================================================================
# Project
# =====================================================================

PROJECT_DIR="/home/maneth14/pytorch-ddp-scaling-lab"

cd "${PROJECT_DIR}"

mkdir -p logs
mkdir -p results/raw
mkdir -p checkpoints


export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"


# We allocate 8 CPU cores total.
#
# Two torchrun processes:
#
#     rank 0
#     rank 1
#
# Each may use up to four OpenMP threads.
export OMP_NUM_THREADS=4


echo "======================================================================"
echo "2-GPU V100 DDP SMOKE TEST"
echo "======================================================================"

echo "Date:"
date

echo

echo "Host:"
hostname

echo

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"

echo


# =====================================================================
# GPU audit
# =====================================================================

nvidia-smi \
    --query-gpu=index,name,memory.total,driver_version \
    --format=csv,noheader


echo


python - <<'PY'
import torch

print("PyTorch        :", torch.__version__)
print("CUDA build     :", torch.version.cuda)
print("CUDA available :", torch.cuda.is_available())
print("Visible GPUs   :", torch.cuda.device_count())
print("NCCL available :", torch.distributed.is_nccl_available())

for i in range(torch.cuda.device_count()):
    print(
        f"GPU {i}:",
        torch.cuda.get_device_name(i),
    )
PY


echo
echo "======================================================================"
echo "U-NET / HEAT — WORLD SIZE 2"
echo "======================================================================"

torchrun \
    --standalone \
    --nproc-per-node=2 \
    scripts/10_train_ddp.py \
    --config configs/unet_heat_v100.yaml \
    --device cuda \
    --backend nccl \
    --num-workers 0 \
    --max-train-batches 8 \
    --max-val-batches 2


echo
echo "======================================================================"
echo "RESNET18 / CIFAR-100 — WORLD SIZE 2"
echo "======================================================================"

torchrun \
    --standalone \
    --nproc-per-node=2 \
    scripts/10_train_ddp.py \
    --config configs/resnet18_v100.yaml \
    --device cuda \
    --backend nccl \
    --num-workers 0 \
    --max-train-batches 8 \
    --max-val-batches 2


echo
echo "======================================================================"
echo "WORLD SIZE 2 SMOKE TEST COMPLETE"
echo "======================================================================"
