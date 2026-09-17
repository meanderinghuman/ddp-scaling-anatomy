#!/bin/bash
#SBATCH --job-name=ddp_w1_smoke
#SBATCH --partition=gpu-v100-int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=logs/ddp_w1_smoke_%j.out
#SBATCH --error=logs/ddp_w1_smoke_%j.err

set -euo pipefail


# =====================================================================
# Project
# =====================================================================

PROJECT_DIR="/home/maneth14/pytorch-ddp-scaling-lab"

cd "${PROJECT_DIR}"

mkdir -p logs
mkdir -p results/raw
mkdir -p checkpoints


# Make the src-layout package importable even if editable installation
# is not active in the compute-node Python environment.
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"


# Four CPU cores are allocated to this Slurm job.
export OMP_NUM_THREADS=4


echo "======================================================================"
echo "1-GPU V100 DDP SMOKE TEST"
echo "======================================================================"

echo "Date:"
date

echo

echo "Host:"
hostname

echo

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

echo

nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader

echo


# =====================================================================
# Basic PyTorch audit
# =====================================================================

python - <<'PY'
import torch

print("PyTorch        :", torch.__version__)
print("CUDA build     :", torch.version.cuda)
print("CUDA available :", torch.cuda.is_available())
print("Visible GPUs   :", torch.cuda.device_count())
print("NCCL available :", torch.distributed.is_nccl_available())

if torch.cuda.is_available():
    print("GPU            :", torch.cuda.get_device_name(0))
    print("BF16 supported :", torch.cuda.is_bf16_supported())
PY


echo
echo "======================================================================"
echo "U-NET / HEAT — WORLD SIZE 1"
echo "======================================================================"

torchrun \
    --standalone \
    --nproc-per-node=1 \
    scripts/10_train_ddp.py \
    --config configs/unet_heat_v100.yaml \
    --device cuda \
    --backend nccl \
    --num-workers 0 \
    --max-train-batches 8 \
    --max-val-batches 2


echo
echo "======================================================================"
echo "RESNET18 / CIFAR-100 — WORLD SIZE 1"
echo "======================================================================"

torchrun \
    --standalone \
    --nproc-per-node=1 \
    scripts/10_train_ddp.py \
    --config configs/resnet18_v100.yaml \
    --device cuda \
    --backend nccl \
    --num-workers 0 \
    --max-train-batches 8 \
    --max-val-batches 2


echo
echo "======================================================================"
echo "WORLD SIZE 1 SMOKE TEST COMPLETE"
echo "======================================================================"
