#!/bin/bash 

#SBATCH --job-name=ddp_env_v100
#SBATCH --partition=gpu-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/env_v100_%j.out
#SBATCH --error=logs/env_v100_%j.err

set -euo pipefail

# project 
PROJECT_DIR="/home/maneth14/pytorch-ddp-scaling-lab"

cd "${PROJECT_DIR}"

mkdir -p logs


echo "======================================================================"
echo "ROSI V100 ENVIRONMENT AUDIT"
echo "======================================================================"

echo "Date:"
date

echo

echo "Hostname:"
hostname

echo

echo "Working directory:"
pwd

echo

echo "SLURM"
echo "----------------------------------------------------------------------"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

echo


# ---------------------------------------------------------------------
# NVIDIA / GPU
# ---------------------------------------------------------------------

echo "NVIDIA-SMI"
echo "----------------------------------------------------------------------"

nvidia-smi

echo


# ---------------------------------------------------------------------
# Python
#
# IMPORTANT:
# If ROSI requires a module/conda/venv activation, we will insert it
# here after seeing what environment is available.
# ---------------------------------------------------------------------

echo "PYTHON"
echo "----------------------------------------------------------------------"

which python || true
python --version || true

echo


echo "PYTORCH"
echo "----------------------------------------------------------------------"

python - <<'PY'
import torch

print("PyTorch version      :", torch.__version__)
print("CUDA build           :", torch.version.cuda)
print("CUDA available       :", torch.cuda.is_available())
print("Visible GPU count    :", torch.cuda.device_count())

print("Distributed available:", torch.distributed.is_available())
print("NCCL available       :", torch.distributed.is_nccl_available())
print("Gloo available       :", torch.distributed.is_gloo_available())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)

        print()
        print(f"GPU {i}")
        print("  name               :", props.name)
        print(
            "  VRAM GiB           :",
            round(props.total_memory / 1024**3, 2),
        )
        print(
            "  compute capability :",
            f"{props.major}.{props.minor}",
        )
        print(
            "  multiprocessors    :",
            props.multi_processor_count,
        )
PY


echo
echo "======================================================================"
echo "ENVIRONMENT AUDIT COMPLETE"
echo "======================================================================"
