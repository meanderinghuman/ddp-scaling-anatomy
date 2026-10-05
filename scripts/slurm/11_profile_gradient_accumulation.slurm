#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

PROJECT_DIR="/home/maneth14/pytorch-ddp-scaling-lab"

cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2


run_profile () {

    MODE="$1"

    echo
    echo "======================================================================"
    echo "Profiling ${MODE}"
    echo "======================================================================"


    torchrun \
        --standalone \
        --nproc-per-node=4 \
        scripts/22_profile_gradient_accumulation.py \
        --config configs/unet_heat_v100.yaml \
        --mode "${MODE}" \
        --tag "unet_world4_${MODE}"
}


run_profile naive
run_profile no_sync
