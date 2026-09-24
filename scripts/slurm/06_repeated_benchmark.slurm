#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail


# =====================================================================
# Required environment variables
# =====================================================================

: "${WORLD_SIZE:?WORLD_SIZE is required}"
: "${CONFIG:?CONFIG is required}"
: "${WORKLOAD:?WORKLOAD is required}"

REPEATS="${REPEATS:-3}"
NUM_WORKERS="${NUM_WORKERS:-2}"
OMP_THREADS="${OMP_THREADS:-2}"


# =====================================================================
# Project
# =====================================================================

PROJECT_DIR="/home/maneth14/pytorch-ddp-scaling-lab"

cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_THREADS}"


OUT_DIR="results/rosi_v100/repeated_runs/raw"

mkdir -p "${OUT_DIR}"


# Trainer writes here first.
SOURCE_JSON="results/raw/${WORKLOAD}_ddp_cuda_nccl_world${WORLD_SIZE}_full.json"


echo "================================================================================"
echo "REPEATED DDP BENCHMARK"
echo "================================================================================"
echo "Date                : $(date)"
echo "Host                : $(hostname)"
echo "Workload            : ${WORKLOAD}"
echo "Config              : ${CONFIG}"
echo "World size          : ${WORLD_SIZE}"
echo "Repetitions         : ${REPEATS}"
echo "Workers / rank      : ${NUM_WORKERS}"
echo "OMP threads         : ${OMP_NUM_THREADS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_JOB_GPUS      : ${SLURM_JOB_GPUS:-}"
echo


echo "GPU inventory"
echo "--------------------------------------------------------------------------------"

nvidia-smi \
    --query-gpu=index,name,memory.total,temperature.gpu,power.draw \
    --format=csv,noheader

echo


# =====================================================================
# Repetitions
# =====================================================================

for REP in $(seq 1 "${REPEATS}"); do

    echo
    echo "================================================================================"
    echo "REPETITION ${REP} / ${REPEATS}"
    echo "================================================================================"
    echo "Start: $(date)"
    echo


    # Remove the trainer's temporary output from any previous repetition.
    # The frozen baseline is already safely stored under results/rosi_v100/raw/.
    rm -f "${SOURCE_JSON}"


    torchrun \
        --standalone \
        --nproc-per-node="${WORLD_SIZE}" \
        scripts/10_train_ddp.py \
        --config "${CONFIG}" \
        --device cuda \
        --backend nccl \
        --num-workers "${NUM_WORKERS}"


    if [[ ! -f "${SOURCE_JSON}" ]]; then
        echo "ERROR: expected result JSON was not created:"
        echo "${SOURCE_JSON}"
        exit 1
    fi


    DEST_JSON="${OUT_DIR}/${WORKLOAD}_world${WORLD_SIZE}_rep${REP}.json"

    cp "${SOURCE_JSON}" "${DEST_JSON}"


    echo
    echo "Saved repetition:"
    echo "${DEST_JSON}"


    python - "${DEST_JSON}" <<'PY'
import json
import sys

path = sys.argv[1]

with open(path, "r", encoding="utf-8") as f:
    d = json.load(f)

print(
    "steady throughput :",
    f"{d['steady_state_throughput_samples_per_sec']:.3f}",
)

print(
    "steady epoch time :",
    f"{d['mean_steady_state_epoch_wall_time_sec']:.3f} s",
)

print(
    "hostname          :",
    d["rank_zero_hostname"],
)

print(
    "world size        :",
    d["world_size"],
)
PY


    echo
    echo "GPU state after repetition ${REP}"
    echo "--------------------------------------------------------------------------------"

    nvidia-smi \
        --query-gpu=index,temperature.gpu,power.draw,memory.used \
        --format=csv,noheader


    echo
    echo "End: $(date)"

done


# =====================================================================
# Verify all expected repetitions
# =====================================================================

echo
echo "================================================================================"
echo "REPETITION AUDIT"
echo "================================================================================"


python - <<PY
import json
from pathlib import Path

workload = "${WORKLOAD}"
world_size = int("${WORLD_SIZE}")
repeats = int("${REPEATS}")

root = Path(
    "results/rosi_v100/repeated_runs/raw"
)

values = []

for rep in range(1, repeats + 1):

    path = root / (
        f"{workload}_world{world_size}_rep{rep}.json"
    )

    if not path.exists():
        raise SystemExit(
            f"Missing repetition: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)

    q = float(
        d["steady_state_throughput_samples_per_sec"]
    )

    values.append(q)

    print(
        f"rep {rep}: {q:.3f} samples/s"
    )

print()
print(
    f"Completed {len(values)} repetitions successfully."
)
PY


echo
echo "================================================================================"
echo "REPEATED BENCHMARK COMPLETE"
echo "================================================================================"
