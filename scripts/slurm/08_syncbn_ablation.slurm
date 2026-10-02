#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail


PROJECT_DIR="/home/maneth14/pytorch-ddp-scaling-lab"

cd "${PROJECT_DIR}"


export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_THREADS:-2}"


REPEATS="${REPEATS:-3}"
WORLD_SIZE=4
NUM_WORKERS="${NUM_WORKERS:-2}"


OUT_DIR="results/rosi_v100/syncbn/raw"

mkdir -p \
    "${OUT_DIR}" \
    results/rosi_v100/syncbn/checkpoints \
    results/rosi_v100/syncbn/logs


SOURCE_JSON="${OUT_DIR}/resnet18_ddp_cuda_nccl_world4_full.json"


echo "================================================================================"
echo "RESNET18 SYNCBATCHNORM ABLATION"
echo "================================================================================"
echo "Date                : $(date)"
echo "Host                : $(hostname)"
echo "World size          : ${WORLD_SIZE}"
echo "Global batch        : 256"
echo "Local batch         : 64"
echo "Repetitions         : ${REPEATS}"
echo "Normalization       : SyncBatchNorm"
echo "Workers/rank        : ${NUM_WORKERS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_JOB_GPUS      : ${SLURM_JOB_GPUS:-}"
echo


nvidia-smi \
    --query-gpu=index,name,memory.total,temperature.gpu,power.draw \
    --format=csv,noheader


for REP in $(seq 1 "${REPEATS}"); do

    echo
    echo "================================================================================"
    echo "SYNCBN REPETITION ${REP} / ${REPEATS}"
    echo "================================================================================"
    echo "Start: $(date)"
    echo


    rm -f "${SOURCE_JSON}"


    torchrun \
        --standalone \
        --nproc-per-node=4 \
        scripts/17_train_syncbn_ablation.py \
        --config configs/resnet18_v100.yaml \
        --device cuda \
        --backend nccl \
        --num-workers "${NUM_WORKERS}"


    if [[ ! -f "${SOURCE_JSON}" ]]; then

        echo "ERROR: result JSON not created:"
        echo "${SOURCE_JSON}"

        exit 1
    fi


    DEST_JSON="${OUT_DIR}/resnet18_world4_syncbn_rep${REP}.json"

    cp \
        "${SOURCE_JSON}" \
        "${DEST_JSON}"


    python - "${DEST_JSON}" <<'PY'
import json
import sys

path = sys.argv[1]

with open(path, "r") as f:
    d = json.load(f)

final = d["history"][-1]

print(
    "Throughput     :",
    f"{d['steady_state_throughput_samples_per_sec']:.3f} img/s",
)

print(
    "Epoch time     :",
    f"{d['mean_steady_state_epoch_wall_time_sec']:.3f} s",
)

print(
    "Final val acc  :",
    f"{100 * final['val_accuracy']:.2f}%",
)

print(
    "Peak VRAM      :",
    f"{max(e['peak_vram_per_rank_gb'] for e in d['history'][1:]):.3f} GiB",
)

print(
    "Hostname       :",
    d["rank_zero_hostname"],
)
PY


    echo
    echo "End: $(date)"

done


echo
echo "================================================================================"
echo "SYNCBN REPETITION AUDIT"
echo "================================================================================"


python - <<'PY'
import json
from pathlib import Path

root = Path(
    "results/rosi_v100/syncbn/raw"
)

for rep in range(1, 4):

    path = root / (
        f"resnet18_world4_syncbn_rep{rep}.json"
    )

    if not path.exists():

        raise SystemExit(
            f"Missing: {path}"
        )

    with path.open() as f:
        d = json.load(f)

    final = d["history"][-1]

    print(
        f"rep {rep}: "
        f"Q={d['steady_state_throughput_samples_per_sec']:.2f} img/s  "
        f"acc={100 * final['val_accuracy']:.2f}%"
    )


print()
print(
    "All 3 SyncBatchNorm repetitions completed successfully."
)
PY


echo
echo "================================================================================"
echo "SYNCBATCHNORM ABLATION COMPLETE"
echo "================================================================================"
