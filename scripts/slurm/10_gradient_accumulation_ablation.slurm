#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail


PROJECT_DIR="/home/maneth14/pytorch-ddp-scaling-lab"

cd "${PROJECT_DIR}"


export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2


OUT="results/rosi_v100/optimization/raw"

mkdir -p \
    "${OUT}" \
    results/rosi_v100/optimization/logs


echo "================================================================================"
echo "PHASE 3 — GRADIENT ACCUMULATION / DDP.NO_SYNC"
echo "================================================================================"
echo "Date                : $(date)"
echo "Host                : $(hostname)"
echo "GPUs                : 4"
echo "Microbatch/rank     : 4"
echo "Accumulation steps  : 4"
echo "Effective batch     : 64"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo


nvidia-smi \
    --query-gpu=index,name,memory.total,temperature.gpu,power.draw \
    --format=csv,noheader


run_experiment () {

    MODE="$1"
    REP="$2"

    RESULT="${OUT}/unet_world4_${MODE}_rep${REP}.json"


    echo
    echo "================================================================================"
    echo "MODE=${MODE} REP=${REP}"
    echo "================================================================================"
    echo "Start: $(date)"


    torchrun \
        --standalone \
        --nproc-per-node=4 \
        scripts/20_train_gradient_accumulation.py \
        --config configs/unet_heat_v100.yaml \
        --mode "${MODE}" \
        --microbatch-size 4 \
        --accumulation-steps 4 \
        --epochs 5 \
        --num-workers 2 \
        --timing-warmup-microbatches 8 \
        --result-path "${RESULT}"


    echo
    echo "Saved: ${RESULT}"
    echo "End: $(date)"
}


# Counterbalanced ordering.
#
# Rep 1: naive -> no_sync
# Rep 2: no_sync -> naive
# Rep 3: naive -> no_sync

run_experiment naive   1
run_experiment no_sync 1

run_experiment no_sync 2
run_experiment naive   2

run_experiment naive   3
run_experiment no_sync 3


echo
echo "================================================================================"
echo "PHASE 3 RESULT AUDIT"
echo "================================================================================"


python - <<'PY'
import json
import statistics
from pathlib import Path

root = Path(
    "results/rosi_v100/optimization/raw"
)

for mode in [
    "naive",
    "no_sync",
]:

    values = []

    print()
    print(mode)
    print("-" * 60)

    for rep in range(
        1,
        4,
    ):

        path = (
            root
            / f"unet_world4_{mode}_rep{rep}.json"
        )

        if not path.exists():

            raise SystemExit(
                f"Missing: {path}"
            )


        with path.open() as f:

            d = json.load(f)


        q = float(
            d[
                "steady_state_throughput_samples_per_sec"
            ]
        )


        mse = float(
            d[
                "history"
            ][-1][
                "val_mse"
            ]
        )


        values.append(
            q
        )


        print(
            f"rep {rep}: "
            f"Q={q:.2f} fields/s  "
            f"valMSE={mse:.6f}"
        )


    print(
        f"mean = {statistics.mean(values):.2f} fields/s"
    )

    print(
        f"std  = {statistics.stdev(values):.2f} fields/s"
    )


print()
print(
    "All 6 Phase 3 runs completed successfully."
)
PY


echo
echo "================================================================================"
echo "PHASE 3 DATA COLLECTION COMPLETE"
echo "================================================================================"
