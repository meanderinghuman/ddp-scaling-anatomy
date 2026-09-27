"""
Analyze Phase 2B PyTorch DDP / NCCL profiler runs.

This analysis deliberately avoids treating summed NCCL kernel time as
pure communication overhead because CUDA/NCCL work can overlap with
backward computation.

Reported metrics include:
    - NCCL all-reduce calls
    - all-reduces per active training step
    - NCCL kernel residency per rank
    - min / mean / max NCCL residency
    - rank-to-rank communication imbalance
    - benchmark-derived mean step time
    - NCCL residency / benchmark step-time diagnostic ratio

The last ratio is a pressure indicator only, NOT an overhead fraction.
"""

import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =====================================================================
# Paths
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]

PROFILE_ROOT = (
    ROOT
    / "results"
    / "rosi_v100"
    / "profiling"
)

RUN_ROOT = (
    PROFILE_ROOT
    / "runs"
)

ANALYSIS = (
    PROFILE_ROOT
    / "analysis"
)

FIGURES = (
    PROFILE_ROOT
    / "figures"
)

REPEATED_SUMMARY = (
    ROOT
    / "results"
    / "rosi_v100"
    / "repeated_runs"
    / "analysis"
    / "repeated_scaling_summary.csv"
)

ANALYSIS.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# Expected profiler matrix
# =====================================================================

RUNS = {
    "resnet_world1": {
        "workload": "resnet18",
        "label": "ResNet18",
        "world_size": 1,
    },

    "resnet_world4": {
        "workload": "resnet18",
        "label": "ResNet18",
        "world_size": 4,
    },

    "unet_world1": {
        "workload": "unet_heat",
        "label": "U-Net",
        "world_size": 1,
    },

    "unet_world4": {
        "workload": "unet_heat",
        "label": "U-Net",
        "world_size": 4,
    },
}


# =====================================================================
# Helpers
# =====================================================================

def load_json(path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def is_nccl_allreduce_kernel(name):

    # Select only the actual NCCL device kernel.
    #
    # Do NOT also add:
    #     nccl:all_reduce
    #     c10d::allreduce_
    #
    # because those are higher-level representations of the same
    # collective and summing them would double-count communication.

    return (
        name.startswith(
            "ncclDevKernel_AllReduce"
        )
    )


# =====================================================================
# Load repeated-run benchmark summary
# =====================================================================

if not REPEATED_SUMMARY.exists():

    raise SystemExit(
        f"Missing repeated benchmark summary:\n"
        f"{REPEATED_SUMMARY}"
    )


with REPEATED_SUMMARY.open(
    "r",
    encoding="utf-8",
) as f:

    benchmark_rows = list(
        csv.DictReader(f)
    )


def benchmark_row(
    workload,
    world_size,
):

    matches = [
        row
        for row in benchmark_rows
        if (
            row["workload"] == workload
            and
            int(row["world_size"])
            == world_size
        )
    ]

    if len(matches) != 1:

        raise RuntimeError(
            f"Expected one repeated benchmark row for "
            f"{workload}, world={world_size}; "
            f"found {len(matches)}"
        )

    return matches[0]


# =====================================================================
# Parse profiler runs
# =====================================================================

per_rank_rows = []

summary_rows = []


for tag, spec in RUNS.items():

    run_dir = (
        RUN_ROOT
        / tag
    )

    if not run_dir.exists():

        raise SystemExit(
            f"Missing profiler run directory:\n"
            f"{run_dir}"
        )


    workload = spec[
        "workload"
    ]

    label = spec[
        "label"
    ]

    world_size = spec[
        "world_size"
    ]


    # -------------------------------------------------------------
    # Benchmark reference
    # -------------------------------------------------------------

    bench = benchmark_row(
        workload,
        world_size,
    )


    throughput = float(
        bench[
            "throughput_mean"
        ]
    )

    global_batch = int(
        bench[
            "global_batch_size"
        ]
    )

    local_batch = int(
        bench[
            "local_batch_size"
        ]
    )


    # Approximate steady-state optimizer-step time.
    #
    # Q = global batch / step time
    #
    # therefore:
    #
    # step time = global batch / Q

    benchmark_step_time_ms = (
        global_batch
        / throughput
        * 1000.0
    )


    # -------------------------------------------------------------
    # Rank profiler data
    # -------------------------------------------------------------

    run_rank_rows = []


    for rank in range(
        world_size
    ):

        metadata_path = (
            run_dir
            / f"metadata_rank{rank}.json"
        )

        communication_path = (
            run_dir
            / f"communication_rank{rank}.json"
        )


        if not metadata_path.exists():

            raise RuntimeError(
                f"Missing {metadata_path}"
            )


        if not communication_path.exists():

            raise RuntimeError(
                f"Missing {communication_path}"
            )


        metadata = load_json(
            metadata_path
        )

        communication = load_json(
            communication_path
        )


        active_steps = int(
            metadata[
                "active_steps"
            ]
        )


        kernels = [
            event
            for event
            in communication
            if is_nccl_allreduce_kernel(
                event[
                    "name"
                ]
            )
        ]


        allreduce_calls = sum(
            int(
                event[
                    "count"
                ]
            )
            for event
            in kernels
        )


        nccl_us = sum(
            float(
                event[
                    "self_device_time_total_us"
                ]
            )
            for event
            in kernels
        )


        nccl_ms = (
            nccl_us
            / 1000.0
        )


        nccl_ms_per_step = (
            nccl_ms
            / active_steps
            if active_steps > 0
            else 0.0
        )


        calls_per_step = (
            allreduce_calls
            / active_steps
            if active_steps > 0
            else 0.0
        )


        pressure_ratio = (
            nccl_ms_per_step
            / benchmark_step_time_ms
            if benchmark_step_time_ms > 0
            else float("nan")
        )


        row = {
            "tag":
                tag,

            "workload":
                workload,

            "workload_label":
                label,

            "world_size":
                world_size,

            "rank":
                rank,

            "hostname":
                metadata[
                    "hostname"
                ],

            "global_batch_size":
                global_batch,

            "local_batch_size":
                local_batch,

            "active_steps":
                active_steps,

            "allreduce_calls":
                allreduce_calls,

            "allreduce_calls_per_step":
                calls_per_step,

            "nccl_kernel_total_ms":
                nccl_ms,

            "nccl_kernel_ms_per_step":
                nccl_ms_per_step,

            "benchmark_throughput":
                throughput,

            "benchmark_step_time_ms":
                benchmark_step_time_ms,

            "nccl_residency_to_step_ratio":
                pressure_ratio,
        }


        run_rank_rows.append(
            row
        )

        per_rank_rows.append(
            row
        )


    # -------------------------------------------------------------
    # Aggregate ranks
    # -------------------------------------------------------------

    nccl_per_step = [
        row[
            "nccl_kernel_ms_per_step"
        ]
        for row
        in run_rank_rows
    ]


    calls_per_step = [
        row[
            "allreduce_calls_per_step"
        ]
        for row
        in run_rank_rows
    ]


    min_nccl = min(
        nccl_per_step
    )

    max_nccl = max(
        nccl_per_step
    )

    mean_nccl = statistics.mean(
        nccl_per_step
    )


    std_nccl = (
        statistics.stdev(
            nccl_per_step
        )
        if len(
            nccl_per_step
        ) > 1
        else 0.0
    )


    imbalance_ratio = (
        max_nccl
        / min_nccl
        if min_nccl > 0
        else None
    )


    max_pressure_ratio = (
        max_nccl
        / benchmark_step_time_ms
        if benchmark_step_time_ms > 0
        else float("nan")
    )


    summary_rows.append(
        {
            "tag":
                tag,

            "workload":
                workload,

            "workload_label":
                label,

            "world_size":
                world_size,

            "global_batch_size":
                global_batch,

            "local_batch_size":
                local_batch,

            "benchmark_throughput":
                throughput,

            "benchmark_step_time_ms":
                benchmark_step_time_ms,

            "allreduce_calls_per_step_mean":
                statistics.mean(
                    calls_per_step
                ),

            "nccl_ms_per_step_min":
                min_nccl,

            "nccl_ms_per_step_mean":
                mean_nccl,

            "nccl_ms_per_step_max":
                max_nccl,

            "nccl_ms_per_step_std":
                std_nccl,

            "rank_imbalance_ratio":
                imbalance_ratio,

            "max_nccl_residency_to_step_ratio":
                max_pressure_ratio,
        }
    )


per_rank_rows.sort(
    key=lambda row: (
        row["workload"],
        row["world_size"],
        row["rank"],
    )
)


summary_rows.sort(
    key=lambda row: (
        row["workload"],
        row["world_size"],
    )
)


# =====================================================================
# CSV — per rank
# =====================================================================

per_rank_csv = (
    ANALYSIS
    / "profiler_per_rank.csv"
)


with per_rank_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            per_rank_rows[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        per_rank_rows
    )


# =====================================================================
# CSV — run summary
# =====================================================================

summary_csv = (
    ANALYSIS
    / "profiler_summary.csv"
)


with summary_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            summary_rows[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        summary_rows
    )


# =====================================================================
# Markdown
# =====================================================================

summary_md = (
    ANALYSIS
    / "profiler_summary.md"
)


with summary_md.open(
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "# Phase 2B — DDP / NCCL Profiling\n\n"
    )

    f.write(
        "NCCL times below represent summed device-kernel residency "
        "from the profiler. They must **not** be interpreted as pure "
        "unhidden communication overhead because NCCL communication "
        "may overlap backward computation.\n\n"
    )

    f.write(
        "The benchmark step time is derived independently from the "
        "three-run mean throughput as "
        "`global_batch_size / throughput`.\n\n"
    )


    f.write(
        "| Workload | GPUs | Local batch | "
        "All-reduces/step | NCCL ms/step mean | "
        "NCCL ms/step max | Rank imbalance | "
        "Benchmark step time |\n"
    )

    f.write(
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )


    for row in summary_rows:

        imbalance = (
            f"{row['rank_imbalance_ratio']:.2f}×"
            if row[
                "rank_imbalance_ratio"
            ] is not None
            else "n/a"
        )

        f.write(
            f"| {row['workload_label']} "
            f"| {row['world_size']} "
            f"| {row['local_batch_size']} "
            f"| {row['allreduce_calls_per_step_mean']:.2f} "
            f"| {row['nccl_ms_per_step_mean']:.3f} "
            f"| {row['nccl_ms_per_step_max']:.3f} "
            f"| {imbalance} "
            f"| {row['benchmark_step_time_ms']:.3f} ms |\n"
        )


    f.write(
        "\n"
    )

    f.write(
        "## Interpretation note\n\n"
    )

    f.write(
        "`max NCCL residency / benchmark step time` is retained in "
        "the CSV as a communication-pressure diagnostic. It is not "
        "an overhead percentage and may be large because CUDA streams "
        "can overlap in time.\n"
    )


# =====================================================================
# Figure 1 — 4-GPU NCCL time per rank
# =====================================================================

world4_rows = [
    row
    for row in per_rank_rows
    if row["world_size"] == 4
]


plt.figure(
    figsize=(8, 5)
)


for workload in [
    "resnet18",
    "unet_heat",
]:

    rows = [
        row
        for row in world4_rows
        if row[
            "workload"
        ] == workload
    ]


    rows.sort(
        key=lambda row:
            row["rank"]
    )


    plt.plot(
        [
            row["rank"]
            for row in rows
        ],
        [
            row[
                "nccl_kernel_ms_per_step"
            ]
            for row in rows
        ],
        marker="o",
        label=rows[0][
            "workload_label"
        ],
    )


plt.xlabel(
    "DDP rank"
)

plt.ylabel(
    "NCCL all-reduce kernel residency (ms/step)"
)

plt.title(
    "4-GPU NCCL Activity by Rank"
)

plt.xticks(
    [0, 1, 2, 3]
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES
    / "01_nccl_activity_by_rank.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# =====================================================================
# Figure 2 — 4-GPU mean/max NCCL activity
# =====================================================================

four_gpu_summary = [
    row
    for row in summary_rows
    if row["world_size"] == 4
]


labels = [
    row[
        "workload_label"
    ]
    for row in four_gpu_summary
]


mean_values = [
    row[
        "nccl_ms_per_step_mean"
    ]
    for row in four_gpu_summary
]


max_values = [
    row[
        "nccl_ms_per_step_max"
    ]
    for row in four_gpu_summary
]


x = list(
    range(
        len(labels)
    )
)


width = 0.35


plt.figure(
    figsize=(7, 5)
)


plt.bar(
    [
        value
        - width / 2
        for value in x
    ],
    mean_values,
    width=width,
    label="Mean across ranks",
)


plt.bar(
    [
        value
        + width / 2
        for value in x
    ],
    max_values,
    width=width,
    label="Slowest / max rank",
)


plt.xticks(
    x,
    labels,
)

plt.ylabel(
    "NCCL all-reduce kernel residency (ms/step)"
)

plt.title(
    "4-GPU NCCL Communication Activity"
)

plt.grid(
    True,
    axis="y",
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES
    / "02_nccl_mean_vs_max.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# =====================================================================
# Figure 3 — rank imbalance
# =====================================================================

imbalance_rows = [
    row
    for row in summary_rows
    if (
        row["world_size"] == 4
        and
        row["rank_imbalance_ratio"]
        is not None
    )
]


plt.figure(
    figsize=(7, 5)
)


plt.bar(
    [
        row[
            "workload_label"
        ]
        for row in imbalance_rows
    ],
    [
        row[
            "rank_imbalance_ratio"
        ]
        for row in imbalance_rows
    ],
)


plt.ylabel(
    "Max-rank / min-rank NCCL time"
)

plt.title(
    "4-GPU Communication Imbalance"
)

plt.grid(
    True,
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    FIGURES
    / "03_nccl_rank_imbalance.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# =====================================================================
# Figure 4 — benchmark step time vs max NCCL activity
# =====================================================================

plt.figure(
    figsize=(7, 5)
)


for row in four_gpu_summary:

    plt.scatter(
        row[
            "benchmark_step_time_ms"
        ],
        row[
            "nccl_ms_per_step_max"
        ],
        s=80,
        label=row[
            "workload_label"
        ],
    )


plt.xlabel(
    "Repeated-benchmark step time (ms)"
)

plt.ylabel(
    "Max-rank NCCL kernel residency (ms/step)"
)

plt.title(
    "Communication Pressure at 4 GPUs"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES
    / "04_nccl_vs_step_time.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# =====================================================================
# Console report
# =====================================================================

print()
print("=" * 105)
print("PHASE 2B — DDP / NCCL PROFILER SUMMARY")
print("=" * 105)


print(
    f"{'Workload':<12}"
    f"{'GPU':>5}"
    f"{'LB':>6}"
    f"{'AR/step':>10}"
    f"{'NCCL mean':>13}"
    f"{'NCCL max':>12}"
    f"{'Imbalance':>12}"
    f"{'Step time':>13}"
)


print(
    "-" * 105
)


for row in summary_rows:

    imbalance = (
        f"{row['rank_imbalance_ratio']:.2f}x"
        if row[
            "rank_imbalance_ratio"
        ] is not None
        else "n/a"
    )


    print(
        f"{row['workload_label']:<12}"
        f"{row['world_size']:>5d}"
        f"{row['local_batch_size']:>6d}"
        f"{row['allreduce_calls_per_step_mean']:>10.2f}"
        f"{row['nccl_ms_per_step_mean']:>12.3f} "
        f"{row['nccl_ms_per_step_max']:>11.3f} "
        f"{imbalance:>11}"
        f"{row['benchmark_step_time_ms']:>12.3f} ms"
    )


print()
print("=" * 105)
print("IMPORTANT")
print("=" * 105)

print(
    "NCCL kernel residency is NOT equivalent to pure communication "
    "overhead because NCCL and backward compute can overlap."
)

print()
print(
    "Per-rank CSV :", per_rank_csv
)

print(
    "Summary CSV  :", summary_csv
)

print(
    "Summary MD   :", summary_md
)

print(
    "Figures      :", FIGURES
)

print("=" * 105)
