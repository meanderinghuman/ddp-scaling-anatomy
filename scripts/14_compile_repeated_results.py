"""
Compile Phase 2A repeated DDP benchmark results.

Experiment:
    2 workloads
    x 3 GPU counts (1, 2, 4)
    x 3 repetitions
    = 18 full runs

Primary statistics:
    mean throughput
    sample standard deviation
    coefficient of variation
    95% confidence interval
    speedup
    scaling efficiency
"""

import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

RAW = (
    ROOT
    / "results"
    / "rosi_v100"
    / "repeated_runs"
    / "raw"
)

ANALYSIS = (
    ROOT
    / "results"
    / "rosi_v100"
    / "repeated_runs"
    / "analysis"
)

FIGURES = (
    ROOT
    / "results"
    / "rosi_v100"
    / "repeated_runs"
    / "figures"
)

ANALYSIS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


WORKLOADS = {
    "resnet18": "ResNet18 / CIFAR-100",
    "unet_heat": "U-Net / Heat2D",
}

WORLD_SIZES = [1, 2, 4]
REPEATS = [1, 2, 3]

EXPECTED_GLOBAL_BATCH = {
    "resnet18": 256,
    "unet_heat": 16,
}

# Student-t critical value for:
# 95% two-sided CI, n=3, df=2.
T_CRIT_95_DF2 = 4.302652729911275


def result_path(workload, world_size, rep):
    return (
        RAW
        / f"{workload}_world{world_size}_rep{rep}.json"
    )


# =====================================================================
# Verify all 18 files
# =====================================================================

expected = [
    result_path(workload, world_size, rep)
    for workload in WORKLOADS
    for world_size in WORLD_SIZES
    for rep in REPEATS
]

missing = [
    path for path in expected
    if not path.exists()
]

if missing:
    print("Missing repeated benchmark files:")
    for path in missing:
        print(" ", path)
    raise SystemExit(1)


print("=" * 100)
print("PHASE 2A — REPEATED DDP BENCHMARK ANALYSIS")
print("=" * 100)
print(f"Found {len(expected)} repeated-run JSON files.")


# =====================================================================
# Read individual runs
# =====================================================================

run_rows = []


for workload in WORKLOADS:

    for world_size in WORLD_SIZES:

        for rep in REPEATS:

            path = result_path(
                workload,
                world_size,
                rep,
            )

            with path.open(
                "r",
                encoding="utf-8",
            ) as f:
                d = json.load(f)

            # ---------------------------------------------------------
            # Integrity checks
            # ---------------------------------------------------------

            assert d["execution"] == "ddp"
            assert d["run_kind"] == "full"
            assert d["workload"] == workload
            assert int(d["world_size"]) == world_size
            assert d["backend"] == "nccl"
            assert d["device_type"] == "cuda"
            assert d["effective_amp"] is True
            assert d["amp_dtype"] == "fp16"
            assert d["grad_scaler_enabled"] is True
            assert d["max_train_batches"] is None
            assert d["max_val_batches"] is None

            assert (
                int(d["global_batch_size"])
                == EXPECTED_GLOBAL_BATCH[workload]
            )

            assert (
                int(d["local_batch_size"])
                * world_size
                == int(d["global_batch_size"])
            )

            history = d["history"]

            # Steady-state peak memory:
            # trainer policy excludes epoch 1.
            steady_history = history[1:]

            peak_vram_steady = max(
                float(row["peak_vram_per_rank_gb"])
                for row in steady_history
            )

            final = history[-1]

            run_rows.append(
                {
                    "workload": workload,
                    "workload_label": WORKLOADS[workload],
                    "world_size": world_size,
                    "repetition": rep,
                    "global_batch_size":
                        int(d["global_batch_size"]),
                    "local_batch_size":
                        int(d["local_batch_size"]),
                    "throughput":
                        float(
                            d[
                                "steady_state_throughput_samples_per_sec"
                            ]
                        ),
                    "epoch_wall_time":
                        float(
                            d[
                                "mean_steady_state_epoch_wall_time_sec"
                            ]
                        ),
                    "peak_vram_steady_gb":
                        peak_vram_steady,
                    "final_train_loss":
                        float(final["train_loss"]),
                    "final_val_accuracy":
                        (
                            float(final["val_accuracy"])
                            if workload == "resnet18"
                            else None
                        ),
                    "final_val_mse":
                        (
                            float(final["val_mse"])
                            if workload == "unet_heat"
                            else None
                        ),
                    "hostname":
                        d["rank_zero_hostname"],
                    "device":
                        d["rank_zero_device"],
                    "pytorch_version":
                        d["pytorch_version"],
                    "cuda_version":
                        d["cuda_version"],
                }
            )


# =====================================================================
# Per-configuration statistics
# =====================================================================

summary_rows = []


for workload in WORKLOADS:

    for world_size in WORLD_SIZES:

        rows = [
            row
            for row in run_rows
            if (
                row["workload"] == workload
                and
                row["world_size"] == world_size
            )
        ]

        if len(rows) != 3:
            raise RuntimeError(
                f"Expected 3 repetitions for "
                f"{workload} world{world_size}, "
                f"found {len(rows)}"
            )

        q = [
            row["throughput"]
            for row in rows
        ]

        epoch_times = [
            row["epoch_wall_time"]
            for row in rows
        ]

        vram = [
            row["peak_vram_steady_gb"]
            for row in rows
        ]

        q_mean = statistics.mean(q)
        q_std = statistics.stdev(q)

        q_sem = (
            q_std
            / math.sqrt(len(q))
        )

        q_ci95 = (
            T_CRIT_95_DF2
            * q_sem
        )

        q_cv = (
            q_std
            / q_mean
            * 100.0
        )

        epoch_mean = statistics.mean(
            epoch_times
        )

        epoch_std = statistics.stdev(
            epoch_times
        )

        vram_mean = statistics.mean(vram)
        vram_std = statistics.stdev(vram)

        if workload == "resnet18":

            final_metrics = [
                row["final_val_accuracy"]
                for row in rows
            ]

        else:

            final_metrics = [
                row["final_val_mse"]
                for row in rows
            ]

        final_metric_mean = statistics.mean(
            final_metrics
        )

        final_metric_std = statistics.stdev(
            final_metrics
        )

        summary_rows.append(
            {
                "workload": workload,
                "workload_label":
                    WORKLOADS[workload],
                "world_size": world_size,
                "global_batch_size":
                    rows[0]["global_batch_size"],
                "local_batch_size":
                    rows[0]["local_batch_size"],
                "n_runs": len(rows),
                "throughput_mean":
                    q_mean,
                "throughput_std":
                    q_std,
                "throughput_sem":
                    q_sem,
                "throughput_ci95_halfwidth":
                    q_ci95,
                "throughput_ci95_low":
                    q_mean - q_ci95,
                "throughput_ci95_high":
                    q_mean + q_ci95,
                "throughput_cv_percent":
                    q_cv,
                "epoch_wall_time_mean":
                    epoch_mean,
                "epoch_wall_time_std":
                    epoch_std,
                "peak_vram_mean_gb":
                    vram_mean,
                "peak_vram_std_gb":
                    vram_std,
                "final_metric_mean":
                    final_metric_mean,
                "final_metric_std":
                    final_metric_std,
            }
        )


# =====================================================================
# Speedup and efficiency from repeated-run means
# =====================================================================

for workload in WORKLOADS:

    workload_rows = [
        row
        for row in summary_rows
        if row["workload"] == workload
    ]

    q1 = next(
        row["throughput_mean"]
        for row in workload_rows
        if row["world_size"] == 1
    )

    for row in workload_rows:

        n = row["world_size"]

        speedup = (
            row["throughput_mean"]
            / q1
        )

        efficiency = (
            speedup
            / n
            * 100.0
        )

        row["speedup"] = speedup

        row[
            "scaling_efficiency_percent"
        ] = efficiency


summary_rows.sort(
    key=lambda row: (
        row["workload"],
        row["world_size"],
    )
)

run_rows.sort(
    key=lambda row: (
        row["workload"],
        row["world_size"],
        row["repetition"],
    )
)


# =====================================================================
# CSV — individual runs
# =====================================================================

per_run_csv = (
    ANALYSIS
    / "per_run_results.csv"
)

per_run_fields = list(
    run_rows[0].keys()
)

with per_run_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=per_run_fields,
    )

    writer.writeheader()
    writer.writerows(run_rows)


# =====================================================================
# CSV — repeated summary
# =====================================================================

summary_csv = (
    ANALYSIS
    / "repeated_scaling_summary.csv"
)

summary_fields = [
    "workload",
    "workload_label",
    "world_size",
    "global_batch_size",
    "local_batch_size",
    "n_runs",
    "throughput_mean",
    "throughput_std",
    "throughput_sem",
    "throughput_ci95_halfwidth",
    "throughput_ci95_low",
    "throughput_ci95_high",
    "throughput_cv_percent",
    "speedup",
    "scaling_efficiency_percent",
    "epoch_wall_time_mean",
    "epoch_wall_time_std",
    "peak_vram_mean_gb",
    "peak_vram_std_gb",
    "final_metric_mean",
    "final_metric_std",
]

with summary_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summary_fields,
    )

    writer.writeheader()
    writer.writerows(summary_rows)


# =====================================================================
# Markdown
# =====================================================================

summary_md = (
    ANALYSIS
    / "repeated_scaling_summary.md"
)

with summary_md.open(
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "# Phase 2A — Repeated V100 Strong-Scaling Results\n\n"
    )

    f.write(
        "Each workload/GPU-count configuration was run three times. "
        "The same model seed and training configuration were retained "
        "to measure system/runtime variability rather than seed variability.\n\n"
    )

    f.write(
        "The 95% confidence interval uses Student's t distribution "
        "with n=3 and df=2. Because only three repetitions are available, "
        "the confidence intervals should be interpreted cautiously; "
        "the raw runs, standard deviation, and coefficient of variation "
        "are also reported.\n\n"
    )

    for workload in WORKLOADS:

        f.write(
            f"## {WORKLOADS[workload]}\n\n"
        )

        f.write(
            "| GPUs | Local batch | Mean throughput | Std | CV | "
            "95% CI | Speedup | Efficiency | Mean VRAM/rank |\n"
        )

        f.write(
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )

        rows = [
            row
            for row in summary_rows
            if row["workload"] == workload
        ]

        for row in rows:

            f.write(
                f"| {row['world_size']} "
                f"| {row['local_batch_size']} "
                f"| {row['throughput_mean']:.2f} "
                f"| {row['throughput_std']:.2f} "
                f"| {row['throughput_cv_percent']:.2f}% "
                f"| ±{row['throughput_ci95_halfwidth']:.2f} "
                f"| {row['speedup']:.3f}× "
                f"| {row['scaling_efficiency_percent']:.2f}% "
                f"| {row['peak_vram_mean_gb']:.3f} GiB |\n"
            )

        f.write("\n")


# =====================================================================
# Plot helpers
# =====================================================================

def get_summary(workload):

    return sorted(
        [
            row
            for row in summary_rows
            if row["workload"] == workload
        ],
        key=lambda row:
            row["world_size"],
    )


def get_runs(workload, world_size):

    return sorted(
        [
            row
            for row in run_rows
            if (
                row["workload"] == workload
                and
                row["world_size"] == world_size
            )
        ],
        key=lambda row:
            row["repetition"],
    )


# =====================================================================
# Figure 1 — ResNet repeated throughput
# =====================================================================

for workload, filename, ylabel in [
    (
        "resnet18",
        "01_resnet_repeated_throughput.png",
        "Throughput (images/s)",
    ),
    (
        "unet_heat",
        "02_unet_repeated_throughput.png",
        "Throughput (fields/s)",
    ),
]:

    rows = get_summary(workload)

    x = [
        row["world_size"]
        for row in rows
    ]

    y = [
        row["throughput_mean"]
        for row in rows
    ]

    yerr = [
        row["throughput_ci95_halfwidth"]
        for row in rows
    ]

    plt.figure(figsize=(7, 5))

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        capsize=5,
        label="Mean ± 95% CI",
    )

    # Plot individual repetitions as raw evidence.
    for world_size in WORLD_SIZES:

        individual = get_runs(
            workload,
            world_size,
        )

        plt.scatter(
            [world_size] * len(individual),
            [
                row["throughput"]
                for row in individual
            ],
            marker="x",
        )

    plt.xlabel("Number of V100 GPUs")
    plt.ylabel(ylabel)

    plt.title(
        f"{WORKLOADS[workload]} — Repeated Strong Scaling"
    )

    plt.xticks(WORLD_SIZES)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES / filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# =====================================================================
# Figure 3 — Mean speedup
# =====================================================================

plt.figure(figsize=(7, 5))

for workload in WORKLOADS:

    rows = get_summary(workload)

    plt.plot(
        [
            row["world_size"]
            for row in rows
        ],
        [
            row["speedup"]
            for row in rows
        ],
        marker="o",
        label=WORKLOADS[workload],
    )

plt.plot(
    WORLD_SIZES,
    WORLD_SIZES,
    linestyle="--",
    label="Ideal linear scaling",
)

plt.xlabel("Number of V100 GPUs")
plt.ylabel("Speedup relative to 1 GPU")
plt.title("Repeated-Run DDP Strong-Scaling Speedup")
plt.xticks(WORLD_SIZES)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES / "03_repeated_speedup.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# =====================================================================
# Figure 4 — Mean scaling efficiency
# =====================================================================

plt.figure(figsize=(7, 5))

for workload in WORKLOADS:

    rows = get_summary(workload)

    plt.plot(
        [
            row["world_size"]
            for row in rows
        ],
        [
            row[
                "scaling_efficiency_percent"
            ]
            for row in rows
        ],
        marker="o",
        label=WORKLOADS[workload],
    )

plt.axhline(
    100.0,
    linestyle="--",
    label="Ideal efficiency",
)

plt.xlabel("Number of V100 GPUs")
plt.ylabel("Scaling efficiency (%)")
plt.title("Repeated-Run DDP Scaling Efficiency")
plt.xticks(WORLD_SIZES)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES / "04_repeated_scaling_efficiency.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# =====================================================================
# Figure 5 — Runtime variability
# =====================================================================

plt.figure(figsize=(7, 5))

for workload in WORKLOADS:

    rows = get_summary(workload)

    plt.plot(
        [
            row["world_size"]
            for row in rows
        ],
        [
            row["throughput_cv_percent"]
            for row in rows
        ],
        marker="o",
        label=WORKLOADS[workload],
    )

plt.xlabel("Number of V100 GPUs")
plt.ylabel("Throughput coefficient of variation (%)")
plt.title("Benchmark Run-to-Run Variability")
plt.xticks(WORLD_SIZES)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES / "05_throughput_variability_cv.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# =====================================================================
# Console report
# =====================================================================

print()
print("=" * 100)
print("REPEATED-RUN STRONG-SCALING SUMMARY")
print("=" * 100)


for workload in WORKLOADS:

    print()
    print(WORKLOADS[workload])
    print("-" * 100)

    print(
        f"{'GPU':>4} "
        f"{'Mean Q':>12} "
        f"{'Std':>10} "
        f"{'CV':>8} "
        f"{'95% CI':>12} "
        f"{'Speedup':>10} "
        f"{'Eff.':>9}"
    )

    for row in get_summary(workload):

        print(
            f"{row['world_size']:>4d} "
            f"{row['throughput_mean']:>12.2f} "
            f"{row['throughput_std']:>10.2f} "
            f"{row['throughput_cv_percent']:>7.2f}% "
            f"±{row['throughput_ci95_halfwidth']:>10.2f} "
            f"{row['speedup']:>9.3f}x "
            f"{row['scaling_efficiency_percent']:>8.2f}%"
        )


print()
print("=" * 100)
print("OUTPUTS")
print("=" * 100)

print("Per-run CSV :", per_run_csv)
print("Summary CSV :", summary_csv)
print("Summary MD  :", summary_md)
print("Figures     :", FIGURES)

print("=" * 100)
