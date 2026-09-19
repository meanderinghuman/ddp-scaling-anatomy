"""
Compile final ROSI V100 strong-scaling benchmark results.

Only *_full.json files are used.

Primary metrics
---------------
Throughput:
    Q_N = saved steady_state_throughput_samples_per_sec

Speedup:
    S_N = Q_N / Q_1

Scaling efficiency:
    E_N = S_N / N * 100

The trainer defines steady state by excluding epoch 1.
Epoch 1 is retained for learning-curve plots but is not used for the
primary scaling performance statistics.
"""

import csv
import json
import shutil
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
# Paths
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "results" / "raw"

OUT_ROOT = ROOT / "results" / "rosi_v100"
OUT_RAW = OUT_ROOT / "raw"
OUT_ANALYSIS = OUT_ROOT / "analysis"
OUT_FIGURES = OUT_ROOT / "figures"

for path in (OUT_RAW, OUT_ANALYSIS, OUT_FIGURES):
    path.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Benchmark definition
# =====================================================================

WORLD_SIZES = [1, 2, 4]

WORKLOAD_LABELS = {
    "resnet18": "ResNet18 / CIFAR-100",
    "unet_heat": "U-Net / Heat2D",
}

EXPECTED_GLOBAL_BATCH = {
    "resnet18": 256,
    "unet_heat": 16,
}


def result_path(workload, world_size):
    return (
        RAW_DIR
        / f"{workload}_ddp_cuda_nccl_world{world_size}_full.json"
    )


# =====================================================================
# Verify all six runs
# =====================================================================

expected_files = [
    result_path(workload, world_size)
    for workload in WORKLOAD_LABELS
    for world_size in WORLD_SIZES
]

missing = [
    path
    for path in expected_files
    if not path.exists()
]

if missing:
    print("Missing final benchmark files:")
    for path in missing:
        print("  ", path)
    raise SystemExit(1)

print("=" * 90)
print("ROSI V100 FINAL BENCHMARK COMPILER")
print("=" * 90)
print(f"Found {len(expected_files)} full benchmark files.")


# =====================================================================
# Load + validate
# =====================================================================

summary_rows = []
epoch_rows = []


for workload in WORKLOAD_LABELS:

    for world_size in WORLD_SIZES:

        path = result_path(workload, world_size)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # -------------------------------------------------------------
        # Integrity checks
        # -------------------------------------------------------------

        assert data["execution"] == "ddp"
        assert data["run_kind"] == "full"
        assert data["workload"] == workload

        assert int(data["world_size"]) == world_size

        assert data["backend"] == "nccl"
        assert data["device_type"] == "cuda"

        assert int(data["global_batch_size"]) == EXPECTED_GLOBAL_BATCH[workload]

        assert (
            int(data["local_batch_size"]) * world_size
            == int(data["global_batch_size"])
        )

        assert data["effective_amp"] is True
        assert data["amp_dtype"] == "fp16"
        assert data["grad_scaler_enabled"] is True

        assert data["max_train_batches"] is None
        assert data["max_val_batches"] is None

        history = data["history"]

        if len(history) != int(data["epochs"]):
            raise RuntimeError(
                f"History length mismatch in {path.name}"
            )

        if len(history) < 2:
            raise RuntimeError(
                f"Need at least two epochs in {path.name}"
            )

        # -------------------------------------------------------------
        # Trainer's authoritative steady-state metrics
        # -------------------------------------------------------------

        throughput = float(
            data["steady_state_throughput_samples_per_sec"]
        )

        epoch_wall_time = float(
            data["mean_steady_state_epoch_wall_time_sec"]
        )

        # Epoch 1 excluded from steady diagnostics.
        steady_history = history[1:]

        steady_throughputs = [
            float(row["throughput_samples_per_sec"])
            for row in steady_history
        ]

        throughput_std = (
            statistics.stdev(steady_throughputs)
            if len(steady_throughputs) > 1
            else 0.0
        )

        steady_wall_times = [
            float(row["epoch_wall_time_sec"])
            for row in steady_history
        ]

        wall_time_std = (
            statistics.stdev(steady_wall_times)
            if len(steady_wall_times) > 1
            else 0.0
        )

        steady_measured_times = [
            float(row["measured_train_time_sec"])
            for row in steady_history
        ]

        mean_measured_train_time = statistics.mean(
            steady_measured_times
        )

        # Peak memory:
        # all epochs catches initialization/startup allocation;
        # steady-state ignores epoch 1.
        peak_vram_all = max(
            float(row["peak_vram_per_rank_gb"])
            for row in history
        )

        peak_vram_steady = max(
            float(row["peak_vram_per_rank_gb"])
            for row in steady_history
        )

        final = history[-1]

        summary_rows.append(
            {
                "workload": workload,
                "workload_label": WORKLOAD_LABELS[workload],
                "world_size": world_size,
                "global_batch_size": int(data["global_batch_size"]),
                "local_batch_size": int(data["local_batch_size"]),
                "throughput_samples_per_sec": throughput,
                "throughput_std": throughput_std,
                "mean_steady_epoch_wall_time_sec": epoch_wall_time,
                "epoch_wall_time_std": wall_time_std,
                "mean_measured_train_time_sec": mean_measured_train_time,
                "peak_vram_all_gb": peak_vram_all,
                "peak_vram_steady_gb": peak_vram_steady,
                "final_train_loss": float(final["train_loss"]),
                "final_val_loss": (
                    float(final["val_loss"])
                    if workload == "resnet18"
                    else None
                ),
                "final_val_accuracy": (
                    float(final["val_accuracy"])
                    if workload == "resnet18"
                    else None
                ),
                "final_val_mse": (
                    float(final["val_mse"])
                    if workload == "unet_heat"
                    else None
                ),
                "num_workers_per_rank": int(data["num_workers_per_rank"]),
                "parameter_count": int(data["parameter_count"]),
                "pytorch_version": data["pytorch_version"],
                "cuda_version": data["cuda_version"],
                "rank_zero_device": data["rank_zero_device"],
                "rank_zero_hostname": data["rank_zero_hostname"],
                "amp_dtype": data["amp_dtype"],
            }
        )

        # -------------------------------------------------------------
        # Per-epoch data
        # -------------------------------------------------------------

        for row in history:

            epoch_rows.append(
                {
                    "workload": workload,
                    "world_size": world_size,
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "throughput_samples_per_sec":
                        float(row["throughput_samples_per_sec"]),
                    "epoch_wall_time_sec":
                        float(row["epoch_wall_time_sec"]),
                    "measured_train_time_sec":
                        float(row["measured_train_time_sec"]),
                    "global_train_examples":
                        int(row["global_train_examples"]),
                    "global_measured_examples":
                        int(row["global_measured_examples"]),
                    "peak_vram_per_rank_gb":
                        float(row["peak_vram_per_rank_gb"]),
                    "val_examples":
                        int(row["val_examples"]),
                    "val_loss":
                        row.get("val_loss"),
                    "val_accuracy":
                        row.get("val_accuracy"),
                    "val_mse":
                        row.get("val_mse"),
                }
            )

        # Preserve only final/full JSONs in repo-ready bundle.
        shutil.copy2(
            path,
            OUT_RAW / path.name,
        )


# =====================================================================
# Speedup + efficiency
# =====================================================================

for workload in WORKLOAD_LABELS:

    rows = [
        row
        for row in summary_rows
        if row["workload"] == workload
    ]

    q1 = next(
        row["throughput_samples_per_sec"]
        for row in rows
        if row["world_size"] == 1
    )

    for row in rows:

        n = row["world_size"]

        speedup = (
            row["throughput_samples_per_sec"]
            / q1
        )

        efficiency = (
            speedup
            / n
            * 100.0
        )

        row["speedup"] = speedup

        row["scaling_efficiency_percent"] = efficiency

        row["parallel_overhead_percent"] = (
            100.0 - efficiency
        )


summary_rows.sort(
    key=lambda r: (
        r["workload"],
        r["world_size"],
    )
)

epoch_rows.sort(
    key=lambda r: (
        r["workload"],
        r["world_size"],
        r["epoch"],
    )
)


# =====================================================================
# CSV outputs
# =====================================================================

summary_csv = OUT_ANALYSIS / "scaling_summary.csv"

summary_fields = [
    "workload",
    "workload_label",
    "world_size",
    "global_batch_size",
    "local_batch_size",
    "throughput_samples_per_sec",
    "throughput_std",
    "speedup",
    "scaling_efficiency_percent",
    "parallel_overhead_percent",
    "mean_steady_epoch_wall_time_sec",
    "epoch_wall_time_std",
    "mean_measured_train_time_sec",
    "peak_vram_all_gb",
    "peak_vram_steady_gb",
    "final_train_loss",
    "final_val_loss",
    "final_val_accuracy",
    "final_val_mse",
    "num_workers_per_rank",
    "parameter_count",
    "pytorch_version",
    "cuda_version",
    "rank_zero_device",
    "rank_zero_hostname",
    "amp_dtype",
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


epoch_csv = OUT_ANALYSIS / "epoch_metrics.csv"

epoch_fields = [
    "workload",
    "world_size",
    "epoch",
    "train_loss",
    "throughput_samples_per_sec",
    "epoch_wall_time_sec",
    "measured_train_time_sec",
    "global_train_examples",
    "global_measured_examples",
    "peak_vram_per_rank_gb",
    "val_examples",
    "val_loss",
    "val_accuracy",
    "val_mse",
]

with epoch_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=epoch_fields,
    )

    writer.writeheader()
    writer.writerows(epoch_rows)


# =====================================================================
# Markdown table
# =====================================================================

summary_md = OUT_ANALYSIS / "scaling_summary.md"

with summary_md.open(
    "w",
    encoding="utf-8",
) as f:

    f.write("# ROSI V100 DDP Strong-Scaling Results\n\n")

    f.write(
        "Primary performance metrics use the trainer's saved "
        "steady-state statistics, which exclude epoch 1.\n\n"
    )

    f.write(
        "Speedup: $S_N = Q_N / Q_1$.  "
        "Efficiency: $E_N = S_N / N \\times 100\\%$.\n\n"
    )

    for workload in WORKLOAD_LABELS:

        f.write(
            f"## {WORKLOAD_LABELS[workload]}\n\n"
        )

        f.write(
            "| GPUs | Global batch | Local batch | "
            "Throughput | Speedup | Efficiency | "
            "Epoch time | Steady VRAM/rank | Final metric |\n"
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

            if workload == "resnet18":

                metric = (
                    f"{100.0 * row['final_val_accuracy']:.2f}% acc"
                )

            else:

                metric = (
                    f"{row['final_val_mse']:.6f} MSE"
                )

            f.write(
                f"| {row['world_size']} "
                f"| {row['global_batch_size']} "
                f"| {row['local_batch_size']} "
                f"| {row['throughput_samples_per_sec']:.2f} "
                f"| {row['speedup']:.3f}x "
                f"| {row['scaling_efficiency_percent']:.2f}% "
                f"| {row['mean_steady_epoch_wall_time_sec']:.3f}s "
                f"| {row['peak_vram_steady_gb']:.3f} GiB "
                f"| {metric} |\n"
            )

        f.write("\n")


# =====================================================================
# Manifest
# =====================================================================

manifest = {
    "benchmark_name":
        "ROSI V100 single-node DDP strong scaling",

    "hardware":
        "NVIDIA Tesla V100-SXM2-32GB",

    "world_sizes":
        WORLD_SIZES,

    "backend":
        "NCCL",

    "precision":
        "FP16 AMP + GradScaler",

    "strong_scaling":
        True,

    "steady_state_policy":
        "Epoch 1 excluded from primary performance statistics.",

    "speedup":
        "S_N = Q_N / Q_1",

    "efficiency":
        "E_N = S_N / N * 100%",

    "workloads":
        WORKLOAD_LABELS,

    "raw_files": [
        path.name
        for path in expected_files
    ],
}

with (
    OUT_ANALYSIS
    / "benchmark_manifest.json"
).open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        manifest,
        f,
        indent=2,
    )


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
        key=lambda r: r["world_size"],
    )


def get_epochs(workload, world_size):

    return sorted(
        [
            row
            for row in epoch_rows
            if (
                row["workload"] == workload
                and
                row["world_size"] == world_size
            )
        ],
        key=lambda r: r["epoch"],
    )


# =====================================================================
# Figure 1: Throughput
# =====================================================================

plt.figure(figsize=(7, 5))

for workload in WORKLOAD_LABELS:

    rows = get_summary(workload)

    plt.errorbar(
        [r["world_size"] for r in rows],
        [r["throughput_samples_per_sec"] for r in rows],
        yerr=[r["throughput_std"] for r in rows],
        marker="o",
        capsize=4,
        label=WORKLOAD_LABELS[workload],
    )

plt.xlabel("Number of GPUs")
plt.ylabel("Throughput (samples/s)")
plt.title("DDP Strong-Scaling Throughput")
plt.xticks(WORLD_SIZES)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT_FIGURES / "01_throughput_vs_gpus.png",
    dpi=180,
)

plt.close()


# =====================================================================
# Figure 2: Speedup
# =====================================================================

plt.figure(figsize=(7, 5))

for workload in WORKLOAD_LABELS:

    rows = get_summary(workload)

    plt.plot(
        [r["world_size"] for r in rows],
        [r["speedup"] for r in rows],
        marker="o",
        label=WORKLOAD_LABELS[workload],
    )

plt.plot(
    WORLD_SIZES,
    WORLD_SIZES,
    linestyle="--",
    label="Ideal linear scaling",
)

plt.xlabel("Number of GPUs")
plt.ylabel("Speedup relative to 1 GPU")
plt.title("DDP Strong-Scaling Speedup")
plt.xticks(WORLD_SIZES)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT_FIGURES / "02_speedup_vs_gpus.png",
    dpi=180,
)

plt.close()


# =====================================================================
# Figure 3: Scaling efficiency
# =====================================================================

plt.figure(figsize=(7, 5))

for workload in WORKLOAD_LABELS:

    rows = get_summary(workload)

    plt.plot(
        [r["world_size"] for r in rows],
        [r["scaling_efficiency_percent"] for r in rows],
        marker="o",
        label=WORKLOAD_LABELS[workload],
    )

plt.axhline(
    100.0,
    linestyle="--",
    label="Ideal",
)

plt.xlabel("Number of GPUs")
plt.ylabel("Scaling efficiency (%)")
plt.title("DDP Strong-Scaling Efficiency")
plt.xticks(WORLD_SIZES)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT_FIGURES / "03_scaling_efficiency.png",
    dpi=180,
)

plt.close()


# =====================================================================
# Figure 4: Epoch wall time
# =====================================================================

plt.figure(figsize=(7, 5))

for workload in WORKLOAD_LABELS:

    rows = get_summary(workload)

    plt.errorbar(
        [r["world_size"] for r in rows],
        [r["mean_steady_epoch_wall_time_sec"] for r in rows],
        yerr=[r["epoch_wall_time_std"] for r in rows],
        marker="o",
        capsize=4,
        label=WORKLOAD_LABELS[workload],
    )

plt.xlabel("Number of GPUs")
plt.ylabel("Steady-state epoch wall time (s)")
plt.title("Epoch Time vs GPU Count")
plt.xticks(WORLD_SIZES)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT_FIGURES / "04_epoch_time_vs_gpus.png",
    dpi=180,
)

plt.close()


# =====================================================================
# Figure 5: VRAM per rank
# =====================================================================

plt.figure(figsize=(7, 5))

for workload in WORKLOAD_LABELS:

    rows = get_summary(workload)

    plt.plot(
        [r["world_size"] for r in rows],
        [r["peak_vram_steady_gb"] for r in rows],
        marker="o",
        label=WORKLOAD_LABELS[workload],
    )

plt.xlabel("Number of GPUs")
plt.ylabel("Peak VRAM per rank (GiB)")
plt.title("Steady-State GPU Memory per DDP Rank")
plt.xticks(WORLD_SIZES)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT_FIGURES / "05_peak_vram_per_rank.png",
    dpi=180,
)

plt.close()


# =====================================================================
# Figure 6: ResNet validation accuracy
# =====================================================================

plt.figure(figsize=(7, 5))

for world_size in WORLD_SIZES:

    rows = get_epochs(
        "resnet18",
        world_size,
    )

    plt.plot(
        [r["epoch"] for r in rows],
        [
            100.0 * float(r["val_accuracy"])
            for r in rows
        ],
        marker="o",
        label=f"{world_size} GPU",
    )

plt.xlabel("Epoch")
plt.ylabel("Validation accuracy (%)")
plt.title("ResNet18 / CIFAR-100 Validation Accuracy")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT_FIGURES / "06_resnet_validation_accuracy.png",
    dpi=180,
)

plt.close()


# =====================================================================
# Figure 7: U-Net validation MSE
# =====================================================================

plt.figure(figsize=(7, 5))

for world_size in WORLD_SIZES:

    rows = get_epochs(
        "unet_heat",
        world_size,
    )

    plt.plot(
        [r["epoch"] for r in rows],
        [
            float(r["val_mse"])
            for r in rows
        ],
        marker="o",
        label=f"{world_size} GPU",
    )

plt.xlabel("Epoch")
plt.ylabel("Validation MSE")
plt.title("U-Net Heat2D Validation MSE")
plt.yscale("log")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT_FIGURES / "07_unet_validation_mse.png",
    dpi=180,
)

plt.close()


# =====================================================================
# Figures 8-9: Training curves
# =====================================================================

for workload, filename, ylabel in [
    (
        "resnet18",
        "08_resnet_training_loss.png",
        "Training loss",
    ),
    (
        "unet_heat",
        "09_unet_training_mse.png",
        "Training MSE",
    ),
]:

    plt.figure(figsize=(7, 5))

    for world_size in WORLD_SIZES:

        rows = get_epochs(
            workload,
            world_size,
        )

        plt.plot(
            [r["epoch"] for r in rows],
            [r["train_loss"] for r in rows],
            marker="o",
            label=f"{world_size} GPU",
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)

    plt.title(
        f"{WORKLOAD_LABELS[workload]} Training"
    )

    if workload == "unet_heat":
        plt.yscale("log")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUT_FIGURES / filename,
        dpi=180,
    )

    plt.close()


# =====================================================================
# Console summary
# =====================================================================

print()
print("=" * 90)
print("FINAL STRONG-SCALING SUMMARY")
print("=" * 90)

for workload in WORKLOAD_LABELS:

    print()
    print(WORKLOAD_LABELS[workload])
    print("-" * 90)

    print(
        f"{'GPU':>4} "
        f"{'GB':>6} "
        f"{'LB':>6} "
        f"{'Throughput':>14} "
        f"{'Speedup':>10} "
        f"{'Efficiency':>12} "
        f"{'VRAM':>10}"
    )

    for row in get_summary(workload):

        print(
            f"{row['world_size']:>4d} "
            f"{row['global_batch_size']:>6d} "
            f"{row['local_batch_size']:>6d} "
            f"{row['throughput_samples_per_sec']:>14.2f} "
            f"{row['speedup']:>9.3f}x "
            f"{row['scaling_efficiency_percent']:>11.2f}% "
            f"{row['peak_vram_steady_gb']:>9.3f}G"
        )

print()
print("=" * 90)
print("OUTPUTS")
print("=" * 90)
print("Summary CSV :", summary_csv)
print("Epoch CSV   :", epoch_csv)
print("Summary MD  :", summary_md)
print("Figures     :", OUT_FIGURES)
print("Raw bundle  :", OUT_RAW)
print("=" * 90)
