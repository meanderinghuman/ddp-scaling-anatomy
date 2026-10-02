"""
Analyze Phase 2C:
ordinary BatchNorm vs SyncBatchNorm on ResNet18 / 4 V100 GPUs.

Baseline:
    results/rosi_v100/repeated_runs/analysis/per_run_results.csv

SyncBN:
    results/rosi_v100/syncbn/raw/
        resnet18_world4_syncbn_rep1.json
        resnet18_world4_syncbn_rep2.json
        resnet18_world4_syncbn_rep3.json
"""

import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

BASELINE_CSV = (
    ROOT
    / "results"
    / "rosi_v100"
    / "repeated_runs"
    / "analysis"
    / "per_run_results.csv"
)

SYNCBN_ROOT = (
    ROOT
    / "results"
    / "rosi_v100"
    / "syncbn"
)

SYNCBN_RAW = (
    SYNCBN_ROOT
    / "raw"
)

ANALYSIS = (
    SYNCBN_ROOT
    / "analysis"
)

FIGURES = (
    SYNCBN_ROOT
    / "figures"
)

ANALYSIS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Read ordinary-BatchNorm baseline
# =====================================================================

with BASELINE_CSV.open(
    "r",
    encoding="utf-8",
) as f:

    rows = list(
        csv.DictReader(f)
    )


baseline_rows = [
    row
    for row in rows
    if (
        row["workload"] == "resnet18"
        and int(row["world_size"]) == 4
    )
]


if len(baseline_rows) != 3:

    raise RuntimeError(
        "Expected exactly 3 ResNet18 world4 baseline repetitions, "
        f"found {len(baseline_rows)}"
    )


baseline = []

for row in baseline_rows:

    baseline.append(
        {
            "normalization": "BatchNorm",
            "repetition": int(
                row["repetition"]
            ),
            "throughput": float(
                row["throughput"]
            ),
            "epoch_wall_time": float(
                row["epoch_wall_time"]
            ),
            "peak_vram_gb": float(
                row["peak_vram_steady_gb"]
            ),
            "val_accuracy": float(
                row["final_val_accuracy"]
            ),
            "train_loss": float(
                row["final_train_loss"]
            ),
            "hostname": row[
                "hostname"
            ],
        }
    )


# =====================================================================
# Read SyncBatchNorm runs
# =====================================================================

syncbn = []


for rep in (1, 2, 3):

    path = (
        SYNCBN_RAW
        / f"resnet18_world4_syncbn_rep{rep}.json"
    )

    if not path.exists():

        raise RuntimeError(
            f"Missing SyncBN run: {path}"
        )


    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        d = json.load(f)


    history = d[
        "history"
    ]

    final = history[-1]

    steady = history[1:]


    syncbn.append(
        {
            "normalization": "SyncBatchNorm",
            "repetition": rep,

            "throughput": float(
                d[
                    "steady_state_throughput_samples_per_sec"
                ]
            ),

            "epoch_wall_time": float(
                d[
                    "mean_steady_state_epoch_wall_time_sec"
                ]
            ),

            "peak_vram_gb": max(
                float(
                    epoch[
                        "peak_vram_per_rank_gb"
                    ]
                )
                for epoch in steady
            ),

            "val_accuracy": float(
                final[
                    "val_accuracy"
                ]
            ),

            "train_loss": float(
                final[
                    "train_loss"
                ]
            ),

            "hostname": d[
                "rank_zero_hostname"
            ],
        }
    )


all_rows = (
    baseline
    + syncbn
)


# =====================================================================
# Save per-run CSV
# =====================================================================

per_run_csv = (
    ANALYSIS
    / "syncbn_per_run.csv"
)


with per_run_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            all_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        all_rows
    )


# =====================================================================
# Statistics
# =====================================================================

def summarize(
    normalization,
    data,
):

    throughput = [
        row["throughput"]
        for row in data
    ]

    accuracy = [
        row["val_accuracy"]
        for row in data
    ]

    epoch_time = [
        row["epoch_wall_time"]
        for row in data
    ]

    memory = [
        row["peak_vram_gb"]
        for row in data
    ]

    train_loss = [
        row["train_loss"]
        for row in data
    ]


    q_mean = statistics.mean(
        throughput
    )

    q_std = statistics.stdev(
        throughput
    )


    return {
        "normalization":
            normalization,

        "n_runs":
            len(data),

        "throughput_mean":
            q_mean,

        "throughput_std":
            q_std,

        "throughput_cv_percent":
            q_std
            / q_mean
            * 100.0,

        "val_accuracy_mean":
            statistics.mean(
                accuracy
            ),

        "val_accuracy_std":
            statistics.stdev(
                accuracy
            ),

        "epoch_time_mean":
            statistics.mean(
                epoch_time
            ),

        "epoch_time_std":
            statistics.stdev(
                epoch_time
            ),

        "peak_vram_mean_gb":
            statistics.mean(
                memory
            ),

        "peak_vram_std_gb":
            statistics.stdev(
                memory
            ),

        "final_train_loss_mean":
            statistics.mean(
                train_loss
            ),

        "final_train_loss_std":
            statistics.stdev(
                train_loss
            ),
    }


summary = [
    summarize(
        "BatchNorm",
        baseline,
    ),

    summarize(
        "SyncBatchNorm",
        syncbn,
    ),
]


bn = summary[0]
sb = summary[1]


throughput_change_percent = (
    (
        sb["throughput_mean"]
        -
        bn["throughput_mean"]
    )
    /
    bn["throughput_mean"]
    *
    100.0
)


epoch_time_change_percent = (
    (
        sb["epoch_time_mean"]
        -
        bn["epoch_time_mean"]
    )
    /
    bn["epoch_time_mean"]
    *
    100.0
)


accuracy_change_pp = (
    sb["val_accuracy_mean"]
    -
    bn["val_accuracy_mean"]
) * 100.0


vram_change_percent = (
    (
        sb["peak_vram_mean_gb"]
        -
        bn["peak_vram_mean_gb"]
    )
    /
    bn["peak_vram_mean_gb"]
    *
    100.0
)


for row in summary:

    row[
        "throughput_change_vs_batchnorm_percent"
    ] = (
        0.0
        if row[
            "normalization"
        ] == "BatchNorm"
        else throughput_change_percent
    )

    row[
        "accuracy_change_vs_batchnorm_pp"
    ] = (
        0.0
        if row[
            "normalization"
        ] == "BatchNorm"
        else accuracy_change_pp
    )


# =====================================================================
# Summary CSV
# =====================================================================

summary_csv = (
    ANALYSIS
    / "syncbn_summary.csv"
)


with summary_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            summary[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        summary
    )


# =====================================================================
# Markdown
# =====================================================================

summary_md = (
    ANALYSIS
    / "syncbn_summary.md"
)


with summary_md.open(
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "# Phase 2C — BatchNorm vs SyncBatchNorm\n\n"
    )

    f.write(
        "ResNet18, 4 V100 GPUs, global batch 256, "
        "local batch 64, FP16 AMP, three runs per condition.\n\n"
    )

    f.write(
        "| Normalization | Throughput | CV | "
        "Val accuracy | Epoch time | VRAM/rank |\n"
    )

    f.write(
        "|---|---:|---:|---:|---:|---:|\n"
    )


    for row in summary:

        f.write(
            f"| {row['normalization']} "
            f"| {row['throughput_mean']:.2f} "
            f"± {row['throughput_std']:.2f} img/s "
            f"| {row['throughput_cv_percent']:.2f}% "
            f"| {100 * row['val_accuracy_mean']:.2f}% "
            f"± {100 * row['val_accuracy_std']:.2f} pp "
            f"| {row['epoch_time_mean']:.3f} "
            f"± {row['epoch_time_std']:.3f} s "
            f"| {row['peak_vram_mean_gb']:.3f} GiB |\n"
        )


    f.write("\n")

    f.write(
        "## SyncBatchNorm change relative to BatchNorm\n\n"
    )

    f.write(
        f"- Throughput: {throughput_change_percent:+.2f}%\n"
    )

    f.write(
        f"- Mean epoch time: {epoch_time_change_percent:+.2f}%\n"
    )

    f.write(
        f"- Validation accuracy: {accuracy_change_pp:+.2f} percentage points\n"
    )

    f.write(
        f"- Peak VRAM/rank: {vram_change_percent:+.2f}%\n"
    )

    f.write(
        "\nThe validation-accuracy comparison is based on only "
        "three five-epoch runs per condition and should not be "
        "interpreted as a definitive generalization result.\n"
    )


# =====================================================================
# Figure 1 — throughput
# =====================================================================

labels = [
    row[
        "normalization"
    ]
    for row in summary
]

means = [
    row[
        "throughput_mean"
    ]
    for row in summary
]

errors = [
    row[
        "throughput_std"
    ]
    for row in summary
]


plt.figure(
    figsize=(7, 5)
)

plt.bar(
    labels,
    means,
    yerr=errors,
    capsize=6,
)

plt.ylabel(
    "Throughput (images/s)"
)

plt.title(
    "ResNet18 / 4 V100 — BatchNorm vs SyncBatchNorm"
)

plt.grid(
    True,
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    FIGURES
    / "01_batchnorm_vs_syncbn_throughput.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# =====================================================================
# Figure 2 — accuracy
# =====================================================================

accuracy_mean = [
    100
    * row[
        "val_accuracy_mean"
    ]
    for row in summary
]

accuracy_std = [
    100
    * row[
        "val_accuracy_std"
    ]
    for row in summary
]


plt.figure(
    figsize=(7, 5)
)

plt.bar(
    labels,
    accuracy_mean,
    yerr=accuracy_std,
    capsize=6,
)

plt.ylabel(
    "Final validation accuracy (%)"
)

plt.title(
    "Normalization and Short-Run Validation Accuracy"
)

plt.grid(
    True,
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    FIGURES
    / "02_batchnorm_vs_syncbn_accuracy.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# =====================================================================
# Figure 3 — epoch time
# =====================================================================

epoch_mean = [
    row[
        "epoch_time_mean"
    ]
    for row in summary
]

epoch_std = [
    row[
        "epoch_time_std"
    ]
    for row in summary
]


plt.figure(
    figsize=(7, 5)
)

plt.bar(
    labels,
    epoch_mean,
    yerr=epoch_std,
    capsize=6,
)

plt.ylabel(
    "Mean steady-state epoch time (s)"
)

plt.title(
    "Normalization Cost at 4 GPUs"
)

plt.grid(
    True,
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    FIGURES
    / "03_batchnorm_vs_syncbn_epoch_time.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# =====================================================================
# Console
# =====================================================================

print()
print("=" * 94)
print("PHASE 2C — BATCHNORM VS SYNCBATCHNORM")
print("=" * 94)

for row in summary:

    print()
    print(
        row[
            "normalization"
        ]
    )

    print(
        f"  Throughput : "
        f"{row['throughput_mean']:.2f} "
        f"± {row['throughput_std']:.2f} img/s"
    )

    print(
        f"  CV         : "
        f"{row['throughput_cv_percent']:.2f}%"
    )

    print(
        f"  Val acc    : "
        f"{100 * row['val_accuracy_mean']:.2f}% "
        f"± {100 * row['val_accuracy_std']:.2f} pp"
    )

    print(
        f"  Epoch time : "
        f"{row['epoch_time_mean']:.3f} "
        f"± {row['epoch_time_std']:.3f} s"
    )

    print(
        f"  VRAM/rank  : "
        f"{row['peak_vram_mean_gb']:.3f} GiB"
    )


print()
print("-" * 94)

print(
    f"SyncBN throughput change : "
    f"{throughput_change_percent:+.2f}%"
)

print(
    f"SyncBN epoch-time change : "
    f"{epoch_time_change_percent:+.2f}%"
)

print(
    f"SyncBN accuracy change   : "
    f"{accuracy_change_pp:+.2f} percentage points"
)

print(
    f"SyncBN VRAM change       : "
    f"{vram_change_percent:+.2f}%"
)

print("=" * 94)

print()
print("Summary :", summary_md)
print("CSV     :", summary_csv)
print("Figures :", FIGURES)
