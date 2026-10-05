"""
Phase 3 analysis:
naive DDP gradient accumulation vs DDP.no_sync().

Controlled variables:
    workload               U-Net / Heat2D
    GPUs                   4
    microbatch/rank        4
    accumulation steps     4
    effective global batch 64
    epochs                 5

Only synchronization strategy changes.
"""

import csv
import json
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
    / "optimization"
    / "raw"
)

ANALYSIS = (
    ROOT
    / "results"
    / "rosi_v100"
    / "optimization"
    / "analysis"
)

FIGURES = (
    ROOT
    / "results"
    / "rosi_v100"
    / "optimization"
    / "figures"
)

ANALYSIS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


MODES = [
    "naive",
    "no_sync",
]

REPEATS = [
    1,
    2,
    3,
]


# =====================================================================
# Load runs
# =====================================================================

run_rows = []


for mode in MODES:

    for rep in REPEATS:

        path = (
            RAW
            / f"unet_world4_{mode}_rep{rep}.json"
        )

        if not path.exists():

            raise RuntimeError(
                f"Missing result: {path}"
            )


        with path.open(
            "r",
            encoding="utf-8",
        ) as f:

            d = json.load(f)


        assert d["mode"] == mode
        assert d["world_size"] == 4
        assert d["microbatch_size_per_rank"] == 4
        assert d["accumulation_steps"] == 4
        assert d["effective_global_batch_size"] == 64


        history = d["history"]

        steady = history[1:]

        final = history[-1]


        run_rows.append(
            {
                "mode":
                    mode,

                "repetition":
                    rep,

                "throughput":
                    float(
                        d[
                            "steady_state_throughput_samples_per_sec"
                        ]
                    ),

                "throughput_within_run_std":
                    float(
                        d[
                            "steady_state_throughput_std"
                        ]
                    ),

                "train_wall_time":
                    float(
                        d[
                            "mean_steady_train_wall_time_sec"
                        ]
                    ),

                "final_val_mse":
                    float(
                        final[
                            "val_mse"
                        ]
                    ),

                "final_train_mse":
                    float(
                        final[
                            "train_loss"
                        ]
                    ),

                "peak_vram_gb":
                    max(
                        float(
                            epoch[
                                "peak_vram_per_rank_gb"
                            ]
                        )
                        for epoch in steady
                    ),

                "optimizer_steps_per_epoch":
                    int(
                        d[
                            "optimizer_steps_per_epoch"
                        ]
                    ),

                "hostname":
                    d[
                        "hostname"
                    ],
            }
        )


# =====================================================================
# Aggregate
# =====================================================================

summary = []


for mode in MODES:

    rows = [
        row
        for row in run_rows
        if row["mode"] == mode
    ]


    throughput = [
        row["throughput"]
        for row in rows
    ]

    wall = [
        row["train_wall_time"]
        for row in rows
    ]

    mse = [
        row["final_val_mse"]
        for row in rows
    ]

    train_mse = [
        row["final_train_mse"]
        for row in rows
    ]

    vram = [
        row["peak_vram_gb"]
        for row in rows
    ]


    q_mean = statistics.mean(
        throughput
    )

    q_std = statistics.stdev(
        throughput
    )


    summary.append(
        {
            "mode":
                mode,

            "n_runs":
                len(rows),

            "throughput_mean":
                q_mean,

            "throughput_std":
                q_std,

            "throughput_cv_percent":
                q_std / q_mean * 100.0,

            "train_wall_mean_sec":
                statistics.mean(
                    wall
                ),

            "train_wall_std_sec":
                statistics.stdev(
                    wall
                ),

            "final_val_mse_mean":
                statistics.mean(
                    mse
                ),

            "final_val_mse_std":
                statistics.stdev(
                    mse
                ),

            "final_train_mse_mean":
                statistics.mean(
                    train_mse
                ),

            "final_train_mse_std":
                statistics.stdev(
                    train_mse
                ),

            "peak_vram_mean_gb":
                statistics.mean(
                    vram
                ),

            "peak_vram_std_gb":
                statistics.stdev(
                    vram
                ),
        }
    )


naive = next(
    row
    for row in summary
    if row["mode"] == "naive"
)

no_sync = next(
    row
    for row in summary
    if row["mode"] == "no_sync"
)


throughput_gain_percent = (
    (
        no_sync[
            "throughput_mean"
        ]
        /
        naive[
            "throughput_mean"
        ]
    )
    - 1.0
) * 100.0


speedup = (
    no_sync[
        "throughput_mean"
    ]
    /
    naive[
        "throughput_mean"
    ]
)


wall_change_percent = (
    (
        no_sync[
            "train_wall_mean_sec"
        ]
        /
        naive[
            "train_wall_mean_sec"
        ]
    )
    - 1.0
) * 100.0


val_mse_change = (
    no_sync[
        "final_val_mse_mean"
    ]
    -
    naive[
        "final_val_mse_mean"
    ]
)


vram_change_percent = (
    (
        no_sync[
            "peak_vram_mean_gb"
        ]
        /
        naive[
            "peak_vram_mean_gb"
        ]
    )
    - 1.0
) * 100.0


# =====================================================================
# CSV files
# =====================================================================

per_run_csv = (
    ANALYSIS
    / "gradient_accumulation_per_run.csv"
)


with per_run_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            run_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(run_rows)


summary_csv = (
    ANALYSIS
    / "gradient_accumulation_summary.csv"
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
    writer.writerows(summary)


# =====================================================================
# Markdown
# =====================================================================

summary_md = (
    ANALYSIS
    / "gradient_accumulation_summary.md"
)


with summary_md.open(
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "# Phase 3 — Gradient Accumulation and DDP.no_sync()\n\n"
    )

    f.write(
        "U-Net / Heat2D, 4 V100 GPUs, microbatch 4/rank, "
        "4 accumulation steps, effective global batch 64, "
        "three runs per synchronization strategy.\n\n"
    )

    f.write(
        "| Mode | Throughput | CV | Train wall | "
        "Final val MSE | VRAM/rank |\n"
    )

    f.write(
        "|---|---:|---:|---:|---:|---:|\n"
    )


    for row in summary:

        label = (
            "Naive accumulation"
            if row["mode"] == "naive"
            else "DDP.no_sync()"
        )

        f.write(
            f"| {label} "
            f"| {row['throughput_mean']:.2f} "
            f"± {row['throughput_std']:.2f} fields/s "
            f"| {row['throughput_cv_percent']:.2f}% "
            f"| {row['train_wall_mean_sec']:.3f} "
            f"± {row['train_wall_std_sec']:.3f} s "
            f"| {row['final_val_mse_mean']:.6f} "
            f"| {row['peak_vram_mean_gb']:.3f} GiB |\n"
        )


    f.write("\n")

    f.write(
        "## no_sync change relative to naive accumulation\n\n"
    )

    f.write(
        f"- Throughput: {throughput_gain_percent:+.2f}%\n"
    )

    f.write(
        f"- Speedup: {speedup:.3f}×\n"
    )

    f.write(
        f"- Mean training wall time: {wall_change_percent:+.2f}%\n"
    )

    f.write(
        f"- Final validation MSE change: {val_mse_change:+.8f}\n"
    )

    f.write(
        f"- Peak VRAM/rank: {vram_change_percent:+.2f}%\n"
    )

    f.write(
        "\nBoth modes use the same effective global batch and optimizer "
        "update frequency. The intended experimental difference is "
        "gradient-synchronization frequency.\n"
    )


# =====================================================================
# Figure 1 — throughput
# =====================================================================

labels = [
    "Naive",
    "DDP.no_sync()",
]

means = [
    naive[
        "throughput_mean"
    ],
    no_sync[
        "throughput_mean"
    ],
]

errors = [
    naive[
        "throughput_std"
    ],
    no_sync[
        "throughput_std"
    ],
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
    "Throughput (fields/s)"
)

plt.title(
    "U-Net / 4 V100 — Gradient Accumulation"
)

plt.grid(
    True,
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    FIGURES
    / "01_naive_vs_no_sync_throughput.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# =====================================================================
# Figure 2 — training wall time
# =====================================================================

wall_means = [
    naive[
        "train_wall_mean_sec"
    ],
    no_sync[
        "train_wall_mean_sec"
    ],
]

wall_errors = [
    naive[
        "train_wall_std_sec"
    ],
    no_sync[
        "train_wall_std_sec"
    ],
]


plt.figure(
    figsize=(7, 5)
)

plt.bar(
    labels,
    wall_means,
    yerr=wall_errors,
    capsize=6,
)

plt.ylabel(
    "Mean steady training-loop time (s)"
)

plt.title(
    "Effect of Suppressing Intermediate DDP Synchronization"
)

plt.grid(
    True,
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    FIGURES
    / "02_naive_vs_no_sync_train_time.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# =====================================================================
# Figure 3 — final validation MSE
# =====================================================================

mse_means = [
    naive[
        "final_val_mse_mean"
    ],
    no_sync[
        "final_val_mse_mean"
    ],
]

mse_errors = [
    naive[
        "final_val_mse_std"
    ],
    no_sync[
        "final_val_mse_std"
    ],
]


plt.figure(
    figsize=(7, 5)
)

plt.bar(
    labels,
    mse_means,
    yerr=mse_errors,
    capsize=6,
)

plt.ylabel(
    "Final validation MSE"
)

plt.title(
    "Learning Outcome Under Equivalent Gradient Accumulation"
)

plt.grid(
    True,
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    FIGURES
    / "03_naive_vs_no_sync_val_mse.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# =====================================================================
# Console
# =====================================================================

print()
print("=" * 96)
print("PHASE 3 — GRADIENT ACCUMULATION ANALYSIS")
print("=" * 96)


for row in summary:

    name = (
        "Naive"
        if row["mode"] == "naive"
        else "DDP.no_sync()"
    )

    print()
    print(name)

    print(
        f"  Throughput : "
        f"{row['throughput_mean']:.2f} "
        f"± {row['throughput_std']:.2f} fields/s"
    )

    print(
        f"  CV         : "
        f"{row['throughput_cv_percent']:.2f}%"
    )

    print(
        f"  Train wall : "
        f"{row['train_wall_mean_sec']:.3f} "
        f"± {row['train_wall_std_sec']:.3f} s"
    )

    print(
        f"  Val MSE    : "
        f"{row['final_val_mse_mean']:.6f}"
    )

    print(
        f"  VRAM/rank  : "
        f"{row['peak_vram_mean_gb']:.3f} GiB"
    )


print()
print("-" * 96)

print(
    f"no_sync throughput gain : "
    f"{throughput_gain_percent:+.2f}%"
)

print(
    f"no_sync speedup         : "
    f"{speedup:.3f}x"
)

print(
    f"training wall change    : "
    f"{wall_change_percent:+.2f}%"
)

print(
    f"validation MSE change   : "
    f"{val_mse_change:+.8f}"
)

print(
    f"VRAM change             : "
    f"{vram_change_percent:+.2f}%"
)

print("=" * 96)

print()
print("Summary :", summary_md)
print("CSV     :", summary_csv)
print("Figures :", FIGURES)
