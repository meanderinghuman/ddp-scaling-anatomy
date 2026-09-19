import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

SUMMARY = (
    ROOT
    / "results"
    / "rosi_v100"
    / "analysis"
    / "scaling_summary.csv"
)

OUT = (
    ROOT
    / "results"
    / "rosi_v100"
    / "figures"
)

with SUMMARY.open("r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))


def select(workload):
    return sorted(
        [r for r in rows if r["workload"] == workload],
        key=lambda r: int(r["world_size"]),
    )


def make_plot(workload, title, ylabel, filename):
    data = select(workload)

    x = [int(r["world_size"]) for r in data]
    y = [float(r["throughput_samples_per_sec"]) for r in data]
    yerr = [float(r["throughput_std"]) for r in data]

    plt.figure(figsize=(7, 5))

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        capsize=4,
    )

    plt.xlabel("Number of V100 GPUs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks([1, 2, 4])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUT / filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


make_plot(
    "resnet18",
    "ResNet18 Strong Scaling",
    "Throughput (images/s)",
    "10_resnet18_throughput.png",
)

make_plot(
    "unet_heat",
    "U-Net Heat2D Strong Scaling",
    "Throughput (fields/s)",
    "11_unet_heat_throughput.png",
)

print("README figures created.")
