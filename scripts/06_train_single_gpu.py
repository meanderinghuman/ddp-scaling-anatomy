"""
Single-GPU reference trainer.

This script establishes the N=1 baseline for the future distributed
training experiments.

Supported workloads
-------------------
1. ResNet18 / CIFAR-100 classification
2. U-Net / 2D heat-equation regression

We record:
    - training loss
    - validation metric
    - measured training time
    - steady-state throughput
    - peak GPU memory

These measurements later become the 1-GPU reference for DDP scaling:

    speedup_N = throughput_N / throughput_1

or, for the same amount of work,

    speedup_N = time_1 / time_N

and

    efficiency_N = speedup_N / N
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ddp_lab.training.common import (
    build_optimizer,
    build_workload,
    load_config,
    set_seed,
)


# =====================================================================
# Paths
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =====================================================================
# Command-line interface
# =====================================================================

parser = argparse.ArgumentParser(
    description="Single-GPU baseline trainer."
)

parser.add_argument(
    "--config",
    type=Path,
    required=True,
    help="Path to workload YAML configuration.",
)

# These two arguments are useful for smoke tests.
#
# Example:
#
#     --max-train-batches 20
#
# means:
#
#     run only the first 20 training batches of each epoch.
parser.add_argument(
    "--max-train-batches",
    type=int,
    default=None,
)

parser.add_argument(
    "--max-val-batches",
    type=int,
    default=None,
)

args = parser.parse_args()


# =====================================================================
# Configuration
# =====================================================================

config = load_config(
    args.config
)

workload = str(
    config["workload"]
)

seed = int(
    config["seed"]
)

global_batch_size = int(
    config["global_batch_size"]
)

num_workers = int(
    config["num_workers"]
)

num_epochs = int(
    config["epochs"]
)

warmup_batches = int(
    config["timing_warmup_batches"]
)

use_amp = bool(
    config["amp"]
)


# =====================================================================
# Hardware
# =====================================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "This reference benchmark requires a CUDA GPU."
    )


device = torch.device(
    "cuda:0"
)


set_seed(
    seed
)


# ---------------------------------------------------------------------
# cuDNN benchmark mode
#
# Our tensor shapes are mostly fixed.
#
# cuDNN can benchmark several convolution implementations and select
# a fast kernel for the current shapes.
#
# This favors performance benchmarking over strict bit-for-bit
# determinism.
# ---------------------------------------------------------------------

torch.backends.cudnn.benchmark = True


torch.cuda.set_device(
    device
)

torch.cuda.empty_cache()

torch.cuda.reset_peak_memory_stats()


# =====================================================================
# Header
# =====================================================================

print("=" * 88)
print("Single-GPU Training Reference")
print("=" * 88)

print(
    f"Workload               : "
    f"{workload}"
)

print(
    f"GPU                    : "
    f"{torch.cuda.get_device_name(0)}"
)

print(
    f"Global batch size      : "
    f"{global_batch_size}"
)

# On one GPU:
#
#     global batch size = local batch size
#
# Later in DDP:
#
#     global batch
#       =
#     local batch × world size
print(
    f"Local batch size       : "
    f"{global_batch_size}"
)

print(
    f"Epochs                 : "
    f"{num_epochs}"
)

print(
    f"Mixed precision        : "
    f"{use_amp}"
)


# =====================================================================
# Build workload
# =====================================================================

(
    train_dataset,
    val_dataset,
    test_dataset,
    model,
    criterion,
) = build_workload(
    workload=workload,
    project_root=PROJECT_ROOT,
    config=config,
)


model = model.to(
    device
)


optimizer = build_optimizer(
    model=model,
    config=config,
)


print()
print("DATASET")
print("-" * 88)

print(
    f"Training samples       : "
    f"{len(train_dataset):,}"
)

print(
    f"Validation samples     : "
    f"{len(val_dataset):,}"
)

print(
    f"Test samples           : "
    f"{len(test_dataset):,}"
)


print()
print("MODEL")
print("-" * 88)


parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)


trainable_parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print(
    f"Parameters             : "
    f"{parameter_count:,}"
)

print(
    f"Trainable parameters   : "
    f"{trainable_parameter_count:,}"
)


# =====================================================================
# DataLoaders
# =====================================================================

# pin_memory=True:
#
# CPU batches are placed in page-locked memory.
#
# Combined with:
#
#     tensor.to(device, non_blocking=True)
#
# this allows more efficient CPU -> GPU transfers.

train_loader = DataLoader(
    train_dataset,
    batch_size=global_batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(
        num_workers > 0
    ),
)


val_loader = DataLoader(
    val_dataset,
    batch_size=global_batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(
        num_workers > 0
    ),
)


# =====================================================================
# Automatic mixed precision
# =====================================================================

def autocast_context():
    """
    Return the BF16 autocast context.

    BF16:
        1 sign bit
        8 exponent bits
        7 fraction bits

    BF16 has much less mantissa precision than FP32, but retains
    the same exponent width.

    That gives BF16 a much larger numerical range than FP16.

    On supported NVIDIA GPUs, convolution and matrix operations can
    often run faster and use less memory under BF16 autocast.
    """

    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=use_amp,
    )


# =====================================================================
# Validation
# =====================================================================

@torch.no_grad()
def validate(
    max_batches=None,
):
    """
    Evaluate the current model.

    ResNet18:
        return validation loss + classification accuracy.

    U-Net:
        return globally aggregated pixel-wise MSE.
    """

    model.eval()


    # -----------------------------------------------------------------
    # ResNet18 / CIFAR-100
    # -----------------------------------------------------------------

    if workload == "resnet18":

        total_correct = 0

        total_examples = 0

        total_loss = 0.0


        for batch_index, (
            inputs,
            targets,
        ) in enumerate(val_loader):

            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break


            inputs = inputs.to(
                device,
                non_blocking=True,
            )

            targets = targets.to(
                device,
                non_blocking=True,
            )


            with autocast_context():

                logits = model(
                    inputs
                )

                loss = criterion(
                    logits,
                    targets,
                )


            batch_size = int(
                inputs.size(0)
            )


            # criterion returns the mean loss for this batch.
            #
            # Multiplying by batch size converts it back to a weighted
            # sum so different final batch sizes are handled correctly.
            total_loss += (
                float(loss.item())
                * batch_size
            )


            predictions = torch.argmax(
                logits,
                dim=1,
            )


            total_correct += int(
                (
                    predictions
                    == targets
                )
                .sum()
                .item()
            )


            total_examples += (
                batch_size
            )


        if total_examples == 0:
            raise RuntimeError(
                "No ResNet validation examples were processed."
            )


        mean_loss = (
            total_loss
            / total_examples
        )


        accuracy = (
            total_correct
            / total_examples
        )


        model.train()


        return {
            "val_loss": float(
                mean_loss
            ),

            "val_accuracy": float(
                accuracy
            ),
        }


    # -----------------------------------------------------------------
    # U-Net / heat equation
    # -----------------------------------------------------------------

    if workload == "unet_heat":

        total_squared_error = 0.0

        total_elements = 0


        for batch_index, (
            inputs,
            targets,
        ) in enumerate(val_loader):

            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break


            inputs = inputs.to(
                device,
                non_blocking=True,
            )

            targets = targets.to(
                device,
                non_blocking=True,
            )


            with autocast_context():

                predictions = model(
                    inputs
                )


            # ---------------------------------------------------------
            # Compute validation MSE manually.
            #
            # We use FP32 for the subtraction and square so the final
            # reported metric is numerically stable even though the
            # model forward pass may use BF16.
            #
            # MSE:
            #
            #        1
            #     ------- Σ (prediction - target)^2
            #        N
            # ---------------------------------------------------------

            squared_error = (
                (
                    predictions.float()
                    - targets
                )
                ** 2
            )


            total_squared_error += float(
                squared_error
                .sum()
                .item()
            )


            total_elements += int(
                targets.numel()
            )


        if total_elements == 0:
            raise RuntimeError(
                "No U-Net validation elements were processed."
            )


        mse = (
            total_squared_error
            / total_elements
        )


        model.train()


        return {
            "val_mse": float(
                mse
            ),
        }


    raise RuntimeError(
        f"Unsupported workload during validation: "
        f"{workload}"
    )


# =====================================================================
# Training
# =====================================================================

history = []


print()
print("TRAINING")
print("-" * 88)


for epoch in range(
    1,
    num_epochs + 1,
):

    model.train()


    # Reset peak-memory counter so each epoch gets its own measurement.
    torch.cuda.reset_peak_memory_stats()


    epoch_loss_sum = 0.0

    epoch_examples = 0


    # -----------------------------------------------------------------
    # Throughput timing
    #
    # We ignore the first few batches because CUDA/cuDNN startup costs
    # are not representative of steady-state training.
    # -----------------------------------------------------------------

    measured_examples = 0

    measured_time = 0.0


    # Wall-clock epoch time includes the full training loop:
    #
    #     DataLoader
    #     CPU -> GPU copies
    #     forward
    #     backward
    #     optimizer
    #
    # This is useful in addition to our GPU-compute-focused timing.
    epoch_wall_start = (
        time.perf_counter()
    )


    for batch_index, (
        inputs,
        targets,
    ) in enumerate(train_loader):

        if (
            args.max_train_batches is not None
            and
            batch_index >= args.max_train_batches
        ):
            break


        # -------------------------------------------------------------
        # CPU -> GPU transfer
        # -------------------------------------------------------------

        inputs = inputs.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )


        # -------------------------------------------------------------
        # Start steady-state timing AFTER transfer.
        #
        # Our primary metric focuses on:
        #
        #     forward
        #     backward
        #     optimizer
        #
        # Later DDP adds gradient communication to this section.
        #
        # torch.cuda.synchronize() is required because CUDA operations
        # are asynchronous relative to Python.
        # -------------------------------------------------------------

        should_measure = (
            batch_index >= warmup_batches
        )


        if should_measure:

            torch.cuda.synchronize()

            batch_start = (
                time.perf_counter()
            )


        # -------------------------------------------------------------
        # Clear old gradients
        # -------------------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )


        # -------------------------------------------------------------
        # Forward + loss
        # -------------------------------------------------------------

        with autocast_context():

            outputs = model(
                inputs
            )

            loss = criterion(
                outputs,
                targets,
            )


        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Non-finite training loss."
            )


        # -------------------------------------------------------------
        # Backward
        #
        # Computes:
        #
        #     ∂L / ∂θ
        #
        # for every trainable parameter θ.
        #
        # Later DDP will synchronize these gradients between GPUs.
        # -------------------------------------------------------------

        loss.backward()


        # -------------------------------------------------------------
        # Optimizer update
        #
        # Example SGD update:
        #
        #     θ_(t+1)
        #       =
        #     θ_t - η ∇L
        #
        # In DDP, gradient synchronization happens before this step.
        # -------------------------------------------------------------

        optimizer.step()


        batch_size = int(
            inputs.size(0)
        )


        # -------------------------------------------------------------
        # Epoch loss accumulation
        #
        # criterion returns the mean loss of the current batch.
        #
        # Multiply by batch size so we can later compute the correctly
        # weighted epoch mean.
        # -------------------------------------------------------------

        epoch_loss_sum += (
            float(loss.item())
            * batch_size
        )


        epoch_examples += (
            batch_size
        )


        # -------------------------------------------------------------
        # Finish steady-state measurement
        # -------------------------------------------------------------

        if should_measure:

            torch.cuda.synchronize()


            batch_elapsed = (
                time.perf_counter()
                - batch_start
            )


            measured_time += (
                batch_elapsed
            )


            measured_examples += (
                batch_size
            )


    # -----------------------------------------------------------------
    # End of training epoch
    # -----------------------------------------------------------------

    torch.cuda.synchronize()


    epoch_wall_time = (
        time.perf_counter()
        - epoch_wall_start
    )


    if epoch_examples == 0:

        raise RuntimeError(
            "No training batches were processed."
        )


    mean_train_loss = (
        epoch_loss_sum
        / epoch_examples
    )


    # If max_train_batches <= timing_warmup_batches, there may be no
    # timed batches. Return NaN rather than dividing by zero.
    if measured_time > 0.0:

        throughput = (
            measured_examples
            / measured_time
        )

    else:

        throughput = float(
            "nan"
        )


    # -----------------------------------------------------------------
    # Validation happens ONCE per epoch.
    #
    # This must be outside the training-batch loop.
    # -----------------------------------------------------------------

    validation = validate(
        max_batches=args.max_val_batches
    )


    peak_vram_gb = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )


    # -----------------------------------------------------------------
    # Machine-readable epoch result
    # -----------------------------------------------------------------

    result = {
        "epoch": int(
            epoch
        ),

        "train_loss": float(
            mean_train_loss
        ),

        "train_examples": int(
            epoch_examples
        ),

        "measured_examples": int(
            measured_examples
        ),

        "measured_train_time_sec": float(
            measured_time
        ),

        "epoch_wall_time_sec": float(
            epoch_wall_time
        ),

        "throughput_samples_per_sec": float(
            throughput
        ),

        "peak_vram_gb": float(
            peak_vram_gb
        ),

        **validation,
    }


    history.append(
        result
    )


    # -----------------------------------------------------------------
    # Human-readable logging
    # -----------------------------------------------------------------

    if workload == "resnet18":

        print(
            f"epoch {epoch:2d} | "
            f"train loss "
            f"{mean_train_loss:.4f} | "
            f"val loss "
            f"{validation['val_loss']:.4f} | "
            f"val acc "
            f"{100.0 * validation['val_accuracy']:.2f}% | "
            f"throughput "
            f"{throughput:.1f} img/s | "
            f"measured "
            f"{measured_time:.2f}s | "
            f"epoch wall "
            f"{epoch_wall_time:.2f}s | "
            f"peak VRAM "
            f"{peak_vram_gb:.2f} GiB"
        )


    elif workload == "unet_heat":

        print(
            f"epoch {epoch:2d} | "
            f"train MSE "
            f"{mean_train_loss:.6f} | "
            f"val MSE "
            f"{validation['val_mse']:.6f} | "
            f"throughput "
            f"{throughput:.1f} fields/s | "
            f"measured "
            f"{measured_time:.2f}s | "
            f"epoch wall "
            f"{epoch_wall_time:.2f}s | "
            f"peak VRAM "
            f"{peak_vram_gb:.2f} GiB"
        )


    else:

        raise RuntimeError(
            f"Unexpected workload: "
            f"{workload}"
        )


# =====================================================================
# Save checkpoint
# =====================================================================

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)


CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


checkpoint_path = (
    CHECKPOINT_DIR
    / f"{workload}_single_gpu.pt"
)


torch.save(
    {
        "workload":
            workload,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "config":
            config,

        "history":
            history,
    },
    checkpoint_path,
)


# =====================================================================
# Save benchmark JSON
# =====================================================================

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "local"
)


RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Fixed typo:
#
#     simgle_gpu
#
# becomes:
#
#     single_gpu
result_path = (
    RESULT_DIR
    / f"{workload}_single_gpu.json"
)


# ---------------------------------------------------------------------
# Summary statistics
#
# Later these make it easier to compare:
#
#     1 GPU
#     2 GPUs
#     4 GPUs
#
# without manually inspecting every epoch.
# ---------------------------------------------------------------------

finite_throughputs = [
    epoch_result[
        "throughput_samples_per_sec"
    ]
    for epoch_result in history
    if math.isfinite(
        epoch_result[
            "throughput_samples_per_sec"
        ]
    )
]


if finite_throughputs:

    mean_throughput = (
        sum(finite_throughputs)
        / len(finite_throughputs)
    )

else:

    mean_throughput = float(
        "nan"
    )


peak_training_vram = max(
    epoch_result[
        "peak_vram_gb"
    ]
    for epoch_result in history
)


summary = {
    "execution":
        "single_gpu",

    "world_size":
        1,

    "gpu":
        torch.cuda.get_device_name(0),

    "workload":
        workload,

    "global_batch_size":
        global_batch_size,

    "local_batch_size":
        global_batch_size,

    "epochs":
        num_epochs,

    "amp":
        use_amp,

    "parameter_count":
        parameter_count,

    "trainable_parameter_count":
        trainable_parameter_count,

    "timing_warmup_batches":
        warmup_batches,

    "max_train_batches":
        args.max_train_batches,

    "max_val_batches":
        args.max_val_batches,

    "mean_throughput_samples_per_sec":
        float(mean_throughput),

    "peak_training_vram_gb":
        float(peak_training_vram),

    "history":
        history,
}


with open(
    result_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        summary,
        file,
        indent=2,
    )


# =====================================================================
# Final output
# =====================================================================

print()
print("=" * 88)
print("SINGLE-GPU TRAINING COMPLETE")
print("=" * 88)

print(
    f"Checkpoint             : "
    f"{checkpoint_path}"
)

print(
    f"Result JSON            : "
    f"{result_path}"
)

print(
    f"Mean throughput        : "
    f"{mean_throughput:.2f} samples/s"
)

print(
    f"Peak training VRAM     : "
    f"{peak_training_vram:.3f} GiB"
)

print("=" * 88)