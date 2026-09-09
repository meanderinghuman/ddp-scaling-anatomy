"""
Smoke test for the heat-equation U-Net surrogate.

We verify:

1. Model construction
2. Correct input/output field shape
3. Finite predictions
4. MSE loss
5. Backpropagation
6. Finite gradients
7. Parameter count
8. GPU memory usage
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ddp_lab.models.unet import HeatUNet


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

SEED = 42

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "heat2d"
)


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 78)
print("Heat-Equation U-Net Model Test")
print("=" * 78)

print(
    f"Device               : {device}"
)

if device.type == "cuda":

    print(
        f"GPU                  : "
        f"{torch.cuda.get_device_name(0)}"
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

model = HeatUNet(
    in_channels=1,
    out_channels=1,
    base_channels=64,
)

model = model.to(device)


total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print()
print("MODEL")
print("-" * 78)

print(
    f"Total parameters     : "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters : "
    f"{trainable_parameters:,}"
)


# ---------------------------------------------------------------------
# Load a small real batch from our audited dataset.
# ---------------------------------------------------------------------

train_inputs = np.load(
    DATA_DIR / "train_inputs.npy",
    mmap_mode="r",
)

train_targets = np.load(
    DATA_DIR / "train_targets.npy",
    mmap_mode="r",
)


batch_size = 8


# np.asarray(...).copy() creates a writable contiguous array.
#
# This avoids warnings from torch.from_numpy when the underlying
# memory-mapped array is read-only.
input_batch = np.asarray(
    train_inputs[:batch_size]
).copy()

target_batch = np.asarray(
    train_targets[:batch_size]
).copy()


inputs = torch.from_numpy(
    input_batch
).to(
    device=device,
    dtype=torch.float32,
)

targets = torch.from_numpy(
    target_batch
).to(
    device=device,
    dtype=torch.float32,
)


print()
print("INPUT")
print("-" * 78)

print(
    f"Input shape           : "
    f"{tuple(inputs.shape)}"
)

print(
    f"Target shape          : "
    f"{tuple(targets.shape)}"
)


# ---------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------

predictions = model(inputs)


print()
print("FORWARD PASS")
print("-" * 78)

print(
    f"Prediction shape      : "
    f"{tuple(predictions.shape)}"
)


assert (
    predictions.shape
    == targets.shape
)


print(
    "Output shape         : PASSED"
)


assert torch.isfinite(
    predictions
).all()


print(
    "Finite predictions   : PASSED"
)


# ---------------------------------------------------------------------
# Regression loss
#
# Mean-squared error:
#
#     MSE = (1/N) * sum_i (prediction_i - target_i)^2
#
# Unlike classification, our target is a continuous physical field.
# ---------------------------------------------------------------------

criterion = nn.MSELoss()


loss = criterion(
    predictions,
    targets,
)


print()
print("LOSS")
print("-" * 78)

print(
    f"MSE loss             : "
    f"{loss.item():.8f}"
)


assert torch.isfinite(
    loss
)


print(
    "Finite loss          : PASSED"
)


# ---------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------

loss.backward()


parameters_with_grad = 0
finite_gradient_tensors = 0


for parameter in model.parameters():

    if parameter.grad is not None:

        parameters_with_grad += 1

        if torch.isfinite(
            parameter.grad
        ).all():

            finite_gradient_tensors += 1


print()
print("BACKWARD PASS")
print("-" * 78)

print(
    f"Tensors with grad    : "
    f"{parameters_with_grad}"
)

print(
    f"Finite grad tensors  : "
    f"{finite_gradient_tensors}"
)


assert parameters_with_grad > 0

assert (
    finite_gradient_tensors
    == parameters_with_grad
)


print(
    "Gradient check       : PASSED"
)


# ---------------------------------------------------------------------
# GPU memory
# ---------------------------------------------------------------------

if device.type == "cuda":

    torch.cuda.synchronize()


    allocated_gb = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved_gb = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    peak_gb = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )


    print()
    print("GPU MEMORY")
    print("-" * 78)

    print(
        f"Allocated            : "
        f"{allocated_gb:.3f} GiB"
    )

    print(
        f"Reserved             : "
        f"{reserved_gb:.3f} GiB"
    )

    print(
        f"Peak allocated       : "
        f"{peak_gb:.3f} GiB"
    )


print()
print("=" * 78)
print("U-NET MODEL TEST PASSED")
print("=" * 78)
