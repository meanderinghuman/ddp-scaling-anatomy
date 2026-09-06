"""
Scientific audit for the generated 2D heat-equation dataset.

Before training a neural network, we verify that the numerical
simulation produced physically and numerically sensible data.

Checks
------
1. Correct array shapes
2. Correct dtype
3. Finite values
4. Non-negative temperatures
5. Zero Dirichlet boundaries
6. Maximum temperature does not increase
7. Total heat does not increase
8. No exact samples are shared between train/val/test
9. Visual inspection of diffusion examples

The goal is to validate the simulator output BEFORE treating it
as ground truth for the U-Net surrogate.
"""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "heat2d"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_PATH = (
    FIGURE_DIR
    / "heat2d_examples.png"
)


# ---------------------------------------------------------------------
# Expected dataset structure
# ---------------------------------------------------------------------

EXPECTED_SPLITS = {
    "train": 4096,
    "val": 512,
    "test": 512,
}

GRID_SIZE = 64

EXPECTED_SHAPE_SUFFIX = (
    1,
    GRID_SIZE,
    GRID_SIZE,
)

BOUNDARY_TOLERANCE = 1e-7

PHYSICS_TOLERANCE = 1e-6


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_split(split_name: str):
    """
    Load one split using memory mapping.

    mmap_mode="r" means NumPy does not need to copy the entire
    array into RAM immediately. Data are read from disk as needed.

    This is similar to how we may later handle larger scientific
    datasets during DDP training.
    """

    inputs = np.load(
        DATA_DIR / f"{split_name}_inputs.npy",
        mmap_mode="r",
    )

    targets = np.load(
        DATA_DIR / f"{split_name}_targets.npy",
        mmap_mode="r",
    )

    return inputs, targets


# ---------------------------------------------------------------------
# Boundary measurement
# ---------------------------------------------------------------------

def maximum_boundary_value(
    array: np.ndarray,
) -> float:
    """
    Return the largest absolute value on all four boundaries.

    Array shape:

        [N, C, H, W]
    """

    top = np.max(
        np.abs(array[:, :, 0, :])
    )

    bottom = np.max(
        np.abs(array[:, :, -1, :])
    )

    left = np.max(
        np.abs(array[:, :, :, 0])
    )

    right = np.max(
        np.abs(array[:, :, :, -1])
    )

    return float(
        max(
            top,
            bottom,
            left,
            right,
        )
    )


# ---------------------------------------------------------------------
# Smoothness metric
# ---------------------------------------------------------------------

def gradient_energy(
    array: np.ndarray,
) -> float:
    """
    Measure average squared spatial gradients.

    A temperature field with rapid spatial changes has higher
    gradient energy.

    Heat diffusion should generally smooth the field, so we expect
    this quantity to decrease after simulation.

    For T(x,y), a continuous analogue is:

        integral |grad T|^2 dx dy

    Here we use a discrete approximation.
    """

    horizontal_difference = (
        array[:, :, :, 1:]
        - array[:, :, :, :-1]
    )

    vertical_difference = (
        array[:, :, 1:, :]
        - array[:, :, :-1, :]
    )

    horizontal_energy = np.mean(
        horizontal_difference**2
    )

    vertical_energy = np.mean(
        vertical_difference**2
    )

    return float(
        horizontal_energy
        + vertical_energy
    )


# ---------------------------------------------------------------------
# Exact-sample fingerprint
# ---------------------------------------------------------------------

def sample_hash(
    sample: np.ndarray,
) -> str:
    """
    Produce a deterministic hash for one input field.

    This allows us to check whether the exact same initial field
    accidentally appears in multiple dataset splits.
    """

    contiguous = np.ascontiguousarray(
        sample
    )

    return hashlib.sha256(
        contiguous.tobytes()
    ).hexdigest()


def split_hashes(
    inputs: np.ndarray,
) -> set[str]:
    """
    Hash every initial condition in one split.
    """

    return {
        sample_hash(inputs[index])
        for index in range(len(inputs))
    }


# ---------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------

print("=" * 78)
print("2D Heat-Equation Dataset Audit")
print("=" * 78)


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------

with open(
    DATA_DIR / "metadata.json",
    "r",
    encoding="utf-8",
) as file:

    metadata = json.load(file)


print()
print("SIMULATION METADATA")
print("-" * 78)

print(
    f"Grid                   : "
    f"{metadata['grid_size']} x "
    f"{metadata['grid_size']}"
)

print(
    f"Thermal diffusivity    : "
    f"{metadata['thermal_diffusivity']}"
)

print(
    f"Stability ratio r      : "
    f"{metadata['stability_ratio']}"
)

print(
    f"Time step dt           : "
    f"{metadata['dt']:.8f}"
)

print(
    f"Simulation steps       : "
    f"{metadata['simulation_steps']}"
)

print(
    f"Final simulated time   : "
    f"{metadata['final_simulated_time']:.6f}"
)


# ---------------------------------------------------------------------
# Stability check
# ---------------------------------------------------------------------

r = metadata[
    "stability_ratio"
]

assert 0.0 < r <= 0.25

print()
print("NUMERICAL STABILITY")
print("-" * 78)

print(
    f"r                      : {r:.4f}"
)

print(
    f"Required               : r <= 0.25"
)

print(
    "Explicit scheme       : PASSED"
)


# ---------------------------------------------------------------------
# Audit each split
# ---------------------------------------------------------------------

loaded_splits = {}


for split_name, expected_count in EXPECTED_SPLITS.items():

    inputs, targets = load_split(
        split_name
    )

    loaded_splits[
        split_name
    ] = (
        inputs,
        targets,
    )


    print()
    print(
        f"{split_name.upper()} SPLIT"
    )
    print("-" * 78)


    # ---------------------------------------------------------
    # Shape checks
    # ---------------------------------------------------------

    expected_shape = (
        expected_count,
        *EXPECTED_SHAPE_SUFFIX,
    )


    print(
        f"Input shape            : "
        f"{inputs.shape}"
    )

    print(
        f"Target shape           : "
        f"{targets.shape}"
    )


    assert (
        inputs.shape
        == expected_shape
    )

    assert (
        targets.shape
        == expected_shape
    )


    print(
        "Shape check            : PASSED"
    )


    # ---------------------------------------------------------
    # Dtype
    # ---------------------------------------------------------

    print(
        f"Input dtype            : "
        f"{inputs.dtype}"
    )

    print(
        f"Target dtype           : "
        f"{targets.dtype}"
    )


    assert inputs.dtype == np.float32
    assert targets.dtype == np.float32


    print(
        "Dtype check            : PASSED"
    )


    # ---------------------------------------------------------
    # Finite values
    # ---------------------------------------------------------

    input_finite = bool(
        np.isfinite(inputs).all()
    )

    target_finite = bool(
        np.isfinite(targets).all()
    )


    assert input_finite
    assert target_finite


    print(
        "Finite values          : PASSED"
    )


    # ---------------------------------------------------------
    # Non-negativity
    # ---------------------------------------------------------

    input_min = float(
        np.min(inputs)
    )

    target_min = float(
        np.min(targets)
    )


    print(
        f"Input minimum          : "
        f"{input_min:.8f}"
    )

    print(
        f"Target minimum         : "
        f"{target_min:.8f}"
    )


    assert (
        input_min
        >= -PHYSICS_TOLERANCE
    )

    assert (
        target_min
        >= -PHYSICS_TOLERANCE
    )


    print(
        "Non-negativity         : PASSED"
    )


    # ---------------------------------------------------------
    # Boundary condition
    # ---------------------------------------------------------

    input_boundary = (
        maximum_boundary_value(
            inputs
        )
    )

    target_boundary = (
        maximum_boundary_value(
            targets
        )
    )


    print(
        f"Input boundary max     : "
        f"{input_boundary:.3e}"
    )

    print(
        f"Target boundary max    : "
        f"{target_boundary:.3e}"
    )


    assert (
        input_boundary
        <= BOUNDARY_TOLERANCE
    )

    assert (
        target_boundary
        <= BOUNDARY_TOLERANCE
    )


    print(
        "Dirichlet boundary     : PASSED"
    )



    initial_max = np.max(
        inputs,
        axis=(1, 2, 3),
    )

    final_max = np.max(
        targets,
        axis=(1, 2, 3),
    )


    max_increase = float(
        np.max(
            final_max
            - initial_max
        )
    )


    print(
        f"Input max mean         : "
        f"{np.mean(initial_max):.6f}"
    )

    print(
        f"Final max mean         : "
        f"{np.mean(final_max):.6f}"
    )

    print(
        f"Largest max increase   : "
        f"{max_increase:.3e}"
    )


    assert (
        max_increase
        <= PHYSICS_TOLERANCE
    )


    print(
        "Maximum principle      : PASSED"
    )


   
    initial_heat = np.sum(
        inputs,
        axis=(1, 2, 3),
        dtype=np.float64,
    )

    final_heat = np.sum(
        targets,
        axis=(1, 2, 3),
        dtype=np.float64,
    )


    heat_increase = float(
        np.max(
            final_heat
            - initial_heat
        )
    )


    mean_heat_ratio = float(
        np.mean(
            final_heat
            / initial_heat
        )
    )


    print(
        f"Mean final/input heat  : "
        f"{mean_heat_ratio:.6f}"
    )

    print(
        f"Largest heat increase  : "
        f"{heat_increase:.3e}"
    )


    assert (
        heat_increase
        <= PHYSICS_TOLERANCE
    )


    print(
        "Total heat check       : PASSED"
    )


    

    initial_gradient_energy = (
        gradient_energy(
            inputs
        )
    )

    final_gradient_energy = (
        gradient_energy(
            targets
        )
    )


    print(
        f"Initial grad energy    : "
        f"{initial_gradient_energy:.8f}"
    )

    print(
        f"Final grad energy      : "
        f"{final_gradient_energy:.8f}"
    )

    print(
        f"Smoothness ratio       : "
        f"{final_gradient_energy / initial_gradient_energy:.6f}"
    )




print()
print("CROSS-SPLIT EXACT DUPLICATE AUDIT")
print("-" * 78)


train_hashes = split_hashes(
    loaded_splits["train"][0]
)

val_hashes = split_hashes(
    loaded_splits["val"][0]
)

test_hashes = split_hashes(
    loaded_splits["test"][0]
)


train_val_overlap = len(
    train_hashes
    & val_hashes
)

train_test_overlap = len(
    train_hashes
    & test_hashes
)

val_test_overlap = len(
    val_hashes
    & test_hashes
)


print(
    f"Train / val overlap    : "
    f"{train_val_overlap}"
)

print(
    f"Train / test overlap   : "
    f"{train_test_overlap}"
)

print(
    f"Val / test overlap     : "
    f"{val_test_overlap}"
)


assert train_val_overlap == 0
assert train_test_overlap == 0
assert val_test_overlap == 0


print(
    "Exact leakage check    : PASSED"
)



print()
print("VISUALIZATION")
print("-" * 78)


train_inputs, train_targets = (
    loaded_splits["train"]
)


# Fixed examples keep the figure reproducible.
sample_indices = [
    0,
    100,
    1000,
]


fig, axes = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(10, 9),
)


for row, sample_index in enumerate(
    sample_indices
):

    initial = np.asarray(
        train_inputs[
            sample_index,
            0,
        ]
    )

    final = np.asarray(
        train_targets[
            sample_index,
            0,
        ]
    )

    change = np.abs(
        initial
        - final
    )


    # Use the same temperature range for initial and final
    # so visual comparison is meaningful.
    temperature_max = float(
        max(
            initial.max(),
            final.max(),
        )
    )


    image_initial = axes[
        row,
        0,
    ].imshow(
        initial,
        origin="lower",
        vmin=0.0,
        vmax=temperature_max,
    )


    axes[
        row,
        0,
    ].set_title(
        f"T0 — sample {sample_index}"
    )


    axes[
        row,
        1,
    ].imshow(
        final,
        origin="lower",
        vmin=0.0,
        vmax=temperature_max,
    )


    axes[
        row,
        1,
    ].set_title(
        "Tf after diffusion"
    )


    axes[
        row,
        2,
    ].imshow(
        change,
        origin="lower",
    )


    axes[
        row,
        2,
    ].set_title(
        "|T0 - Tf|"
    )


    for column in range(3):

        axes[
            row,
            column,
        ].set_xticks([])

        axes[
            row,
            column,
        ].set_yticks([])


fig.suptitle(
    "2D Heat-Equation Dataset Examples"
)

fig.tight_layout()

fig.savefig(
    FIGURE_PATH,
    dpi=160,
)

plt.close(fig)


print(
    f"Saved figure           : "
    f"{FIGURE_PATH}"
)


print()
print("=" * 78)
print("HEAT DATASET AUDIT PASSED")
print("=" * 78)