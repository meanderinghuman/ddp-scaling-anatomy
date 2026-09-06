"""
Generate the 2D heat-equation surrogate dataset.

Each sample contains:

    input  = initial temperature field
    target = temperature field after diffusion

The arrays are saved as .npy files so they can later be
memory-mapped efficiently during single-GPU and DDP training.
"""

import json
from pathlib import Path

import numpy as np

from ddp_lab.data.heat2d import (
    HeatConfig,
    grid_spacing,
    make_initial_field,
    simulate_heat,
    time_step,
    validate_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "heat2d"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SEED = 42


CONFIG = HeatConfig(
    grid_size=64,
    thermal_diffusivity=0.01,
    stability_ratio=0.20,
    simulation_steps=50,
)


SPLIT_SIZES = {
    "train": 4096,
    "val": 512,
    "test": 512,
}


def generate_split(
    split_name: str,
    num_samples: int,
    rng: np.random.Generator,
) -> None:
    """
    Generate one dataset split and save it to disk.
    """

    n = CONFIG.grid_size


    inputs = np.empty(
        (
            num_samples,
            1,
            n,
            n,
        ),
        dtype=np.float32,
    )

    targets = np.empty_like(
        inputs
    )


    for sample_index in range(
        num_samples
    ):

        initial = make_initial_field(
            rng=rng,
            config=CONFIG,
        )

        final = simulate_heat(
            initial_field=initial,
            config=CONFIG,
        )


        # Add channel dimension:
        #
        # [H, W] -> [1, H, W]
        #
        # U-Net later expects:
        #
        # [batch, channels, height, width]
        inputs[
            sample_index,
            0,
        ] = initial

        targets[
            sample_index,
            0,
        ] = final


        if (
            (sample_index + 1) % 512 == 0
            or
            sample_index + 1 == num_samples
        ):

            print(
                f"{split_name:<5} | "
                f"{sample_index + 1:>5}/"
                f"{num_samples:<5}"
            )


    np.save(
        OUTPUT_DIR
        / f"{split_name}_inputs.npy",
        inputs,
    )

    np.save(
        OUTPUT_DIR
        / f"{split_name}_targets.npy",
        targets,
    )


    print(
        f"{split_name:<5} saved."
    )


def main():

    validate_config(
        CONFIG
    )


    dx = grid_spacing(
        CONFIG.grid_size
    )

    dt = time_step(
        CONFIG
    )

    final_time = (
        dt
        * CONFIG.simulation_steps
    )


    print("=" * 78)
    print("2D Heat-Equation Dataset Generation")
    print("=" * 78)


    print()
    print("NUMERICAL CONFIGURATION")
    print("-" * 78)

    print(
        f"Grid size             : "
        f"{CONFIG.grid_size} x "
        f"{CONFIG.grid_size}"
    )

    print(
        f"Thermal diffusivity   : "
        f"{CONFIG.thermal_diffusivity}"
    )

    print(
        f"dx                     : "
        f"{dx:.8f}"
    )

    print(
        f"Stability ratio r      : "
        f"{CONFIG.stability_ratio:.4f}"
    )

    print(
        f"dt                     : "
        f"{dt:.8f}"
    )

    print(
        f"Simulation steps       : "
        f"{CONFIG.simulation_steps}"
    )

    print(
        f"Final simulated time   : "
        f"{final_time:.6f}"
    )


   
    rng = np.random.default_rng(
        SEED
    )


    print()
    print("GENERATING DATA")
    print("-" * 78)


    for split_name, num_samples in SPLIT_SIZES.items():

        generate_split(
            split_name=split_name,
            num_samples=num_samples,
            rng=rng,
        )


    metadata = {
        "seed": SEED,
        "grid_size": CONFIG.grid_size,
        "thermal_diffusivity": (
            CONFIG.thermal_diffusivity
        ),
        "stability_ratio": (
            CONFIG.stability_ratio
        ),
        "dx": dx,
        "dt": dt,
        "simulation_steps": (
            CONFIG.simulation_steps
        ),
        "final_simulated_time": (
            final_time
        ),
        "boundary_condition": (
            "zero_dirichlet"
        ),
        "split_sizes": SPLIT_SIZES,
        "dtype": "float32",
    }


    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )


    print()
    print(
        f"Metadata              : "
        f"{OUTPUT_DIR / 'metadata.json'}"
    )

    print()
    print("=" * 78)
    print("HEAT DATASET GENERATION COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()