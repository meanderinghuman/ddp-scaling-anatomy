"""
2D heat-equation simulator and dataset utilities.

We solve

    dT/dt = alpha * Laplacian(T)

on a square domain using an explicit finite-difference method.

The generated pairs

    initial temperature field -> final temperature field

will later be used to train a U-Net surrogate model.
"""

from dataclasses import dataclass 

import numpy as np 

@dataclass(frozen=True)
class HeatConfig:
    """
    Numerical config for 2d heat Eq.
    """
    grid_size: int = 64 
    thermal_diffusivity: float = 0.01 
    stability_ratio: float = 0.20 
    simulation_steps: int = 50 

def grid_spacing(grid_size: int,) -> float:
    """
    domain [0,1] , dx = 1 /(N-1)
    """
    return 1.0 / (grid_size -1)

def time_step(config: HeatConfig,) -> float:
    dx = grid_spacing(config.grid_size)
    return (
        config.stability_ratio * dx ** 2 / config.thermal_diffusivity
    )

def validate_config(config: HeatConfig,) -> None:
    if config.grid_size < 8:
        raise ValueError(
            "grid_size must be >= 8."
        )

    if config.thermal_diffusivity <= 0:
        raise ValueError(
            "thermal_diffusivity must be positive."
        )

    if config.simulation_steps <= 0:
        raise ValueError(
            "simulation_steps must be positive."
        )

    # Explicit 2D heat equation stability condition.
    if not (
        0.0
        < config.stability_ratio
        <= 0.25
    ):
        raise ValueError(
            "For the 2D explicit heat-equation scheme, "
            "stability_ratio must satisfy "
            "0 < r <= 0.25."
        )

def make_initial_field(rng: np.random.Generator, config: HeatConfig,) -> np.ndarray:
    n = config.grid_size

    coordinates = np.linspace(
        0.0,
        1.0,
        n,
        dtype=np.float32,
    )

    x, y = np.meshgrid(
        coordinates,
        coordinates,
        indexing="xy",
    )

    field = np.zeros(
        (n, n),
        dtype=np.float32,
    )

    # Use between one and four independent heat sources.
    num_sources = int(
        rng.integers(
            low=1,
            high=5,
        )
    )

    for _ in range(num_sources):

        # Keep sources mostly away from the fixed boundary.
        center_x = rng.uniform(
            0.15,
            0.85,
        )

        center_y = rng.uniform(
            0.15,
            0.85,
        )

        amplitude = rng.uniform(
            0.5,
            1.5,
        )

        sigma = rng.uniform(
            0.035,
            0.12,
        )

        squared_distance = (
            (x - center_x) ** 2
            +
            (y - center_y) ** 2
        )

        gaussian = (
            amplitude
            * np.exp(
                -squared_distance
                / (2.0 * sigma**2)
            )
        )

        field += gaussian.astype(
            np.float32
        )

    # Dirichlet boundary condition:
   
    field[0, :] = 0.0
    field[-1, :] = 0.0
    field[:, 0] = 0.0
    field[:, -1] = 0.0


    return field

def simulate_heat(initial_field: np.ndarray, config: HeatConfig,) -> np.ndarray:
    validate_config(
        config
    )

    expected_shape = (
        config.grid_size,
        config.grid_size,
    )

    if initial_field.shape != expected_shape:
        raise ValueError(
            f"Expected field shape {expected_shape}, "
            f"got {initial_field.shape}."
        )


    temperature = initial_field.astype(
        np.float32,
        copy=True,
    )


    r = np.float32(
        config.stability_ratio
    )


    for _ in range(
        config.simulation_steps
    ):

        # Work from the previous time step.
        #
        # We copy because every T^(n+1) value must be computed
        # from T^n, not partly updated values.
       

        previous = temperature

        updated = previous.copy()


        # -----------------------------------------------------
        # Five-point finite-difference stencil:
        #
        #                   north
        #                     |
        #          west -- center -- east
        #                     |
        #                   south
        #
        # Laplacian:
        #
        #     north + south + east + west - 4*center
        # -----------------------------------------------------

        center = previous[
            1:-1,
            1:-1,
        ]

        north = previous[
            :-2,
            1:-1,
        ]

        south = previous[
            2:,
            1:-1,
        ]

        west = previous[
            1:-1,
            :-2,
        ]

        east = previous[
            1:-1,
            2:,
        ]


        laplacian_term = (
            north
            + south
            + west
            + east
            - 4.0 * center
        )


        updated[
            1:-1,
            1:-1,
        ] = (
            center
            + r * laplacian_term
        )


        # Fixed zero-temperature boundary.
        updated[0, :] = 0.0
        updated[-1, :] = 0.0
        updated[:, 0] = 0.0
        updated[:, -1] = 0.0


        temperature = updated


    return temperature.astype(
        np.float32
    )
