"""Control logic and automation functions for aircraft simulation."""

from dataclasses import replace

import jax.numpy as jnp

from .config import AircraftConfig, PhysicsConfig
from .interface import calculate_g_force
from .primitives import FLOAT_DTYPE, FloatScalar
from .state import Aircraft, Controls, PDControllerState


def apply_g_limiter(
    aircraft: Aircraft,
    controls: Controls,
    aircraft_config: AircraftConfig,
    physics_config: PhysicsConfig,
    dt: FloatScalar,
) -> tuple[Controls, PDControllerState]:
    """
    Apply G-force limiting to elevator input using PD control.

    Parameters
    ----------
    aircraft : Aircraft
        Current aircraft state.
    controls : Controls
        Raw pilot control inputs.
    aircraft_config : AircraftConfig
        Aircraft configuration including G-limits and PD controller config.
    physics_config : PhysicsConfig
        Physics configuration for G-force calculation.
    dt : FloatScalar
        Time step for derivative calculation.

    Returns
    -------
    tuple[Controls, PDControllerState]
        Adjusted controls with G-limiting applied and updated PD controller state.
    """
    g_forces = calculate_g_force(aircraft, aircraft_config, physics_config)

    # Negate for pilot convention: positive G = pulling up
    current_g = -g_forces[2]

    g_min = jnp.array(aircraft_config.g_limit_min, dtype=FLOAT_DTYPE)
    g_max = jnp.array(aircraft_config.g_limit_max, dtype=FLOAT_DTYPE)

    saturated_g = jnp.clip(current_g, g_min, g_max)
    error = current_g - saturated_g

    kp = jnp.array(aircraft_config.g_limiter_controller_config.kp, dtype=FLOAT_DTYPE)
    kd = jnp.array(aircraft_config.g_limiter_controller_config.kd, dtype=FLOAT_DTYPE)
    max_correction = jnp.array(
        aircraft_config.g_limiter_controller_config.max_correction, dtype=FLOAT_DTYPE
    )

    # Apply proportional derivative formula
    derivative = (error - aircraft.g_limiter_pd.previous_error) / dt
    correction = kp * error + kd * derivative
    correction = jnp.clip(correction, -max_correction, max_correction)

    adjusted_elevator = jnp.clip(
        controls.elevator - correction,
        jnp.array(-1.0, dtype=FLOAT_DTYPE),
        jnp.array(1.0, dtype=FLOAT_DTYPE),
    )

    adjusted_controls = replace(controls, elevator=adjusted_elevator)

    new_pd_state = PDControllerState(previous_error=error)

    return adjusted_controls, new_pd_state
