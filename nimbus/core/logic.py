"""Control logic and automation functions for aircraft simulation."""

from dataclasses import replace

import jax.numpy as jnp

from .config import AircraftConfig, PhysicsConfig
from .interface import calculate_g_force
from .primitives import FLOAT_DTYPE, FloatScalar, Vector3
from .state import Aircraft, Controls, PIDControllerState


def apply_g_limiter(
    aircraft: Aircraft,
    controls: Controls,
    wind_velocity: Vector3,
    aircraft_config: AircraftConfig,
    physics_config: PhysicsConfig,
    dt: FloatScalar,
) -> tuple[Controls, PIDControllerState]:
    """
    Apply G-force limiting to elevator input using PID control.

    Parameters
    ----------
    aircraft : Aircraft
        Current aircraft state.
    controls : Controls
        Raw pilot control inputs.
    wind_velocity : Vector3
        Wind velocity in NED world frame [m/s].
    aircraft_config : AircraftConfig
        Aircraft configuration including G-limits and PID controller config.
    physics_config : PhysicsConfig
        Physics configuration for G-force calculation.
    dt : FloatScalar
        Time step for derivative calculation.

    Returns
    -------
    tuple[Controls, PIDControllerState]
        Adjusted controls with G-limiting applied and updated PID controller state.
    """
    g_forces = calculate_g_force(
        aircraft, wind_velocity, aircraft_config, physics_config
    )

    # Negate for pilot convention: positive G = pulling up
    current_g = -g_forces[2]

    g_min = jnp.array(aircraft_config.g_limit_min, dtype=FLOAT_DTYPE)
    g_max = jnp.array(aircraft_config.g_limit_max, dtype=FLOAT_DTYPE)

    saturated_g = jnp.clip(current_g, g_min, g_max)
    error = current_g - saturated_g

    kp = jnp.array(aircraft_config.g_limiter_controller_config.kp, dtype=FLOAT_DTYPE)
    ki = jnp.array(aircraft_config.g_limiter_controller_config.ki, dtype=FLOAT_DTYPE)
    kd = jnp.array(aircraft_config.g_limiter_controller_config.kd, dtype=FLOAT_DTYPE)
    max_correction = jnp.array(
        aircraft_config.g_limiter_controller_config.max_correction, dtype=FLOAT_DTYPE
    )
    integral_limit = jnp.array(
        aircraft_config.g_limiter_controller_config.integral_limit, dtype=FLOAT_DTYPE
    )

    # Calculate PID terms
    derivative = (error - aircraft.g_limiter_pid.previous_error) / dt

    # Only accumulate integral when there's a G-limit violation
    # Reset to zero when within acceptable tolerance
    integral = jnp.where(
        jnp.abs(error) > jnp.array(0.0, dtype=FLOAT_DTYPE),
        aircraft.g_limiter_pid.integral + error * dt,
        jnp.array(0.0, dtype=FLOAT_DTYPE),
    )

    # Apply integral windup prevention
    integral = jnp.clip(integral, -integral_limit, integral_limit)

    # Apply PID formula
    correction = kp * error + ki * integral + kd * derivative
    correction = jnp.clip(correction, -max_correction, max_correction)

    adjusted_elevator = jnp.clip(
        controls.elevator - correction,
        jnp.array(-1.0, dtype=FLOAT_DTYPE),
        jnp.array(1.0, dtype=FLOAT_DTYPE),
    )

    adjusted_controls = replace(controls, elevator=adjusted_elevator)

    new_pid_state = PIDControllerState(previous_error=error, integral=integral)

    return adjusted_controls, new_pid_state
