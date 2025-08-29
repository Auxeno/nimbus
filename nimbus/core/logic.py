"""Control logic and automation functions for aircraft simulation."""

from dataclasses import replace

import jax.numpy as jnp

from .config import AircraftConfig, PhysicsConfig, PIDControllerConfig
from .interface import calculate_g_force
from .primitives import FLOAT_DTYPE, FloatScalar
from .state import Aircraft, Controls, PIDControllerState, Wind


def update_pid(
    error: FloatScalar,
    pid_state: PIDControllerState,
    pid_config: PIDControllerConfig,
    dt: FloatScalar,
) -> tuple[FloatScalar, PIDControllerState]:
    """
    Update PID controller state and compute control output.

    Parameters
    ----------
    error : FloatScalar
        Current error signal.
    pid_state : PIDControllerState
        Current PID controller state.
    pid_config : PIDControllerConfig
        PID controller configuration with gains and limits.
    dt : FloatScalar
        Time step for integral and derivative calculations.

    Returns
    -------
    tuple[FloatScalar, PIDControllerState]
        Control output and updated PID state.
    """
    kp = jnp.array(pid_config.kp, dtype=FLOAT_DTYPE)
    ki = jnp.array(pid_config.ki, dtype=FLOAT_DTYPE)
    kd = jnp.array(pid_config.kd, dtype=FLOAT_DTYPE)
    max_correction = jnp.array(pid_config.max_correction, dtype=FLOAT_DTYPE)
    integral_limit = jnp.array(pid_config.integral_limit, dtype=FLOAT_DTYPE)

    derivative = (error - pid_state.previous_error) / dt

    # Update integral with anti-windup
    integral = jnp.where(
        jnp.abs(error) > jnp.array(0.0, dtype=FLOAT_DTYPE),
        pid_state.integral + error * dt,
        jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    integral = jnp.clip(integral, -integral_limit, integral_limit)

    # PID output
    output = kp * error + ki * integral + kd * derivative
    output = jnp.clip(output, -max_correction, max_correction)

    new_state = PIDControllerState(previous_error=error, integral=integral)

    return output, new_state


def apply_g_limiter(
    aircraft: Aircraft,
    controls: Controls,
    wind: Wind,
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
    wind : Wind
        Wind state with mean and gust components.
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
    g_forces = calculate_g_force(aircraft, wind, aircraft_config, physics_config)

    # Negate for pilot convention: positive G = pulling up
    current_g = -g_forces[2]

    g_min = jnp.array(aircraft_config.g_limit_min, dtype=FLOAT_DTYPE)
    g_max = jnp.array(aircraft_config.g_limit_max, dtype=FLOAT_DTYPE)

    saturated_g = jnp.clip(current_g, g_min, g_max)
    error = current_g - saturated_g

    correction, new_pid_state = update_pid(
        error,
        aircraft.g_limiter_pid,
        aircraft_config.g_limiter_controller_config,
        dt,
    )

    adjusted_elevator = jnp.clip(
        controls.elevator - correction,
        jnp.array(-1.0, dtype=FLOAT_DTYPE),
        jnp.array(1.0, dtype=FLOAT_DTYPE),
    )

    adjusted_controls = replace(controls, elevator=adjusted_elevator)

    return adjusted_controls, new_pid_state
