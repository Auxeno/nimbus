"""Control logic functions for simulation."""

from dataclasses import replace

import jax.numpy as jnp

from .config import PIDControllerConfig
from .primitives import FLOAT_DTYPE, FloatScalar
from .state import Controls, PIDControllerState


def step_controls(
    current: Controls,
    commanded: Controls,
    engine_spool_time: FloatScalar,
    actuator_time: FloatScalar,
    dt: FloatScalar,
) -> Controls:
    """
    Slew current control values toward commanded values with simple rate limits.

    Parameters
    ----------
    current : Controls
        Current (actuated) control values used by the physics step.
    commanded : Controls
        Target control values produced by pilot/autopilot/G-limiter.
    engine_spool_time : FloatScalar
        Seconds for throttle to move from 0 → 1 under full step input.
    actuator_time : FloatScalar
        Seconds for a surface to move by 1.0 (e.g., from -1 → 0 or 0 → +1).
    dt : FloatScalar
        Simulation time step [s].

    Returns
    -------
    Controls
        Updated current controls after applying rate limits and clamping.
    """
    throttle_step = dt / engine_spool_time
    throttle_delta = jnp.clip(
        commanded.throttle - current.throttle,
        -throttle_step,
        throttle_step,
    )
    throttle = jnp.clip(current.throttle + throttle_delta, 0.0, 1.0)

    actuator_step = dt / actuator_time

    aileron_delta = jnp.clip(
        commanded.aileron - current.aileron,
        -actuator_step,
        actuator_step,
    )
    aileron = jnp.clip(current.aileron + aileron_delta, -1.0, 1.0)

    elevator_delta = jnp.clip(
        commanded.elevator - current.elevator,
        -actuator_step,
        actuator_step,
    )
    elevator = jnp.clip(current.elevator + elevator_delta, -1.0, 1.0)

    rudder_delta = jnp.clip(
        commanded.rudder - current.rudder,
        -actuator_step,
        actuator_step,
    )
    rudder = jnp.clip(current.rudder + rudder_delta, -1.0, 1.0)

    return Controls(
        throttle=throttle,
        aileron=aileron,
        elevator=elevator,
        rudder=rudder,
    )


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

    new_state = replace(pid_state, previous_error=error, integral=integral)

    return output, new_state
