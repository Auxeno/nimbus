"""Control logic functions for simulation."""

import jax.numpy as jnp

from .config import PIDControllerConfig
from .primitives import FLOAT_DTYPE, FloatScalar
from .state import PIDControllerState


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
