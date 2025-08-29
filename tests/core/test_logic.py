"""Tests for logic module."""

import jax.numpy as jnp

from nimbus.core.config import PIDControllerConfig
from nimbus.core.logic import update_pid
from nimbus.core.primitives import EPS, FLOAT_DTYPE
from nimbus.core.state import PIDControllerState


def test_update_pid_proportional_only(jit_mode: str) -> None:
    """Test PID controller with only proportional term."""
    error = jnp.array(1.0, dtype=FLOAT_DTYPE)
    pid_state = PIDControllerState(
        previous_error=jnp.array(1.0, dtype=FLOAT_DTYPE),  # Same as current to zero derivative
        integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    pid_config = PIDControllerConfig(
        kp=0.5, ki=0.0, kd=0.0, max_correction=2.0, integral_limit=5.0
    )
    dt = jnp.array(0.01, dtype=FLOAT_DTYPE)

    output, new_state = update_pid(error, pid_state, pid_config, dt)

    # Only P term: kp * error = 0.5 * 1.0 = 0.5
    assert jnp.abs(output - 0.5) < EPS
    assert jnp.abs(new_state.previous_error - 1.0) < EPS


def test_update_pid_output_clamping(jit_mode: str) -> None:
    """Test PID controller output clamping."""
    error = jnp.array(10.0, dtype=FLOAT_DTYPE)  # Large error
    pid_state = PIDControllerState(
        previous_error=jnp.array(10.0, dtype=FLOAT_DTYPE),  # Same to zero derivative
        integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    pid_config = PIDControllerConfig(
        kp=1.0, ki=0.0, kd=0.0, max_correction=0.5, integral_limit=5.0  # Low max
    )
    dt = jnp.array(0.01, dtype=FLOAT_DTYPE)

    output, new_state = update_pid(error, pid_state, pid_config, dt)

    # Output should be clamped to max_correction
    assert jnp.abs(output) <= 0.5 + EPS
    assert new_state.previous_error == error


def test_update_pid_state_updates(jit_mode: str) -> None:
    """Test PID controller state updates correctly."""
    error = jnp.array(2.0, dtype=FLOAT_DTYPE)
    pid_state = PIDControllerState(
        previous_error=jnp.array(1.0, dtype=FLOAT_DTYPE),
        integral=jnp.array(0.5, dtype=FLOAT_DTYPE),
    )
    pid_config = PIDControllerConfig(
        kp=0.0, ki=0.0, kd=0.0, max_correction=10.0, integral_limit=10.0
    )
    dt = jnp.array(0.01, dtype=FLOAT_DTYPE)

    output, new_state = update_pid(error, pid_state, pid_config, dt)

    # State should be updated
    assert new_state.previous_error == error
    assert jnp.isfinite(new_state.integral)


def test_update_pid_zero_error_resets_integral(jit_mode: str) -> None:
    """Test PID controller resets integral on zero error (anti-windup)."""
    error = jnp.array(0.0, dtype=FLOAT_DTYPE)
    pid_state = PIDControllerState(
        previous_error=jnp.array(1.0, dtype=FLOAT_DTYPE),
        integral=jnp.array(2.0, dtype=FLOAT_DTYPE),
    )
    pid_config = PIDControllerConfig(
        kp=1.0, ki=0.5, kd=0.1, max_correction=2.0, integral_limit=5.0
    )
    dt = jnp.array(0.01, dtype=FLOAT_DTYPE)

    output, new_state = update_pid(error, pid_state, pid_config, dt)

    # With zero error, integral should reset to zero (anti-windup behavior)
    assert new_state.integral == 0.0
    assert new_state.previous_error == 0.0