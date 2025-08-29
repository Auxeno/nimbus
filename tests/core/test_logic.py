"""Tests for logic module."""

from dataclasses import replace

import jax
import jax.numpy as jnp

from nimbus.core.config import AircraftConfig, PIDControllerConfig, PhysicsConfig
from nimbus.core.logic import apply_g_limiter
from nimbus.core.primitives import EPS, FLOAT_DTYPE, INT_DTYPE
from nimbus.core.quaternion import from_euler_zyx
from nimbus.core.state import Aircraft, Body, Controls, Meta, PIDControllerState


def test_apply_g_limiter_no_correction(jit_mode: str) -> None:
    """Test G-limiter with aircraft within G-limits (no correction needed)."""
    # Standard case 1 - aircraft in level flight within G-limits
    meta = Meta(
        active=jnp.array(True, dtype=bool),
        id=jnp.array(0, dtype=INT_DTYPE),
    )
    body = Body(
        position=jnp.array([0.0, 0.0, -500.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([100.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
        orientation=from_euler_zyx(
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
        angular_velocity=jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
    )
    controls = Controls(
        throttle=jnp.array(0.5, dtype=FLOAT_DTYPE),
        aileron=jnp.array(0.0, dtype=FLOAT_DTYPE),
        elevator=jnp.array(0.2, dtype=FLOAT_DTYPE),
        rudder=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    aircraft = Aircraft(
        meta=meta,
        body=body,
        controls=controls,
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )
    aircraft_config = AircraftConfig()
    physics_config = PhysicsConfig()
    dt = jnp.array(0.01, dtype=FLOAT_DTYPE)
    wind_velocity = jnp.zeros(3, dtype=FLOAT_DTYPE)

    adjusted_controls, new_pd_state = apply_g_limiter(
        aircraft, controls, wind_velocity, aircraft_config, physics_config, dt
    )

    # Should not modify elevator when within limits
    assert jnp.abs(adjusted_controls.elevator - controls.elevator) < 0.1
    assert jnp.abs(new_pd_state.previous_error) < 0.1

    # Standard case 2 - moderate elevator input
    controls_moderate = Controls(
        throttle=jnp.array(0.5, dtype=FLOAT_DTYPE),
        aileron=jnp.array(0.0, dtype=FLOAT_DTYPE),
        elevator=jnp.array(0.1, dtype=FLOAT_DTYPE),
        rudder=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    aircraft_moderate = Aircraft(
        meta=meta,
        body=body,
        controls=controls_moderate,
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )

    adjusted_controls_2, new_pd_state_2 = apply_g_limiter(
        aircraft_moderate,
        controls_moderate,
        wind_velocity,
        aircraft_config,
        physics_config,
        dt,
    )

    # Should have minimal correction
    assert jnp.abs(adjusted_controls_2.elevator - controls_moderate.elevator) < 0.1


def test_apply_g_limiter_positive_g_violation(jit_mode: str) -> None:
    """Test G-limiter when exceeding positive G-limit."""
    # Aircraft already in high-G turn with angle of attack
    meta = Meta(
        active=jnp.array(True, dtype=bool),
        id=jnp.array(0, dtype=INT_DTYPE),
    )
    body = Body(
        position=jnp.array([0.0, 0.0, -1000.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array(
            [200.0, 0.0, -20.0], dtype=FLOAT_DTYPE
        ),  # High speed with vertical component
        orientation=from_euler_zyx(
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.2, dtype=FLOAT_DTYPE),  # Pitched up significantly
            jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
        angular_velocity=jnp.array([0.0, 1.0, 0.0], dtype=FLOAT_DTYPE),  # Pitching up
    )
    controls = Controls(
        throttle=jnp.array(0.9, dtype=FLOAT_DTYPE),
        aileron=jnp.array(0.0, dtype=FLOAT_DTYPE),
        elevator=jnp.array(0.8, dtype=FLOAT_DTYPE),  # High elevator
        rudder=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    aircraft = Aircraft(
        meta=meta,
        body=body,
        controls=controls,
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )
    # Very low G-limit to ensure violation
    aircraft_config = AircraftConfig(
        g_limit_max=0.5,  # Extremely low limit
        g_limiter_controller_config=PIDControllerConfig(
            kp=0.5, ki=0.0, kd=0.0, max_correction=1.0, integral_limit=2.0
        ),
    )
    physics_config = PhysicsConfig()
    dt = jnp.array(0.01, dtype=FLOAT_DTYPE)
    wind_velocity = jnp.zeros(3, dtype=FLOAT_DTYPE)

    adjusted_controls, new_pd_state = apply_g_limiter(
        aircraft, controls, wind_velocity, aircraft_config, physics_config, dt
    )

    # Should reduce elevator input when G-limit is violated (or no change if not violated)
    assert adjusted_controls.elevator <= controls.elevator + EPS
    # PD state should be updated with current error (may be zero if no violation)
    assert jnp.isfinite(new_pd_state.previous_error)

    # Standard case 2 - even more extreme high-G scenario
    body_extreme = Body(
        position=jnp.array([0.0, 0.0, -1000.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array(
            [250.0, 0.0, -40.0], dtype=FLOAT_DTYPE
        ),  # Higher speed and more vertical component
        orientation=from_euler_zyx(
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.35, dtype=FLOAT_DTYPE),  # Even higher pitch angle (~20 degrees)
            jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
        angular_velocity=jnp.array(
            [0.0, 2.0, 0.0], dtype=FLOAT_DTYPE
        ),  # High pitch rate
    )
    aircraft_extreme = Aircraft(
        meta=meta,
        body=body_extreme,
        controls=controls,
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )

    adjusted_controls_2, new_pd_state_2 = apply_g_limiter(
        aircraft_extreme, controls, wind_velocity, aircraft_config, physics_config, dt
    )

    # Should apply stronger correction (more negative elevator = stronger correction)
    assert adjusted_controls_2.elevator <= adjusted_controls.elevator


def test_apply_g_limiter_negative_g_violation(jit_mode: str) -> None:
    """Test G-limiter when exceeding negative G-limit."""
    # Aircraft in negative-G pushover maneuver
    meta = Meta(
        active=jnp.array(True, dtype=bool),
        id=jnp.array(0, dtype=INT_DTYPE),
    )
    body = Body(
        position=jnp.array([0.0, 0.0, -1000.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([200.0, 0.0, 20.0], dtype=FLOAT_DTYPE),  # High speed diving
        orientation=from_euler_zyx(
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(-0.3, dtype=FLOAT_DTYPE),  # Pitched down significantly
            jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
        angular_velocity=jnp.array(
            [0.0, -1.5, 0.0], dtype=FLOAT_DTYPE
        ),  # Pitching down
    )
    controls = Controls(
        throttle=jnp.array(0.2, dtype=FLOAT_DTYPE),  # Low thrust
        aileron=jnp.array(0.0, dtype=FLOAT_DTYPE),
        elevator=jnp.array(-0.8, dtype=FLOAT_DTYPE),  # Negative elevator
        rudder=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    aircraft = Aircraft(
        meta=meta,
        body=body,
        controls=controls,
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )
    # Very tight negative G-limit
    aircraft_config = AircraftConfig(
        g_limit_min=0.2,  # Positive limit to force negative-G violation
        g_limiter_controller_config=PIDControllerConfig(
            kp=0.5, ki=0.0, kd=0.0, max_correction=1.0, integral_limit=2.0
        ),
    )
    physics_config = PhysicsConfig()
    dt = jnp.array(0.01, dtype=FLOAT_DTYPE)
    wind_velocity = jnp.zeros(3, dtype=FLOAT_DTYPE)

    adjusted_controls, new_pd_state = apply_g_limiter(
        aircraft, controls, wind_velocity, aircraft_config, physics_config, dt
    )

    # Should increase elevator input (less negative) when negative G-limit violated
    assert adjusted_controls.elevator >= controls.elevator - EPS
    # PD state should be updated with current error
    assert jnp.isfinite(new_pd_state.previous_error)


def test_apply_g_limiter_pd_controller(jit_mode: str) -> None:
    """Test PD controller behavior including proportional and derivative terms."""
    # Aircraft in high-G scenario that actually produces G-force violations
    meta = Meta(
        active=jnp.array(True, dtype=bool),
        id=jnp.array(0, dtype=INT_DTYPE),
    )
    body = Body(
        position=jnp.array([0.0, 0.0, -1000.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array(
            [250.0, 0.0, -30.0], dtype=FLOAT_DTYPE
        ),  # High speed with angle of attack
        orientation=from_euler_zyx(
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.3, dtype=FLOAT_DTYPE),  # Significant pitch angle
            jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
        angular_velocity=jnp.array(
            [0.0, 2.0, 0.0], dtype=FLOAT_DTYPE
        ),  # High pitch rate
    )
    controls = Controls(
        throttle=jnp.array(0.9, dtype=FLOAT_DTYPE),
        aileron=jnp.array(0.0, dtype=FLOAT_DTYPE),
        elevator=jnp.array(0.7, dtype=FLOAT_DTYPE),  # High elevator
        rudder=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    aircraft = Aircraft(
        meta=meta,
        body=body,
        controls=controls,
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )

    # Low proportional gain with very low G-limit
    config_low_kp = AircraftConfig(
        g_limit_max=0.3,  # Extremely low limit to force violation
        g_limiter_controller_config=PIDControllerConfig(
            kp=0.1, ki=0.0, kd=0.0, max_correction=1.0, integral_limit=2.0
        ),
    )
    # High proportional gain
    config_high_kp = AircraftConfig(
        g_limit_max=0.3,  # Same limit
        g_limiter_controller_config=PIDControllerConfig(
            kp=0.5, ki=0.0, kd=0.0, max_correction=1.0, integral_limit=2.0
        ),
    )
    physics_config = PhysicsConfig()
    dt = jnp.array(0.01, dtype=FLOAT_DTYPE)
    wind_velocity = jnp.zeros(3, dtype=FLOAT_DTYPE)

    controls_low_kp, _ = apply_g_limiter(
        aircraft, controls, wind_velocity, config_low_kp, physics_config, dt
    )
    controls_high_kp, _ = apply_g_limiter(
        aircraft, controls, wind_velocity, config_high_kp, physics_config, dt
    )

    # Higher Kp should result in larger correction (more reduction in elevator) if violation occurs
    correction_low = controls.elevator - controls_low_kp.elevator
    correction_high = controls.elevator - controls_high_kp.elevator
    assert correction_high >= correction_low - EPS

    # Standard case 2 - test derivative term
    aircraft_with_error = Aircraft(
        meta=meta,
        body=body,
        controls=controls,
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(1.0, dtype=FLOAT_DTYPE),  # Previous error
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )
    config_with_kd = AircraftConfig(
        g_limiter_controller_config=PIDControllerConfig(
            kp=0.1, ki=0.0, kd=0.05, max_correction=1.0, integral_limit=2.0
        ),
    )

    adjusted_controls, new_pd_state = apply_g_limiter(
        aircraft_with_error, controls, wind_velocity, config_with_kd, physics_config, dt
    )

    # Should update previous error (may be zero if no G-limit violation)
    assert jnp.isfinite(new_pd_state.previous_error)

    # Standard case 3 - test max correction clamping
    aircraft_extreme = Aircraft(
        meta=meta,
        body=Body(
            position=jnp.array([0.0, 0.0, -1000.0], dtype=FLOAT_DTYPE),
            velocity=jnp.array([100.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
            orientation=from_euler_zyx(
                jnp.array(0.0, dtype=FLOAT_DTYPE),
                jnp.array(0.0, dtype=FLOAT_DTYPE),
                jnp.array(0.0, dtype=FLOAT_DTYPE),
            ),
            angular_velocity=jnp.array([0.0, 5.0, 0.0], dtype=FLOAT_DTYPE),  # Extreme G
        ),
        controls=Controls(
            throttle=jnp.array(0.5, dtype=FLOAT_DTYPE),
            aileron=jnp.array(0.0, dtype=FLOAT_DTYPE),
            elevator=jnp.array(0.9, dtype=FLOAT_DTYPE),
            rudder=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )
    config_low_max = AircraftConfig(
        g_limiter_controller_config=PIDControllerConfig(
            kp=1.0,
            ki=0.0,
            kd=0.0,
            max_correction=0.1,  # High Kp, low max
            integral_limit=0.0,
        ),
    )

    controls_clamped, _ = apply_g_limiter(
        aircraft_extreme,
        aircraft_extreme.controls,
        wind_velocity,
        config_low_max,
        physics_config,
        dt,
    )

    # Correction should be limited by max_correction
    correction = jnp.abs(aircraft_extreme.controls.elevator - controls_clamped.elevator)
    assert correction <= config_low_max.g_limiter_controller_config.max_correction + EPS


def test_apply_g_limiter_elevator_saturation(jit_mode: str) -> None:
    """Test elevator saturation at ±1.0 limits."""
    # Edge case 1 - near positive limit
    meta = Meta(
        active=jnp.array(True, dtype=bool),
        id=jnp.array(0, dtype=INT_DTYPE),
    )
    body = Body(
        position=jnp.array([0.0, 0.0, -1000.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([100.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
        orientation=from_euler_zyx(
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
        angular_velocity=jnp.array([0.0, -3.0, 0.0], dtype=FLOAT_DTYPE),
    )
    controls_high = Controls(
        throttle=jnp.array(0.5, dtype=FLOAT_DTYPE),
        aileron=jnp.array(0.0, dtype=FLOAT_DTYPE),
        elevator=jnp.array(0.95, dtype=FLOAT_DTYPE),  # Near positive limit
        rudder=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    aircraft_high = Aircraft(
        meta=meta,
        body=body,
        controls=controls_high,
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )
    aircraft_config = AircraftConfig(
        g_limiter_controller_config=PIDControllerConfig(
            kp=0.5, ki=0.0, kd=0.0, max_correction=0.8, integral_limit=2.0
        ),
    )
    physics_config = PhysicsConfig()
    dt = jnp.array(0.01, dtype=FLOAT_DTYPE)
    wind_velocity = jnp.zeros(3, dtype=FLOAT_DTYPE)

    adjusted_controls, _ = apply_g_limiter(
        aircraft_high, controls_high, wind_velocity, aircraft_config, physics_config, dt
    )

    # Elevator should be clamped to [-1, 1]
    assert adjusted_controls.elevator >= -1.0 - EPS
    assert adjusted_controls.elevator <= 1.0 + EPS

    # Edge case 2 - near negative limit
    controls_low = Controls(
        throttle=jnp.array(0.5, dtype=FLOAT_DTYPE),
        aileron=jnp.array(0.0, dtype=FLOAT_DTYPE),
        elevator=jnp.array(-0.95, dtype=FLOAT_DTYPE),  # Near negative limit
        rudder=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    body_neg = Body(
        position=jnp.array([0.0, 0.0, -1000.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([100.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
        orientation=from_euler_zyx(
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
        angular_velocity=jnp.array([0.0, 3.0, 0.0], dtype=FLOAT_DTYPE),
    )
    aircraft_low = Aircraft(
        meta=meta,
        body=body_neg,
        controls=controls_low,
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )

    adjusted_controls_2, _ = apply_g_limiter(
        aircraft_low, controls_low, wind_velocity, aircraft_config, physics_config, dt
    )

    # Should remain within bounds
    assert adjusted_controls_2.elevator >= -1.0 - EPS
    assert adjusted_controls_2.elevator <= 1.0 + EPS


def test_apply_g_limiter_timestep_stability(jit_mode: str) -> None:
    """Test controller stability with various timesteps."""
    # Standard case 1 - small timestep
    meta = Meta(
        active=jnp.array(True, dtype=bool),
        id=jnp.array(0, dtype=INT_DTYPE),
    )
    body = Body(
        position=jnp.array([0.0, 0.0, -1000.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([100.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
        orientation=from_euler_zyx(
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
        angular_velocity=jnp.array([0.0, 1.0, 0.0], dtype=FLOAT_DTYPE),
    )
    controls = Controls(
        throttle=jnp.array(0.5, dtype=FLOAT_DTYPE),
        aileron=jnp.array(0.0, dtype=FLOAT_DTYPE),
        elevator=jnp.array(0.3, dtype=FLOAT_DTYPE),
        rudder=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    aircraft = Aircraft(
        meta=meta,
        body=body,
        controls=controls,
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.5, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )
    aircraft_config = AircraftConfig()
    physics_config = PhysicsConfig()
    wind_velocity = jnp.zeros(3, dtype=FLOAT_DTYPE)

    # Very small timestep
    dt_small = jnp.array(1e-6, dtype=FLOAT_DTYPE)
    adjusted_controls_small, new_pd_state_small = apply_g_limiter(
        aircraft, controls, wind_velocity, aircraft_config, physics_config, dt_small
    )

    # Should still produce valid results
    assert jnp.isfinite(adjusted_controls_small.elevator)
    assert jnp.isfinite(new_pd_state_small.previous_error)

    # Standard case 2 - large timestep
    dt_large = jnp.array(0.1, dtype=FLOAT_DTYPE)
    adjusted_controls_large, new_pd_state_large = apply_g_limiter(
        aircraft, controls, wind_velocity, aircraft_config, physics_config, dt_large
    )

    # Should remain stable and bounded
    assert jnp.abs(adjusted_controls_large.elevator) <= 1.0 + EPS
    assert jnp.isfinite(new_pd_state_large.previous_error)

    # Edge case - normal timestep for comparison
    dt_normal = jnp.array(0.01, dtype=FLOAT_DTYPE)
    adjusted_controls_normal, new_pd_state_normal = apply_g_limiter(
        aircraft, controls, wind_velocity, aircraft_config, physics_config, dt_normal
    )

    # All should be finite
    assert jnp.isfinite(adjusted_controls_normal.elevator)
    assert jnp.isfinite(new_pd_state_normal.previous_error)


def test_apply_g_limiter_aircraft_configurations(jit_mode: str) -> None:
    """Test with different aircraft configurations (fighter vs transport)."""
    # Standard case 1 - fighter aircraft with high G-limits
    meta = Meta(
        active=jnp.array(True, dtype=bool),
        id=jnp.array(0, dtype=INT_DTYPE),
    )
    body = Body(
        position=jnp.array([0.0, 0.0, -1000.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([150.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
        orientation=from_euler_zyx(
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
        angular_velocity=jnp.array([0.0, 2.0, 0.0], dtype=FLOAT_DTYPE),
    )
    controls = Controls(
        throttle=jnp.array(0.7, dtype=FLOAT_DTYPE),
        aileron=jnp.array(0.0, dtype=FLOAT_DTYPE),
        elevator=jnp.array(0.6, dtype=FLOAT_DTYPE),
        rudder=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    aircraft = Aircraft(
        meta=meta,
        body=body,
        controls=controls,
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )

    # Fighter config - high G-limits
    fighter_config = AircraftConfig(
        g_limit_min=-3.0,
        g_limit_max=9.0,
        g_limiter_controller_config=PIDControllerConfig(
            kp=0.1, ki=0.0, kd=0.01, max_correction=0.5, integral_limit=2.0
        ),
    )
    # Transport config - lower G-limits
    transport_config = AircraftConfig(
        g_limit_min=-1.0,
        g_limit_max=2.5,
        g_limiter_controller_config=PIDControllerConfig(
            kp=0.1, ki=0.0, kd=0.01, max_correction=0.5, integral_limit=2.0
        ),
    )
    physics_config = PhysicsConfig()
    dt = jnp.array(0.01, dtype=FLOAT_DTYPE)
    wind_velocity = jnp.zeros(3, dtype=FLOAT_DTYPE)

    fighter_controls, _ = apply_g_limiter(
        aircraft, controls, wind_velocity, fighter_config, physics_config, dt
    )
    transport_controls, _ = apply_g_limiter(
        aircraft, controls, wind_velocity, transport_config, physics_config, dt
    )

    # Transport should have more aggressive limiting
    fighter_correction = jnp.abs(controls.elevator - fighter_controls.elevator)
    transport_correction = jnp.abs(controls.elevator - transport_controls.elevator)
    assert transport_correction >= fighter_correction - EPS

    # Standard case 2 - aerobatic aircraft
    aerobatic_config = AircraftConfig(
        g_limit_min=-6.0,
        g_limit_max=12.0,
        g_limiter_controller_config=PIDControllerConfig(
            kp=0.05, ki=0.0, kd=0.005, max_correction=0.3, integral_limit=2.0
        ),
    )

    aerobatic_controls, _ = apply_g_limiter(
        aircraft, controls, wind_velocity, aerobatic_config, physics_config, dt
    )

    # Aerobatic should have least limiting
    aerobatic_correction = jnp.abs(controls.elevator - aerobatic_controls.elevator)
    assert aerobatic_correction <= fighter_correction + EPS


def test_apply_g_limiter_multi_step(jit_mode: str) -> None:
    """Test PD controller state evolution over multiple time steps."""
    # Standard case - simulate multiple time steps
    meta = Meta(
        active=jnp.array(True, dtype=bool),
        id=jnp.array(0, dtype=INT_DTYPE),
    )
    body = Body(
        position=jnp.array([0.0, 0.0, -1000.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([100.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
        orientation=from_euler_zyx(
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
        angular_velocity=jnp.array([0.0, 1.5, 0.0], dtype=FLOAT_DTYPE),
    )
    controls = Controls(
        throttle=jnp.array(0.5, dtype=FLOAT_DTYPE),
        aileron=jnp.array(0.0, dtype=FLOAT_DTYPE),
        elevator=jnp.array(0.5, dtype=FLOAT_DTYPE),
        rudder=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    aircraft = Aircraft(
        meta=meta,
        body=body,
        controls=controls,
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )
    aircraft_config = AircraftConfig()
    physics_config = PhysicsConfig()
    dt = jnp.array(0.01, dtype=FLOAT_DTYPE)
    wind_velocity = jnp.zeros(3, dtype=FLOAT_DTYPE)

    # Simulate 5 time steps
    current_aircraft = aircraft
    for i in range(5):
        adjusted_controls, new_pd_state = apply_g_limiter(
            current_aircraft,
            controls,
            wind_velocity,
            aircraft_config,
            physics_config,
            dt,
        )

        # Update aircraft PD state for next iteration
        current_aircraft = replace(current_aircraft, g_limiter_pid=new_pd_state)

        # Verify state consistency
        assert jnp.isfinite(new_pd_state.previous_error)
        assert jnp.abs(adjusted_controls.elevator) <= 1.0 + EPS
        # Other controls should remain unchanged
        assert jnp.allclose(adjusted_controls.throttle, controls.throttle, atol=EPS)
        assert jnp.allclose(adjusted_controls.aileron, controls.aileron, atol=EPS)
        assert jnp.allclose(adjusted_controls.rudder, controls.rudder, atol=EPS)


def test_apply_g_limiter_vmap(jit_mode: str) -> None:
    """Test vmap compatibility for batch processing."""
    # Create batch of aircraft states
    batch_size = 3

    # Create different metas for batch
    metas = jax.vmap(
        lambda i: Meta(
            active=jnp.array(True, dtype=bool),
            id=i,
        )
    )(jnp.arange(batch_size, dtype=INT_DTYPE))

    # Create different bodies with varying angular velocities
    angular_vels = jnp.array(
        [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.5, 0.0]], dtype=FLOAT_DTYPE
    )
    bodies = jax.vmap(
        lambda ang_vel: Body(
            position=jnp.array([0.0, 0.0, -1000.0], dtype=FLOAT_DTYPE),
            velocity=jnp.array([100.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
            orientation=from_euler_zyx(
                jnp.array(0.0, dtype=FLOAT_DTYPE),
                jnp.array(0.0, dtype=FLOAT_DTYPE),
                jnp.array(0.0, dtype=FLOAT_DTYPE),
            ),
            angular_velocity=ang_vel,
        )
    )(angular_vels)

    # Create different control inputs
    elevator_inputs = jnp.array([0.3, 0.5, 0.7], dtype=FLOAT_DTYPE)
    controls_batch = jax.vmap(
        lambda elev: Controls(
            throttle=jnp.array(0.5, dtype=FLOAT_DTYPE),
            aileron=jnp.array(0.0, dtype=FLOAT_DTYPE),
            elevator=elev,
            rudder=jnp.array(0.0, dtype=FLOAT_DTYPE),
        )
    )(elevator_inputs)

    # Create PD states
    pd_states = jax.vmap(
        lambda _: PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        )
    )(jnp.arange(batch_size))

    # Create batch of aircraft
    aircraft_batch = jax.vmap(Aircraft)(metas, bodies, controls_batch, pd_states)

    aircraft_config = AircraftConfig()
    physics_config = PhysicsConfig()
    dt = jnp.array(0.01, dtype=FLOAT_DTYPE)
    wind_velocity = jnp.zeros(3, dtype=FLOAT_DTYPE)

    # Apply vmap
    vmapped_g_limiter = jax.vmap(
        apply_g_limiter, in_axes=(0, 0, None, None, None, None)
    )

    batched_adjusted_controls, batched_pd_states = vmapped_g_limiter(
        aircraft_batch,
        controls_batch,
        wind_velocity,
        aircraft_config,
        physics_config,
        dt,
    )

    # Verify results
    assert batched_adjusted_controls.elevator.shape == (batch_size,)
    assert batched_pd_states.previous_error.shape == (batch_size,)
    assert jnp.all(jnp.isfinite(batched_adjusted_controls.elevator))
    assert jnp.all(jnp.isfinite(batched_pd_states.previous_error))
    # All elevators should be within bounds
    assert jnp.all(batched_adjusted_controls.elevator >= -1.0 - EPS)
    assert jnp.all(batched_adjusted_controls.elevator <= 1.0 + EPS)


def test_apply_g_limiter_extreme_values(jit_mode: str) -> None:
    """Test with extreme but realistic values."""
    # Edge case 1 - extreme maneuver scenario
    meta = Meta(
        active=jnp.array(True, dtype=bool),
        id=jnp.array(0, dtype=INT_DTYPE),
    )
    body = Body(
        position=jnp.array([0.0, 0.0, -5000.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([200.0, 0.0, 0.0], dtype=FLOAT_DTYPE),  # High speed
        orientation=from_euler_zyx(
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.2, dtype=FLOAT_DTYPE),  # Pitched up
            jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
        angular_velocity=jnp.array(
            [0.0, 10.0, 0.0], dtype=FLOAT_DTYPE
        ),  # Very high pitch rate
    )
    controls = Controls(
        throttle=jnp.array(1.0, dtype=FLOAT_DTYPE),  # Max throttle
        aileron=jnp.array(0.0, dtype=FLOAT_DTYPE),
        elevator=jnp.array(1.0, dtype=FLOAT_DTYPE),  # Maximum elevator
        rudder=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    aircraft = Aircraft(
        meta=meta,
        body=body,
        controls=controls,
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(5.0, dtype=FLOAT_DTYPE),  # Large previous error
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )
    aircraft_config = AircraftConfig(
        g_limit_max=12.0,
        g_limiter_controller_config=PIDControllerConfig(
            kp=0.5, ki=0.0, kd=0.05, max_correction=1.0, integral_limit=2.0
        ),
    )
    physics_config = PhysicsConfig()
    dt = jnp.array(0.01, dtype=FLOAT_DTYPE)
    wind_velocity = jnp.array(0.0, dtype=FLOAT_DTYPE)

    adjusted_controls, new_pd_state = apply_g_limiter(
        aircraft, controls, wind_velocity, aircraft_config, physics_config, dt
    )

    # Should remain bounded and stable
    assert adjusted_controls.elevator >= -1.0 - EPS
    assert adjusted_controls.elevator <= 1.0 + EPS
    assert jnp.isfinite(new_pd_state.previous_error)

    # Edge case 2 - near-zero velocity
    body_slow = Body(
        position=jnp.array([0.0, 0.0, -100.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([1.0, 0.0, 0.0], dtype=FLOAT_DTYPE),  # Very slow
        orientation=from_euler_zyx(
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
        angular_velocity=jnp.array([0.0, 0.1, 0.0], dtype=FLOAT_DTYPE),
    )
    aircraft_slow = Aircraft(
        meta=meta,
        body=body_slow,
        controls=controls,
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )

    adjusted_controls_2, new_pd_state_2 = apply_g_limiter(
        aircraft_slow, controls, wind_velocity, aircraft_config, physics_config, dt
    )

    # Should handle low velocity gracefully
    assert jnp.isfinite(adjusted_controls_2.elevator)
    assert jnp.isfinite(new_pd_state_2.previous_error)
