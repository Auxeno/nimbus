"""Tests for physics module."""

import jax
import jax.numpy as jnp

from nimbus.core.physics import (
    calculate_aero_forces,
    calculate_air_density,
    calculate_angle_of_attack,
    calculate_angle_of_sideslip,
    calculate_control_moments,
    calculate_damping_moments,
    calculate_drag,
    calculate_dynamic_pressure,
    calculate_dynamic_pressure_scale,
    calculate_lift,
    calculate_sideslip,
    calculate_thrust,
    calculate_weight,
    estimate_inertia,
)
from nimbus.core.primitives import EPS, FLOAT_DTYPE
from nimbus.core.quaternion import from_euler_zyx

pi = jnp.array(jnp.pi, dtype=FLOAT_DTYPE)


def test_calculate_air_density(jit_mode: str) -> None:
    """Test air density calculation."""
    # Standard case 1 - sea level
    altitude_1 = jnp.array(0.0, dtype=FLOAT_DTYPE)
    rho_0_1 = jnp.array(1.225, dtype=FLOAT_DTYPE)
    rho_decay_1 = jnp.array(5500.0, dtype=FLOAT_DTYPE)
    result_1 = calculate_air_density(altitude_1, rho_0_1, rho_decay_1)
    expected_1 = jnp.array(1.225, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_1, expected_1, atol=EPS)

    # Standard case 2 - half density altitude
    altitude_2 = jnp.array(5500.0, dtype=FLOAT_DTYPE)
    result_2 = calculate_air_density(altitude_2, rho_0_1, rho_decay_1)
    expected_2 = rho_0_1 / jnp.array(2.0, dtype=FLOAT_DTYPE)
    tolerance_2 = jnp.array(1e-5, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_2, expected_2, atol=tolerance_2)

    # Standard case 3 - typical altitude
    altitude_3 = jnp.array(1000.0, dtype=FLOAT_DTYPE)
    result_3 = calculate_air_density(altitude_3, rho_0_1, rho_decay_1)
    zero_val = jnp.array(0.0, dtype=FLOAT_DTYPE)
    assert result_3 < rho_0_1 and result_3 > zero_val

    # Edge case 1 - negative altitude (below sea level)
    altitude_neg = jnp.array(-100.0, dtype=FLOAT_DTYPE)
    result_4 = calculate_air_density(altitude_neg, rho_0_1, rho_decay_1)
    assert result_4 > rho_0_1

    # Edge case 2 - very high altitude
    altitude_high = jnp.array(20000.0, dtype=FLOAT_DTYPE)
    result_5 = calculate_air_density(altitude_high, rho_0_1, rho_decay_1)
    point_one = jnp.array(0.1, dtype=FLOAT_DTYPE)
    assert result_5 < point_one and result_5 > zero_val

    # Edge case 3 - zero decay rate guard
    one_val = jnp.array(1.0, dtype=FLOAT_DTYPE)
    decay_large = jnp.array(1e10, dtype=FLOAT_DTYPE)
    result_6 = calculate_air_density(altitude_3, one_val, decay_large)
    assert jnp.isclose(result_6, one_val, atol=EPS)

    # Test with vmap
    altitudes = jnp.array([0.0, 5500.0, 11000.0, -500.0], dtype=FLOAT_DTYPE)
    rho_0s = jnp.array([1.225, 1.225, 1.225, 1.225], dtype=FLOAT_DTYPE)
    rho_decays = jnp.array([5500.0, 5500.0, 5500.0, 5500.0], dtype=FLOAT_DTYPE)

    calculate_air_density_vmap = jax.vmap(calculate_air_density)
    vmap_results = calculate_air_density_vmap(altitudes, rho_0s, rho_decays)

    assert vmap_results.shape == (4,)
    assert jnp.isclose(vmap_results[0], 1.225, atol=EPS)
    assert jnp.isclose(vmap_results[1], 1.225 / 2.0, atol=EPS)
    assert jnp.isclose(vmap_results[2], 1.225 / 4.0, atol=EPS)
    assert vmap_results[3] > 1.225


def test_calculate_dynamic_pressure(jit_mode: str) -> None:
    """Test dynamic pressure calculation."""
    # Standard case 1 - zero airspeed
    zero_airspeed = jnp.array(0.0, dtype=FLOAT_DTYPE)
    rho_std = jnp.array(1.225, dtype=FLOAT_DTYPE)
    result_1 = calculate_dynamic_pressure(zero_airspeed, rho_std)
    expected_1 = jnp.array(0.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_1, expected_1, atol=EPS)

    # Standard case 2 - typical airspeed
    airspeed_50 = jnp.array(50.0, dtype=FLOAT_DTYPE)
    result_2 = calculate_dynamic_pressure(airspeed_50, rho_std)
    expected_2 = jnp.array(0.5, dtype=FLOAT_DTYPE) * rho_std * airspeed_50**2
    assert jnp.isclose(result_2, expected_2, atol=EPS)

    # Standard case 3 - unit values
    unit_airspeed = jnp.array(1.0, dtype=FLOAT_DTYPE)
    unit_density = jnp.array(1.0, dtype=FLOAT_DTYPE)
    result_3 = calculate_dynamic_pressure(unit_airspeed, unit_density)
    expected_3 = jnp.array(0.5, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_3, expected_3, atol=EPS)

    # Edge case 1 - negative airspeed (should still give positive pressure)
    airspeed_neg = jnp.array(-10.0, dtype=FLOAT_DTYPE)
    result_4 = calculate_dynamic_pressure(airspeed_neg, rho_std)
    expected_4 = (
        jnp.array(0.5, dtype=FLOAT_DTYPE)
        * rho_std
        * jnp.array(100.0, dtype=FLOAT_DTYPE)
    )
    assert jnp.isclose(result_4, expected_4, atol=EPS)

    # Edge case 2 - very high airspeed
    airspeed_high = jnp.array(1000.0, dtype=FLOAT_DTYPE)
    result_5 = calculate_dynamic_pressure(airspeed_high, rho_std)
    assert result_5 > jnp.array(0.0, dtype=FLOAT_DTYPE)

    # Edge case 3 - zero air density
    zero_density = jnp.array(0.0, dtype=FLOAT_DTYPE)
    result_6 = calculate_dynamic_pressure(airspeed_50, zero_density)
    expected_6 = jnp.array(0.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_6, expected_6, atol=EPS)

    # Test with vmap
    airspeeds = jnp.array([0.0, 10.0, 50.0, 100.0], dtype=FLOAT_DTYPE)
    air_densities = jnp.array([1.225, 1.225, 1.0, 0.5], dtype=FLOAT_DTYPE)

    calculate_dynamic_pressure_vmap = jax.vmap(calculate_dynamic_pressure)
    vmap_results = calculate_dynamic_pressure_vmap(airspeeds, air_densities)

    assert vmap_results.shape == (4,)
    assert jnp.isclose(vmap_results[0], 0.0, atol=EPS)
    assert vmap_results[1] > 0.0


def test_calculate_dynamic_pressure_scale(jit_mode: str) -> None:
    """Test dynamic pressure scaling."""
    # Standard case 1 - below half reference speed
    rho_std = jnp.array(1.225, dtype=FLOAT_DTYPE)
    speed_5 = jnp.array(5.0, dtype=FLOAT_DTYPE)
    speed_10 = jnp.array(10.0, dtype=FLOAT_DTYPE)
    q_half = calculate_dynamic_pressure(speed_5, rho_std)
    q_below_half = q_half - jnp.array(1.0, dtype=FLOAT_DTYPE)
    result_1 = calculate_dynamic_pressure_scale(q_below_half, speed_10, rho_std)
    expected_1 = jnp.array(0.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_1, expected_1, atol=EPS)

    # Standard case 2 - at reference speed
    q_ref = calculate_dynamic_pressure(speed_10, rho_std)
    result_2 = calculate_dynamic_pressure_scale(q_ref, speed_10, rho_std)
    expected_2 = jnp.array(1.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_2, expected_2, atol=EPS)

    # Standard case 3 - between half and reference
    q_half_new = calculate_dynamic_pressure(speed_5, rho_std)
    q_ref_new = calculate_dynamic_pressure(speed_10, rho_std)
    q_mid = (q_half_new + q_ref_new) / jnp.array(2.0, dtype=FLOAT_DTYPE)
    result_3 = calculate_dynamic_pressure_scale(q_mid, speed_10, rho_std)
    expected_3 = jnp.array(0.5, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_3, expected_3, atol=EPS)

    # Edge case 1 - above reference speed
    speed_20 = jnp.array(20.0, dtype=FLOAT_DTYPE)
    q_high = calculate_dynamic_pressure(speed_20, rho_std)
    result_4 = calculate_dynamic_pressure_scale(q_high, speed_10, rho_std)
    expected_4 = jnp.array(1.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_4, expected_4, atol=EPS)

    # Edge case 2 - zero dynamic pressure
    zero_q = jnp.array(0.0, dtype=FLOAT_DTYPE)
    result_5 = calculate_dynamic_pressure_scale(zero_q, speed_10, rho_std)
    expected_5 = jnp.array(0.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_5, expected_5, atol=EPS)

    # Test with vmap
    qs = jnp.array([0.0, 30.0, 120.0, 500.0], dtype=FLOAT_DTYPE)
    q_falloff_speeds = jnp.array([10.0, 10.0, 10.0, 10.0], dtype=FLOAT_DTYPE)
    air_densities = jnp.array([1.225, 1.225, 1.225, 1.225], dtype=FLOAT_DTYPE)

    calculate_dynamic_pressure_scale_vmap = jax.vmap(calculate_dynamic_pressure_scale)
    vmap_results = calculate_dynamic_pressure_scale_vmap(
        qs, q_falloff_speeds, air_densities
    )

    assert vmap_results.shape == (4,)
    assert jnp.isclose(vmap_results[0], 0.0, atol=EPS)
    assert jnp.isclose(vmap_results[3], 1.0, atol=EPS)


def test_calculate_angle_of_attack(jit_mode: str) -> None:
    """Test angle of attack calculation."""
    # Standard case 1 - level flight
    velocity = jnp.array([10.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    zero_angle = jnp.array(0.0, dtype=FLOAT_DTYPE)
    orientation = from_euler_zyx(zero_angle, zero_angle, zero_angle)
    result_1 = calculate_angle_of_attack(velocity, orientation)
    expected_1 = jnp.array(0.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_1, expected_1, atol=EPS)

    # Standard case 2 - pitched up aircraft
    velocity = jnp.array([10.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    ten_deg = jnp.deg2rad(jnp.array(10.0, dtype=FLOAT_DTYPE))
    orientation = from_euler_zyx(zero_angle, ten_deg, zero_angle)
    result_2 = calculate_angle_of_attack(velocity, orientation)
    zero_val = jnp.array(0.0, dtype=FLOAT_DTYPE)
    assert result_2 > zero_val

    # Standard case 3 - descending flight
    velocity = jnp.array([10.0, 0.0, 5.0], dtype=FLOAT_DTYPE)
    orientation = from_euler_zyx(zero_angle, zero_angle, zero_angle)
    result_3 = calculate_angle_of_attack(velocity, orientation)
    assert result_3 > zero_val

    # Edge case 1 - pure vertical velocity
    velocity = jnp.array([0.0, 0.0, 10.0], dtype=FLOAT_DTYPE)
    orientation = from_euler_zyx(zero_angle, zero_angle, zero_angle)
    result_4 = calculate_angle_of_attack(velocity, orientation)
    expected_4 = pi / jnp.array(2.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_4, expected_4, atol=EPS)

    # Edge case 2 - backward velocity
    velocity = jnp.array([-10.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    orientation = from_euler_zyx(zero_angle, zero_angle, zero_angle)
    result_5 = calculate_angle_of_attack(velocity, orientation)
    assert jnp.abs(result_5) > pi / jnp.array(2.0, dtype=FLOAT_DTYPE)

    # Test with vmap
    velocities = jnp.array(
        [
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 5.0],
            [0.0, 0.0, 10.0],
            [-10.0, 0.0, 0.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    zero_angle = jnp.array(0.0, dtype=FLOAT_DTYPE)
    orientations = jax.vmap(
        lambda _: from_euler_zyx(zero_angle, zero_angle, zero_angle)
    )(jnp.arange(4))

    calculate_angle_of_attack_vmap = jax.vmap(calculate_angle_of_attack)
    vmap_results = calculate_angle_of_attack_vmap(velocities, orientations)

    assert vmap_results.shape == (4,)
    assert jnp.isclose(vmap_results[0], zero_val, atol=EPS)


def test_calculate_angle_of_sideslip(jit_mode: str) -> None:
    """Test sideslip angle calculation."""
    # Standard case 1 - no sideslip
    zero_angle = jnp.array(0.0, dtype=FLOAT_DTYPE)
    velocity = jnp.array([10.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    orientation = from_euler_zyx(zero_angle, zero_angle, zero_angle)
    result_1 = calculate_angle_of_sideslip(velocity, orientation)
    expected_1 = jnp.array(0.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_1, expected_1, atol=EPS)

    # Standard case 2 - pure sideways velocity
    velocity = jnp.array([0.0, 10.0, 0.0], dtype=FLOAT_DTYPE)
    orientation = from_euler_zyx(zero_angle, zero_angle, zero_angle)
    result_2 = calculate_angle_of_sideslip(velocity, orientation)
    expected_2 = pi / jnp.array(2.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_2, expected_2, atol=EPS)

    # Standard case 3 - diagonal velocity
    velocity = jnp.array([10.0, 10.0, 0.0], dtype=FLOAT_DTYPE)
    orientation = from_euler_zyx(zero_angle, zero_angle, zero_angle)
    result_3 = calculate_angle_of_sideslip(velocity, orientation)
    expected_3 = pi / jnp.array(4.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_3, expected_3, atol=EPS)

    # Edge case 1 - yawed aircraft
    velocity = jnp.array([10.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    thirty_deg = jnp.deg2rad(jnp.array(30.0, dtype=FLOAT_DTYPE))
    orientation = from_euler_zyx(thirty_deg, zero_angle, zero_angle)
    result_4 = calculate_angle_of_sideslip(velocity, orientation)
    assert result_4 < jnp.array(0.0, dtype=FLOAT_DTYPE)  # Negative beta for right yaw

    # Edge case 2 - backward velocity
    velocity = jnp.array([-10.0, 5.0, 0.0], dtype=FLOAT_DTYPE)
    orientation = from_euler_zyx(zero_angle, zero_angle, zero_angle)
    result_5 = calculate_angle_of_sideslip(velocity, orientation)
    assert jnp.abs(result_5) > pi / jnp.array(2.0, dtype=FLOAT_DTYPE)

    # Test with vmap
    velocities = jnp.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [10.0, 10.0, 0.0],
            [-10.0, 5.0, 0.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    zero_angle = jnp.array(0.0, dtype=FLOAT_DTYPE)
    orientations = jax.vmap(
        lambda _: from_euler_zyx(zero_angle, zero_angle, zero_angle)
    )(jnp.arange(4))

    calculate_angle_of_sideslip_vmap = jax.vmap(calculate_angle_of_sideslip)
    vmap_results = calculate_angle_of_sideslip_vmap(velocities, orientations)

    assert vmap_results.shape == (4,)
    assert jnp.isclose(vmap_results[0], jnp.array(0.0, dtype=FLOAT_DTYPE), atol=EPS)
    assert jnp.isclose(
        vmap_results[1], pi / jnp.array(2.0, dtype=FLOAT_DTYPE), atol=EPS
    )


def test_calculate_weight(jit_mode: str) -> None:
    """Test weight force calculation."""
    # Standard case 1 - typical values
    mass = jnp.array(100.0, dtype=FLOAT_DTYPE)
    gravity = jnp.array(9.81, dtype=FLOAT_DTYPE)
    result_1 = calculate_weight(mass, gravity)
    expected_1 = jnp.array([0.0, 0.0, 981.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_1, expected_1, atol=EPS)

    # Standard case 2 - unit values
    unit_mass = jnp.array(1.0, dtype=FLOAT_DTYPE)
    unit_gravity = jnp.array(1.0, dtype=FLOAT_DTYPE)
    result_2 = calculate_weight(unit_mass, unit_gravity)
    expected_2 = jnp.array([0.0, 0.0, 1.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_2, expected_2, atol=EPS)

    # Edge case 1 - zero mass
    zero_mass = jnp.array(0.0, dtype=FLOAT_DTYPE)
    result_3 = calculate_weight(zero_mass, gravity)
    expected_3 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_3, expected_3, atol=EPS)

    # Edge case 2 - zero gravity
    zero_gravity = jnp.array(0.0, dtype=FLOAT_DTYPE)
    result_4 = calculate_weight(mass, zero_gravity)
    expected_4 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_4, expected_4, atol=EPS)

    # Edge case 3 - negative gravity (should still work)
    light_mass = jnp.array(10.0, dtype=FLOAT_DTYPE)
    negative_gravity = jnp.array(-9.81, dtype=FLOAT_DTYPE)
    result_5 = calculate_weight(light_mass, negative_gravity)
    expected_5 = jnp.array([0.0, 0.0, -98.1], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_5, expected_5, atol=EPS)

    # Test with vmap
    masses = jnp.array([0.0, 1.0, 100.0, 1000.0], dtype=FLOAT_DTYPE)
    gravities = jnp.array([9.81, 9.81, 9.81, 9.81], dtype=FLOAT_DTYPE)

    calculate_weight_vmap = jax.vmap(calculate_weight)
    vmap_results = calculate_weight_vmap(masses, gravities)

    assert vmap_results.shape == (4, 3)
    assert jnp.allclose(vmap_results[0], jnp.zeros(3), atol=EPS)
    assert jnp.isclose(vmap_results[2, 2], 981.0, atol=EPS)


def test_calculate_thrust(jit_mode: str) -> None:
    """Test thrust force calculation."""
    # Standard case 1 - full throttle at sea level
    full_throttle = jnp.array(1.0, dtype=FLOAT_DTYPE)
    max_thrust = jnp.array(1000.0, dtype=FLOAT_DTYPE)
    sea_level_density = jnp.array(1.225, dtype=FLOAT_DTYPE)
    result_1 = calculate_thrust(
        full_throttle, max_thrust, sea_level_density, sea_level_density
    )
    expected_1 = jnp.array([1000.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_1, expected_1, atol=EPS)

    # Standard case 2 - half throttle
    half_throttle = jnp.array(0.5, dtype=FLOAT_DTYPE)
    result_2 = calculate_thrust(
        half_throttle, max_thrust, sea_level_density, sea_level_density
    )
    expected_2 = jnp.array([500.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_2, expected_2, atol=EPS)

    # Standard case 3 - altitude effect
    altitude_density = jnp.array(0.6125, dtype=FLOAT_DTYPE)
    result_3 = calculate_thrust(
        full_throttle, max_thrust, altitude_density, sea_level_density
    )
    expected_3 = jnp.array([500.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_3, expected_3, atol=EPS)

    # Edge case 1 - zero throttle
    zero_throttle = jnp.array(0.0, dtype=FLOAT_DTYPE)
    result_4 = calculate_thrust(
        zero_throttle, max_thrust, sea_level_density, sea_level_density
    )
    expected_4 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_4, expected_4, atol=EPS)

    # Edge case 2 - zero air density
    zero_density = jnp.array(0.0, dtype=FLOAT_DTYPE)
    result_5 = calculate_thrust(
        full_throttle, max_thrust, zero_density, sea_level_density
    )
    expected_5 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_5, expected_5, atol=EPS)

    # Edge case 3 - over throttle (should still work)
    over_throttle = jnp.array(1.5, dtype=FLOAT_DTYPE)
    result_6 = calculate_thrust(
        over_throttle, max_thrust, sea_level_density, sea_level_density
    )
    expected_6 = jnp.array([1500.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_6, expected_6, atol=EPS)

    # Test with vmap
    throttles = jnp.array([0.0, 0.5, 1.0, 1.0], dtype=FLOAT_DTYPE)
    max_thrusts = jnp.array([1000.0, 1000.0, 1000.0, 2000.0], dtype=FLOAT_DTYPE)
    air_densities = jnp.array([1.225, 1.225, 0.6125, 1.225], dtype=FLOAT_DTYPE)
    rho_0s = jnp.array([1.225, 1.225, 1.225, 1.225], dtype=FLOAT_DTYPE)

    calculate_thrust_vmap = jax.vmap(calculate_thrust)
    vmap_results = calculate_thrust_vmap(throttles, max_thrusts, air_densities, rho_0s)

    assert vmap_results.shape == (4, 3)
    assert jnp.allclose(vmap_results[0], jnp.zeros(3), atol=EPS)
    assert jnp.isclose(vmap_results[1, 0], 500.0, atol=EPS)


def test_calculate_lift(jit_mode: str) -> None:
    """Test lift force calculation."""
    # Standard case 1 - zero angle of attack
    zero_alpha = jnp.array(0.0, dtype=FLOAT_DTYPE)
    dynamic_pressure = jnp.array(100.0, dtype=FLOAT_DTYPE)
    coef_lift = jnp.array(5.0, dtype=FLOAT_DTYPE)
    wing_area = jnp.array(20.0, dtype=FLOAT_DTYPE)
    max_angle = jnp.array(15.0, dtype=FLOAT_DTYPE)
    result_1 = calculate_lift(
        zero_alpha, dynamic_pressure, coef_lift, wing_area, max_angle
    )
    expected_1 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_1, expected_1, atol=EPS)

    # Standard case 2 - small positive angle
    small_angle_deg = jnp.array(5.0, dtype=FLOAT_DTYPE)
    alpha = jnp.deg2rad(small_angle_deg)
    result_2 = calculate_lift(alpha, dynamic_pressure, coef_lift, wing_area, max_angle)
    assert result_2[2] < 0.0  # Lift is negative (upward) in FRD

    # Standard case 3 - at max lift angle
    max_angle_deg = jnp.array(15.0, dtype=FLOAT_DTYPE)
    alpha = jnp.deg2rad(max_angle_deg)
    result_3 = calculate_lift(alpha, dynamic_pressure, coef_lift, wing_area, max_angle)
    assert result_3[2] < 0.0

    # Edge case 1 - beyond max angle (in falloff region)
    beyond_max_deg = jnp.array(20.0, dtype=FLOAT_DTYPE)
    alpha = jnp.deg2rad(beyond_max_deg)
    result_4 = calculate_lift(alpha, dynamic_pressure, coef_lift, wing_area, max_angle)
    assert jnp.abs(result_4[2]) < jnp.abs(result_3[2])

    # Edge case 2 - beyond 2x max angle (zero lift)
    zero_lift_deg = jnp.array(31.0, dtype=FLOAT_DTYPE)
    alpha = jnp.deg2rad(zero_lift_deg)
    result_5 = calculate_lift(alpha, dynamic_pressure, coef_lift, wing_area, max_angle)
    expected_5 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_5, expected_5, atol=EPS)

    # Edge case 3 - negative angle
    neg_angle_deg = jnp.array(-10.0, dtype=FLOAT_DTYPE)
    alpha = jnp.deg2rad(neg_angle_deg)
    result_6 = calculate_lift(alpha, dynamic_pressure, coef_lift, wing_area, max_angle)
    assert result_6[2] > 0.0  # Downward lift for negative alpha

    # Test with vmap
    alphas = jnp.array(
        [0.0, jnp.deg2rad(5.0), jnp.deg2rad(15.0), jnp.deg2rad(31.0)], dtype=FLOAT_DTYPE
    )
    dynamic_pressures = jnp.array([100.0, 100.0, 100.0, 100.0], dtype=FLOAT_DTYPE)
    coef_lifts = jnp.array([5.0, 5.0, 5.0, 5.0], dtype=FLOAT_DTYPE)
    wing_areas = jnp.array([20.0, 20.0, 20.0, 20.0], dtype=FLOAT_DTYPE)
    max_angles = jnp.array([15.0, 15.0, 15.0, 15.0], dtype=FLOAT_DTYPE)

    calculate_lift_vmap = jax.vmap(calculate_lift)
    vmap_results = calculate_lift_vmap(
        alphas, dynamic_pressures, coef_lifts, wing_areas, max_angles
    )

    assert vmap_results.shape == (4, 3)
    assert jnp.allclose(vmap_results[0], jnp.zeros(3), atol=EPS)
    assert jnp.allclose(vmap_results[3], jnp.zeros(3), atol=EPS)


def test_calculate_sideslip(jit_mode: str) -> None:
    """Test sideslip force calculation."""
    # Standard case 1 - zero sideslip
    zero_beta = jnp.array(0.0, dtype=FLOAT_DTYPE)
    dynamic_pressure = jnp.array(100.0, dtype=FLOAT_DTYPE)
    coef_sideslip = jnp.array(3.0, dtype=FLOAT_DTYPE)
    side_area = jnp.array(10.0, dtype=FLOAT_DTYPE)
    max_angle = jnp.array(15.0, dtype=FLOAT_DTYPE)
    result_1 = calculate_sideslip(
        zero_beta, dynamic_pressure, coef_sideslip, side_area, max_angle
    )
    expected_1 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_1, expected_1, atol=EPS)

    # Standard case 2 - small positive sideslip
    small_angle_deg = jnp.array(5.0, dtype=FLOAT_DTYPE)
    beta = jnp.deg2rad(small_angle_deg)
    result_2 = calculate_sideslip(
        beta, dynamic_pressure, coef_sideslip, side_area, max_angle
    )
    assert result_2[1] < jnp.array(
        0.0, dtype=FLOAT_DTYPE
    )  # Negative Y force for positive beta

    # Standard case 3 - at max sideslip angle
    max_angle_deg = jnp.array(15.0, dtype=FLOAT_DTYPE)
    beta = jnp.deg2rad(max_angle_deg)
    result_3 = calculate_sideslip(
        beta, dynamic_pressure, coef_sideslip, side_area, max_angle
    )
    assert result_3[1] < 0.0

    # Edge case 1 - beyond max angle (in falloff region)
    beyond_max_deg = jnp.array(20.0, dtype=FLOAT_DTYPE)
    beta = jnp.deg2rad(beyond_max_deg)
    result_4 = calculate_sideslip(
        beta, dynamic_pressure, coef_sideslip, side_area, max_angle
    )
    assert jnp.abs(result_4[1]) < jnp.abs(result_3[1])

    # Edge case 2 - beyond 2x max angle (zero force)
    zero_force_deg = jnp.array(31.0, dtype=FLOAT_DTYPE)
    beta = jnp.deg2rad(zero_force_deg)
    result_5 = calculate_sideslip(
        beta, dynamic_pressure, coef_sideslip, side_area, max_angle
    )
    expected_5 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_5, expected_5, atol=EPS)

    # Edge case 3 - negative sideslip
    neg_angle_deg = jnp.array(-10.0, dtype=FLOAT_DTYPE)
    beta = jnp.deg2rad(neg_angle_deg)
    result_6 = calculate_sideslip(
        beta, dynamic_pressure, coef_sideslip, side_area, max_angle
    )
    assert result_6[1] > 0.0  # Positive Y force for negative beta

    # Test with vmap
    betas = jnp.array(
        [0.0, jnp.deg2rad(5.0), jnp.deg2rad(15.0), jnp.deg2rad(31.0)], dtype=FLOAT_DTYPE
    )
    dynamic_pressures = jnp.array([100.0, 100.0, 100.0, 100.0], dtype=FLOAT_DTYPE)
    coef_sideslips = jnp.array([3.0, 3.0, 3.0, 3.0], dtype=FLOAT_DTYPE)
    side_areas = jnp.array([10.0, 10.0, 10.0, 10.0], dtype=FLOAT_DTYPE)
    max_angles = jnp.array([15.0, 15.0, 15.0, 15.0], dtype=FLOAT_DTYPE)

    calculate_sideslip_vmap = jax.vmap(calculate_sideslip)
    vmap_results = calculate_sideslip_vmap(
        betas, dynamic_pressures, coef_sideslips, side_areas, max_angles
    )

    assert vmap_results.shape == (4, 3)
    assert jnp.allclose(vmap_results[0], jnp.zeros(3), atol=EPS)
    assert jnp.allclose(vmap_results[3], jnp.zeros(3), atol=EPS)


def test_calculate_drag(jit_mode: str) -> None:
    """Test drag force calculation."""
    # Standard case 1 - forward velocity
    zero_angle = jnp.array(0.0, dtype=FLOAT_DTYPE)
    velocity = jnp.array([10.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    orientation = from_euler_zyx(zero_angle, zero_angle, zero_angle)
    airspeed = jnp.array(10.0, dtype=FLOAT_DTYPE)
    dynamic_pressure = jnp.array(100.0, dtype=FLOAT_DTYPE)
    coef_drag = jnp.array(0.5, dtype=FLOAT_DTYPE)
    surface_areas = jnp.array([2.0, 3.0, 4.0], dtype=FLOAT_DTYPE)
    result_1 = calculate_drag(
        velocity, orientation, airspeed, dynamic_pressure, coef_drag, surface_areas
    )
    assert result_1[0] < jnp.array(0.0, dtype=FLOAT_DTYPE)  # Drag opposes velocity
    assert jnp.allclose(result_1[1:], jnp.zeros(2), atol=EPS)

    # Standard case 2 - sideways velocity
    velocity = jnp.array([0.0, 10.0, 0.0], dtype=FLOAT_DTYPE)
    orientation = from_euler_zyx(zero_angle, zero_angle, zero_angle)
    result_2 = calculate_drag(
        velocity, orientation, airspeed, dynamic_pressure, coef_drag, surface_areas
    )
    assert result_2[1] < jnp.array(0.0, dtype=FLOAT_DTYPE)
    assert jnp.allclose(
        result_2[jnp.array([0, 2])],
        jnp.zeros(2),
        atol=EPS,
    )

    # Standard case 3 - diagonal velocity
    velocity = jnp.array([10.0, 10.0, 0.0], dtype=FLOAT_DTYPE)
    orientation = from_euler_zyx(zero_angle, zero_angle, zero_angle)
    diagonal_airspeed = jnp.sqrt(jnp.array(200.0, dtype=FLOAT_DTYPE))
    result_3 = calculate_drag(
        velocity,
        orientation,
        diagonal_airspeed,
        dynamic_pressure,
        coef_drag,
        surface_areas,
    )
    zero = jnp.array(0.0, dtype=FLOAT_DTYPE)
    assert result_3[0] < zero and result_3[1] < zero

    # Edge case 1 - zero airspeed
    zero_velocity = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    zero_airspeed = jnp.array(0.0, dtype=FLOAT_DTYPE)
    result_4 = calculate_drag(
        zero_velocity,
        orientation,
        zero_airspeed,
        dynamic_pressure,
        coef_drag,
        surface_areas,
    )
    expected_4 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_4, expected_4, atol=EPS)

    # Edge case 2 - very small airspeed
    tiny_velocity = jnp.array([EPS / 2, 0.0, 0.0], dtype=FLOAT_DTYPE)
    tiny_airspeed = EPS / jnp.array(2.0, dtype=FLOAT_DTYPE)
    result_5 = calculate_drag(
        tiny_velocity,
        orientation,
        tiny_airspeed,
        dynamic_pressure,
        coef_drag,
        surface_areas,
    )
    expected_5 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_5, expected_5, atol=EPS)

    # Test with vmap
    velocities = jnp.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [10.0, 10.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    zero_angle = jnp.array(0.0, dtype=FLOAT_DTYPE)
    orientations = jax.vmap(
        lambda _: from_euler_zyx(zero_angle, zero_angle, zero_angle)
    )(jnp.arange(4))
    airspeeds = jnp.array([10.0, 10.0, jnp.sqrt(200.0), 0.0], dtype=FLOAT_DTYPE)
    qs = jnp.array([100.0, 100.0, 100.0, 100.0], dtype=FLOAT_DTYPE)
    coef_drags = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=FLOAT_DTYPE)
    surface_areas_batch = jnp.tile(surface_areas, (4, 1))

    calculate_drag_vmap = jax.vmap(calculate_drag)
    vmap_results = calculate_drag_vmap(
        velocities, orientations, airspeeds, qs, coef_drags, surface_areas_batch
    )

    assert vmap_results.shape == (4, 3)
    assert vmap_results[0, 0] < jnp.array(0.0, dtype=FLOAT_DTYPE)
    assert jnp.allclose(vmap_results[3], jnp.zeros(3), atol=EPS)


def test_calculate_aero_forces(jit_mode: str) -> None:
    """Test combined aerodynamic forces calculation."""
    # Standard case 1 - level forward flight, no sideslip
    zero_angle = jnp.array(0.0, dtype=FLOAT_DTYPE)
    velocity = jnp.array([20.0, 0.0, 0.0], dtype=FLOAT_DTYPE)  # Relative velocity
    orientation = from_euler_zyx(zero_angle, zero_angle, zero_angle)
    coef_drag = jnp.array(0.5, dtype=FLOAT_DTYPE)
    coef_lift = jnp.array(2.0, dtype=FLOAT_DTYPE)
    coef_sideslip = jnp.array(1.0, dtype=FLOAT_DTYPE)
    max_attack_angle = jnp.array(15.0, dtype=FLOAT_DTYPE)  # degrees
    max_sideslip_angle = jnp.array(20.0, dtype=FLOAT_DTYPE)  # degrees
    surface_areas = jnp.array([3.0, 2.0, 4.0], dtype=FLOAT_DTYPE)  # front, side, top
    air_density = jnp.array(1.225, dtype=FLOAT_DTYPE)
    result_1 = calculate_aero_forces(
        velocity,
        orientation,
        air_density,
        coef_drag,
        coef_lift,
        coef_sideslip,
        max_attack_angle,
        max_sideslip_angle,
        surface_areas,
    )
    # Should only have drag in -X direction (opposes forward motion)
    zero_val = jnp.array(0.0, dtype=FLOAT_DTYPE)
    assert result_1[0] < zero_val  # Drag opposes velocity
    assert jnp.abs(result_1[1]) < jnp.abs(result_1[0])  # Minimal sideslip force
    assert jnp.abs(result_1[2]) < jnp.abs(result_1[0])  # Minimal lift at zero AoA

    # Standard case 2 - pitched up flight (positive angle of attack)
    pitch_angle = jnp.deg2rad(jnp.array(5.0, dtype=FLOAT_DTYPE))
    orientation_pitched = from_euler_zyx(zero_angle, pitch_angle, zero_angle)
    result_2 = calculate_aero_forces(
        velocity,
        orientation_pitched,
        air_density,
        coef_drag,
        coef_lift,
        coef_sideslip,
        max_attack_angle,
        max_sideslip_angle,
        surface_areas,
    )
    # Should have lift (negative Z in FRD frame) and drag
    assert result_2[0] < zero_val  # Still have drag
    assert result_2[2] < zero_val  # Lift is upward (negative Z)
    assert jnp.abs(result_2[2]) > jnp.abs(result_1[2])  # More lift than level flight

    # Standard case 3 - yawed flight (sideslip)
    yaw_angle = jnp.deg2rad(jnp.array(10.0, dtype=FLOAT_DTYPE))
    orientation_yawed = from_euler_zyx(yaw_angle, zero_angle, zero_angle)
    result_3 = calculate_aero_forces(
        velocity,
        orientation_yawed,
        air_density,
        coef_drag,
        coef_lift,
        coef_sideslip,
        max_attack_angle,
        max_sideslip_angle,
        surface_areas,
    )
    # Should have sideslip force (Y direction)
    assert result_3[0] < zero_val  # Still have drag
    assert jnp.abs(result_3[1]) > jnp.abs(
        result_1[1]
    )  # More sideslip than level flight

    # Standard case 4 - combined pitch and yaw
    orientation_combined = from_euler_zyx(yaw_angle, pitch_angle, zero_angle)
    result_4 = calculate_aero_forces(
        velocity,
        orientation_combined,
        air_density,
        coef_drag,
        coef_lift,
        coef_sideslip,
        max_attack_angle,
        max_sideslip_angle,
        surface_areas,
    )
    # Should have all three force components
    assert result_4[0] < zero_val  # Drag
    assert jnp.abs(result_4[1]) > zero_val  # Sideslip
    assert result_4[2] < zero_val  # Lift

    # Edge case 1 - zero velocity (no aerodynamic forces)
    velocity_zero = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_5 = calculate_aero_forces(
        velocity_zero,
        orientation,
        air_density,
        coef_drag,
        coef_lift,
        coef_sideslip,
        max_attack_angle,
        max_sideslip_angle,
        surface_areas,
    )
    expected_zero = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_5, expected_zero, atol=EPS)

    # Edge case 2 - very low speed (below falloff speed)
    velocity_low = jnp.array([5.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_6 = calculate_aero_forces(
        velocity_low,
        orientation_pitched,
        air_density,
        coef_drag,
        coef_lift,
        coef_sideslip,
        max_attack_angle,
        max_sideslip_angle,
        surface_areas,
    )
    # Should have reduced forces compared to higher speed
    assert jnp.linalg.norm(result_6) < jnp.linalg.norm(result_2)

    # Edge case 3 - high angle of attack (beyond max)
    high_angle_deg = jnp.array(20.0, dtype=FLOAT_DTYPE)
    high_pitch = jnp.deg2rad(high_angle_deg)  # 20 degrees, beyond max of 15
    orientation_high_aoa = from_euler_zyx(zero_angle, high_pitch, zero_angle)
    result_7 = calculate_aero_forces(
        velocity,
        orientation_high_aoa,
        air_density,
        coef_drag,
        coef_lift,
        coef_sideslip,
        max_attack_angle,
        max_sideslip_angle,
        surface_areas,
    )
    # Should have reduced lift compared to max angle but more than 5 degrees
    # At 20 deg (beyond max 15 deg), we're in falloff region, still generating significant lift
    assert jnp.abs(result_7[2]) > jnp.abs(result_2[2])  # More lift than 5 deg AoA

    # Edge case 4 - zero air density (space flight)
    zero_air_density = jnp.array(0.0, dtype=FLOAT_DTYPE)
    result_8 = calculate_aero_forces(
        velocity,
        orientation,
        zero_air_density,
        coef_drag,
        coef_lift,
        coef_sideslip,
        max_attack_angle,
        max_sideslip_angle,
        surface_areas,
    )
    assert jnp.allclose(result_8, expected_zero, atol=EPS)

    # Test with vmap - different flight conditions
    velocities = jnp.array(
        [
            [20.0, 0.0, 0.0],  # Level flight
            [0.0, 0.0, 0.0],  # Hover
            [15.0, 5.0, 0.0],  # Sideways motion
            [10.0, 0.0, 5.0],  # Descending
        ],
        dtype=FLOAT_DTYPE,
    )

    orientations = jnp.array(
        [
            from_euler_zyx(zero_angle, zero_angle, zero_angle),
            from_euler_zyx(zero_angle, zero_angle, zero_angle),
            from_euler_zyx(zero_angle, zero_angle, zero_angle),
            from_euler_zyx(zero_angle, zero_angle, zero_angle),
        ]
    )

    coef_drags = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=FLOAT_DTYPE)
    coef_lifts = jnp.array([2.0, 2.0, 2.0, 2.0], dtype=FLOAT_DTYPE)
    coef_sideslips = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=FLOAT_DTYPE)
    max_attack_angles = jnp.array([15.0, 15.0, 15.0, 15.0], dtype=FLOAT_DTYPE)
    max_sideslip_angles = jnp.array([20.0, 20.0, 20.0, 20.0], dtype=FLOAT_DTYPE)
    surface_areas_batch = jnp.tile(surface_areas, (4, 1))
    air_densities = jnp.array([1.225, 1.225, 1.225, 1.225], dtype=FLOAT_DTYPE)

    calculate_aero_forces_vmap = jax.vmap(calculate_aero_forces)
    vmap_results = calculate_aero_forces_vmap(
        velocities,
        orientations,
        air_densities,
        coef_drags,
        coef_lifts,
        coef_sideslips,
        max_attack_angles,
        max_sideslip_angles,
        surface_areas_batch,
    )

    assert vmap_results.shape == (4, 3)
    # Level flight should have negative X force (drag)
    assert vmap_results[0, 0] < zero_val
    # Hover should have near-zero forces
    assert jnp.allclose(vmap_results[1], expected_zero, atol=EPS)
    # Sideways motion should have Y component
    assert jnp.abs(vmap_results[2, 1]) > zero_val
    # Descending flight should have Z component from angle of attack
    assert jnp.abs(vmap_results[3, 2]) > zero_val


def test_calculate_control_moments(jit_mode: str) -> None:
    """Test control surface moments calculation."""
    # Standard case 1 - forward velocity, aileron input
    zero_angle = jnp.array(0.0, dtype=FLOAT_DTYPE)
    velocity = jnp.array([10.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    orientation = from_euler_zyx(zero_angle, zero_angle, zero_angle)
    airspeed = jnp.array(10.0, dtype=FLOAT_DTYPE)
    dynamic_pressure = jnp.array(100.0, dtype=FLOAT_DTYPE)
    aileron = jnp.array(1.0, dtype=FLOAT_DTYPE)
    elevator = jnp.array(0.0, dtype=FLOAT_DTYPE)
    rudder = jnp.array(0.0, dtype=FLOAT_DTYPE)
    coefs_torque = jnp.array([1.0, 2.0, 3.0], dtype=FLOAT_DTYPE)
    result_1 = calculate_control_moments(
        velocity,
        orientation,
        airspeed,
        dynamic_pressure,
        aileron,
        elevator,
        rudder,
        coefs_torque,
    )
    zero_val = jnp.array(0.0, dtype=FLOAT_DTYPE)
    assert result_1[0] > zero_val  # Positive roll moment
    assert jnp.allclose(result_1[1:], jnp.zeros(2), atol=EPS)

    # Standard case 2 - elevator input
    elevator = jnp.array(1.0, dtype=FLOAT_DTYPE)
    aileron = jnp.array(0.0, dtype=FLOAT_DTYPE)
    result_2 = calculate_control_moments(
        velocity,
        orientation,
        airspeed,
        dynamic_pressure,
        aileron,
        elevator,
        rudder,
        coefs_torque,
    )
    assert result_2[1] > zero_val  # Positive pitch moment
    assert jnp.allclose(
        result_2[jnp.array([0, 2])],
        jnp.zeros(2),
        atol=EPS,
    )

    # Standard case 3 - rudder input
    rudder = jnp.array(1.0, dtype=FLOAT_DTYPE)
    elevator = jnp.array(0.0, dtype=FLOAT_DTYPE)
    result_3 = calculate_control_moments(
        velocity,
        orientation,
        airspeed,
        dynamic_pressure,
        aileron,
        elevator,
        rudder,
        coefs_torque,
    )
    assert result_3[2] > zero_val  # Positive yaw moment
    assert jnp.allclose(result_3[:2], jnp.zeros(2), atol=EPS)

    # Edge case 1 - zero airspeed
    zero_velocity = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    zero_airspeed = jnp.array(0.0, dtype=FLOAT_DTYPE)
    aileron = jnp.array(1.0, dtype=FLOAT_DTYPE)
    result_4 = calculate_control_moments(
        zero_velocity,
        orientation,
        zero_airspeed,
        dynamic_pressure,
        aileron,
        elevator,
        rudder,
        coefs_torque,
    )
    expected_4 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_4, expected_4, atol=EPS)

    # Edge case 2 - backward velocity (reduces effectiveness)
    backward_velocity = jnp.array([-10.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_5 = calculate_control_moments(
        backward_velocity,
        orientation,
        airspeed,
        dynamic_pressure,
        aileron,
        elevator,
        rudder,
        coefs_torque,
    )
    assert result_5[0] < zero_val  # Reversed effect

    # Test with vmap
    velocities = jnp.array(
        [
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    zero_angle = jnp.array(0.0, dtype=FLOAT_DTYPE)
    orientations = jax.vmap(
        lambda _: from_euler_zyx(zero_angle, zero_angle, zero_angle)
    )(jnp.arange(4))
    airspeeds = jnp.array([10.0, 10.0, 10.0, 0.0], dtype=FLOAT_DTYPE)
    qs = jnp.array([100.0, 100.0, 100.0, 100.0], dtype=FLOAT_DTYPE)
    ailerons = jnp.array([1.0, 0.0, 0.0, 1.0], dtype=FLOAT_DTYPE)
    elevators = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    rudders = jnp.array([0.0, 0.0, 1.0, 0.0], dtype=FLOAT_DTYPE)
    coefs_torques = jnp.tile(coefs_torque, (4, 1))

    calculate_control_moments_vmap = jax.vmap(calculate_control_moments)
    vmap_results = calculate_control_moments_vmap(
        velocities,
        orientations,
        airspeeds,
        qs,
        ailerons,
        elevators,
        rudders,
        coefs_torques,
    )

    assert vmap_results.shape == (4, 3)
    assert vmap_results[0, 0] > zero_val
    assert vmap_results[1, 1] > zero_val
    assert vmap_results[2, 2] > zero_val
    assert jnp.allclose(vmap_results[3], jnp.zeros(3), atol=EPS)


def test_calculate_damping_moments(jit_mode: str) -> None:
    """Test angular damping moments calculation."""
    # Standard case 1 - roll damping
    angular_velocity = jnp.array([1.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    q = jnp.array(100.0, dtype=FLOAT_DTYPE)
    coef_damping = jnp.array(10.0, dtype=FLOAT_DTYPE)
    result_1 = calculate_damping_moments(angular_velocity, q, coef_damping)
    expected_1 = jnp.array([-1000.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_1, expected_1, atol=EPS)

    # Standard case 2 - all axis damping
    angular_velocity = jnp.array([1.0, 2.0, 3.0], dtype=FLOAT_DTYPE)
    result_2 = calculate_damping_moments(angular_velocity, q, coef_damping)
    expected_2 = jnp.array([-1000.0, -2000.0, -3000.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_2, expected_2, atol=EPS)

    # Edge case 1 - zero angular velocity
    angular_velocity = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_3 = calculate_damping_moments(angular_velocity, q, coef_damping)
    expected_3 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_3, expected_3, atol=EPS)

    # Edge case 2 - negative angular velocity
    angular_velocity = jnp.array([-1.0, -1.0, -1.0], dtype=FLOAT_DTYPE)
    result_4 = calculate_damping_moments(angular_velocity, q, coef_damping)
    expected_4 = jnp.array([1000.0, 1000.0, 1000.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_4, expected_4, atol=EPS)

    # Edge case 3 - zero dynamic pressure
    angular_velocity = jnp.array([1.0, 1.0, 1.0], dtype=FLOAT_DTYPE)
    q = jnp.array(0.0, dtype=FLOAT_DTYPE)
    result_5 = calculate_damping_moments(angular_velocity, q, coef_damping)
    expected_5 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_5, expected_5, atol=EPS)

    # Test with vmap
    angular_velocities = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    qs = jnp.array([100.0, 100.0, 100.0, 100.0], dtype=FLOAT_DTYPE)
    coef_dampings = jnp.array([10.0, 10.0, 10.0, 10.0], dtype=FLOAT_DTYPE)

    calculate_damping_moments_vmap = jax.vmap(calculate_damping_moments)
    vmap_results = calculate_damping_moments_vmap(angular_velocities, qs, coef_dampings)

    assert vmap_results.shape == (4, 3)
    assert jnp.isclose(vmap_results[0, 0], -1000.0, atol=EPS)
    assert jnp.isclose(vmap_results[1, 1], -1000.0, atol=EPS)
    assert jnp.isclose(vmap_results[2, 2], -1000.0, atol=EPS)


def test_estimate_inertia(jit_mode: str) -> None:
    """Test inertia estimation."""
    # Standard case 1 - uniform surface areas
    mass = jnp.array(100.0, dtype=FLOAT_DTYPE)
    uniform_areas = jnp.array([4.0, 4.0, 4.0], dtype=FLOAT_DTYPE)
    result_1 = estimate_inertia(mass, uniform_areas)
    expected_1 = jnp.array([200.0, 200.0, 200.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_1, expected_1, atol=EPS)

    # Standard case 2 - different surface areas
    varied_areas = jnp.array([1.0, 4.0, 9.0], dtype=FLOAT_DTYPE)
    result_2 = estimate_inertia(mass, varied_areas)
    expected_2 = jnp.array([100.0, 200.0, 300.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_2, expected_2, atol=EPS)

    # Edge case 1 - zero mass
    zero_mass = jnp.array(0.0, dtype=FLOAT_DTYPE)
    result_3 = estimate_inertia(zero_mass, uniform_areas)
    expected_3 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_3, expected_3, atol=EPS)

    # Edge case 2 - zero surface areas
    zero_areas = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_4 = estimate_inertia(mass, zero_areas)
    expected_4 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_4, expected_4, atol=EPS)

    # Edge case 3 - very small values
    unit_mass = jnp.array(1.0, dtype=FLOAT_DTYPE)
    tiny_areas = jnp.array([EPS, EPS, EPS], dtype=FLOAT_DTYPE)
    result_5 = estimate_inertia(unit_mass, tiny_areas)
    zero_val = jnp.array(0.0, dtype=FLOAT_DTYPE)
    assert jnp.all(result_5 > zero_val)

    # Test with vmap
    masses = jnp.array([100.0, 200.0, 0.0, 50.0], dtype=FLOAT_DTYPE)
    surface_areas_batch = jnp.array(
        [
            [4.0, 4.0, 4.0],
            [1.0, 4.0, 9.0],
            [4.0, 4.0, 4.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=FLOAT_DTYPE,
    )

    estimate_inertia_vmap = jax.vmap(estimate_inertia)
    vmap_results = estimate_inertia_vmap(masses, surface_areas_batch)

    assert vmap_results.shape == (4, 3)
    assert jnp.allclose(
        vmap_results[0],
        jnp.array([200.0, 200.0, 200.0]),
        atol=EPS,
    )
    assert jnp.allclose(vmap_results[2], jnp.zeros(3), atol=EPS)
