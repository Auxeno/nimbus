"""Tests for interface module."""

from dataclasses import replace

import jax
import jax.numpy as jnp

from nimbus.core.config import (
    AircraftConfig,
    MapConfig,
    PhysicsConfig,
    PIDControllerConfig,
    RouteConfig,
    TerrainConfig,
)
from nimbus.core.interface import (
    aircraft_state_derivatives,
    apply_g_limiter,
    calculate_angular_acceleration,
    calculate_g_force,
    calculate_translational_acceleration,
    next_waypoint,
    terrain_collision,
    waypoint_hit,
)
from nimbus.core.primitives import EPS, FLOAT_DTYPE, INT_DTYPE
from nimbus.core.quaternion import from_euler_zyx
from nimbus.core.scenario import InitialConditions, generate_route
from nimbus.core.state import Aircraft, Body, Controls, Meta, PIDControllerState, Wind
from nimbus.core.terrain import generate_heightmap


def test_calculate_translational_acceleration(jit_mode: str) -> None:
    """Test translational acceleration calculation."""
    # Standard case 1 - aircraft in level flight
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
        elevator=jnp.array(0.0, dtype=FLOAT_DTYPE),
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

    wind = Wind(
        mean=jnp.zeros(3, dtype=FLOAT_DTYPE),
        gust=jnp.zeros(3, dtype=FLOAT_DTYPE),
    )
    result_1 = calculate_translational_acceleration(aircraft, wind, aircraft_config, physics_config)

    # Check shape
    assert result_1.shape == (3,)
    # Should have non-zero acceleration due to gravity, thrust, and drag
    assert jnp.linalg.norm(result_1) > 0.0
    # Gravity component should be positive in Z (down in NED)
    assert result_1[2] > 0.0

    # Standard case 2 - aircraft at rest (hovering)
    body_hover = Body(
        position=jnp.array([0.0, 0.0, -1000.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
        orientation=body.orientation,
        angular_velocity=jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
    )
    aircraft_hover = Aircraft(
        meta=meta,
        body=body_hover,
        controls=controls,
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )
    result_2 = calculate_translational_acceleration(
        aircraft_hover, wind, aircraft_config, physics_config
    )

    # Should have acceleration mainly from gravity
    assert result_2[2] > 0.0  # Falling down

    # Standard case 3 - high altitude (lower air density)
    body_high = Body(
        position=jnp.array([0.0, 0.0, -10000.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([100.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
        orientation=body.orientation,
        angular_velocity=jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
    )
    aircraft_high = Aircraft(
        meta=meta,
        body=body_high,
        controls=controls,
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )
    result_3 = calculate_translational_acceleration(
        aircraft_high, wind, aircraft_config, physics_config
    )

    # At high altitude, air density is lower, so drag and lift are reduced
    assert result_3.shape == (3,)

    # Test with vmap - different throttle settings
    throttles = jnp.array([0.0, 0.5, 1.0], dtype=FLOAT_DTYPE)
    controls_batch = jax.vmap(
        lambda t: Controls(
            throttle=t,
            aileron=jnp.array(0.0, dtype=FLOAT_DTYPE),
            elevator=jnp.array(0.0, dtype=FLOAT_DTYPE),
            rudder=jnp.array(0.0, dtype=FLOAT_DTYPE),
        )
    )(throttles)
    bodies = jax.vmap(lambda _: body)(jnp.arange(3))
    metas = jax.vmap(lambda i: Meta(jnp.array(True, dtype=bool), i))(jnp.arange(3))
    g_limiter_pid_batch = jax.vmap(
        lambda _: PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        )
    )(jnp.arange(3))
    aircraft_batch = jax.vmap(Aircraft)(metas, bodies, controls_batch, g_limiter_pid_batch)

    accel_vmap = jax.vmap(
        lambda a: calculate_translational_acceleration(a, wind, aircraft_config, physics_config)
    )
    vmap_results = accel_vmap(aircraft_batch)

    assert vmap_results.shape == (3, 3)
    # Higher throttle should generally produce more forward acceleration
    # (less negative or more positive in x-direction due to thrust)


def test_calculate_angular_acceleration(jit_mode: str) -> None:
    """Test angular acceleration calculation."""
    # Standard case 1 - aircraft with control inputs
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
        aileron=jnp.array(0.5, dtype=FLOAT_DTYPE),
        elevator=jnp.array(-0.3, dtype=FLOAT_DTYPE),
        rudder=jnp.array(0.2, dtype=FLOAT_DTYPE),
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
    wind = Wind(
        mean=jnp.zeros(3, dtype=FLOAT_DTYPE),
        gust=jnp.zeros(3, dtype=FLOAT_DTYPE),
    )
    aircraft_config = AircraftConfig()
    physics_config = PhysicsConfig()

    result_1 = calculate_angular_acceleration(aircraft, wind, aircraft_config, physics_config)

    # Check shape
    assert result_1.shape == (3,)
    # Should have angular acceleration due to control inputs
    assert jnp.linalg.norm(result_1) > 0.0

    # Standard case 2 - no control inputs
    controls_neutral = Controls(
        throttle=jnp.array(0.5, dtype=FLOAT_DTYPE),
        aileron=jnp.array(0.0, dtype=FLOAT_DTYPE),
        elevator=jnp.array(0.0, dtype=FLOAT_DTYPE),
        rudder=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    aircraft_neutral = Aircraft(
        meta=meta,
        body=body,
        controls=controls_neutral,
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )
    result_2 = calculate_angular_acceleration(
        aircraft_neutral, wind, aircraft_config, physics_config
    )

    # Should have minimal angular acceleration
    assert jnp.linalg.norm(result_2) < jnp.linalg.norm(result_1)

    # Standard case 3 - with angular velocity (damping)
    body_rotating = Body(
        position=body.position,
        velocity=body.velocity,
        orientation=body.orientation,
        angular_velocity=jnp.array([0.1, 0.05, 0.02], dtype=FLOAT_DTYPE),
    )
    aircraft_rotating = Aircraft(
        meta=meta,
        body=body_rotating,
        controls=controls_neutral,
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )
    result_3 = calculate_angular_acceleration(
        aircraft_rotating, wind, aircraft_config, physics_config
    )

    # Should have damping moments opposing rotation
    assert result_3.shape == (3,)

    # Test with vmap - different control surfaces
    ailerons = jnp.array([-0.5, 0.0, 0.5], dtype=FLOAT_DTYPE)
    controls_batch = jax.vmap(
        lambda a: Controls(
            throttle=jnp.array(0.5, dtype=FLOAT_DTYPE),
            aileron=a,
            elevator=jnp.array(0.0, dtype=FLOAT_DTYPE),
            rudder=jnp.array(0.0, dtype=FLOAT_DTYPE),
        )
    )(ailerons)
    bodies = jax.vmap(lambda _: body)(jnp.arange(3))
    metas = jax.vmap(lambda i: Meta(jnp.array(True, dtype=bool), i))(jnp.arange(3))
    g_limiter_pid_batch = jax.vmap(
        lambda _: PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        )
    )(jnp.arange(3))
    aircraft_batch = jax.vmap(Aircraft)(metas, bodies, controls_batch, g_limiter_pid_batch)

    ang_accel_vmap = jax.vmap(
        lambda a: calculate_angular_acceleration(a, wind, aircraft_config, physics_config)
    )
    vmap_results = ang_accel_vmap(aircraft_batch)

    assert vmap_results.shape == (3, 3)
    # Different aileron inputs should produce different roll accelerations


def test_aircraft_state_derivatives(jit_mode: str) -> None:
    """Test complete state derivatives calculation."""
    # Standard case - aircraft in flight
    meta = Meta(
        active=jnp.array(True, dtype=bool),
        id=jnp.array(0, dtype=INT_DTYPE),
    )
    body = Body(
        position=jnp.array([100.0, 50.0, -800.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([120.0, 10.0, -5.0], dtype=FLOAT_DTYPE),
        orientation=from_euler_zyx(
            jnp.array(0.1, dtype=FLOAT_DTYPE),
            jnp.array(0.05, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
        angular_velocity=jnp.array([0.01, 0.02, 0.005], dtype=FLOAT_DTYPE),
    )
    controls = Controls(
        throttle=jnp.array(0.6, dtype=FLOAT_DTYPE),
        aileron=jnp.array(0.1, dtype=FLOAT_DTYPE),
        elevator=jnp.array(-0.05, dtype=FLOAT_DTYPE),
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

    wind = Wind(
        mean=jnp.zeros(3, dtype=FLOAT_DTYPE),
        gust=jnp.zeros(3, dtype=FLOAT_DTYPE),
    )
    dx, dv, dq, dw = aircraft_state_derivatives(aircraft, wind, aircraft_config, physics_config)

    # Check shapes
    assert dx.shape == (3,)  # Position derivative (velocity)
    assert dv.shape == (3,)  # Velocity derivative (acceleration)
    assert dq.shape == (4,)  # Quaternion derivative
    assert dw.shape == (3,)  # Angular velocity derivative (angular acceleration)

    # Position derivative should match velocity (in world frame)
    assert jnp.allclose(dx, body.velocity, atol=EPS)

    # Should have non-zero derivatives
    assert jnp.linalg.norm(dv) > 0.0  # Acceleration
    assert jnp.linalg.norm(dq) > 0.0  # Quaternion change
    assert jnp.linalg.norm(dw) > 0.0  # Angular acceleration

    # Test with vmap - batch of aircraft
    positions = jnp.array(
        [
            [0.0, 0.0, -100.0],
            [100.0, 100.0, -500.0],
            [200.0, -50.0, -1000.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    velocities = jnp.array(
        [
            [50.0, 0.0, 0.0],
            [100.0, 10.0, -5.0],
            [150.0, -20.0, 10.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    zero_angle = jnp.array(0.0, dtype=FLOAT_DTYPE)
    orientations = jax.vmap(lambda _: from_euler_zyx(zero_angle, zero_angle, zero_angle))(
        jnp.arange(3)
    )
    angular_velocities = jnp.zeros((3, 3), dtype=FLOAT_DTYPE)
    bodies = jax.vmap(Body)(positions, velocities, orientations, angular_velocities)
    metas = jax.vmap(lambda i: Meta(jnp.array(True, dtype=bool), i))(jnp.arange(3))
    controls_batch = jax.vmap(lambda _: Controls.default())(jnp.arange(3))
    g_limiter_pid_batch = jax.vmap(
        lambda _: PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        )
    )(jnp.arange(3))
    aircraft_batch = jax.vmap(Aircraft)(metas, bodies, controls_batch, g_limiter_pid_batch)

    derivatives_vmap = jax.vmap(
        lambda a: aircraft_state_derivatives(a, wind, aircraft_config, physics_config)
    )
    dx_batch, dv_batch, dq_batch, dw_batch = derivatives_vmap(aircraft_batch)

    assert dx_batch.shape == (3, 3)
    assert dv_batch.shape == (3, 3)
    assert dq_batch.shape == (3, 4)
    assert dw_batch.shape == (3, 3)


def test_terrain_collision(jit_mode: str) -> None:
    """Test terrain collision detection."""
    key = jax.random.PRNGKey(42)
    terrain_config = TerrainConfig(resolution=64)
    heightmap = generate_heightmap(
        key,
        resolution=jnp.array(terrain_config.resolution, dtype=INT_DTYPE),
        base_scale=jnp.array(terrain_config.base_scale, dtype=FLOAT_DTYPE),
        octaves=jnp.array(terrain_config.octaves, dtype=INT_DTYPE),
        persistence=jnp.array(terrain_config.persistence, dtype=FLOAT_DTYPE),
        lacunarity=jnp.array(terrain_config.lacunarity, dtype=FLOAT_DTYPE),
        mountain_gain=jnp.array(terrain_config.mountain_gain, dtype=FLOAT_DTYPE),
        bump_gain=jnp.array(terrain_config.bump_gain, dtype=FLOAT_DTYPE),
        padding=jnp.array(terrain_config.padding, dtype=INT_DTYPE),
    )
    map_config = MapConfig()

    # Standard case 1 - aircraft above terrain
    meta = Meta(
        active=jnp.array(True, dtype=bool),
        id=jnp.array(0, dtype=INT_DTYPE),
    )
    body_above = Body(
        position=jnp.array([0.0, 0.0, -500.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([50.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
        orientation=from_euler_zyx(
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
        angular_velocity=jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
    )
    aircraft_above = Aircraft(
        meta=meta,
        body=body_above,
        controls=Controls.default(),
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )

    result_1 = terrain_collision(aircraft_above, heightmap, map_config)
    # Should not be colliding at -500m altitude
    assert result_1 == jnp.array(False, dtype=bool)

    # Standard case 2 - aircraft underground
    body_underground = Body(
        position=jnp.array([0.0, 0.0, 1000.0], dtype=FLOAT_DTYPE),  # Deep underground
        velocity=jnp.array([50.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
        orientation=body_above.orientation,
        angular_velocity=jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
    )
    aircraft_underground = Aircraft(
        meta=meta,
        body=body_underground,
        controls=Controls.default(),
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )

    result_2 = terrain_collision(aircraft_underground, heightmap, map_config)
    # Should be colliding when underground
    assert result_2 == jnp.array(True, dtype=bool)

    # Standard case 3 - aircraft at edge of map
    body_edge = Body(
        position=jnp.array([4900.0, 4900.0, -500.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([50.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
        orientation=body_above.orientation,
        angular_velocity=jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
    )
    aircraft_edge = Aircraft(
        meta=meta,
        body=body_edge,
        controls=Controls.default(),
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )

    result_3 = terrain_collision(aircraft_edge, heightmap, map_config)
    # Should handle edge cases properly
    assert isinstance(result_3, jnp.ndarray)
    assert result_3.dtype == bool

    # Test with vmap - different altitudes
    altitudes = jnp.array([-1000.0, -100.0, 100.0, 1000.0], dtype=FLOAT_DTYPE)
    positions = jnp.stack(
        [
            jnp.zeros(4, dtype=FLOAT_DTYPE),
            jnp.zeros(4, dtype=FLOAT_DTYPE),
            altitudes,
        ],
        axis=1,
    )
    velocities = jnp.tile(jnp.array([50.0, 0.0, 0.0], dtype=FLOAT_DTYPE), (4, 1))
    zero_angle = jnp.array(0.0, dtype=FLOAT_DTYPE)
    orientations = jax.vmap(lambda _: from_euler_zyx(zero_angle, zero_angle, zero_angle))(
        jnp.arange(4)
    )
    angular_velocities = jnp.zeros((4, 3), dtype=FLOAT_DTYPE)
    bodies = jax.vmap(Body)(positions, velocities, orientations, angular_velocities)
    metas = jax.vmap(lambda i: Meta(jnp.array(True, dtype=bool), i))(jnp.arange(4))
    controls_batch = jax.vmap(lambda _: Controls.default())(jnp.arange(4))
    g_limiter_pid_batch = jax.vmap(
        lambda _: PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        )
    )(jnp.arange(4))
    aircraft_batch = jax.vmap(Aircraft)(metas, bodies, controls_batch, g_limiter_pid_batch)

    collision_vmap = jax.vmap(lambda a: terrain_collision(a, heightmap, map_config))
    vmap_results = collision_vmap(aircraft_batch)

    assert vmap_results.shape == (4,)
    # Higher aircraft should not collide, lower ones might
    assert vmap_results[0] == jnp.array(False, dtype=bool)  # -1000m should be safe
    assert vmap_results[3] == jnp.array(True, dtype=bool)  # 1000m underground will hit


def test_calculate_g_force(jit_mode: str) -> None:
    """Test G-force calculation."""
    # Standard case 1 - level flight
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
        elevator=jnp.array(0.0, dtype=FLOAT_DTYPE),
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

    wind = Wind(
        mean=jnp.zeros(3, dtype=FLOAT_DTYPE),
        gust=jnp.zeros(3, dtype=FLOAT_DTYPE),
    )
    result_1 = calculate_g_force(aircraft, wind, aircraft_config, physics_config)

    # Check shape
    assert result_1.shape == (3,)
    # Should experience some G-force
    assert jnp.linalg.norm(result_1) > 0.0

    # Standard case 2 - high-G maneuver
    controls_high_g = Controls(
        throttle=jnp.array(1.0, dtype=FLOAT_DTYPE),
        aileron=jnp.array(0.0, dtype=FLOAT_DTYPE),
        elevator=jnp.array(-1.0, dtype=FLOAT_DTYPE),  # Full up elevator
        rudder=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    aircraft_high_g = Aircraft(
        meta=meta,
        body=body,
        controls=controls_high_g,
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )
    result_2 = calculate_g_force(aircraft_high_g, wind, aircraft_config, physics_config)

    # Should experience higher G-force with control inputs
    assert jnp.linalg.norm(result_2) != jnp.linalg.norm(result_1)

    # Test with vmap
    throttles = jnp.array([0.0, 0.5, 1.0], dtype=FLOAT_DTYPE)
    controls_batch = jax.vmap(
        lambda t: Controls(
            throttle=t,
            aileron=jnp.array(0.0, dtype=FLOAT_DTYPE),
            elevator=jnp.array(0.0, dtype=FLOAT_DTYPE),
            rudder=jnp.array(0.0, dtype=FLOAT_DTYPE),
        )
    )(throttles)
    bodies = jax.vmap(lambda _: body)(jnp.arange(3))
    metas = jax.vmap(lambda i: Meta(jnp.array(True, dtype=bool), i))(jnp.arange(3))
    g_limiter_pid_batch = jax.vmap(
        lambda _: PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        )
    )(jnp.arange(3))
    aircraft_batch = jax.vmap(Aircraft)(metas, bodies, controls_batch, g_limiter_pid_batch)

    g_force_vmap = jax.vmap(lambda a: calculate_g_force(a, wind, aircraft_config, physics_config))
    vmap_results = g_force_vmap(aircraft_batch)

    assert vmap_results.shape == (3, 3)


def test_waypoint_hit(jit_mode: str) -> None:
    """Test waypoint hit detection."""
    route_config = RouteConfig()

    # Standard case 1 - aircraft at waypoint
    meta = Meta(
        active=jnp.array(True, dtype=bool),
        id=jnp.array(0, dtype=INT_DTYPE),
    )
    body = Body(
        position=jnp.array([1500.0, 0.0, -500.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([50.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
        orientation=from_euler_zyx(
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
        angular_velocity=jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
    )
    aircraft = Aircraft(
        meta=meta,
        body=body,
        controls=Controls.default(),
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )
    key = jax.random.PRNGKey(42)
    initial_conditions = InitialConditions.default()
    route = generate_route(key, initial_conditions)

    result_1 = waypoint_hit(aircraft, route, route_config)
    # Should hit the first waypoint at [1500, 0, -500]
    assert result_1 == jnp.array(True, dtype=bool)

    # Standard case 2 - aircraft far from waypoint
    body_far = Body(
        position=jnp.array([0.0, 0.0, -500.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([50.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
        orientation=body.orientation,
        angular_velocity=jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
    )
    aircraft_far = Aircraft(
        meta=meta,
        body=body_far,
        controls=Controls.default(),
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )

    result_2 = waypoint_hit(aircraft_far, route, route_config)
    # Should not hit waypoint when far away
    assert result_2 == jnp.array(False, dtype=bool)

    # Standard case 3 - waypoint already visited
    route_visited = replace(route, visited=route.visited.at[0].set(True))
    result_3 = waypoint_hit(aircraft, route_visited, route_config)
    # Should not hit already visited waypoint
    assert result_3 == jnp.array(False, dtype=bool)

    # Standard case 4 - near but outside radius
    body_near = Body(
        position=jnp.array([1500.0 + route_config.radius + 10.0, 0.0, -500.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([50.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
        orientation=body.orientation,
        angular_velocity=jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
    )
    aircraft_near = Aircraft(
        meta=meta,
        body=body_near,
        controls=Controls.default(),
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )

    result_4 = waypoint_hit(aircraft_near, route, route_config)
    # Should not hit when just outside radius
    assert result_4 == jnp.array(False, dtype=bool)


def test_next_waypoint(jit_mode: str) -> None:
    """Test waypoint advancement."""
    # Standard case 1 - advance to next waypoint
    key = jax.random.PRNGKey(42)
    initial_conditions = InitialConditions.default()
    route = generate_route(key, initial_conditions)
    loop = jnp.array(False, dtype=bool)

    result_1 = next_waypoint(route, loop)
    # Should mark first waypoint as visited
    assert result_1.visited[0] == jnp.array(True, dtype=bool)
    # Should advance to second waypoint
    assert result_1.current_idx == 1
    # Other waypoints should remain unvisited
    assert result_1.visited[1] == jnp.array(False, dtype=bool)

    # Standard case 2 - advance through all waypoints
    current_route = route
    for _ in range(route.positions.shape[0]):
        current_route = next_waypoint(current_route, loop)

    # All waypoints should be visited
    assert jnp.all(current_route.visited)
    # Should be at index 0 (wrapped around)
    assert current_route.current_idx == 0

    # Standard case 3 - looping enabled
    route_last = replace(
        route,
        current_idx=jnp.array(route.positions.shape[0] - 1, dtype=INT_DTYPE),
        visited=jnp.ones(route.positions.shape[0], dtype=bool).at[-1].set(False),
    )
    loop_true = jnp.array(True, dtype=bool)

    result_3 = next_waypoint(route_last, loop_true)
    # Should loop back to start
    assert result_3.current_idx == 0
    # All waypoints should be reset to unvisited
    assert jnp.all(~result_3.visited)

    # Standard case 4 - no looping at end
    result_4 = next_waypoint(route_last, loop)
    # Should wrap to 0 but keep visited status
    assert result_4.current_idx == 0
    assert result_4.visited[-1] == jnp.array(True, dtype=bool)

    # Test with vmap - multiple routes
    keys = jax.random.split(jax.random.PRNGKey(42), 3)
    initial_conditions = InitialConditions.default()
    routes = jax.vmap(lambda k: generate_route(k, initial_conditions))(keys)
    loops = jnp.array([False, True, False], dtype=bool)

    next_vmap = jax.vmap(next_waypoint)
    vmap_results = next_vmap(routes, loops)

    assert vmap_results.current_idx.shape == (3,)
    assert vmap_results.visited.shape == (3, routes.positions.shape[1])
    # All should have first waypoint visited
    assert jnp.all(vmap_results.visited[:, 0])


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
    wind = Wind(
        mean=jnp.zeros(3, dtype=FLOAT_DTYPE),
        gust=jnp.zeros(3, dtype=FLOAT_DTYPE),
    )

    adjusted_controls, new_pd_state = apply_g_limiter(
        aircraft, controls, wind, aircraft_config, physics_config, dt
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
        wind,
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
    wind = Wind(
        mean=jnp.zeros(3, dtype=FLOAT_DTYPE),
        gust=jnp.zeros(3, dtype=FLOAT_DTYPE),
    )

    adjusted_controls, new_pd_state = apply_g_limiter(
        aircraft, controls, wind, aircraft_config, physics_config, dt
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
        angular_velocity=jnp.array([0.0, 2.0, 0.0], dtype=FLOAT_DTYPE),  # High pitch rate
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
        aircraft_extreme, controls, wind, aircraft_config, physics_config, dt
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
        angular_velocity=jnp.array([0.0, -1.5, 0.0], dtype=FLOAT_DTYPE),  # Pitching down
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
    wind = Wind(
        mean=jnp.zeros(3, dtype=FLOAT_DTYPE),
        gust=jnp.zeros(3, dtype=FLOAT_DTYPE),
    )

    adjusted_controls, new_pd_state = apply_g_limiter(
        aircraft, controls, wind, aircraft_config, physics_config, dt
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
        angular_velocity=jnp.array([0.0, 2.0, 0.0], dtype=FLOAT_DTYPE),  # High pitch rate
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
    wind = Wind(
        mean=jnp.zeros(3, dtype=FLOAT_DTYPE),
        gust=jnp.zeros(3, dtype=FLOAT_DTYPE),
    )

    controls_low_kp, _ = apply_g_limiter(
        aircraft, controls, wind, config_low_kp, physics_config, dt
    )
    controls_high_kp, _ = apply_g_limiter(
        aircraft, controls, wind, config_high_kp, physics_config, dt
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
        aircraft_with_error, controls, wind, config_with_kd, physics_config, dt
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
        wind,
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
    wind = Wind(
        mean=jnp.zeros(3, dtype=FLOAT_DTYPE),
        gust=jnp.zeros(3, dtype=FLOAT_DTYPE),
    )

    adjusted_controls, _ = apply_g_limiter(
        aircraft_high, controls_high, wind, aircraft_config, physics_config, dt
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
        aircraft_low, controls_low, wind, aircraft_config, physics_config, dt
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
    wind = Wind(
        mean=jnp.zeros(3, dtype=FLOAT_DTYPE),
        gust=jnp.zeros(3, dtype=FLOAT_DTYPE),
    )

    # Very small timestep
    dt_small = jnp.array(1e-6, dtype=FLOAT_DTYPE)
    adjusted_controls_small, new_pd_state_small = apply_g_limiter(
        aircraft, controls, wind, aircraft_config, physics_config, dt_small
    )

    # Should still produce valid results
    assert jnp.isfinite(adjusted_controls_small.elevator)
    assert jnp.isfinite(new_pd_state_small.previous_error)

    # Standard case 2 - large timestep
    dt_large = jnp.array(0.1, dtype=FLOAT_DTYPE)
    adjusted_controls_large, new_pd_state_large = apply_g_limiter(
        aircraft, controls, wind, aircraft_config, physics_config, dt_large
    )

    # Should remain stable and bounded
    assert jnp.abs(adjusted_controls_large.elevator) <= 1.0 + EPS
    assert jnp.isfinite(new_pd_state_large.previous_error)

    # Edge case - normal timestep for comparison
    dt_normal = jnp.array(0.01, dtype=FLOAT_DTYPE)
    adjusted_controls_normal, new_pd_state_normal = apply_g_limiter(
        aircraft, controls, wind, aircraft_config, physics_config, dt_normal
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
    wind = Wind(
        mean=jnp.zeros(3, dtype=FLOAT_DTYPE),
        gust=jnp.zeros(3, dtype=FLOAT_DTYPE),
    )

    fighter_controls, _ = apply_g_limiter(
        aircraft, controls, wind, fighter_config, physics_config, dt
    )
    transport_controls, _ = apply_g_limiter(
        aircraft, controls, wind, transport_config, physics_config, dt
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
        aircraft, controls, wind, aerobatic_config, physics_config, dt
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
    wind = Wind(
        mean=jnp.zeros(3, dtype=FLOAT_DTYPE),
        gust=jnp.zeros(3, dtype=FLOAT_DTYPE),
    )

    # Simulate 5 time steps
    current_aircraft = aircraft
    for _ in range(5):
        adjusted_controls, new_pd_state = apply_g_limiter(
            current_aircraft,
            controls,
            wind,
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
    angular_vels = jnp.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.5, 0.0]], dtype=FLOAT_DTYPE)
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
    wind = Wind(
        mean=jnp.zeros(3, dtype=FLOAT_DTYPE),
        gust=jnp.zeros(3, dtype=FLOAT_DTYPE),
    )

    # Apply vmap
    vmapped_g_limiter = jax.vmap(apply_g_limiter, in_axes=(0, 0, None, None, None, None))

    batched_adjusted_controls, batched_pd_states = vmapped_g_limiter(
        aircraft_batch,
        controls_batch,
        wind,
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
        angular_velocity=jnp.array([0.0, 10.0, 0.0], dtype=FLOAT_DTYPE),  # Very high pitch rate
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
    wind = Wind(
        mean=jnp.zeros(3, dtype=FLOAT_DTYPE),
        gust=jnp.zeros(3, dtype=FLOAT_DTYPE),
    )

    adjusted_controls, new_pd_state = apply_g_limiter(
        aircraft, controls, wind, aircraft_config, physics_config, dt
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
        aircraft_slow, controls, wind, aircraft_config, physics_config, dt
    )

    # Should handle low velocity gracefully
    assert jnp.isfinite(adjusted_controls_2.elevator)
    assert jnp.isfinite(new_pd_state_2.previous_error)
