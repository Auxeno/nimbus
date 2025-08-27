"""Tests for interface module."""

from dataclasses import replace

import jax
import jax.numpy as jnp

from nimbus.core.config import (
    AircraftConfig,
    MapConfig,
    PhysicsConfig,
    RouteConfig,
    TerrainConfig,
)
from nimbus.core.interface import (
    aircraft_state_derivatives,
    calculate_angular_acceleration,
    calculate_g_force,
    calculate_translational_acceleration,
    next_waypoint,
    terrain_collision,
    waypoint_hit,
)
from nimbus.core.primitives import EPS, FLOAT_DTYPE, INT_DTYPE
from nimbus.core.quaternion import from_euler_zyx
from nimbus.core.scenario import ScenarioConfig, generate_route
from nimbus.core.state import Aircraft, Body, Controls, Meta, PDControllerState
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
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
        ),
    )
    aircraft_config = AircraftConfig()
    physics_config = PhysicsConfig()

    result_1 = calculate_translational_acceleration(
        aircraft, aircraft_config, physics_config
    )

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
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
        ),
    )
    result_2 = calculate_translational_acceleration(
        aircraft_hover, aircraft_config, physics_config
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
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
        ),
    )
    result_3 = calculate_translational_acceleration(
        aircraft_high, aircraft_config, physics_config
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
    g_limiter_pd_batch = jax.vmap(
        lambda _: PDControllerState(previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE))
    )(jnp.arange(3))
    aircraft_batch = jax.vmap(Aircraft)(
        metas, bodies, controls_batch, g_limiter_pd_batch
    )

    accel_vmap = jax.vmap(
        lambda a: calculate_translational_acceleration(
            a, aircraft_config, physics_config
        )
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
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
        ),
    )
    aircraft_config = AircraftConfig()
    physics_config = PhysicsConfig()

    result_1 = calculate_angular_acceleration(aircraft, aircraft_config, physics_config)

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
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
        ),
    )
    result_2 = calculate_angular_acceleration(
        aircraft_neutral, aircraft_config, physics_config
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
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
        ),
    )
    result_3 = calculate_angular_acceleration(
        aircraft_rotating, aircraft_config, physics_config
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
    g_limiter_pd_batch = jax.vmap(
        lambda _: PDControllerState(previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE))
    )(jnp.arange(3))
    aircraft_batch = jax.vmap(Aircraft)(
        metas, bodies, controls_batch, g_limiter_pd_batch
    )

    ang_accel_vmap = jax.vmap(
        lambda a: calculate_angular_acceleration(a, aircraft_config, physics_config)
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
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
        ),
    )
    aircraft_config = AircraftConfig()
    physics_config = PhysicsConfig()

    dx, dv, dq, dw = aircraft_state_derivatives(
        aircraft, aircraft_config, physics_config
    )

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
    orientations = jax.vmap(
        lambda _: from_euler_zyx(zero_angle, zero_angle, zero_angle)
    )(jnp.arange(3))
    angular_velocities = jnp.zeros((3, 3), dtype=FLOAT_DTYPE)
    bodies = jax.vmap(Body)(positions, velocities, orientations, angular_velocities)
    metas = jax.vmap(lambda i: Meta(jnp.array(True, dtype=bool), i))(jnp.arange(3))
    controls_batch = jax.vmap(lambda _: Controls.default())(jnp.arange(3))
    g_limiter_pd_batch = jax.vmap(
        lambda _: PDControllerState(previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE))
    )(jnp.arange(3))
    aircraft_batch = jax.vmap(Aircraft)(
        metas, bodies, controls_batch, g_limiter_pd_batch
    )

    derivatives_vmap = jax.vmap(
        lambda a: aircraft_state_derivatives(a, aircraft_config, physics_config)
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
        resolution=terrain_config.resolution,
        base_scale=terrain_config.base_scale,
        octaves=terrain_config.octaves,
        persistence=terrain_config.persistence,
        lacunarity=terrain_config.lacunarity,
        mountain_gain=terrain_config.mountain_gain,
        bump_gain=terrain_config.bump_gain,
        padding=terrain_config.padding,
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
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
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
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
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
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
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
    orientations = jax.vmap(
        lambda _: from_euler_zyx(zero_angle, zero_angle, zero_angle)
    )(jnp.arange(4))
    angular_velocities = jnp.zeros((4, 3), dtype=FLOAT_DTYPE)
    bodies = jax.vmap(Body)(positions, velocities, orientations, angular_velocities)
    metas = jax.vmap(lambda i: Meta(jnp.array(True, dtype=bool), i))(jnp.arange(4))
    controls_batch = jax.vmap(lambda _: Controls.default())(jnp.arange(4))
    g_limiter_pd_batch = jax.vmap(
        lambda _: PDControllerState(previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE))
    )(jnp.arange(4))
    aircraft_batch = jax.vmap(Aircraft)(
        metas, bodies, controls_batch, g_limiter_pd_batch
    )

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
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
        ),
    )
    aircraft_config = AircraftConfig()
    physics_config = PhysicsConfig()

    result_1 = calculate_g_force(aircraft, aircraft_config, physics_config)

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
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
        ),
    )
    result_2 = calculate_g_force(aircraft_high_g, aircraft_config, physics_config)

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
    g_limiter_pd_batch = jax.vmap(
        lambda _: PDControllerState(previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE))
    )(jnp.arange(3))
    aircraft_batch = jax.vmap(Aircraft)(
        metas, bodies, controls_batch, g_limiter_pd_batch
    )

    g_force_vmap = jax.vmap(
        lambda a: calculate_g_force(a, aircraft_config, physics_config)
    )
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
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
        ),
    )
    key = jax.random.PRNGKey(42)
    scenario_config = ScenarioConfig()
    route = generate_route(key, scenario_config)

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
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
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
        position=jnp.array(
            [1500.0 + route_config.radius + 10.0, 0.0, -500.0], dtype=FLOAT_DTYPE
        ),
        velocity=jnp.array([50.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
        orientation=body.orientation,
        angular_velocity=jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
    )
    aircraft_near = Aircraft(
        meta=meta,
        body=body_near,
        controls=Controls.default(),
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
        ),
    )

    result_4 = waypoint_hit(aircraft_near, route, route_config)
    # Should not hit when just outside radius
    assert result_4 == jnp.array(False, dtype=bool)


def test_next_waypoint(jit_mode: str) -> None:
    """Test waypoint advancement."""
    # Standard case 1 - advance to next waypoint
    key = jax.random.PRNGKey(42)
    scenario_config = ScenarioConfig()
    route = generate_route(key, scenario_config)
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
    for i in range(route.positions.shape[0]):
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
    scenario_config = ScenarioConfig()
    routes = jax.vmap(lambda k: generate_route(k, scenario_config))(keys)
    loops = jnp.array([False, True, False], dtype=bool)

    next_vmap = jax.vmap(next_waypoint)
    vmap_results = next_vmap(routes, loops)

    assert vmap_results.current_idx.shape == (3,)
    assert vmap_results.visited.shape == (3, routes.positions.shape[1])
    # All should have first waypoint visited
    assert jnp.all(vmap_results.visited[:, 0])
