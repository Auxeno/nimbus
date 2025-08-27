"""Tests for simulation module."""

import jax
import jax.numpy as jnp

from nimbus.core.config import (
    AircraftConfig,
    PhysicsConfig,
    SimulationConfig,
    TerrainConfig,
)
from nimbus.core.primitives import EPS, FLOAT_DTYPE, INT_DTYPE
from nimbus.core.quaternion import from_euler_zyx
from nimbus.core.scenario import ScenarioConfig, generate_route
from nimbus.core.simulation import (
    freeze_aircraft,
    set_controls,
    step,
    step_aircraft_euler,
    step_aircraft_rk4,
)
from nimbus.core.state import (
    Aircraft,
    Body,
    Controls,
    Meta,
    PDControllerState,
    Simulation,
)
from nimbus.core.terrain import generate_heightmap

pi = jnp.array(jnp.pi, dtype=FLOAT_DTYPE)


def test_set_controls(jit_mode: str) -> None:
    """Test control update for simulation."""
    # Standard case 1 - update all controls
    meta = Meta(
        active=jnp.array(True, dtype=bool),
        id=jnp.array(0, dtype=INT_DTYPE),
    )
    body = Body(
        position=jnp.array([0.0, 0.0, 100.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([50.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
        orientation=from_euler_zyx(
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
        angular_velocity=jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
    )
    old_controls = Controls(
        throttle=jnp.array(0.5, dtype=FLOAT_DTYPE),
        aileron=jnp.array(0.0, dtype=FLOAT_DTYPE),
        elevator=jnp.array(0.0, dtype=FLOAT_DTYPE),
        rudder=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    aircraft = Aircraft(
        meta=meta,
        body=body,
        controls=old_controls,
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
        ),
    )
    simulation = Simulation(
        aircraft=aircraft,
        time=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    new_controls = Controls(
        throttle=jnp.array(1.0, dtype=FLOAT_DTYPE),
        aileron=jnp.array(0.5, dtype=FLOAT_DTYPE),
        elevator=jnp.array(-0.3, dtype=FLOAT_DTYPE),
        rudder=jnp.array(0.1, dtype=FLOAT_DTYPE),
    )
    result_1 = set_controls(simulation, new_controls)
    assert jnp.isclose(result_1.aircraft.controls.throttle, 1.0, atol=EPS)
    assert jnp.isclose(result_1.aircraft.controls.aileron, 0.5, atol=EPS)
    assert jnp.isclose(result_1.aircraft.controls.elevator, -0.3, atol=EPS)
    assert jnp.isclose(result_1.aircraft.controls.rudder, 0.1, atol=EPS)
    # Body and meta should remain unchanged
    assert jnp.allclose(result_1.aircraft.body.position, body.position, atol=EPS)
    assert result_1.aircraft.meta.active == meta.active

    # Standard case 2 - partial control update
    partial_controls = Controls(
        throttle=jnp.array(0.75, dtype=FLOAT_DTYPE),
        aileron=jnp.array(0.0, dtype=FLOAT_DTYPE),
        elevator=jnp.array(0.0, dtype=FLOAT_DTYPE),
        rudder=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    result_2 = set_controls(simulation, partial_controls)
    assert jnp.isclose(result_2.aircraft.controls.throttle, 0.75, atol=EPS)
    assert jnp.isclose(result_2.aircraft.controls.aileron, 0.0, atol=EPS)

    # Edge case 1 - extreme control values
    extreme_controls = Controls(
        throttle=jnp.array(0.0, dtype=FLOAT_DTYPE),
        aileron=jnp.array(1.0, dtype=FLOAT_DTYPE),
        elevator=jnp.array(-1.0, dtype=FLOAT_DTYPE),
        rudder=jnp.array(1.0, dtype=FLOAT_DTYPE),
    )
    result_3 = set_controls(simulation, extreme_controls)
    assert jnp.isclose(result_3.aircraft.controls.throttle, 0.0, atol=EPS)
    assert jnp.isclose(result_3.aircraft.controls.aileron, 1.0, atol=EPS)
    assert jnp.isclose(result_3.aircraft.controls.elevator, -1.0, atol=EPS)
    assert jnp.isclose(result_3.aircraft.controls.rudder, 1.0, atol=EPS)

    # Test with vmap
    simulations = jax.vmap(lambda _: simulation)(jnp.arange(3))
    control_throttles = jnp.array([0.25, 0.5, 0.75], dtype=FLOAT_DTYPE)
    control_ailerons = jnp.array([0.0, 0.5, -0.5], dtype=FLOAT_DTYPE)
    control_elevators = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    control_rudders = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    controls_batch = jax.vmap(Controls)(
        control_throttles, control_ailerons, control_elevators, control_rudders
    )

    set_controls_vmap = jax.vmap(set_controls)
    vmap_results = set_controls_vmap(simulations, controls_batch)

    assert vmap_results.aircraft.controls.throttle.shape == (3,)
    assert jnp.isclose(vmap_results.aircraft.controls.throttle[0], 0.25, atol=EPS)
    assert jnp.isclose(vmap_results.aircraft.controls.throttle[2], 0.75, atol=EPS)
    assert jnp.isclose(vmap_results.aircraft.controls.aileron[1], 0.5, atol=EPS)


def test_freeze_aircraft(jit_mode: str) -> None:
    """Test aircraft freezing (zero velocities)."""
    # Standard case 1 - freeze moving aircraft
    meta = Meta(
        active=jnp.array(True, dtype=bool),
        id=jnp.array(1, dtype=INT_DTYPE),
    )
    body_moving = Body(
        position=jnp.array([100.0, 200.0, 300.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([50.0, 10.0, -5.0], dtype=FLOAT_DTYPE),
        orientation=from_euler_zyx(
            jnp.array(0.1, dtype=FLOAT_DTYPE),
            jnp.array(0.2, dtype=FLOAT_DTYPE),
            jnp.array(0.3, dtype=FLOAT_DTYPE),
        ),
        angular_velocity=jnp.array([0.5, -0.3, 0.1], dtype=FLOAT_DTYPE),
    )
    controls = Controls.default()
    aircraft = Aircraft(
        meta=meta,
        body=body_moving,
        controls=controls,
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
        ),
    )
    result_1 = freeze_aircraft(aircraft)
    # Position and orientation should remain unchanged
    assert jnp.allclose(result_1.body.position, body_moving.position, atol=EPS)
    assert jnp.allclose(result_1.body.orientation, body_moving.orientation, atol=EPS)
    # Velocities should be zero
    expected_zero_vel = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_1.body.velocity, expected_zero_vel, atol=EPS)
    assert jnp.allclose(result_1.body.angular_velocity, expected_zero_vel, atol=EPS)
    # Meta and controls should remain unchanged
    assert result_1.meta.active == meta.active
    assert result_1.meta.id == meta.id
    assert jnp.isclose(result_1.controls.throttle, controls.throttle, atol=EPS)

    # Standard case 2 - freeze already stationary aircraft
    body_stationary = Body(
        position=jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
        orientation=from_euler_zyx(
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
        angular_velocity=jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
    )
    aircraft_stationary = Aircraft(
        meta=meta,
        body=body_stationary,
        controls=controls,
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
        ),
    )
    result_2 = freeze_aircraft(aircraft_stationary)
    assert jnp.allclose(result_2.body.velocity, expected_zero_vel, atol=EPS)
    assert jnp.allclose(result_2.body.angular_velocity, expected_zero_vel, atol=EPS)

    # Edge case 1 - very high velocities
    body_fast = Body(
        position=jnp.array([0.0, 0.0, 1000.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([500.0, 100.0, -50.0], dtype=FLOAT_DTYPE),
        orientation=body_moving.orientation,
        angular_velocity=jnp.array([10.0, -5.0, 2.0], dtype=FLOAT_DTYPE),
    )
    aircraft_fast = Aircraft(
        meta=meta,
        body=body_fast,
        controls=controls,
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
        ),
    )
    result_3 = freeze_aircraft(aircraft_fast)
    assert jnp.allclose(result_3.body.velocity, expected_zero_vel, atol=EPS)
    assert jnp.allclose(result_3.body.angular_velocity, expected_zero_vel, atol=EPS)

    # Test with vmap
    positions = jnp.array(
        [
            [0.0, 0.0, 100.0],
            [100.0, 100.0, 200.0],
            [200.0, -100.0, 50.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    velocities = jnp.array(
        [
            [50.0, 0.0, 0.0],
            [100.0, 20.0, -10.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    zero_angle = jnp.array(0.0, dtype=FLOAT_DTYPE)
    orientations = jax.vmap(
        lambda _: from_euler_zyx(zero_angle, zero_angle, zero_angle)
    )(jnp.arange(3))
    angular_velocities = jnp.array(
        [
            [0.1, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.2],
        ],
        dtype=FLOAT_DTYPE,
    )
    bodies = jax.vmap(Body)(positions, velocities, orientations, angular_velocities)
    metas = jax.vmap(lambda i: Meta(jnp.array(True, dtype=bool), i))(jnp.arange(3))
    controls_batch = jax.vmap(lambda _: Controls.default())(jnp.arange(3))
    g_limiter_pd_batch = jax.vmap(
        lambda _: PDControllerState(previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE))
    )(jnp.arange(3))
    aircraft_batch = jax.vmap(Aircraft)(
        metas, bodies, controls_batch, g_limiter_pd_batch
    )

    freeze_aircraft_vmap = jax.vmap(freeze_aircraft)
    vmap_results = freeze_aircraft_vmap(aircraft_batch)

    assert vmap_results.body.velocity.shape == (3, 3)
    assert vmap_results.body.angular_velocity.shape == (3, 3)
    assert jnp.allclose(vmap_results.body.velocity, jnp.zeros((3, 3)), atol=EPS)
    assert jnp.allclose(vmap_results.body.angular_velocity, jnp.zeros((3, 3)), atol=EPS)
    # Positions should be unchanged
    assert jnp.allclose(vmap_results.body.position, positions, atol=EPS)


def test_step_aircraft_rk4(jit_mode: str) -> None:
    """Test aircraft state integration."""
    # Standard case 1 - simple forward flight
    meta = Meta(
        active=jnp.array(True, dtype=bool),
        id=jnp.array(0, dtype=INT_DTYPE),
    )
    body = Body(
        position=jnp.array([0.0, 0.0, 1000.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([50.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
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
    dt = jnp.array(0.01, dtype=FLOAT_DTYPE)

    result_1 = step_aircraft_rk4(aircraft, aircraft_config, physics_config, dt)
    # Aircraft should have moved forward
    assert result_1.body.position[0] > aircraft.body.position[0]
    # Orientation should be normalized
    orientation_norm = jnp.linalg.norm(result_1.body.orientation)
    assert jnp.isclose(orientation_norm, 1.0, atol=1e-5)
    # Aircraft should still be active
    assert result_1.meta.active == aircraft.meta.active

    # Standard case 2 - aircraft with control inputs
    controls_active = Controls(
        throttle=jnp.array(0.8, dtype=FLOAT_DTYPE),
        aileron=jnp.array(0.5, dtype=FLOAT_DTYPE),
        elevator=jnp.array(-0.3, dtype=FLOAT_DTYPE),
        rudder=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    aircraft_controlled = Aircraft(
        meta=meta,
        body=body,
        controls=controls_active,
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
        ),
    )
    result_2 = step_aircraft_rk4(
        aircraft_controlled, aircraft_config, physics_config, dt
    )
    # Should have angular velocity due to control inputs
    assert jnp.linalg.norm(result_2.body.angular_velocity) > 0.0
    # Orientation should change
    assert not jnp.allclose(result_2.body.orientation, body.orientation, atol=EPS)

    # Standard case 3 - hovering aircraft (zero initial velocity)
    body_hover = Body(
        position=jnp.array([0.0, 0.0, 1000.0], dtype=FLOAT_DTYPE),
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
    result_3 = step_aircraft_rk4(aircraft_hover, aircraft_config, physics_config, dt)
    # Should fall under gravity
    assert result_3.body.position[2] > body_hover.position[2]  # Positive down in NED

    # Edge case 1 - very small timestep
    dt_small = jnp.array(0.0001, dtype=FLOAT_DTYPE)
    result_4 = step_aircraft_rk4(aircraft, aircraft_config, physics_config, dt_small)
    # Changes should be very small
    position_change = jnp.linalg.norm(result_4.body.position - body.position)
    assert position_change < 0.01

    # Edge case 2 - larger timestep
    dt_large = jnp.array(0.1, dtype=FLOAT_DTYPE)
    result_5 = step_aircraft_rk4(aircraft, aircraft_config, physics_config, dt_large)
    # Changes should be larger
    position_change_large = jnp.linalg.norm(result_5.body.position - body.position)
    assert position_change_large > position_change

    # Test with vmap - different initial velocities
    velocities = jnp.array(
        [
            [30.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    positions = jnp.tile(jnp.array([0.0, 0.0, 1000.0], dtype=FLOAT_DTYPE), (3, 1))
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

    step_aircraft_rk4_vmap = jax.vmap(
        lambda a: step_aircraft_rk4(a, aircraft_config, physics_config, dt)
    )
    vmap_results = step_aircraft_rk4_vmap(aircraft_batch)

    assert vmap_results.body.position.shape == (3, 3)
    # All aircraft should move forward
    assert jnp.all(vmap_results.body.position[:, 0] > positions[:, 0])
    # Faster aircraft should move further
    assert vmap_results.body.position[2, 0] > vmap_results.body.position[1, 0]
    assert vmap_results.body.position[1, 0] > vmap_results.body.position[0, 0]


def test_step(jit_mode: str) -> None:
    """Test full simulation step."""
    # Setup heightmap and route for testing
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
    key_route = jax.random.PRNGKey(100)
    scenario_config = ScenarioConfig()
    route = generate_route(key_route, scenario_config)

    # Standard case 1 - normal flight step
    meta = Meta(
        active=jnp.array(True, dtype=bool),
        id=jnp.array(0, dtype=INT_DTYPE),
    )
    body = Body(
        position=jnp.array([0.0, 0.0, -500.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([50.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
        orientation=from_euler_zyx(
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
            jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
        angular_velocity=jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
    )
    controls = Controls.default()
    aircraft = Aircraft(
        meta=meta,
        body=body,
        controls=controls,
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
        ),
    )
    simulation = Simulation(
        aircraft=aircraft,
        time=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    config = SimulationConfig(dt=0.01)

    result_1, route_1 = step(simulation, heightmap, route, config)
    # Time should advance
    assert result_1.time > simulation.time
    assert jnp.isclose(result_1.time, 0.01, atol=EPS)
    # Aircraft should still be active (high above terrain)
    assert result_1.aircraft.meta.active == jnp.array(True, dtype=bool)
    # Aircraft should have moved
    assert not jnp.allclose(result_1.aircraft.body.position, body.position, atol=EPS)

    # Standard case 2 - aircraft near ground but not colliding
    body_low = Body(
        position=jnp.array([0.0, 0.0, -50.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([50.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
        orientation=body.orientation,
        angular_velocity=jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
    )
    aircraft_low = Aircraft(
        meta=meta,
        body=body_low,
        controls=controls,
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
        ),
    )
    simulation_low = Simulation(
        aircraft=aircraft_low,
        time=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    result_2, route_2 = step(simulation_low, heightmap, route, config)
    assert result_2.aircraft.meta.active == jnp.array(True, dtype=bool)
    # Should still be active if above terrain
    # With terrain at 0.9 (normalized), actual height is negative (mountain)
    # so aircraft at 50m down might collide depending on terrain height config

    # Standard case 3 - multiple steps
    current_sim = simulation
    current_route = route
    time_steps = 10
    for i in range(time_steps):
        current_sim, current_route = step(current_sim, heightmap, current_route, config)
    # Time should have advanced by dt * steps
    expected_time = jnp.array(0.01 * time_steps, dtype=FLOAT_DTYPE)
    assert jnp.isclose(current_sim.time, expected_time, atol=EPS)
    # Aircraft should have moved significantly
    position_change = jnp.linalg.norm(
        current_sim.aircraft.body.position - body.position
    )
    assert position_change > 1.0

    # Edge case 1 - aircraft starting underground (immediate collision)
    body_underground = Body(
        position=jnp.array([0.0, 0.0, 2000.0], dtype=FLOAT_DTYPE),  # Deep underground
        velocity=jnp.array([50.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
        orientation=body.orientation,
        angular_velocity=jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
    )
    aircraft_underground = Aircraft(
        meta=meta,
        body=body_underground,
        controls=controls,
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
        ),
    )
    simulation_underground = Simulation(
        aircraft=aircraft_underground,
        time=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    result_3, route_3 = step(simulation_underground, heightmap, route, config)
    # Should be inactive due to collision
    assert result_3.aircraft.meta.active == jnp.array(False, dtype=bool)
    # Velocity should be zero (frozen)
    assert jnp.allclose(
        result_3.aircraft.body.velocity, jnp.zeros(3, dtype=FLOAT_DTYPE), atol=EPS
    )
    assert jnp.allclose(
        result_3.aircraft.body.angular_velocity,
        jnp.zeros(3, dtype=FLOAT_DTYPE),
        atol=EPS,
    )

    # Edge case 2 - inactive aircraft should remain frozen
    meta_inactive = Meta(
        active=jnp.array(False, dtype=bool),
        id=jnp.array(0, dtype=INT_DTYPE),
    )
    aircraft_inactive = Aircraft(
        meta=meta_inactive,
        body=body,
        controls=controls,
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
        ),
    )
    simulation_inactive = Simulation(
        aircraft=aircraft_inactive,
        time=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    result_4, route_4 = step(simulation_inactive, heightmap, route, config)
    # Should remain inactive
    assert result_4.aircraft.meta.active == jnp.array(False, dtype=bool)
    # Time should still advance
    assert result_4.time > simulation_inactive.time

    # Test with vmap - different starting altitudes
    altitudes = jnp.array([100.0, 500.0, 1000.0], dtype=FLOAT_DTYPE)
    positions = jnp.stack(
        [jnp.zeros(3, dtype=FLOAT_DTYPE), jnp.zeros(3, dtype=FLOAT_DTYPE), altitudes],
        axis=1,
    )
    velocities = jnp.tile(jnp.array([50.0, 0.0, 0.0], dtype=FLOAT_DTYPE), (3, 1))
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
    times = jnp.zeros(3, dtype=FLOAT_DTYPE)
    simulations = jax.vmap(Simulation)(aircraft_batch, times)

    # Create routes for batch processing
    route_keys = jax.random.split(jax.random.PRNGKey(200), 3)
    scenario_config = ScenarioConfig()
    routes = jax.vmap(lambda k: generate_route(k, scenario_config))(route_keys)
    heightmaps = jnp.tile(heightmap[None, :, :], (3, 1, 1))

    step_vmap = jax.vmap(lambda sim, hm, rt: step(sim, hm, rt, config))
    vmap_results, vmap_routes = step_vmap(simulations, heightmaps, routes)

    assert vmap_results.time.shape == (3,)
    assert jnp.all(vmap_results.time == 0.01)
    # All aircraft positions should have changed
    assert vmap_results.aircraft.body.position.shape == (3, 3)


def test_step_aircraft_euler(jit_mode: str) -> None:
    """Test aircraft state integration using Euler method."""
    # Standard case 1 - simple forward flight
    meta = Meta(
        active=jnp.array(True, dtype=bool),
        id=jnp.array(0, dtype=INT_DTYPE),
    )
    body = Body(
        position=jnp.array([0.0, 0.0, 1000.0], dtype=FLOAT_DTYPE),
        velocity=jnp.array([50.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
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
    dt = jnp.array(0.01, dtype=FLOAT_DTYPE)

    result_1 = step_aircraft_euler(aircraft, aircraft_config, physics_config, dt)
    # Aircraft should have moved forward
    assert result_1.body.position[0] > aircraft.body.position[0]
    # Orientation should be normalized
    orientation_norm = jnp.linalg.norm(result_1.body.orientation)
    assert jnp.isclose(orientation_norm, 1.0, atol=1e-5)
    # Aircraft should still be active
    assert result_1.meta.active == aircraft.meta.active

    # Standard case 2 - aircraft with control inputs
    controls_active = Controls(
        throttle=jnp.array(0.8, dtype=FLOAT_DTYPE),
        aileron=jnp.array(0.5, dtype=FLOAT_DTYPE),
        elevator=jnp.array(-0.3, dtype=FLOAT_DTYPE),
        rudder=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )
    aircraft_controlled = Aircraft(
        meta=meta,
        body=body,
        controls=controls_active,
        g_limiter_pd=PDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE)
        ),
    )
    result_2 = step_aircraft_euler(
        aircraft_controlled, aircraft_config, physics_config, dt
    )
    # Should have angular velocity due to control inputs
    assert jnp.linalg.norm(result_2.body.angular_velocity) > 0.0

    # Explicit Euler uses the *old* ω to update q; with ω0 = 0, q won't change on the first step.
    # Take a second step, then require orientation to have changed.
    result_3 = step_aircraft_euler(result_2, aircraft_config, physics_config, dt)

    assert not jnp.allclose(result_3.body.orientation, body.orientation, atol=1e-6)
    assert jnp.isclose(jnp.linalg.norm(result_3.body.orientation), 1.0, atol=1e-5)

    # Standard case 3 - hovering aircraft (zero initial velocity)
    body_hover = Body(
        position=jnp.array([0.0, 0.0, 1000.0], dtype=FLOAT_DTYPE),
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
    result = step_aircraft_euler(aircraft_hover, aircraft_config, physics_config, dt)
    result_3 = step_aircraft_euler(result, aircraft_config, physics_config, dt)
    # Should fall under gravity
    assert result_3.body.position[2] > body_hover.position[2]  # Positive down in NED

    # Edge case 1 - very small timestep
    dt_small = jnp.array(0.0001, dtype=FLOAT_DTYPE)
    result_4 = step_aircraft_euler(aircraft, aircraft_config, physics_config, dt_small)
    # Changes should be very small
    position_change = jnp.linalg.norm(result_4.body.position - body.position)
    assert position_change < 0.01

    # Edge case 2 - larger timestep
    dt_large = jnp.array(0.1, dtype=FLOAT_DTYPE)
    result_5 = step_aircraft_euler(aircraft, aircraft_config, physics_config, dt_large)
    # Changes should be larger
    position_change_large = jnp.linalg.norm(result_5.body.position - body.position)
    assert position_change_large > position_change

    # Test with vmap - different initial velocities
    velocities = jnp.array(
        [
            [30.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    positions = jnp.tile(jnp.array([0.0, 0.0, 1000.0], dtype=FLOAT_DTYPE), (3, 1))
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

    step_aircraft_euler_vmap = jax.vmap(
        lambda a: step_aircraft_euler(a, aircraft_config, physics_config, dt)
    )
    vmap_results = step_aircraft_euler_vmap(aircraft_batch)

    assert vmap_results.body.position.shape == (3, 3)
    # All aircraft should move forward
    assert jnp.all(vmap_results.body.position[:, 0] > positions[:, 0])
    # Faster aircraft should move further
    assert vmap_results.body.position[2, 0] > vmap_results.body.position[1, 0]
    assert vmap_results.body.position[1, 0] > vmap_results.body.position[0, 0]
