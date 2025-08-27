"""Tests for scenario module."""

import jax
import jax.numpy as jnp

from nimbus.core.config import TerrainConfig
from nimbus.core.primitives import EPS, FLOAT_DTYPE, INT_DTYPE
from nimbus.core.scenario import (
    ScenarioConfig,
    generate_simulation,
    generate_terrain_map,
)
from nimbus.core.state import Controls


def test_generate_simulation(jit_mode: str) -> None:
    """Test simulation generation."""
    key = jax.random.PRNGKey(42)

    # Standard case 1 - single simulation generation
    scenario_config = ScenarioConfig()
    result_1 = generate_simulation(key, scenario_config)

    # Check aircraft state
    assert result_1.aircraft.meta.active == jnp.array(True, dtype=bool)
    assert result_1.aircraft.meta.id == jnp.array(0, dtype=INT_DTYPE)

    # Check position is within expected bounds
    assert result_1.aircraft.body.position.shape == (3,)
    assert result_1.aircraft.body.position[0] >= -1000.0
    assert result_1.aircraft.body.position[0] <= 1000.0
    assert result_1.aircraft.body.position[1] >= -1000.0
    assert result_1.aircraft.body.position[1] <= 1000.0
    # Altitude is now -500.0 based on the updated loader.py
    assert jnp.isclose(result_1.aircraft.body.position[2], -500.0, atol=EPS)

    # Check velocity - updated to 150.0 based on loader.py
    expected_velocity = jnp.array([150.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_1.aircraft.body.velocity, expected_velocity, atol=EPS)

    # Check orientation (should be facing North)
    expected_orientation = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(
        result_1.aircraft.body.orientation, expected_orientation, atol=EPS
    )

    # Check angular velocity
    expected_ang_vel = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(
        result_1.aircraft.body.angular_velocity, expected_ang_vel, atol=EPS
    )

    # Check controls are default
    default_controls = Controls.default()
    assert result_1.aircraft.controls.throttle == default_controls.throttle
    assert result_1.aircraft.controls.aileron == default_controls.aileron
    assert result_1.aircraft.controls.elevator == default_controls.elevator
    assert result_1.aircraft.controls.rudder == default_controls.rudder

    # Check initial time
    assert result_1.time == jnp.array(0.0, dtype=FLOAT_DTYPE)

    # Standard case 2 - deterministic with same key
    result_2 = generate_simulation(key, scenario_config)

    # Should produce identical results with same key
    assert jnp.allclose(
        result_2.aircraft.body.position, result_1.aircraft.body.position, atol=EPS
    )

    # Standard case 3 - different with different key
    key2 = jax.random.PRNGKey(123)
    result_3 = generate_simulation(key2, scenario_config)

    # Position should be different (except altitude)
    assert not jnp.isclose(
        result_3.aircraft.body.position[0],
        result_1.aircraft.body.position[0],
        atol=1e-3,
    )
    assert not jnp.isclose(
        result_3.aircraft.body.position[1],
        result_1.aircraft.body.position[1],
        atol=1e-3,
    )
    assert jnp.isclose(
        result_3.aircraft.body.position[2], result_1.aircraft.body.position[2], atol=EPS
    )

    # Test with vmap - generate 5 simulations
    keys = jax.random.split(key, 5)
    generate_simulation_vmap = jax.vmap(
        lambda k: generate_simulation(k, scenario_config)
    )
    vmap_results = generate_simulation_vmap(keys)

    # Check shapes
    assert vmap_results.aircraft.body.position.shape == (5, 3)
    assert vmap_results.aircraft.body.velocity.shape == (5, 3)
    assert vmap_results.aircraft.body.orientation.shape == (5, 4)
    assert vmap_results.aircraft.body.angular_velocity.shape == (5, 3)
    assert vmap_results.time.shape == (5,)

    # All aircraft should have same altitude
    assert jnp.all(vmap_results.aircraft.body.position[:, 2] == -500.0)

    # All aircraft should have same velocity
    for i in range(5):
        assert jnp.allclose(
            vmap_results.aircraft.body.velocity[i], expected_velocity, atol=EPS
        )

    # Positions should be different (x, y coordinates)
    for i in range(5):
        for j in range(i + 1, 5):
            # At least one of x or y should be different between any two scenarios
            x_diff = jnp.abs(
                vmap_results.aircraft.body.position[i, 0]
                - vmap_results.aircraft.body.position[j, 0]
            )
            y_diff = jnp.abs(
                vmap_results.aircraft.body.position[i, 1]
                - vmap_results.aircraft.body.position[j, 1]
            )
            assert (x_diff > 1e-3) or (y_diff > 1e-3)


def test_generate_terrain_map(jit_mode: str) -> None:
    """Test terrain map generation with terrain configuration."""
    key = jax.random.PRNGKey(42)

    # Standard case 1 - default terrain config
    default_config = TerrainConfig()
    result_1 = generate_terrain_map(key, default_config)

    # Check shape matches resolution
    assert result_1.shape == (default_config.resolution, default_config.resolution)
    # Values should be normalized between 0 and 1
    assert jnp.all(result_1 >= 0.0)
    assert jnp.all(result_1 <= 1.0)
    # Should have some variation (not all the same value)
    assert jnp.var(result_1) > 0.0

    # Standard case 2 - custom resolution
    custom_config = TerrainConfig(resolution=128)
    result_2 = generate_terrain_map(key, custom_config)

    assert result_2.shape == (128, 128)
    assert jnp.all(result_2 >= 0.0)
    assert jnp.all(result_2 <= 1.0)

    # Standard case 3 - different parameters
    varied_config = TerrainConfig(
        resolution=64,
        base_scale=0.02,
        octaves=3,
        persistence=0.7,
        lacunarity=1.8,
        mountain_gain=2.0,
        bump_gain=0.5,
    )
    result_3 = generate_terrain_map(key, varied_config)

    assert result_3.shape == (64, 64)
    assert jnp.all(result_3 >= 0.0)
    assert jnp.all(result_3 <= 1.0)
    # Different parameters should produce different terrain
    assert not jnp.allclose(result_3[:64, :64], result_1[:64, :64], atol=1e-3)

    # Standard case 4 - deterministic with same key
    result_4 = generate_terrain_map(key, default_config)
    # Same key and config should produce identical results
    assert jnp.allclose(result_4, result_1, atol=EPS)

    # Standard case 5 - different with different key
    key2 = jax.random.PRNGKey(123)
    result_5 = generate_terrain_map(key2, default_config)
    # Different key should produce different terrain
    assert not jnp.allclose(result_5, result_1, atol=1e-3)

    # Edge case 1 - small resolution
    small_config = TerrainConfig(resolution=32)
    result_6 = generate_terrain_map(key, small_config)
    assert result_6.shape == (32, 32)
    assert jnp.all(result_6 >= 0.0)
    assert jnp.all(result_6 <= 1.0)

    # Edge case 2 - large resolution
    large_config = TerrainConfig(resolution=512)
    result_7 = generate_terrain_map(key, large_config)
    assert result_7.shape == (512, 512)
    assert jnp.all(result_7 >= 0.0)
    assert jnp.all(result_7 <= 1.0)

    # Test with vmap - multiple keys
    keys = jax.random.split(key, 3)

    generate_terrain_map_vmap = jax.vmap(generate_terrain_map, in_axes=(0, None))
    vmap_results = generate_terrain_map_vmap(keys, default_config)

    assert vmap_results.shape == (
        3,
        default_config.resolution,
        default_config.resolution,
    )
    # All should be properly normalized
    assert jnp.all(vmap_results >= 0.0)
    assert jnp.all(vmap_results <= 1.0)
    # Each should be different
    for i in range(3):
        for j in range(i + 1, 3):
            assert not jnp.allclose(vmap_results[i], vmap_results[j], atol=1e-3)
