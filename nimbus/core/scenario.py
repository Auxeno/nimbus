"""Scenario generation utilities for flight simulation initialisation."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from chex import PRNGKey

from . import quaternion
from .config import TerrainConfig
from .primitives import FLOAT_DTYPE, INT_DTYPE, FloatScalar, Matrix
from .state import Aircraft, Body, Controls, Meta, PIDControllerState, Route, Simulation
from .terrain import generate_heightmap


@dataclass(frozen=True)
class Fixed:
    x: float


@dataclass(frozen=True)
class Uniform:
    low: float
    high: float


ScalarSpec = Fixed | Uniform
Vector3Spec = tuple[ScalarSpec, ScalarSpec, ScalarSpec]


def _sample_scalar(key: PRNGKey, spec: ScalarSpec) -> FloatScalar:
    if isinstance(spec, Fixed):
        return jnp.array(spec.x, dtype=FLOAT_DTYPE)
    elif isinstance(spec, Uniform):
        return jax.random.uniform(
            key, (), minval=spec.low, maxval=spec.high, dtype=FLOAT_DTYPE
        )
    else:
        raise TypeError(f"Unknown ScalarSpec: {type(spec)}")


def _sample_vec3(key: PRNGKey, spec: Vector3Spec) -> jnp.ndarray:
    key_x, key_y, key_z = jax.random.split(key, 3)
    x = _sample_scalar(key_x, spec[0])
    y = _sample_scalar(key_y, spec[1])
    z = _sample_scalar(key_z, spec[2])
    return jnp.stack([x, y, z])


@dataclass(frozen=True)
class ScenarioConfig:
    position: Vector3Spec = (
        Uniform(-1000.0, 1000.0),
        Uniform(-1000.0, 1000.0),
        Fixed(-500.0),
    )

    velocity: Vector3Spec = (Fixed(150.0), Fixed(0.0), Fixed(0.0))

    # Euler angles: yaw (Z), pitch (Y), roll (X) [degrees]
    orientation_euler: Vector3Spec = (Fixed(0.0), Fixed(0.0), Fixed(0.0))

    angular_velocity: Vector3Spec = (Fixed(0.0), Fixed(0.0), Fixed(0.0))

    waypoints: tuple[Vector3Spec, ...] = (
        (Fixed(1500.0), Fixed(0.0), Fixed(-500.0)),
        (Fixed(2000.0), Fixed(0.0), Fixed(-500.0)),
        (Fixed(2500.0), Fixed(0.0), Fixed(-500.0)),
        (Fixed(3000.0), Fixed(0.0), Fixed(-500.0)),
    )


def generate_terrain_map(key: PRNGKey, terrain_config: TerrainConfig) -> Matrix:
    return generate_heightmap(
        key=key,
        resolution=jnp.array(terrain_config.resolution, dtype=INT_DTYPE),
        base_scale=jnp.array(terrain_config.base_scale, dtype=FLOAT_DTYPE),
        octaves=jnp.array(terrain_config.octaves, dtype=INT_DTYPE),
        persistence=jnp.array(terrain_config.persistence, dtype=FLOAT_DTYPE),
        lacunarity=jnp.array(terrain_config.lacunarity, dtype=FLOAT_DTYPE),
        mountain_gain=jnp.array(terrain_config.mountain_gain, dtype=FLOAT_DTYPE),
        bump_gain=jnp.array(terrain_config.bump_gain, dtype=FLOAT_DTYPE),
        padding=jnp.array(terrain_config.padding, dtype=INT_DTYPE),
    )


def generate_route(key: PRNGKey, scenario_config: ScenarioConfig) -> Route:
    num_waypoints = len(scenario_config.waypoints)
    keys = jax.random.split(key, num_waypoints)

    sampled = [_sample_vec3(k, wp) for k, wp in zip(keys, scenario_config.waypoints)]
    positions = jnp.stack(sampled, axis=0).astype(FLOAT_DTYPE)

    return Route(
        positions=positions,
        visited=jnp.zeros(num_waypoints, dtype=bool),
        current_idx=jnp.array(0, dtype=INT_DTYPE),
    )


def generate_simulation(key: PRNGKey, scenario_config: ScenarioConfig) -> Simulation:
    key_position, key_velocity, key_omega, key_orientation = jax.random.split(key, 4)

    position = _sample_vec3(key_position, scenario_config.position)
    velocity = _sample_vec3(key_velocity, scenario_config.velocity)
    angular_velocity = _sample_vec3(key_omega, scenario_config.angular_velocity)

    eul_deg = _sample_vec3(key_orientation, scenario_config.orientation_euler)
    yaw, pitch, roll = jnp.deg2rad(eul_deg)
    quat = quaternion.from_euler_zyx(yaw, pitch, roll).astype(FLOAT_DTYPE)

    aircraft = Aircraft(
        meta=Meta(active=jnp.array(True, dtype=bool), id=jnp.array(0, dtype=INT_DTYPE)),
        body=Body(
            position=position.astype(FLOAT_DTYPE),
            velocity=velocity.astype(FLOAT_DTYPE),
            orientation=quat,
            angular_velocity=angular_velocity.astype(FLOAT_DTYPE),
        ),
        controls=Controls.default(),
        g_limiter_pid=PIDControllerState(
            previous_error=jnp.array(0.0, dtype=FLOAT_DTYPE),
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE)
        ),
    )

    return Simulation(aircraft=aircraft, time=jnp.array(0.0, dtype=FLOAT_DTYPE))


def generate_scenario(
    key: PRNGKey,
    scenario_config: ScenarioConfig,
    terrain_config: TerrainConfig,
) -> tuple[Simulation, Matrix, Route]:
    key_map, key_route, key_simulation = jax.random.split(key, 3)
    heightmap = generate_terrain_map(key_map, terrain_config)
    route = generate_route(key_route, scenario_config)
    simulation = generate_simulation(key_simulation, scenario_config)
    return simulation, heightmap, route
