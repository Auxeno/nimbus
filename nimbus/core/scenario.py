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
class InitialConditions:
    position: Vector3Spec
    """Initial aircraft position in NED world-frame [m]."""

    velocity: Vector3Spec
    """Initial aircraft velocity in NED world-frame [m/s]."""

    orientation_euler: Vector3Spec
    """Initial aircraft Euler angles: yaw (Z), pitch (Y), roll (X) [degrees]."""

    angular_velocity: Vector3Spec
    """Initial aircraft angular velocity in FRD body-frame [rad/s]"""

    wind_speed: ScalarSpec
    """Wind speed [m/s]."""

    wind_direction: ScalarSpec
    """Wind direction [degrees] - direction wind is coming FROM (meteorological convention)."""

    waypoints: tuple[Vector3Spec, ...]
    """Waypoint positions in NED world-frame [m]."""

    @classmethod
    def default(cls) -> "InitialConditions":
        return cls(
            position=(
                Uniform(-1000.0, 1000.0),
                Uniform(-1000.0, 1000.0),
                Fixed(-500.0),
            ),
            velocity=(Fixed(150.0), Fixed(0.0), Fixed(0.0)),
            orientation_euler=(Fixed(0.0), Fixed(0.0), Fixed(0.0)),
            angular_velocity=(Fixed(0.0), Fixed(0.0), Fixed(0.0)),
            wind_speed=Uniform(0.0, 10.0),
            wind_direction=Uniform(0.0, 360.0),
            waypoints=(
                (Fixed(1500.0), Fixed(0.0), Fixed(-500.0)),
                (Fixed(2000.0), Fixed(0.0), Fixed(-500.0)),
                (Fixed(2500.0), Fixed(0.0), Fixed(-500.0)),
                (Fixed(3000.0), Fixed(0.0), Fixed(-500.0)),
            ),
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


def generate_route(key: PRNGKey, initial_conditions: InitialConditions) -> Route:
    num_waypoints = len(initial_conditions.waypoints)
    keys = jax.random.split(key, num_waypoints)

    sampled = [_sample_vec3(k, wp) for k, wp in zip(keys, initial_conditions.waypoints)]
    positions = jnp.stack(sampled, axis=0).astype(FLOAT_DTYPE)

    return Route(
        positions=positions,
        visited=jnp.zeros(num_waypoints, dtype=bool),
        current_idx=jnp.array(0, dtype=INT_DTYPE),
    )


def generate_simulation(
    key: PRNGKey,
    initial_conditions: InitialConditions,
) -> Simulation:
    (
        key_position,
        key_velocity,
        key_omega,
        key_orientation,
        key_wind_speed,
        key_wind_dir,
    ) = jax.random.split(key, 6)

    position = _sample_vec3(key_position, initial_conditions.position)
    velocity = _sample_vec3(key_velocity, initial_conditions.velocity)
    angular_velocity = _sample_vec3(key_omega, initial_conditions.angular_velocity)

    wind_speed = _sample_scalar(key_wind_speed, initial_conditions.wind_speed)
    wind_dir = jnp.deg2rad(
        _sample_scalar(key_wind_dir, initial_conditions.wind_direction)
    )
    wind_north = wind_speed * jnp.cos(wind_dir + jnp.pi)
    wind_east = wind_speed * jnp.sin(wind_dir + jnp.pi)
    wind_velocity = jnp.array([wind_north, wind_east, 0.0], dtype=FLOAT_DTYPE)

    eul_deg = _sample_vec3(key_orientation, initial_conditions.orientation_euler)
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
            integral=jnp.array(0.0, dtype=FLOAT_DTYPE),
        ),
    )

    return Simulation(
        aircraft=aircraft,
        wind_velocity=wind_velocity,
        time=jnp.array(0.0, dtype=FLOAT_DTYPE),
    )


def generate_scenario(
    key: PRNGKey,
    initial_conditions: InitialConditions,
    terrain_config: TerrainConfig,
) -> tuple[Simulation, Matrix, Route]:
    key_map, key_route, key_simulation = jax.random.split(key, 3)
    heightmap = generate_terrain_map(key_map, terrain_config)
    route = generate_route(key_route, initial_conditions)
    simulation = generate_simulation(key_simulation, initial_conditions)
    return simulation, heightmap, route
