"""State classes for aircraft, simulation, and control data."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .primitives import (
    FLOAT_DTYPE,
    BoolScalar,
    FloatScalar,
    IntScalar,
    Matrix,
    Quaternion,
    Vector,
    Vector3,
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Meta:
    """Metadata for a simulation entity."""

    active: BoolScalar
    """Whether the entity is active in the simulation."""

    id: IntScalar
    """Unique identifier for the entity."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Body:
    """Physical state of a rigid body in 6DOF simulation."""

    position: Vector3
    """World position vector [x, y, z] in NED world frame [m]."""

    velocity: Vector3
    """Linear velocity vector [vx, vy, vz] in FRD body frame [m/s]."""

    orientation: Quaternion
    """Unit quaternion [w, x, y, z] representing body to world rotation."""

    angular_velocity: Vector3
    """Angular velocity vector [p, q, r] in FRD body frame [rad/s]."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PIDControllerState:
    """Generic PID controller state."""

    previous: FloatScalar
    """Previous value of quantity being controlled."""

    previous_error: FloatScalar
    """Previous error value for derivative term."""

    integral: FloatScalar
    """Accumulated error for integral term."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Controls:
    """Pilot control input values."""

    throttle: FloatScalar
    """Throttle setting [0, 1], fraction of available thrust."""

    aileron: FloatScalar
    """Aileron input [-1, 1], controls roll, rotates about x-axis."""

    elevator: FloatScalar
    """Elevator input [-1, 1], controls pitch, rotates about y-axis."""

    rudder: FloatScalar
    """Rudder input [-1, 1], controls yaw, rotates about z-axis."""

    @classmethod
    def default(cls) -> "Controls":
        """Return default control values (zero throttle, neutral control surfaces)."""
        return cls(
            throttle=jnp.array(0.5, dtype=FLOAT_DTYPE),
            aileron=jnp.array(0.0, dtype=FLOAT_DTYPE),
            elevator=jnp.array(0.0, dtype=FLOAT_DTYPE),
            rudder=jnp.array(0.0, dtype=FLOAT_DTYPE),
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Aircraft:
    """Entity representing a single aircraft."""

    meta: Meta
    """Metadata including ID and activity flag."""

    body: Body
    """Rigid-body physical state (position, velocity, attitude)."""

    controls: Controls
    """Current control surface positions (throttle, aileron, elevator, rudder)."""

    commanded_controls: Controls
    """Commanded control surface positions (throttle, aileron, elevator, rudder)."""

    g_limiter_pid: PIDControllerState
    """PID controller state for G-force limiter."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Route:
    """Container for a waypoint route."""

    positions: Matrix
    """Waypoint world position matrix [N, 3] in NED world frame [m]."""

    visited: Vector
    """Vector of bools [N] indicating which waypoints have been hit."""

    current_idx: IntScalar
    """Current waypoint index."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Wind:
    """Wind state with mean and gust components."""

    mean: Vector3
    """Mean wind velocity in NED world frame [m/s]."""

    gust: Vector3
    """Gust/turbulence velocity centered at zero [m/s]."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Simulation:
    """Top-level container for simulation state."""

    aircraft: Aircraft
    """Aircraft entity in the simulation."""

    wind: Wind
    """Wind state with mean and gust components."""

    time: FloatScalar
    """Simulation time in seconds."""
