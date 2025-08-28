"""Flight simulation control and numerical integration routines."""

from dataclasses import replace

import jax
import jax.numpy as jnp

from . import quaternion
from .config import AircraftConfig, PhysicsConfig, SimulationConfig
from .interface import (
    aircraft_state_derivatives,
    next_waypoint,
    terrain_collision,
    waypoint_hit,
)
from .logic import apply_g_limiter
from .primitives import FLOAT_DTYPE, FloatScalar, Matrix, Vector3
from .state import Aircraft, Body, Controls, Meta, Route, Simulation


def set_controls(simulation: Simulation, controls: Controls) -> Simulation:
    """
    Update simulation with new control inputs.

    Parameters
    ----------
    simulation : Simulation
        Current simulation state.
    controls : Controls
        New control inputs (throttle, aileron, elevator, rudder).

    Returns
    -------
    Simulation
        Updated simulation with new controls.
    """
    aircraft = replace(simulation.aircraft, controls=controls)
    return replace(simulation, aircraft=aircraft)


def step_aircraft_euler(
    aircraft: Aircraft,
    wind_velocity: Vector3,
    aircraft_config: AircraftConfig,
    physics_config: PhysicsConfig,
    dt: FloatScalar,
) -> Aircraft:
    """
    Advance aircraft state by one timestep using Euler integration.

    Parameters
    ----------
    aircraft : Aircraft
        Current aircraft state.
    wind_velocity : Vector3
        Wind velocity in NED world frame [m/s].
    aircraft_config : AircraftConfig
        Aircraft configuration parameters.
    physics_config : PhysicsConfig
        Physics configuration parameters.
    dt : FloatScalar
        Time step for integration.

    Returns
    -------
    Aircraft
        Aircraft state after one time step.
    """
    # Derivatives at the current state
    dx, dv, dq, dw = aircraft_state_derivatives(
        aircraft, wind_velocity, aircraft_config, physics_config
    )

    return replace(
        aircraft,
        body=Body(
            position=aircraft.body.position + dt * dx,
            velocity=aircraft.body.velocity + dt * dv,
            orientation=quaternion.normalize(aircraft.body.orientation + dt * dq),
            angular_velocity=aircraft.body.angular_velocity + dt * dw,
        ),
    )


def step_aircraft_rk4(
    aircraft: Aircraft,
    wind_velocity: Vector3,
    aircraft_config: AircraftConfig,
    physics_config: PhysicsConfig,
    dt: FloatScalar,
) -> Aircraft:
    """
    Advance aircraft state by one timestep using RK4 integration.

    Parameters
    ----------
    aircraft : Aircraft
        Current aircraft state.
    wind_velocity : Vector3
        Wind velocity in NED world frame [m/s].
    aircraft_config : AircraftConfig
        Aircraft configuration parameters.
    physics_config : PhysicsConfig
        Physics configuration parameters.
    dt : FloatScalar
        Time step for integration.

    Returns
    -------
    Aircraft
        Aircraft state after one time step.
    """
    # k1: derivatives at the current state
    dx_1, dv_1, dq_1, dw_1 = aircraft_state_derivatives(
        aircraft, wind_velocity, aircraft_config, physics_config
    )

    # k2: derivatives at half-step using k1
    aircraft_2 = replace(
        aircraft,
        body=Body(
            position=aircraft.body.position + 0.5 * dt * dx_1,
            velocity=aircraft.body.velocity + 0.5 * dt * dv_1,
            orientation=aircraft.body.orientation + 0.5 * dt * dq_1,
            angular_velocity=aircraft.body.angular_velocity + 0.5 * dt * dw_1,
        ),
    )
    dx_2, dv_2, dq_2, dw_2 = aircraft_state_derivatives(
        aircraft_2, wind_velocity, aircraft_config, physics_config
    )

    # k3: derivatives at half-step using k2
    aircraft_3 = replace(
        aircraft,
        body=Body(
            position=aircraft.body.position + 0.5 * dt * dx_2,
            velocity=aircraft.body.velocity + 0.5 * dt * dv_2,
            orientation=aircraft.body.orientation + 0.5 * dt * dq_2,
            angular_velocity=aircraft.body.angular_velocity + 0.5 * dt * dw_2,
        ),
    )
    dx_3, dv_3, dq_3, dw_3 = aircraft_state_derivatives(
        aircraft_3, wind_velocity, aircraft_config, physics_config
    )

    # k4: derivatives at full step using k3
    aircraft_4 = replace(
        aircraft,
        body=Body(
            position=aircraft.body.position + dt * dx_3,
            velocity=aircraft.body.velocity + dt * dv_3,
            orientation=aircraft.body.orientation + dt * dq_3,
            angular_velocity=aircraft.body.angular_velocity + dt * dw_3,
        ),
    )
    dx_4, dv_4, dq_4, dw_4 = aircraft_state_derivatives(
        aircraft_4, wind_velocity, aircraft_config, physics_config
    )

    # Combine weighted sum (RK4 formula)
    position = aircraft.body.position + (dt / 6.0) * (dx_1 + 2 * dx_2 + 2 * dx_3 + dx_4)
    velocity = aircraft.body.velocity + (dt / 6.0) * (dv_1 + 2 * dv_2 + 2 * dv_3 + dv_4)
    orientation = quaternion.normalize(
        aircraft.body.orientation + (dt / 6.0) * (dq_1 + 2 * dq_2 + 2 * dq_3 + dq_4)
    )
    angular_velocity = aircraft.body.angular_velocity + (dt / 6.0) * (
        dw_1 + 2 * dw_2 + 2 * dw_3 + dw_4
    )

    return replace(
        aircraft,
        body=Body(
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
        ),
    )


def freeze_aircraft(aircraft: Aircraft) -> Aircraft:
    """
    Freeze aircraft motion by zeroing all velocities.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft to freeze.

    Returns
    -------
    Aircraft
        Aircraft with zeroed linear and angular velocities.
    """
    body = Body(
        position=aircraft.body.position,
        velocity=jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
        orientation=aircraft.body.orientation,
        angular_velocity=jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
    )
    return replace(aircraft, body=body)


def step(
    simulation: Simulation,
    heightmap: Matrix,
    route: Route,
    config: SimulationConfig,
) -> tuple[Simulation, Route]:
    # Terrain collision
    colliding = terrain_collision(
        aircraft=simulation.aircraft,
        heightmap=heightmap,
        map_config=config.map,
    )
    active = jnp.logical_and(
        simulation.aircraft.meta.active, jnp.logical_not(colliding)
    )
    aircraft = replace(
        simulation.aircraft,
        meta=Meta(
            active=active,
            id=simulation.aircraft.meta.id,
        ),
    )

    # Waypoint collision
    route = jax.lax.cond(
        waypoint_hit(aircraft=aircraft, route=route, route_config=config.route),
        lambda args: next_waypoint(*args),
        lambda args: route,
        operand=(route, jnp.array(config.route.loop, dtype=bool)),
    )

    # Apply G-limiter to aircraft controls
    adjusted_controls, new_pid_state = apply_g_limiter(
        aircraft=aircraft,
        controls=aircraft.controls,
        wind_velocity=simulation.wind_velocity,
        aircraft_config=config.aircraft,
        physics_config=config.physics,
        dt=jnp.array(config.dt, dtype=FLOAT_DTYPE),
    )
    aircraft = replace(
        aircraft, controls=adjusted_controls, g_limiter_pid=new_pid_state
    )

    # Aircraft dynamics update
    aircraft = jax.lax.cond(
        aircraft.meta.active,
        lambda: step_aircraft_rk4(
            aircraft=aircraft,
            wind_velocity=simulation.wind_velocity,
            aircraft_config=config.aircraft,
            physics_config=config.physics,
            dt=jnp.array(config.dt, dtype=FLOAT_DTYPE),
        ),
        lambda: freeze_aircraft(aircraft),
    )

    time = simulation.time + jnp.array(config.dt, dtype=FLOAT_DTYPE)

    return replace(simulation, aircraft=aircraft, time=time), route
