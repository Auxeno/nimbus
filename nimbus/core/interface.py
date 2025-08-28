"""Flight dynamics interface between lower-level modules and simulation.py."""

from dataclasses import replace

import jax
import jax.numpy as jnp

from . import physics, quaternion, spatial
from .config import AircraftConfig, MapConfig, PhysicsConfig, RouteConfig
from .primitives import FLOAT_DTYPE, INT_DTYPE, BoolScalar, Matrix, Vector3, norm_3
from .state import Aircraft, Route


def calculate_translational_acceleration(
    aircraft: Aircraft,
    wind_velocity: Vector3,
    aircraft_config: AircraftConfig,
    physics_config: PhysicsConfig,
) -> Vector3:
    """
    Calculate translational acceleration for an aircraft.

    Parameters
    ----------
    aircraft : Aircraft
        Current aircraft state including position, velocity, and orientation.
    wind_velocity : Vector3
        Wind velocity in NED world frame [m/s].
    aircraft_config : AircraftConfig
        Aircraft configuration parameters including mass, drag coefficients.
    physics_config : PhysicsConfig
        Physics configuration including gravity and air density parameters.

    Returns
    -------
    Vector3
        Translational acceleration in NED world frame [m/s^2].
    """
    air_density = physics.calculate_air_density(
        altitude=-aircraft.body.position[2],
        rho_0=jnp.array(physics_config.rho_0, dtype=FLOAT_DTYPE),
        rho_decay=jnp.array(physics_config.rho_decay, dtype=FLOAT_DTYPE),
    )

    aero_forces = physics.calculate_aero_forces(
        velocity=aircraft.body.velocity,
        orientation=aircraft.body.orientation,
        wind_velocity=wind_velocity,
        air_density=air_density,
        coef_drag=jnp.array(aircraft_config.coef_drag, dtype=FLOAT_DTYPE),
        coef_lift=jnp.array(aircraft_config.coef_lift, dtype=FLOAT_DTYPE),
        coef_sideslip=jnp.array(aircraft_config.coef_sideslip, dtype=FLOAT_DTYPE),
        max_attack_angle=jnp.array(aircraft_config.max_attack_angle, dtype=FLOAT_DTYPE),
        max_sideslip_angle=jnp.array(
            aircraft_config.max_sideslip_angle, dtype=FLOAT_DTYPE
        ),
        surface_areas=jnp.array(aircraft_config.surface_areas, dtype=FLOAT_DTYPE),
    )

    thrust_force = physics.calculate_thrust(
        throttle=jnp.array(aircraft.controls.throttle, dtype=FLOAT_DTYPE),
        max_thrust=jnp.array(aircraft_config.max_thrust, dtype=FLOAT_DTYPE),
        air_density=air_density,
        rho_0=jnp.array(physics_config.rho_0, dtype=FLOAT_DTYPE),
    )

    body_forces = aero_forces + thrust_force

    world_forces = physics.calculate_weight(
        mass=jnp.array(aircraft_config.mass, dtype=FLOAT_DTYPE),
        gravity=jnp.array(physics_config.gravity, dtype=FLOAT_DTYPE),
    )

    world_forces += quaternion.rotate_vector(body_forces, aircraft.body.orientation)

    acceleration = world_forces / aircraft_config.mass

    return acceleration


def calculate_angular_acceleration(
    aircraft: Aircraft,
    aircraft_config: AircraftConfig,
    physics_config: PhysicsConfig,
) -> Vector3:
    """
    Calculate angular acceleration for an aircraft.

    Parameters
    ----------
    aircraft : Aircraft
        Current aircraft state including controls and angular velocity.
    aircraft_config : AircraftConfig
        Aircraft configuration including torque and damping coefficients.
    physics_config : PhysicsConfig
        Physics configuration for air density calculations.

    Returns
    -------
    Vector3
        Angular acceleration in FRD body frame [rad/s^2].
    """
    airspeed = norm_3(aircraft.body.velocity)

    air_density = physics.calculate_air_density(
        -aircraft.body.position[2],
        jnp.array(physics_config.rho_0, dtype=FLOAT_DTYPE),
        jnp.array(physics_config.rho_decay, dtype=FLOAT_DTYPE),
    )

    q = physics.calculate_dynamic_pressure(airspeed=airspeed, air_density=air_density)

    control_moments = physics.calculate_control_moments(
        velocity=aircraft.body.velocity,
        orientation=aircraft.body.orientation,
        airspeed=airspeed,
        dynamic_pressure=q,
        aileron=aircraft.controls.aileron,
        elevator=aircraft.controls.elevator,
        rudder=aircraft.controls.rudder,
        coefs_torque=jnp.array(aircraft_config.coefs_torque),
    )

    damping_moments = physics.calculate_damping_moments(
        angular_velocity=aircraft.body.angular_velocity,
        dynamic_pressure=q,
        coef_rot_damping=jnp.array(aircraft_config.coef_rot_damping),
    )

    body_moments = control_moments + damping_moments

    inertia = physics.estimate_inertia(
        mass=jnp.array(aircraft_config.mass, dtype=FLOAT_DTYPE),
        surface_areas=jnp.array(aircraft_config.surface_areas, dtype=FLOAT_DTYPE),
    )

    angular_acceleration = body_moments / inertia

    return angular_acceleration


def aircraft_state_derivatives(
    aircraft: Aircraft,
    wind_velocity: Vector3,
    aircraft_config: AircraftConfig,
    physics_config: PhysicsConfig,
) -> tuple[Vector3, Vector3, Vector3, Vector3]:
    """
    Calculate complete aircraft state derivatives for numerical integration.

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

    Returns
    -------
    tuple[Vector3, Vector3, Vector3, Vector3]
        State derivatives: (d_position, d_velocity, d_orientation, d_angular_velocity).
        - d_position: velocity in NED world frame [m/s]
        - d_velocity: acceleration in NED world frame [m/s^2]
        - d_orientation: quaternion derivative [1/s]
        - d_angular_velocity: angular acceleration in FRD body frame [rad/s^2]
    """
    acceleration = calculate_translational_acceleration(
        aircraft, wind_velocity, aircraft_config, physics_config
    )

    angular_acceleration = calculate_angular_acceleration(
        aircraft, aircraft_config, physics_config
    )

    quaternion_derivative = quaternion.derivative(
        aircraft.body.orientation, aircraft.body.angular_velocity
    )

    return (
        aircraft.body.velocity,
        acceleration,
        quaternion_derivative,
        angular_acceleration,
    )


def terrain_collision(
    aircraft: Aircraft,
    heightmap: Matrix,
    map_config: MapConfig,
) -> BoolScalar:
    """
    Check if an aircraft is colliding with terrain.

    Parameters
    ----------
    aircraft : Aircraft
        Current aircraft state with position in NED world frame.
    heightmap : Matrix
        Normalized terrain heightmap with values in range [0, 1].
    map_config : MapConfig
        Terrain configuration including map size and max elevation [m].

    Returns
    -------
    BoolScalar
        True if the aircraft is below ground level, else False.
    """
    return spatial.calculate_terrain_collision(
        heightmap=heightmap,
        position=aircraft.body.position,
        map_size=jnp.array(map_config.size, dtype=FLOAT_DTYPE),
        terrain_height=jnp.array(map_config.terrain_height, dtype=FLOAT_DTYPE),
        use_bilinear=jnp.array(map_config.use_bilinear, dtype=bool),
    )


def calculate_g_force(
    aircraft: Aircraft,
    wind_velocity: Vector3,
    aircraft_config: AircraftConfig,
    physics_config: PhysicsConfig,
) -> Vector3:
    """
    Calculate G-forces experienced by the aircraft in body frame.

    Parameters
    ----------
    aircraft : Aircraft
        Current aircraft state including position, velocity, and orientation.
    wind_velocity : Vector3
        Wind velocity in NED world frame [m/s].
    aircraft_config : AircraftConfig
        Aircraft configuration parameters including mass and force coefficients.
    physics_config : PhysicsConfig
        Physics configuration including gravity and air density parameters.

    Returns
    -------
    Vector3
        G-force vector in FRD body frame [g], where 1g = 9.81 m/s^2.
    """
    air_density = physics.calculate_air_density(
        altitude=-aircraft.body.position[2],
        rho_0=jnp.array(physics_config.rho_0, dtype=FLOAT_DTYPE),
        rho_decay=jnp.array(physics_config.rho_decay, dtype=FLOAT_DTYPE),
    )

    aero_forces = physics.calculate_aero_forces(
        velocity=aircraft.body.velocity,
        orientation=aircraft.body.orientation,
        wind_velocity=wind_velocity,
        air_density=air_density,
        coef_drag=jnp.array(aircraft_config.coef_drag, dtype=FLOAT_DTYPE),
        coef_lift=jnp.array(aircraft_config.coef_lift, dtype=FLOAT_DTYPE),
        coef_sideslip=jnp.array(aircraft_config.coef_sideslip, dtype=FLOAT_DTYPE),
        max_attack_angle=jnp.array(aircraft_config.max_attack_angle, dtype=FLOAT_DTYPE),
        max_sideslip_angle=jnp.array(
            aircraft_config.max_sideslip_angle, dtype=FLOAT_DTYPE
        ),
        surface_areas=jnp.array(aircraft_config.surface_areas, dtype=FLOAT_DTYPE),
    )

    thrust_force = physics.calculate_thrust(
        throttle=jnp.array(aircraft.controls.throttle, dtype=FLOAT_DTYPE),
        max_thrust=jnp.array(aircraft_config.max_thrust, dtype=FLOAT_DTYPE),
        air_density=air_density,
        rho_0=jnp.array(physics_config.rho_0, dtype=FLOAT_DTYPE),
    )

    body_forces = aero_forces + thrust_force

    world_forces = physics.calculate_weight(
        mass=jnp.array(aircraft_config.mass, dtype=FLOAT_DTYPE),
        gravity=jnp.array(physics_config.gravity, dtype=FLOAT_DTYPE),
    )

    body_forces += quaternion.rotate_vector(
        world_forces, quaternion.inverse(aircraft.body.orientation)
    )

    acceleration = body_forces / aircraft_config.mass
    acceleration_g = acceleration / physics_config.gravity

    return acceleration_g


def waypoint_hit(
    aircraft: Aircraft,
    route: Route,
    route_config: RouteConfig,
) -> BoolScalar:
    """
    Check if the aircraft has reached the current waypoint.

    Parameters
    ----------
    aircraft : Aircraft
        Current aircraft state with position in NED world frame.
    route : Route
        Current route state including waypoint positions and visit status.
    route_config : RouteConfig
        Route configuration including waypoint hit radius.

    Returns
    -------
    BoolScalar
        True if aircraft is within radius of current unvisited waypoint, else False.
    """
    return jnp.logical_and(
        spatial.spherical_collision(
            position_1=aircraft.body.position,
            position_2=route.positions[route.current_idx],
            distance=jnp.array(route_config.radius, dtype=FLOAT_DTYPE),
        ),
        jnp.logical_not(route.visited[route.current_idx]),
    )


def next_waypoint(route: Route, loop: BoolScalar) -> Route:
    """
    Advance to the next waypoint in the route sequence.

    Parameters
    ----------
    route : Route
        Current route state with waypoint positions and visit status.
    loop : BoolScalar
        If True, reset to first waypoint after reaching the last one.

    Returns
    -------
    Route
        Updated route with current waypoint marked as visited and index advanced.
        If looping is enabled and last waypoint is reached, all waypoints are
        reset to unvisited and index returns to 0.
    """
    visited = route.visited.at[route.current_idx].set(True)

    is_last = route.current_idx == route.positions.shape[0] - 1
    next_idx = jax.lax.cond(
        is_last,
        lambda: jnp.array(0, dtype=INT_DTYPE),
        lambda: route.current_idx + 1,
    )

    visited = jax.lax.cond(
        jnp.logical_and(next_idx == 0, loop),
        lambda: jnp.zeros_like(route.visited),
        lambda: visited,
    )

    return replace(route, current_idx=next_idx, visited=visited)
