"""Flight dynamics interface between lower-level modules and simulation.py."""

from dataclasses import replace

import jax
import jax.numpy as jnp

from . import logic, physics, quaternion, spatial
from .config import AircraftConfig, MapConfig, PhysicsConfig, RouteConfig, WindConfig
from .primitives import (
    FLOAT_DTYPE,
    INT_DTYPE,
    BoolScalar,
    FloatScalar,
    Matrix,
    PRNGKey,
    Vector3,
    norm_3,
)
from .state import Aircraft, Controls, PIDControllerState, Route, Wind
from .wind import calculate_wind


def calculate_translational_acceleration(
    aircraft: Aircraft,
    wind: Wind,
    aircraft_config: AircraftConfig,
    physics_config: PhysicsConfig,
) -> Vector3:
    """
    Calculate translational acceleration for an aircraft.

    Parameters
    ----------
    aircraft : Aircraft
        Current aircraft state including position, velocity, and orientation.
    wind : Wind
        Wind state with mean and gust components.
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

    # Calculate relative velocity (aircraft velocity minus wind)
    wind_velocity = wind.mean + wind.gust
    relative_velocity = aircraft.body.velocity - wind_velocity

    aero_forces = physics.calculate_aero_forces(
        velocity=relative_velocity,
        orientation=aircraft.body.orientation,
        air_density=air_density,
        surface_areas=jnp.array(aircraft_config.surface_areas, dtype=FLOAT_DTYPE),
        coef_drag=jnp.array(aircraft_config.coef_drag, dtype=FLOAT_DTYPE),
        coef_sideslip=jnp.array(aircraft_config.coef_sideslip, dtype=FLOAT_DTYPE),
        max_attack_angle=jnp.array(aircraft_config.max_attack_angle, dtype=FLOAT_DTYPE),
        zero_lift_attack_angle=jnp.array(
            aircraft_config.zero_lift_attack_angle, dtype=FLOAT_DTYPE
        ),
        lift_slope=jnp.array(aircraft_config.lift_slope, dtype=FLOAT_DTYPE),
        aspect_ratio=jnp.array(aircraft_config.aspect_ratio, dtype=FLOAT_DTYPE),
        oswald_efficiency=jnp.array(
            aircraft_config.oswald_efficiency, dtype=FLOAT_DTYPE
        ),
        max_sideslip_angle=jnp.array(
            aircraft_config.max_sideslip_angle, dtype=FLOAT_DTYPE
        ),
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
    wind: Wind,
    aircraft_config: AircraftConfig,
    physics_config: PhysicsConfig,
) -> Vector3:
    """
    Calculate angular acceleration for an aircraft.

    Parameters
    ----------
    aircraft : Aircraft
        Current aircraft state including controls and angular velocity.
    wind : Wind
        Wind state with mean and gust components.
    aircraft_config : AircraftConfig
        Aircraft configuration including torque and damping coefficients.
    physics_config : PhysicsConfig
        Physics configuration for air density calculations.

    Returns
    -------
    Vector3
        Angular acceleration in FRD body frame [rad/s^2].
    """
    # Calculate relative velocity (aircraft velocity minus wind)
    wind_velocity = wind.mean + wind.gust
    relative_velocity = aircraft.body.velocity - wind_velocity
    airspeed = norm_3(relative_velocity)

    air_density = physics.calculate_air_density(
        -aircraft.body.position[2],
        jnp.array(physics_config.rho_0, dtype=FLOAT_DTYPE),
        jnp.array(physics_config.rho_decay, dtype=FLOAT_DTYPE),
    )

    q = physics.calculate_dynamic_pressure(airspeed=airspeed, air_density=air_density)

    control_moments = physics.calculate_control_moments(
        velocity=relative_velocity,
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
    wind: Wind,
    aircraft_config: AircraftConfig,
    physics_config: PhysicsConfig,
) -> tuple[Vector3, Vector3, Vector3, Vector3]:
    """
    Calculate complete aircraft state derivatives for numerical integration.

    Parameters
    ----------
    aircraft : Aircraft
        Current aircraft state.
    wind : Wind
        Wind state with mean and gust components.
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
        aircraft, wind, aircraft_config, physics_config
    )

    angular_acceleration = calculate_angular_acceleration(
        aircraft, wind, aircraft_config, physics_config
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
    wind: Wind,
    aircraft_config: AircraftConfig,
    physics_config: PhysicsConfig,
) -> Vector3:
    """
    Calculate G-forces experienced by the aircraft in body frame.

    Parameters
    ----------
    aircraft : Aircraft
        Current aircraft state including position, velocity, and orientation.
    wind : Wind
        Wind state with mean and gust components.
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

    # Relative air velocity (NED) and aero forces (FRD)
    wind_velocity = wind.mean + wind.gust
    relative_velocity = aircraft.body.velocity - wind_velocity

    aero_forces = physics.calculate_aero_forces(
        velocity=relative_velocity,
        orientation=aircraft.body.orientation,
        air_density=air_density,
        surface_areas=jnp.array(aircraft_config.surface_areas, dtype=FLOAT_DTYPE),
        coef_drag=jnp.array(aircraft_config.coef_drag, dtype=FLOAT_DTYPE),
        coef_sideslip=jnp.array(aircraft_config.coef_sideslip, dtype=FLOAT_DTYPE),
        max_attack_angle=jnp.array(aircraft_config.max_attack_angle, dtype=FLOAT_DTYPE),
        zero_lift_attack_angle=jnp.array(
            aircraft_config.zero_lift_attack_angle, dtype=FLOAT_DTYPE
        ),
        lift_slope=jnp.array(aircraft_config.lift_slope, dtype=FLOAT_DTYPE),
        aspect_ratio=jnp.array(aircraft_config.aspect_ratio, dtype=FLOAT_DTYPE),
        oswald_efficiency=jnp.array(
            aircraft_config.oswald_efficiency, dtype=FLOAT_DTYPE
        ),
        max_sideslip_angle=jnp.array(
            aircraft_config.max_sideslip_angle, dtype=FLOAT_DTYPE
        ),
    )

    thrust_force = physics.calculate_thrust(
        throttle=jnp.array(aircraft.controls.throttle, dtype=FLOAT_DTYPE),
        max_thrust=jnp.array(aircraft_config.max_thrust, dtype=FLOAT_DTYPE),
        air_density=air_density,
        rho_0=jnp.array(physics_config.rho_0, dtype=FLOAT_DTYPE),
    )

    # Specific force in body (FRD)
    body_specific_force = (aero_forces + thrust_force) / jnp.array(
        aircraft_config.mass, dtype=FLOAT_DTYPE
    )

    # Convert to g-units
    acceleration_g = body_specific_force / jnp.array(
        physics_config.gravity, dtype=FLOAT_DTYPE
    )

    return acceleration_g


def apply_g_limiter(
    aircraft: Aircraft,
    controls: Controls,
    wind: Wind,
    aircraft_config: AircraftConfig,
    physics_config: PhysicsConfig,
    dt: FloatScalar,
) -> tuple[Controls, PIDControllerState]:
    """
    Apply G-force limiting to elevator input using PID control.

    Parameters
    ----------
    aircraft : Aircraft
        Current aircraft state.
    controls : Controls
        Raw pilot control inputs.
    wind : Wind
        Wind state with mean and gust components.
    aircraft_config : AircraftConfig
        Aircraft configuration including G-limits and PID controller config.
    physics_config : PhysicsConfig
        Physics configuration for G-force calculation.
    dt : FloatScalar
        Time step for derivative calculation.

    Returns
    -------
    tuple[Controls, PIDControllerState]
        Adjusted controls with G-limiting applied and updated PID controller state.
    """
    g_forces = calculate_g_force(aircraft, wind, aircraft_config, physics_config)

    # Negate for pilot convention: positive G = pulling up
    current_g = -g_forces[2]

    g_min = jnp.array(aircraft_config.g_limit_min, dtype=FLOAT_DTYPE)
    g_max = jnp.array(aircraft_config.g_limit_max, dtype=FLOAT_DTYPE)

    saturated_g = jnp.clip(current_g, g_min, g_max)
    error = current_g - saturated_g

    correction, new_pid_state = logic.update_pid(
        error,
        aircraft.g_limiter_pid,
        aircraft_config.g_limiter_controller_config,
        dt,
    )

    adjusted_elevator = jnp.clip(
        controls.elevator - correction,
        jnp.array(-1.0, dtype=FLOAT_DTYPE),
        jnp.array(1.0, dtype=FLOAT_DTYPE),
    )

    adjusted_controls = replace(controls, elevator=adjusted_elevator)

    return adjusted_controls, new_pid_state


def apply_aoa_limiter(
    aircraft: Aircraft,
    controls: Controls,
    wind: Wind,
    aircraft_config: AircraftConfig,
    dt: FloatScalar,
) -> tuple[Controls, PIDControllerState]:
    """ """
    velocity = aircraft.body.velocity - (wind.mean + wind.gust)
    alpha = jnp.rad2deg(
        physics.calculate_angle_of_attack(velocity, aircraft.body.orientation)
    )

    aoa_limit = jnp.array(aircraft_config.aoa_limit, dtype=FLOAT_DTYPE)

    saturated_aoa = jnp.clip(alpha, -aoa_limit, aoa_limit)
    error = alpha - saturated_aoa

    correction, new_pid_state = logic.update_pid(
        error,
        aircraft.aoa_limiter_pid,
        aircraft_config.aoa_limiter_controller_config,
        dt,
    )

    adjusted_elevator = jnp.clip(
        controls.elevator - correction,
        jnp.array(-1.0, dtype=FLOAT_DTYPE),
        jnp.array(1.0, dtype=FLOAT_DTYPE),
    )

    adjusted_controls = replace(controls, elevator=adjusted_elevator)

    return adjusted_controls, new_pid_state


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
    return (
        spatial.spherical_collision(
            position_1=aircraft.body.position,
            position_2=route.positions[route.current_idx],
            distance=jnp.array(route_config.radius, dtype=FLOAT_DTYPE),
        )
        & ~route.visited[route.current_idx]
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
        (next_idx == 0) & loop,
        lambda: jnp.zeros_like(route.visited),
        lambda: visited,
    )

    return replace(route, current_idx=next_idx, visited=visited)


def update_wind(
    key: PRNGKey,
    wind: Wind,
    wind_config: WindConfig,
    dt: FloatScalar,
) -> "Wind":
    """
    Update wind state by evolving the gust component.

    Parameters
    ----------
    key : PRNGKey
        JAX random key for stochastic updates.
    wind : Wind
        Current wind state with mean and gust components.
    wind_config : WindConfig
        Configuration for wind gust generation.
    dt : FloatScalar
        Time step [s].

    Returns
    -------
    Wind
        Updated wind state with new gust component.
    """

    new_gust = calculate_wind(
        key,
        wind.gust,
        jnp.array(wind_config.gust_intensity, dtype=FLOAT_DTYPE),
        jnp.array(wind_config.gust_duration, dtype=FLOAT_DTYPE),
        jnp.array(wind_config.vertical_damping, dtype=FLOAT_DTYPE),
        dt,
    )

    return Wind(mean=wind.mean, gust=new_gust)
