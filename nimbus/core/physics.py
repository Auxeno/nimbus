"""
Physics module for computing forces, torques and motion.

All quantities use SI units, NED world-frame and FRD body-frame unless noted.
"""

import jax
import jax.numpy as jnp

from . import quaternion
from .primitives import EPS, FLOAT_DTYPE, FloatScalar, Quaternion, Vector3, norm_3


def calculate_air_density(
    altitude: FloatScalar,
    rho_0: FloatScalar,
    rho_decay: FloatScalar,
) -> FloatScalar:
    """
    Calculate air density from altitude with an exponential decay model.

    Parameters
    ----------
    altitude: FloatScalar
        Altitude above sea level [m]
    rho_0: FloatScalar
        Air density at sea level [kg/m^3].
    rho_decay: FloatScalar
        Air density halves when altitude increases by this much [m].

    Returns
    -------
    air_density: FloatScalar
        Density of air at specified altitude [kg/m^3].

    Notes
    -----
    - rho_0 = 1.225 and rho_decay = 5500.0 are rough true approximations under ISA
    conditions.
    """
    decay_rate = jnp.log(2.0) / rho_decay
    air_density = rho_0 * jnp.exp(-decay_rate * altitude)
    return air_density


def calculate_dynamic_pressure(
    airspeed: FloatScalar,
    air_density: FloatScalar,
) -> FloatScalar:
    """
    Calculate dynamic pressure from airspeed and air density.

    Parameters
    ----------
    airspeed: FloatScalar
        True airspeed [m/s].
    air_density: FloatScalar
        Density of air [kg/m^3].

    Returns
    -------
    q: FloatScalar
        Dynamic pressure [N/m^2].
    """
    q = 0.5 * air_density * airspeed**2
    return q


def calculate_dynamic_pressure_scale(
    q: FloatScalar,
    q_falloff_speed: FloatScalar,
    air_density: FloatScalar,
) -> FloatScalar:
    """
    Apply low-speed aero effectiveness scaling based on a reference speed.

    Parameters
    ----------
    q: FloatScalar
        Dynamic pressure [N/m^2].
    q_falloff_speed: FloatScalar
        Used to scale down control surface effectiveness at low speeds [m/s].
    air_density: FloatScalar
        Density of air [kg/m^3].

    Returns
    -------
    scale: FloatScalar
        Scale of dynamic pressure [0, 1].
    """

    # Dynamic pressure at the reference and half reference speeds
    q_ref = calculate_dynamic_pressure(q_falloff_speed, air_density)
    q_half = calculate_dynamic_pressure(0.5 * q_falloff_speed, air_density)

    scale = (q - q_half) / (q_ref - q_half)
    scale = jnp.where(q <= q_half, 0.0, scale)
    scale = jnp.where(q >= q_ref, 1.0, scale)
    return scale


def calculate_angle_of_attack(
    velocity: Vector3,
    orientation: Quaternion,
) -> FloatScalar:
    """
    Calculate angle of attack (alpha) in radians from velocity and orientation.

    Parameters
    ----------
    velocity: Vector3
        Relative velocity in NED world-frame [m/s].
    orientation: Quaternion
        Body to world orientation quaternion.

    Returns
    -------
    alpha: FloatScalar
        Angle of attack between body forward axis and
        velocity vector in XZ plane [radians].
    """
    velocity_body = quaternion.rotate_vector(velocity, quaternion.inverse(orientation))
    alpha = jnp.arctan2(velocity_body[2], velocity_body[0])
    return alpha


def calculate_angle_of_sideslip(
    velocity: Vector3,
    orientation: Quaternion,
) -> FloatScalar:
    """
    Calculate sideslip angle (beta) in radians from velocity and orientation.

    Parameters
    ----------
    velocity: Vector3
        Relative velocity in NED world-frame [m/s].
    orientation: Quaternion
        Body to world orientation quaternion.

    Returns
    -------
    beta: FloatScalar
        Angle of sideslip [radians].
    """
    velocity_body = quaternion.rotate_vector(velocity, quaternion.inverse(orientation))
    beta = jnp.arctan2(velocity_body[1], velocity_body[0])
    return beta


def calculate_weight(mass: FloatScalar, gravity: FloatScalar) -> Vector3:
    """
    Calculate gravitational force in NED world-frame.

    Parameters
    ----------
    mass: FloatScalar
        Body mass [kg].
    gravity: FloatScalar
        Gravitational acceleration rate, positive down [m/s^2].

    Returns
    -------
    weight: Vector3
        Weight force vector in NED world-frame [N].
    """
    weight = jnp.array([0.0, 0.0, mass * gravity], dtype=FLOAT_DTYPE)
    return weight


def calculate_thrust(
    throttle: FloatScalar,
    max_thrust: FloatScalar,
    air_density: FloatScalar,
    rho_0: FloatScalar,
) -> Vector3:
    """
    Calculate engine thrust in FRD body-frame.

    Parameters
    ----------
    throttle: FloatScalar
        Throttle input value [0, 1].
    max_thrust: FloatScalar
        Maximum engine thrust [N].
    air_density: FloatScalar
        Density of air [kg/m^3].
    rho_0: FloatScalar
        Density of air at sea level [kg/m^3].

    Returns
    -------
    thrust: Vector3
        Engine thrust force vector in FRD body-frame [N].
    """
    thrust_scalar = throttle * max_thrust * (air_density / rho_0)
    thrust = jnp.array([thrust_scalar, 0.0, 0.0], dtype=FLOAT_DTYPE)
    return thrust


def calculate_lift(
    alpha: FloatScalar,
    dynamic_pressure: FloatScalar,
    coef_lift: FloatScalar,
    wing_area: FloatScalar,
    max_angle_deg: FloatScalar,
) -> FloatScalar:
    """
    Scalar lift (signed by alpha), no direction.
    Positive value corresponds to +lift_axis (up) in FRD.

    Parameters
    ----------
    alpha : FloatScalar
        Angle of attack [rad].
    dynamic_pressure : FloatScalar
        Aerodynamic pressure [N/m^2].
    coef_lift : FloatScalar
        Lift effectiveness coefficient.
    wing_area : FloatScalar
        Wing planform area [m^2].
    max_angle_deg : FloatScalar
        AoA that yields maximum lift [deg].

    Returns
    -------
    lift : FloatScalar
        Signed lift magnitude [N], to be applied along +lift_axis.
    """
    max_angle_rad = jnp.deg2rad(max_angle_deg)
    zero_lift_angle_rad = 2 * max_angle_rad
    abs_alpha = jnp.abs(alpha)

    falloff = (zero_lift_angle_rad - abs_alpha) / jnp.maximum(
        zero_lift_angle_rad - max_angle_rad, EPS
    )

    alpha_gain = jnp.where(
        abs_alpha <= max_angle_rad,
        coef_lift * alpha,
        coef_lift * max_angle_rad * falloff * jnp.sign(alpha),
    )
    alpha_gain = jnp.where(abs_alpha > zero_lift_angle_rad, 0.0, alpha_gain)

    lift = dynamic_pressure * alpha_gain * wing_area

    return lift


def calculate_sideslip(
    beta: FloatScalar,
    dynamic_pressure: FloatScalar,
    coef_sideslip: FloatScalar,
    side_area: FloatScalar,
    max_angle_deg: FloatScalar,
) -> FloatScalar:
    """
    Scalar lateral force from sideslip (signed), no direction.
    Positive return value is along +side_axis (right) in FRD.

    Parameters
    ----------
    beta : FloatScalar
        Sideslip angle [rad].
    dynamic_pressure : FloatScalar
        Aerodynamic pressure [N/m^2].
    coef_sideslip : FloatScalar
        Sideslip effectiveness coefficient.
    side_area : FloatScalar
        Effective lateral area [m^2].
    max_angle_deg : FloatScalar
        Beta that yields maximum lateral force [deg].

    Returns
    -------
    sideslip : FloatScalar
        Signed sideslip force [N], to be applied along +side_axis.
    """
    max_angle_rad = jnp.deg2rad(max_angle_deg)
    zero_sideslip_angle_rad = 2.0 * max_angle_rad
    abs_beta = jnp.abs(beta)

    falloff = (zero_sideslip_angle_rad - abs_beta) / jnp.maximum(
        zero_sideslip_angle_rad - max_angle_rad, EPS
    )

    beta_gain = jnp.where(
        abs_beta <= max_angle_rad,
        coef_sideslip * beta,
        coef_sideslip * max_angle_rad * falloff * jnp.sign(beta),
    )
    beta_gain = jnp.where(abs_beta > zero_sideslip_angle_rad, 0.0, beta_gain)

    sideslip = -dynamic_pressure * beta_gain * side_area

    return sideslip


def calculate_drag(
    velocity: Vector3,
    orientation: Quaternion,
    airspeed: FloatScalar,
    dynamic_pressure: FloatScalar,
    coef_drag: FloatScalar,
    surface_areas: Vector3,
) -> FloatScalar:
    """
    Scalar form-drag magnitude (non-negative), no direction.

    Uses the same projected-area cuboid model as the previous vector version.
    The returned scalar should be applied along +drag_axis (opposes relative wind).

    Parameters
    ----------
    velocity : Vector3
        Relative air velocity in NED [m/s].
    orientation : Quaternion
        Body->world orientation quaternion.
    airspeed : FloatScalar
        |velocity| [m/s].
    dynamic_pressure : FloatScalar
        Aerodynamic pressure [N/m^2].
    coef_drag : FloatScalar
        Drag coefficient.
    surface_areas : Vector3
        Front, side, top areas [m^2].

    Returns
    -------
    drag : FloatScalar
        Non-negative drag magnitude [N], to be applied along +drag_axis.
    """
    velocity_body = quaternion.rotate_vector(velocity, quaternion.inverse(orientation))
    velocity_dir_body = velocity_body / jnp.maximum(airspeed, EPS)

    # Projected area of a cuboid against the flow direction
    projected_area = jnp.dot(surface_areas, jnp.abs(velocity_dir_body))
    drag = coef_drag * dynamic_pressure * projected_area

    return drag


def calculate_aero_axes(
    velocity: Vector3,
    orientation: Quaternion,
) -> tuple[Vector3, Vector3, Vector3]:
    """
    Build aerodynamic unit axes in the FRD body-frame from relative air velocity.

    Parameters
    ----------
    velocity : Vector3
        Aircraft velocity relative to the airmass in NED world-frame [m/s].
    orientation : Quaternion
        Body-to-world orientation quaternion.

    Returns
    -------
    drag_axis : Vector3
        Unit vector in FRD body-frame pointing along +drag (opposes the relative wind).
    side_axis : Vector3
        Unit vector in FRD body-frame pointing laterally (positive right).
    lift_axis : Vector3
        Unit vector in FRD body-frame perpendicular to the airflow (positive up).

    Notes
    -----
    - Axes are returned in FRD convention (+x forward, +y right, +z down).
    - +drag is defined to oppose the airflow. The plane orthogonal to +drag
      spans the side/lift directions.
    - To fix the in-plane rotation about +drag in a smooth, roll-agnostic way,
      we construct the **minimal rotation** that maps body forward to the airflow
      direction, then carry body right through that rotation to define +side.
      Finally, +lift is formed with a normalized cross product to ensure an
      orthonormal, right-handed basis.
    """

    # Velocity in body-frame and corresponding +drag direction
    velocity_body = quaternion.rotate_vector(velocity, quaternion.inverse(orientation))
    airspeed = jnp.maximum(norm_3(velocity_body), EPS)
    drag_axis = -velocity_body / airspeed  # +drag opposes relative wind in FRD

    # Body-frame basis vectors (FRD: +x forward, +y right, +z down)
    body_forward = jnp.array([1.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    body_right = jnp.array([0.0, 1.0, 0.0], dtype=FLOAT_DTYPE)
    body_down = jnp.array([0.0, 0.0, 1.0], dtype=FLOAT_DTYPE)

    # Minimal-rotation quaternion that maps body_forward to target_forward
    target_forward = -drag_axis
    source_forward = body_forward

    # Two-vector quaternion (generic case)
    q_vec = jnp.cross(source_forward, target_forward)
    q_w = 1.0 + jnp.dot(source_forward, target_forward)

    # Degenerate case: source_forward ≈ -target_forward (180 deg)
    # Choose any axis orthogonal to source_forward, prefer body right
    near_pi = q_w <= EPS
    fallback_axis = jnp.where(
        jnp.abs(jnp.dot(body_right, source_forward)) < 0.9, body_right, -body_down
    )
    q_w_alt = 0.0
    q_vec_alt = fallback_axis

    # Select generic or fallback quaternion components
    q_w = jnp.where(near_pi, q_w_alt, q_w)
    q_vec = jnp.where(near_pi, q_vec_alt, q_vec)

    # Normalise quaternion [w, x, y, z]
    q_norm = jnp.sqrt(q_w * q_w + jnp.dot(q_vec, q_vec))
    q_w = q_w / jnp.maximum(q_norm, EPS)
    q_vec = q_vec / jnp.maximum(q_norm, EPS)
    q_minrot = jnp.concatenate([jnp.array([q_w], dtype=FLOAT_DTYPE), q_vec], axis=0)

    # Carry body right through the minimal rotation to define +side
    side_axis = quaternion.rotate_vector(body_right, q_minrot)
    side_axis = side_axis / jnp.maximum(norm_3(side_axis), EPS)

    # Hemisphere stabilisation toward +body_right
    side_axis = jnp.where(jnp.dot(side_axis, body_right) < 0.0, -side_axis, side_axis)

    # +lift from cross product ensures orthogonality and right-handedness
    lift_axis = jnp.cross(drag_axis, side_axis)
    lift_axis = lift_axis / jnp.maximum(norm_3(lift_axis), EPS)

    # Point +lift up in FRD (opposite body down)
    lift_axis = jnp.where(jnp.dot(lift_axis, -body_down) < 0.0, -lift_axis, lift_axis)

    return drag_axis, side_axis, lift_axis


def calculate_aero_forces(
    velocity: Vector3,
    orientation: Quaternion,
    air_density: FloatScalar,
    coef_drag: FloatScalar,
    coef_lift: FloatScalar,
    coef_sideslip: FloatScalar,
    max_attack_angle: FloatScalar,
    max_sideslip_angle: FloatScalar,
    surface_areas: Vector3,
) -> Vector3:
    """
    Calculate aerodynamic forces in FRD body-frame using drag, lift and sideslip.

    Parameters
    ----------
    velocity: Vector3
        Relative velocity in NED world-frame [m/s].
    orientation: Quaternion
        Body to world orientation quaternion.
    air_density: FloatScalar
        Density of air [kg/m^3].
    coef_drag: FloatScalar
        Coefficient scaling scale of drag.
    coef_lift: FloatScalar
        Coefficient scaling effectiveness of lift.
    coef_sideslip: FloatScalar
        Coefficient scaling effectiveness of sideslip.
    max_attack_angle: FloatScalar
        Angle of attack that generates maximum lift [degrees].
    max_sideslip_angle: FloatScalar
        Angle of sideslip that generates maximum sideslip [degrees].
    surface_areas: Vector3
        Surface areas of front, side and top areas of body [m^2].

    Returns
    -------
    force_body: Vector3
        Resultant aerodynamic force (lift + drag + sideslip) in FRD body-frame [N].
    """
    airspeed = norm_3(velocity)

    def aero_forces():
        q = calculate_dynamic_pressure(airspeed, air_density)
        alpha = calculate_angle_of_attack(velocity, orientation)
        beta = calculate_angle_of_sideslip(velocity, orientation)

        drag_axis, side_axis, lift_axis = calculate_aero_axes(velocity, orientation)
        drag = calculate_drag(
            velocity, orientation, airspeed, q, coef_drag, surface_areas
        )
        lift = calculate_lift(alpha, q, coef_lift, surface_areas[2], max_attack_angle)
        sideslip = calculate_sideslip(
            beta, q, coef_sideslip, surface_areas[1], max_sideslip_angle
        )
        force_body = drag * drag_axis + lift * lift_axis + sideslip * side_axis
        return force_body

    force_body = jax.lax.cond(
        airspeed >= EPS,
        lambda: aero_forces(),
        lambda: jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
    )

    return force_body


def calculate_control_moments(
    velocity: Vector3,
    orientation: Quaternion,
    airspeed: FloatScalar,
    dynamic_pressure: FloatScalar,
    aileron: FloatScalar,
    elevator: FloatScalar,
    rudder: FloatScalar,
    coefs_torque: Vector3,
) -> Vector3:
    """
    Calculate control surface moments in FRD body-frame.

    Parameters
    ----------
    velocity: Vector3
        Relative velocity in NED world-frame [m/s].
    orientation: Quaternion
        Body to world orientation quaternion.
    airspeed: FloatScalar
        True airspeed (magnitude of relative velocity) [m/s].
    dynamic_pressure: FloatScalar
        Aerodynamic pressure on body [N/m^2].
    aileron: FloatScalar
        Aileron (roll) input [-1, 1]
    elevator: FloatScalar
        Elevator (pitch) input [-1, 1]
    rudder: FloatScalar
        Rudder (yaw) input [-1, 1]
    coefs_torque: Vector3
        Torque effectiveness coefficients for roll, pitch and yaw.

    Returns
    -------
    moments: Vector3
        Moments about roll, pitch and yaw axes [N/m].
    """

    def moments() -> Vector3:
        # Velocity in body frame
        velocity_body = quaternion.rotate_vector(
            velocity, quaternion.inverse(orientation)
        )
        velocity_dir_body = velocity_body / airspeed

        # Cosine of angle between velocity and forward axis
        alignment = jnp.dot(velocity_dir_body, jnp.array([1.0, 0.0, 0.0]))

        # Inputs scaled by velocity and orientation alignment
        inputs = jnp.array([aileron, elevator, rudder], dtype=FLOAT_DTYPE)
        moments = alignment * dynamic_pressure * coefs_torque * inputs
        return moments

    return jax.lax.cond(
        airspeed > EPS,
        moments,
        lambda: jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
    )


def calculate_damping_moments(
    angular_velocity: Vector3,
    dynamic_pressure: FloatScalar,
    coef_rot_damping: FloatScalar,
) -> Vector3:
    """
    Calculate angular damping moments in FRD body-frame.

    Parameters
    ----------
    angular_velocity: Vector3
        Angular velocity about roll, pitch and yaw axes [radians/s].
    dynamic_pressure: FloatScalar
        Aerodynamic pressure on body [N/m^2].
    coef_rot_damping: FloatScalar
        Angular velocity damping rate coefficient.

    Returns
    -------
    moments: Vector3
        Moments about roll, pitch and yaw axes [N/m].
    """
    moments = -angular_velocity * dynamic_pressure * coef_rot_damping
    return moments


def estimate_inertia(mass: FloatScalar, surface_areas: Vector3) -> Vector3:
    """
    Cheaply approximate XX YY ZZ inertia diagonal.

    Parameters
    ----------
    mass: FloatScalar
        Body mass [kg].
    surface_areas: Vector3
        Surface areas of front, side and top areas of body [m^2].

    Returns
    -------
    inertia_diagonal: Vector3
        Diagonal of inertia matrix [kg m^2].
    """
    inertia_diagonal = mass * jnp.sqrt(surface_areas)
    return inertia_diagonal
