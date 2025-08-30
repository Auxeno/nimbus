"""
Quaternion package for 3D orientation and rotation.

Uses scalar-first format [w, x, y, z] with Hamilton product, right-handed
rotations, and ZYX (yaw-pitch-roll) Euler angle conventions.
All angles in radians.
https://blog.mbedded.ninja/mathematics/geometry/quaternions
"""

import jax
import jax.numpy as jnp

from .primitives import (
    EPS,
    FLOAT_DTYPE,
    FloatScalar,
    Matrix,
    Quaternion,
    Vector3,
    norm_3,
    norm_4,
)


def canonicalize(q: Quaternion) -> Quaternion:
    """
    Ensures a quaternion has a non-negative scalar component (w >= 0).

    Parameters
    ----------
    q: (4,) Quaternion
        Input quaternion [w, x, y, z].

    Returns
    -------
    (4,) Quaternion
        Canonical quaternion with w >= 0.
    """
    return jax.lax.cond(
        q[0] < 0.0,
        lambda: -q,
        lambda: q,
    )


def normalize(q: Quaternion) -> Quaternion:
    """
    Normalise a quaternion to unit length.

    Parameters
    ----------
    q : (4,) Quaternion
        Input quaternion [w, x, y, z].

    Returns
    -------
    (4,) Quaternion
        Unit quaternion, or identity if input magnitude is near zero.

    Notes
    -----
    If the quaternion's magnitude is very close to zero (below EPS),
    return the identity quaternion [1, 0, 0, 0].
    """
    magnitude = norm_4(q)
    q_normalized = jax.lax.cond(
        magnitude > EPS,
        lambda: q / magnitude,
        lambda: jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
    )
    return canonicalize(q_normalized)


def multiply(q1: Quaternion, q2: Quaternion) -> Quaternion:
    """
    Multiply two quaternions (Hamilton product).

    Parameters
    ----------
    q1 : (4,) Quaternion
        First quaternion in [w, x, y, z] order.
    q2 : (4,) Quaternion
        Second quaternion in [w, x, y, z] order.

    Returns
    -------
    (4,) Quaternion
        The quaternion product q1 * q2.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=FLOAT_DTYPE,
    )


def conjugate(q: Quaternion) -> Quaternion:
    """
    Return the conjugate of a quaternion.

    Parameters
    ----------
    q : (4,) Quaternion
        Input quaternion [w, x, y, z].

    Returns
    -------
    (4,) Quaternion
        Conjugated quaternion [w, -x, -y, -z].
    """
    w, x, y, z = q
    return jnp.array([w, -x, -y, -z], dtype=FLOAT_DTYPE)


def inverse(q: Quaternion) -> Quaternion:
    """
    Return the inverse of a quaternion.

    Parameters
    ----------
    q : (4,) Quaternion
        Input quaternion [w, x, y, z].

    Returns
    -------
    (4,) Quaternion
        Inverse quaternion, or identity if input magnitude is near zero.

    Notes
    -----
    If the quaternion's magnitude is very close to zero (below EPS),
    return the identity quaternion [1, 0, 0, 0].
    """
    magnitude_sq = q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2
    return jax.lax.cond(
        magnitude_sq > EPS,
        lambda: conjugate(q) / magnitude_sq,
        lambda: jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
    )


def rotate_vector(v: Vector3, q: Quaternion) -> Vector3:
    """
    Rotate a 3D vector using a quaternion.

    Parameters
    ----------
    v : (3,) Vector3
        3D vector to rotate.
    q : (4,) Quaternion
        Unit quaternion representing the rotation.

    Returns
    -------
    (3,) Vector3
        Rotated 3D vector.
    """
    # Convert 3D vector to  pure quaternion
    v_quat = jnp.array([0.0, *v], dtype=FLOAT_DTYPE)

    # Quaternion multiplication formula
    w, x, y, z = multiply(multiply(q, v_quat), conjugate(q))
    return jnp.array([x, y, z], dtype=FLOAT_DTYPE)


def derivative(q: Quaternion, omega: Vector3) -> Quaternion:
    """
    Compute the quaternion derivative given angular velocity in body frame.

    Parameters
    ----------
    q : (4,) Quaternion
        Orientation quaternion [w, x, y, z].
    omega : (3,) Vector3
        Angular velocity vector [ωx, ωy, ωz] in radians/sec.

    Returns
    -------
    (4,) Quaternion
        Quaternion time derivative (dq/dt).
    """
    omega_quat = jnp.array([0.0, *omega], dtype=FLOAT_DTYPE)
    q_dot = 0.5 * multiply(q, omega_quat)
    return q_dot


def slerp(q1: Quaternion, q2: Quaternion, t: FloatScalar) -> Quaternion:
    """
    Perform spherical linear interpolation (SLERP) between two unit quaternions.

    Parameters
    ----------
    q1 : (4,) Quaternion
        Initial quaternion [w, x, y, z], must be unit length.
    q2 : (4,) Quaternion
        Final quaternion [w, x, y, z], must be unit length.
    t : FloatScalar
        Interpolation factor in range [0, 1].

    Returns
    -------
    (4,) Quaternion
        Interpolated unit quaternion.

    Notes
    -----
    If the dot product is negative, we negate q2 to ensure shortest path.
    If the angle between quaternions is very small (dot product > 1 - EPS),
    we fallback to linear interpolation to avoid division by a near-zero sine.
    """
    # Compute the dot product (cosine of half-angle)
    cos_theta = jnp.dot(q1, q2)

    # Use shortest interpolation path
    q2 = jnp.where(cos_theta < 0.0, -q2, q2)
    cos_theta = jnp.abs(cos_theta)

    # If angle is small, use linear interpolation to avoid division by zero
    def linear_interp() -> Quaternion:
        return (1.0 - t) * q1 + t * q2

    def spherical_interp() -> Quaternion:
        # Compute angle between quaternions
        theta = jnp.arccos(jnp.clip(cos_theta, -1.0, 1.0))
        sin_theta = jnp.sin(theta)

        # Compute interpolation weights
        weight1 = jnp.sin((1.0 - t) * theta) / sin_theta
        weight2 = jnp.sin(t * theta) / sin_theta
        return weight1 * q1 + weight2 * q2

    q_interp = jax.lax.cond(
        cos_theta > 1.0 - EPS,
        linear_interp,
        spherical_interp,
    )
    return normalize(q_interp)


def from_axis_angle(axis: Vector3, angle: FloatScalar) -> Quaternion:
    """
    Create a quaternion from axis-angle representation.

    Parameters
    ----------
    axis : (3,) Vector3
        3D rotation axis.
    angle : FloatScalar
        Rotation angle in radians.

    Returns
    -------
    (4,) Quaternion
        Quaternion representing the rotation.

    Notes
    -----
    If the axis has near-zero magnitude, it defaults to [1, 0, 0] (x-axis),
    producing a valid quaternion for the given angle.
    """
    # Normalise axis vector
    magnitude = norm_3(axis)
    x, y, z = jax.lax.cond(
        magnitude > EPS,
        lambda: axis / magnitude,
        lambda: jnp.array([1.0, 0.0, 0.0], dtype=FLOAT_DTYPE),
    )

    # Axis angle to quaternion formula
    s, c = jnp.sin(0.5 * angle), jnp.cos(0.5 * angle)
    return jnp.array([c, x * s, y * s, z * s], dtype=FLOAT_DTYPE)


def from_euler_zyx(
    yaw: FloatScalar,
    pitch: FloatScalar,
    roll: FloatScalar,
) -> Quaternion:
    """
    Create a quaternion from ZYX Euler angles (yaw, pitch, roll), in radians.

    Parameters
    ----------
    yaw : FloatScalar
        Rotation about the Z-axis (ψ, heading).
    pitch : FloatScalar
        Rotation about the Y-axis (θ, attitude).
    roll : FloatScalar
        Rotation about the X-axis (φ, bank).

    Returns
    -------
    (4,) Quaternion
        Unit quaternion [w, x, y, z] representing the same orientation.
    """
    # Pre-compute half angles
    hy, hp, hr = 0.5 * yaw, 0.5 * pitch, 0.5 * roll
    cy, sy = jnp.cos(hy), jnp.sin(hy)
    cp, sp = jnp.cos(hp), jnp.sin(hp)
    cr, sr = jnp.cos(hr), jnp.sin(hr)

    # Standard ZYX to quaternion formula
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return normalize(jnp.array([w, x, y, z], dtype=FLOAT_DTYPE))


def to_euler_zyx(q: Quaternion) -> Vector3:
    """
    Convert a quaternion to ZYX Euler angles (yaw, pitch, roll), in radians.

    Parameters
    ----------
    q : (4,) Quaternion
        Unit quaternion [w, x, y, z].

    Returns
    -------
    (3,) Vector3
        [yaw, pitch, roll] angles in radians.
    """
    w, x, y, z = q
    yaw = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    pitch = jnp.arcsin(jnp.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    roll = jnp.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return jnp.array([yaw, pitch, roll], dtype=FLOAT_DTYPE)


def to_rotation_matrix(q: Quaternion) -> Matrix:
    """
    Convert a quaternion to a 3x3 rotation matrix.

    Parameters
    ----------
    q : (4,) Quaternion
        Unit quaternion [w, x, y, z].

    Returns
    -------
    (3, 3) Matrix
        3x3 rotation matrix.
    """
    w, x, y, z = q
    return jnp.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=FLOAT_DTYPE,
    )
