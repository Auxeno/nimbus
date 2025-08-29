"""
Primitives module for shared vector aliases, numerical constants, and basic functions.
"""

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, Scalar

# Project precision settings
FLOAT_DTYPE = jnp.float32
INT_DTYPE = jnp.int32
EPS = 1e-8

# Project type aliases
Scalar = Scalar
BoolScalar = Bool[Array, ""]
IntScalar = Int[Array, ""]
FloatScalar = Float[Array, ""]
Vector = Float[Array, "N"]
Vector2 = Float[Array, "2"]
Vector3 = Float[Array, "3"]
Quaternion = Float[Array, "4"]
Matrix = Float[Array, "N M"]
Array = Array
PRNGKey = PRNGKeyArray


def norm_2(v: Vector2) -> FloatScalar:
    """
    Compute the Euclidean norm of a 2D vector.

    Parameters
    ----------
    v : Vector2
        2D vector [x, y].

    Returns
    -------
    norm : FloatScalar
        L2 norm (magnitude) of the vector.
    """
    return jnp.sqrt(v[0] ** 2 + v[1] ** 2)


def norm_3(v: Vector3) -> FloatScalar:
    """
    Compute the Euclidean norm of a 3D vector.

    Parameters
    ----------
    v : Vector3
        3D vector [x, y, z].

    Returns
    -------
    norm : FloatScalar
        L2 norm (magnitude) of the vector.
    """
    return jnp.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def norm_4(v: Quaternion) -> FloatScalar:
    """
    Compute the Euclidean norm of a quaternion.

    Parameters
    ----------
    v : Quaternion
        Quaternion [w, x, y, z].

    Returns
    -------
    norm : FloatScalar
        L2 norm (magnitude) of the quaternion.
    """
    return jnp.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2 + v[3] ** 2)
