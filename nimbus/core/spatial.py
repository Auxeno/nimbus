"""
Spatial module.

All quantities use SI units, NED world-frame and FRD body-frame unless noted.
"""

import jax
import jax.numpy as jnp

from .primitives import FLOAT_DTYPE, BoolScalar, FloatScalar, Matrix, Vector3, norm_3


def spherical_collision(
    position_1: Vector3,
    position_2: Vector3,
    distance: FloatScalar,
) -> BoolScalar:
    """
    Spherical collision test between two 3D positions.

    Parameters
    ----------
    position_1 : Vector3
        First position in NED world-frame [m].
    position_2 : Vector3
        Second position in NED world-frame [m].
    distance : FloatScalar
        Collision threshold radius [m].

    Returns
    -------
    collision : BoolScalar
        True if distance between positions is <= threshold.
    """
    return norm_3(position_2 - position_1) <= distance


def interpolate_nearest(
    heightmap: Matrix,
    u: FloatScalar,
    v: FloatScalar,
) -> FloatScalar:
    """
    Nearest-neighbour interpolation from a normalised heightmap.

    Parameters
    ----------
    heightmap : Matrix
        2D heightmap with values in [0, 1].
    u : FloatScalar
        Row index in continuous map coordinates.
    v : FloatScalar
        Column index in continuous map coordinates.

    Returns
    -------
    height : FloatScalar
        Heightmap value at nearest integer coordinate [0, 1].
    """
    u_i, v_i = round(u), round(v)
    return heightmap[u_i, v_i]


def interpolate_bilinear(
    heightmap: Matrix,
    u: FloatScalar,
    v: FloatScalar,
) -> FloatScalar:
    """
    Bilinear interpolation from a normalised heightmap.

    Parameters
    ----------
    heightmap : Matrix
        2D heightmap with values in [0, 1].
    u : FloatScalar
        Row index in continuous map coordinates.
    v : FloatScalar
        Column index in continuous map coordinates.

    Returns
    -------
    height : FloatScalar
        Interpolated height value [0, 1].

    Notes
    -----
    Values are clamped to valid bounds.
    """
    rows, cols = heightmap.shape

    # Clamp continuous indices
    u_clamp = jnp.clip(u, 0.0, rows - 1.0)
    v_clamp = jnp.clip(v, 0.0, cols - 1.0)

    # Integer cell corners
    u0 = jnp.floor(u_clamp).astype(jnp.int32)
    v0 = jnp.floor(v_clamp).astype(jnp.int32)
    u1 = jnp.minimum(u0 + 1, rows - 1)
    v1 = jnp.minimum(v0 + 1, cols - 1)

    # Fractional mix factors
    tu = u_clamp - u0
    tv = v_clamp - v0

    # Corner samples
    h00 = heightmap[u0, v0]
    h10 = heightmap[u1, v0]
    h01 = heightmap[u0, v1]
    h11 = heightmap[u1, v1]

    # Bilinear blend
    return h00 * (1 - tu) * (1 - tv) + h10 * tu * (1 - tv) + h01 * (1 - tu) * tv + h11 * tu * tv


def calculate_height_diff(
    heightmap: Matrix,
    position: Vector3,
    map_size: FloatScalar,
    terrain_height: FloatScalar,
    use_bilinear: BoolScalar,
) -> FloatScalar:
    """
    Compute vertical difference between entity and terrain elevation.

    Parameters
    ----------
    heightmap : Matrix
        Normalised terrain heightmap [0, 1].
    position : Vector3
        Entity position in NED world-frame [m].
    map_size : FloatScalar
        Total terrain width/length [m].
    terrain_height : FloatScalar
        Maximum terrain elevation [m].
    use_bilinear : BoolScalar, optional
        If True, use bilinear interpolation; else nearest-neighbour.

    Returns
    -------
    height_diff : FloatScalar
        Difference in down-axis [m]; positive if entity is above ground.
    """
    rows, cols = heightmap.shape
    n, e, d = position

    rows_f = jnp.array(rows, dtype=FLOAT_DTYPE)
    cols_f = jnp.array(cols, dtype=FLOAT_DTYPE)

    # Continuous map coords (fractional indices in [0, rows-1], [0, cols-1])
    u = (n + 0.5 * map_size) / map_size * (rows_f - 1.0)
    v = (e + 0.5 * map_size) / map_size * (cols_f - 1.0)

    def calculate_terrain_height() -> FloatScalar:
        h_norm = jax.lax.cond(
            use_bilinear,
            lambda args: interpolate_bilinear(*args),
            lambda args: interpolate_nearest(*args),
            operand=(heightmap, u, v),
        )
        # Map [0, 1] → [-1, +1] -> [terrain_height, -terrain_height] (NED)
        h_signed = 2.0 * (h_norm - 0.5)
        ground_down = -h_signed * terrain_height
        return d - ground_down

    # Out-of-bounds check (oob terrain height = 0)
    out_of_bounds = jnp.logical_or(
        jnp.logical_or(u < 0.0, u >= rows_f),
        jnp.logical_or(v < 0.0, v >= cols_f),
    )

    return jax.lax.cond(out_of_bounds, lambda: d, lambda: calculate_terrain_height())


def calculate_terrain_collision(
    heightmap: Matrix,
    position: Vector3,
    map_size: FloatScalar,
    terrain_height: FloatScalar,
    use_bilinear: BoolScalar,
) -> BoolScalar:
    """
    Check if a position is colliding with (underneath) terrain surface.

    Parameters
    ----------
    heightmap : Matrix
        Normalised terrain heightmap [0, 1].
    position : Vector3
        Position in NED world-frame [m].
    map_size : FloatScalar
        Total terrain width/length [m].
    terrain_height : FloatScalar
        Maximum terrain elevation [m].
    use_bilinear : BoolScalar, optional
        Whether to use bilinear interpolation for terrain height.

    Returns
    -------
    collision : BoolScalar
        True if position is at or below terrain surface.
    """
    diff = calculate_height_diff(
        heightmap=heightmap,
        position=position,
        map_size=map_size,
        terrain_height=terrain_height,
        use_bilinear=use_bilinear,
    )
    return diff >= 0.0
