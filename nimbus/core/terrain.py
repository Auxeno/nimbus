"""Terrain generation module for procedural heightmaps.

Generate heightmaps using fractal noise for flight simulation environments.
Terrain values represent normalized heights in range [0, 1], where higher
values correspond to increased terrain elevation in world coordinates.
"""

import jax
import jax.numpy as jnp

from .primitives import (
    EPS,
    FLOAT_DTYPE,
    INT_DTYPE,
    FloatScalar,
    IntScalar,
    Matrix,
    PRNGKey,
    Vector,
)

# Gradient vectors for simplex noise
# Unit vectors pointing to 8 directions (4 cardinal + 4 inter-cardinal)
_GRADIENTS: Matrix = jnp.array(
    [
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [-1.0, -1.0],
    ],
    dtype=FLOAT_DTYPE,
) / jnp.sqrt(2.0)


def simplex_noise_2d(x: Matrix, y: Matrix, permutation: Vector) -> Matrix:
    """
    Generate 2D simplex noise at given coordinates.

    Parameters
    ----------
    x : Matrix
        X coordinates to sample noise at.
    y : Matrix
        Y coordinates to sample noise at.
    permutation : Vector
        Permutation table of length 512 for gradient hashing.

    Returns
    -------
    Matrix
        Noise values in the range [-1, 1] at each coordinate.
    """
    skew_factor = (jnp.sqrt(3.0) - 1.0) / 2.0
    unskew_factor = (3.0 - jnp.sqrt(3.0)) / 6.0

    # Skew input space so the grid becomes a simplex grid
    skew = (x + y) * skew_factor
    cell_x = jnp.floor(x + skew).astype(INT_DTYPE)
    cell_y = jnp.floor(y + skew).astype(INT_DTYPE)

    # Unskew to the cell origin in (x, y)
    unskew = (cell_x + cell_y) * unskew_factor
    origin_x = cell_x - unskew
    origin_y = cell_y - unskew

    # Distances from cell origin
    dx0 = x - origin_x
    dy0 = y - origin_y

    # Which triangle of the cell
    step_x = jnp.where(dx0 > dy0, 1, 0)
    step_y = 1 - step_x

    # Offsets for the other corners
    dx1 = dx0 - step_x + unskew_factor
    dy1 = dy0 - step_y + unskew_factor
    dx2 = dx0 - 1.0 + 2.0 * unskew_factor
    dy2 = dy0 - 1.0 + 2.0 * unskew_factor

    # Hash gradient indices for the three corners
    ii = cell_x & 255
    jj = cell_y & 255

    gi0 = permutation[permutation[ii] + jj] & 7
    gi1 = permutation[permutation[ii + step_x] + jj + step_y] & 7
    gi2 = permutation[permutation[ii + 1] + jj + 1] & 7

    def corner(grad_index: Matrix, dx: Matrix, dy: Matrix) -> Matrix:
        t = 0.5 - dx * dx - dy * dy
        # Only contribute where inside the circle; raise to 4th power for smooth fade
        t4 = jnp.where(t > 0, t**4, 0.0)
        g = _GRADIENTS[grad_index]  # (..., 2)
        dot = g[..., 0] * dx + g[..., 1] * dy
        return t4 * dot

    n0 = corner(gi0, dx0, dy0)
    n1 = corner(gi1, dx1, dy1)
    n2 = corner(gi2, dx2, dy2)

    # Empirical scale so output is [-1, 1]
    return 70.0 * (n0 + n1 + n2)


def generate_heightmap(
    key: PRNGKey,
    *,
    resolution: IntScalar,
    base_scale: FloatScalar,
    octaves: IntScalar,
    persistence: FloatScalar,
    lacunarity: FloatScalar,
    mountain_gain: FloatScalar,
    bump_gain: FloatScalar,
    padding: IntScalar,
) -> Matrix:
    """
    Generate a square heightmap using fractional Brownian motion with radial tapering.

    Parameters
    ----------
    key : PRNGKey
        JAX random key for noise generation.
    resolution : IntScalar
        Width and height of the square heightmap in pixels.
    base_scale : FloatScalar
        Base frequency scale for the noise function.
    octaves : IntScalar
        Number of noise layers to combine.
    persistence : FloatScalar
        Amplitude reduction factor for each octave.
    lacunarity : FloatScalar
        Frequency increase factor for each octave.
    mountain_gain : FloatScalar
        Amplitude multiplier for lower-frequency octaves (mountains).
    bump_gain : FloatScalar
        Amplitude multiplier for higher-frequency octaves (surface detail).
    padding : IntScalar
        Number of pixels to pad around edges with constant value 0.5.

    Returns
    -------
    Matrix
        Square heightmap with normalized height values in range [0, 1].
        Higher values represent elevated terrain in world coordinates.
        Shape is (resolution, resolution) or (resolution + 2*padding,
        resolution + 2*padding) if padding > 0.

    Notes
    -----
    The heightmap uses a two-stage octave system where lower octaves
    form large-scale features (mountains) and higher octaves add
    surface detail (bumps). A radial taper smoothly transitions
    heights to 0.5 at the boundaries to create natural-looking edges.

    Terrain values map to world elevation: higher values = terrain up,
    lower values = terrain down, relative to reference level.
    """

    # Width of outer rim [pixels]
    rim_width = resolution // 8

    # Octave at which surface detail starts
    split_octave = octaves // 2

    # Build a 512 entry permutation table
    permutation = jnp.tile(jax.random.permutation(key, 256), 2)

    # Make grid in array (row-major) coordinates
    ys = jnp.arange(resolution, dtype=FLOAT_DTYPE)
    xs = jnp.arange(resolution, dtype=FLOAT_DTYPE)
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing="ij")  # (resolution, resolution)

    height = jnp.zeros((resolution, resolution), dtype=FLOAT_DTYPE)
    total_amp = jnp.array(0.0, dtype=FLOAT_DTYPE)
    amplitude = jnp.array(1.0, dtype=FLOAT_DTYPE)
    frequency = jnp.array(1.0, dtype=FLOAT_DTYPE)

    for octave_idx in range(octaves):
        layer = simplex_noise_2d(
            grid_x * (base_scale * frequency),
            grid_y * (base_scale * frequency),
            permutation,
        )
        gain = mountain_gain if octave_idx < split_octave else bump_gain
        layer_amp = amplitude * jnp.asarray(gain, dtype=FLOAT_DTYPE)
        height = height + layer_amp * layer
        total_amp = total_amp + layer_amp
        amplitude = amplitude * persistence
        frequency = frequency * lacunarity

    # Normalise to [0, 1]
    height = (height + total_amp) / (2.0 * total_amp + EPS)

    # Radial taper to 0.5 near the edge
    centre = (resolution - 1) / 2.0
    distance = jnp.hypot(grid_x - centre, grid_y - centre)
    edge_mask = jnp.clip((centre - rim_width - distance) / rim_width, 0.0, 1.0)
    height = height * edge_mask + 0.5 * (1.0 - edge_mask)

    # Optional padding
    if padding > 0:
        height = jnp.pad(
            height,
            pad_width=((padding, padding), (padding, padding)),
            mode="constant",
            constant_values=0.5,
        )

    return height
