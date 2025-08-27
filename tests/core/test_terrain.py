"""Tests for terrain module."""

import jax
import jax.numpy as jnp

from nimbus.core.config import TerrainConfig
from nimbus.core.primitives import EPS, FLOAT_DTYPE
from nimbus.core.terrain import generate_heightmap, simplex_noise_2d

pi = jnp.array(jnp.pi, dtype=FLOAT_DTYPE)


def test_simplex_noise_2d(jit_mode: str) -> None:
    """Test 2D simplex noise generation."""
    # Create a simple permutation table for testing
    key = jax.random.PRNGKey(42)
    permutation = jnp.tile(jax.random.permutation(key, 256), 2)

    # Standard case 1 - single point
    x1 = jnp.array([0.0], dtype=FLOAT_DTYPE)
    y1 = jnp.array([0.0], dtype=FLOAT_DTYPE)
    result_1 = simplex_noise_2d(x1, y1, permutation)
    assert result_1.shape == (1,)
    assert jnp.abs(result_1[0]) <= 1.0  # Should be in range [-1, 1]

    # Standard case 2 - multiple points in a grid
    x2 = jnp.array([[0.0, 1.0], [0.0, 1.0]], dtype=FLOAT_DTYPE)
    y2 = jnp.array([[0.0, 0.0], [1.0, 1.0]], dtype=FLOAT_DTYPE)
    result_2 = simplex_noise_2d(x2, y2, permutation)
    assert result_2.shape == (2, 2)
    assert jnp.all(jnp.abs(result_2) <= 1.0)  # All values in range [-1, 1]

    # Standard case 3 - deterministic behavior with same inputs
    x3 = jnp.array([0.5, 1.5, 2.5], dtype=FLOAT_DTYPE)
    y3 = jnp.array([0.5, 1.5, 2.5], dtype=FLOAT_DTYPE)
    result_3a = simplex_noise_2d(x3, y3, permutation)
    result_3b = simplex_noise_2d(x3, y3, permutation)
    assert jnp.allclose(result_3a, result_3b, atol=EPS)

    # Edge case 1 - negative coordinates
    x4 = jnp.array([-1.0, -2.0], dtype=FLOAT_DTYPE)
    y4 = jnp.array([-1.0, -2.0], dtype=FLOAT_DTYPE)
    result_4 = simplex_noise_2d(x4, y4, permutation)
    assert result_4.shape == (2,)
    assert jnp.all(jnp.abs(result_4) <= 1.0)

    # Edge case 2 - large coordinates
    x5 = jnp.array([100.0, 1000.0], dtype=FLOAT_DTYPE)
    y5 = jnp.array([100.0, 1000.0], dtype=FLOAT_DTYPE)
    result_5 = simplex_noise_2d(x5, y5, permutation)
    assert result_5.shape == (2,)
    assert jnp.all(jnp.abs(result_5) <= 1.0)

    # Test with vmap - batch processing multiple coordinate pairs
    x_coords = jnp.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=FLOAT_DTYPE)
    y_coords = jnp.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=FLOAT_DTYPE)
    permutations = jnp.tile(permutation[None, :], (2, 1))  # Same permutation for both

    simplex_noise_2d_vmap = jax.vmap(simplex_noise_2d)
    vmap_results = simplex_noise_2d_vmap(x_coords, y_coords, permutations)

    assert vmap_results.shape == (2, 3)
    assert jnp.all(jnp.abs(vmap_results) <= 1.0)

    # Check deterministic behavior
    vmap_results_2 = simplex_noise_2d_vmap(x_coords, y_coords, permutations)
    assert jnp.allclose(vmap_results, vmap_results_2, atol=EPS)


def test_generate_heightmap(jit_mode: str) -> None:
    """Test heightmap generation."""
    key = jax.random.PRNGKey(123)
    config = TerrainConfig()

    # Standard case 1 - default parameters from config
    result_1 = generate_heightmap(
        key,
        resolution=config.resolution,
        base_scale=config.base_scale,
        octaves=config.octaves,
        persistence=config.persistence,
        lacunarity=config.lacunarity,
        mountain_gain=config.mountain_gain,
        bump_gain=config.bump_gain,
        padding=config.padding,
        rim_width=config.rim_width,
        split_octave=config.split_octave,
    )
    assert result_1.shape == (config.resolution, config.resolution)
    assert jnp.all(result_1 >= 0.0) and jnp.all(result_1 <= 1.0)  # Values in [0, 1]
    assert result_1.dtype == FLOAT_DTYPE

    # Standard case 2 - custom size
    custom_config = TerrainConfig(resolution=64)
    result_2 = generate_heightmap(
        key,
        resolution=custom_config.resolution,
        base_scale=custom_config.base_scale,
        octaves=custom_config.octaves,
        persistence=custom_config.persistence,
        lacunarity=custom_config.lacunarity,
        mountain_gain=custom_config.mountain_gain,
        bump_gain=custom_config.bump_gain,
        padding=custom_config.padding,
        rim_width=custom_config.rim_width,
        split_octave=custom_config.split_octave,
    )
    assert result_2.shape == (64, 64)
    assert jnp.all(result_2 >= 0.0) and jnp.all(result_2 <= 1.0)

    # Standard case 3 - with padding
    padded_config = TerrainConfig(resolution=32, padding=4)
    result_3 = generate_heightmap(
        key,
        resolution=padded_config.resolution,
        base_scale=padded_config.base_scale,
        octaves=padded_config.octaves,
        persistence=padded_config.persistence,
        lacunarity=padded_config.lacunarity,
        mountain_gain=padded_config.mountain_gain,
        bump_gain=padded_config.bump_gain,
        padding=padded_config.padding,
        rim_width=padded_config.rim_width,
        split_octave=padded_config.split_octave,
    )
    assert result_3.shape == (40, 40)  # 32 + 2*4
    assert jnp.all(result_3 >= 0.0) and jnp.all(result_3 <= 1.0)

    # Standard case 4 - deterministic with same key
    small_config = TerrainConfig(resolution=16)
    result_4a = generate_heightmap(
        key,
        resolution=small_config.resolution,
        base_scale=small_config.base_scale,
        octaves=small_config.octaves,
        persistence=small_config.persistence,
        lacunarity=small_config.lacunarity,
        mountain_gain=small_config.mountain_gain,
        bump_gain=small_config.bump_gain,
        padding=small_config.padding,
        rim_width=small_config.rim_width,
        split_octave=small_config.split_octave,
    )
    result_4b = generate_heightmap(
        key,
        resolution=small_config.resolution,
        base_scale=small_config.base_scale,
        octaves=small_config.octaves,
        persistence=small_config.persistence,
        lacunarity=small_config.lacunarity,
        mountain_gain=small_config.mountain_gain,
        bump_gain=small_config.bump_gain,
        padding=small_config.padding,
        rim_width=small_config.rim_width,
        split_octave=small_config.split_octave,
    )
    assert jnp.allclose(result_4a, result_4b, atol=EPS)

    # Standard case 5 - different with different keys
    key2 = jax.random.PRNGKey(456)
    result_5a = generate_heightmap(
        key,
        resolution=small_config.resolution,
        base_scale=small_config.base_scale,
        octaves=small_config.octaves,
        persistence=small_config.persistence,
        lacunarity=small_config.lacunarity,
        mountain_gain=small_config.mountain_gain,
        bump_gain=small_config.bump_gain,
        padding=small_config.padding,
        rim_width=small_config.rim_width,
        split_octave=small_config.split_octave,
    )
    result_5b = generate_heightmap(
        key2,
        resolution=small_config.resolution,
        base_scale=small_config.base_scale,
        octaves=small_config.octaves,
        persistence=small_config.persistence,
        lacunarity=small_config.lacunarity,
        mountain_gain=small_config.mountain_gain,
        bump_gain=small_config.bump_gain,
        padding=small_config.padding,
        rim_width=small_config.rim_width,
        split_octave=small_config.split_octave,
    )
    assert not jnp.allclose(result_5a, result_5b, atol=1e-3)  # Should be different

    # Edge case 1 - single octave
    single_octave_config = TerrainConfig(resolution=16, octaves=1)
    result_6 = generate_heightmap(
        key,
        resolution=single_octave_config.resolution,
        base_scale=single_octave_config.base_scale,
        octaves=single_octave_config.octaves,
        persistence=single_octave_config.persistence,
        lacunarity=single_octave_config.lacunarity,
        mountain_gain=single_octave_config.mountain_gain,
        bump_gain=single_octave_config.bump_gain,
        padding=single_octave_config.padding,
        rim_width=single_octave_config.rim_width,
        split_octave=single_octave_config.split_octave,
    )
    assert result_6.shape == (16, 16)
    assert jnp.all(result_6 >= 0.0) and jnp.all(result_6 <= 1.0)

    # Edge case 2 - high frequency
    high_freq_config = TerrainConfig(resolution=16, base_scale=1.0)
    result_7 = generate_heightmap(
        key,
        resolution=high_freq_config.resolution,
        base_scale=high_freq_config.base_scale,
        octaves=high_freq_config.octaves,
        persistence=high_freq_config.persistence,
        lacunarity=high_freq_config.lacunarity,
        mountain_gain=high_freq_config.mountain_gain,
        bump_gain=high_freq_config.bump_gain,
        padding=high_freq_config.padding,
        rim_width=high_freq_config.rim_width,
        split_octave=high_freq_config.split_octave,
    )
    assert result_7.shape == (16, 16)
    assert jnp.all(result_7 >= 0.0) and jnp.all(result_7 <= 1.0)

    # Edge case 3 - different gain settings
    gain_config = TerrainConfig(resolution=16, mountain_gain=2.0, bump_gain=0.5)
    result_8 = generate_heightmap(
        key,
        resolution=gain_config.resolution,
        base_scale=gain_config.base_scale,
        octaves=gain_config.octaves,
        persistence=gain_config.persistence,
        lacunarity=gain_config.lacunarity,
        mountain_gain=gain_config.mountain_gain,
        bump_gain=gain_config.bump_gain,
        padding=gain_config.padding,
        rim_width=gain_config.rim_width,
        split_octave=gain_config.split_octave,
    )
    assert result_8.shape == (16, 16)
    assert jnp.all(result_8 >= 0.0) and jnp.all(result_8 <= 1.0)

    # Edge case 4 - radial taper verification (edges should be close to 0.5)
    taper_config = TerrainConfig(resolution=32, rim_width=4)
    result_9 = generate_heightmap(
        key,
        resolution=taper_config.resolution,
        base_scale=taper_config.base_scale,
        octaves=taper_config.octaves,
        persistence=taper_config.persistence,
        lacunarity=taper_config.lacunarity,
        mountain_gain=taper_config.mountain_gain,
        bump_gain=taper_config.bump_gain,
        padding=taper_config.padding,
        rim_width=taper_config.rim_width,
        split_octave=taper_config.split_octave,
    )
    edges = jnp.concatenate(
        [
            result_9[0, :],  # top edge
            result_9[-1, :],  # bottom edge
            result_9[:, 0],  # left edge
            result_9[:, -1],  # right edge
        ]
    )
    # Edge values should be closer to 0.5 than center values
    center_val = result_9[16, 16]
    edge_mean = jnp.mean(edges)
    assert jnp.abs(edge_mean - 0.5) < jnp.abs(center_val - 0.5)

    # Test with vmap - batch generate multiple heightmaps
    keys = jax.random.split(key, 3)
    vmap_config = TerrainConfig(resolution=64, octaves=3)

    # Create vmap function for generate_heightmap with varying base_scale
    def generate_heightmap_batch(key_i):
        return generate_heightmap(
            key_i,
            resolution=vmap_config.resolution,
            base_scale=vmap_config.base_scale,
            octaves=vmap_config.octaves,
            persistence=vmap_config.persistence,
            lacunarity=vmap_config.lacunarity,
            mountain_gain=vmap_config.mountain_gain,
            bump_gain=vmap_config.bump_gain,
            padding=vmap_config.padding,
            rim_width=vmap_config.rim_width,
            split_octave=vmap_config.split_octave,
        )

    generate_heightmap_vmap = jax.vmap(generate_heightmap_batch)
    vmap_results = generate_heightmap_vmap(keys)

    assert vmap_results.shape == (3, 64, 64)
    assert jnp.all(vmap_results >= 0.0) and jnp.all(vmap_results <= 1.0)

    # Each heightmap should be different due to different keys and scales
    assert not jnp.allclose(vmap_results[0], vmap_results[1], atol=1e-3)
    assert not jnp.allclose(vmap_results[1], vmap_results[2], atol=1e-3)

    # Test deterministic behavior with same inputs
    vmap_results_2 = generate_heightmap_vmap(keys)
    assert jnp.allclose(vmap_results, vmap_results_2, atol=EPS)


def test_generate_heightmap_parameters(jit_mode: str) -> None:
    """Test heightmap generation with various parameter combinations."""
    key = jax.random.PRNGKey(789)

    # Test persistence effect
    low_pers_config = TerrainConfig(resolution=16, persistence=0.1, octaves=3)
    high_pers_config = TerrainConfig(resolution=16, persistence=0.9, octaves=3)

    low_persistence = generate_heightmap(
        key,
        resolution=low_pers_config.resolution,
        base_scale=low_pers_config.base_scale,
        octaves=low_pers_config.octaves,
        persistence=low_pers_config.persistence,
        lacunarity=low_pers_config.lacunarity,
        mountain_gain=low_pers_config.mountain_gain,
        bump_gain=low_pers_config.bump_gain,
        padding=low_pers_config.padding,
        rim_width=low_pers_config.rim_width,
        split_octave=low_pers_config.split_octave,
    )
    high_persistence = generate_heightmap(
        key,
        resolution=high_pers_config.resolution,
        base_scale=high_pers_config.base_scale,
        octaves=high_pers_config.octaves,
        persistence=high_pers_config.persistence,
        lacunarity=high_pers_config.lacunarity,
        mountain_gain=high_pers_config.mountain_gain,
        bump_gain=high_pers_config.bump_gain,
        padding=high_pers_config.padding,
        rim_width=high_pers_config.rim_width,
        split_octave=high_pers_config.split_octave,
    )
    assert jnp.all(low_persistence >= 0.0) and jnp.all(low_persistence <= 1.0)
    assert jnp.all(high_persistence >= 0.0) and jnp.all(high_persistence <= 1.0)

    # Test lacunarity effect
    low_lac_config = TerrainConfig(resolution=16, lacunarity=1.5, octaves=3)
    high_lac_config = TerrainConfig(resolution=16, lacunarity=3.0, octaves=3)

    low_lacunarity = generate_heightmap(
        key,
        resolution=low_lac_config.resolution,
        base_scale=low_lac_config.base_scale,
        octaves=low_lac_config.octaves,
        persistence=low_lac_config.persistence,
        lacunarity=low_lac_config.lacunarity,
        mountain_gain=low_lac_config.mountain_gain,
        bump_gain=low_lac_config.bump_gain,
        padding=low_lac_config.padding,
        rim_width=low_lac_config.rim_width,
        split_octave=low_lac_config.split_octave,
    )
    high_lacunarity = generate_heightmap(
        key,
        resolution=high_lac_config.resolution,
        base_scale=high_lac_config.base_scale,
        octaves=high_lac_config.octaves,
        persistence=high_lac_config.persistence,
        lacunarity=high_lac_config.lacunarity,
        mountain_gain=high_lac_config.mountain_gain,
        bump_gain=high_lac_config.bump_gain,
        padding=high_lac_config.padding,
        rim_width=high_lac_config.rim_width,
        split_octave=high_lac_config.split_octave,
    )
    assert jnp.all(low_lacunarity >= 0.0) and jnp.all(low_lacunarity <= 1.0)
    assert jnp.all(high_lacunarity >= 0.0) and jnp.all(high_lacunarity <= 1.0)

    # Test split_octave parameter
    split_early_config = TerrainConfig(resolution=16, octaves=4, split_octave=1)
    split_late_config = TerrainConfig(resolution=16, octaves=4, split_octave=3)

    split_early = generate_heightmap(
        key,
        resolution=split_early_config.resolution,
        base_scale=split_early_config.base_scale,
        octaves=split_early_config.octaves,
        persistence=split_early_config.persistence,
        lacunarity=split_early_config.lacunarity,
        mountain_gain=split_early_config.mountain_gain,
        bump_gain=split_early_config.bump_gain,
        padding=split_early_config.padding,
        rim_width=split_early_config.rim_width,
        split_octave=split_early_config.split_octave,
    )
    split_late = generate_heightmap(
        key,
        resolution=split_late_config.resolution,
        base_scale=split_late_config.base_scale,
        octaves=split_late_config.octaves,
        persistence=split_late_config.persistence,
        lacunarity=split_late_config.lacunarity,
        mountain_gain=split_late_config.mountain_gain,
        bump_gain=split_late_config.bump_gain,
        padding=split_late_config.padding,
        rim_width=split_late_config.rim_width,
        split_octave=split_late_config.split_octave,
    )
    assert jnp.all(split_early >= 0.0) and jnp.all(split_early <= 1.0)
    assert jnp.all(split_late >= 0.0) and jnp.all(split_late <= 1.0)

    # Test rim_width parameter
    narrow_rim_config = TerrainConfig(resolution=32, rim_width=2)
    wide_rim_config = TerrainConfig(resolution=32, rim_width=12)

    narrow_rim = generate_heightmap(
        key,
        resolution=narrow_rim_config.resolution,
        base_scale=narrow_rim_config.base_scale,
        octaves=narrow_rim_config.octaves,
        persistence=narrow_rim_config.persistence,
        lacunarity=narrow_rim_config.lacunarity,
        mountain_gain=narrow_rim_config.mountain_gain,
        bump_gain=narrow_rim_config.bump_gain,
        padding=narrow_rim_config.padding,
        rim_width=narrow_rim_config.rim_width,
        split_octave=narrow_rim_config.split_octave,
    )
    wide_rim = generate_heightmap(
        key,
        resolution=wide_rim_config.resolution,
        base_scale=wide_rim_config.base_scale,
        octaves=wide_rim_config.octaves,
        persistence=wide_rim_config.persistence,
        lacunarity=wide_rim_config.lacunarity,
        mountain_gain=wide_rim_config.mountain_gain,
        bump_gain=wide_rim_config.bump_gain,
        padding=wide_rim_config.padding,
        rim_width=wide_rim_config.rim_width,
        split_octave=wide_rim_config.split_octave,
    )
    assert jnp.all(narrow_rim >= 0.0) and jnp.all(narrow_rim <= 1.0)
    assert jnp.all(wide_rim >= 0.0) and jnp.all(wide_rim <= 1.0)
