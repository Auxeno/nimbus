"""Tests for spatial module."""

import jax
import jax.numpy as jnp

from nimbus.core.primitives import EPS, FLOAT_DTYPE
from nimbus.core.spatial import (
    calculate_height_diff,
    calculate_terrain_collision,
    interpolate_bilinear,
    interpolate_nearest,
    spherical_collision,
)


def test_spherical_collision(jit_mode: str) -> None:
    """Test spherical collision detection."""
    # Common values
    one_val = jnp.array(1.0, dtype=FLOAT_DTYPE)
    point_one_val = jnp.array(0.1, dtype=FLOAT_DTYPE)
    zero_val = jnp.array(0.0, dtype=FLOAT_DTYPE)

    # Standard case 1 - no collision
    pos1 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    pos2 = jnp.array([2.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    distance = one_val
    result_1 = spherical_collision(pos1, pos2, distance)
    expected_1 = False
    assert result_1 == expected_1

    # Standard case 2 - collision at exact distance
    pos3 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    pos4 = jnp.array([1.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    distance = one_val
    result_2 = spherical_collision(pos3, pos4, distance)
    expected_2 = True
    assert result_2 == expected_2

    # Standard case 3 - collision within distance
    pos5 = jnp.array([1.0, 1.0, 1.0], dtype=FLOAT_DTYPE)
    pos6 = jnp.array([1.5, 1.5, 1.5], dtype=FLOAT_DTYPE)
    distance = one_val
    result_3 = spherical_collision(pos5, pos6, distance)
    expected_3 = True
    assert result_3 == expected_3

    # Edge case 1 - identical positions
    pos7 = jnp.array([5.0, 3.0, 1.0], dtype=FLOAT_DTYPE)
    pos8 = jnp.array([5.0, 3.0, 1.0], dtype=FLOAT_DTYPE)
    distance = point_one_val
    result_4 = spherical_collision(pos7, pos8, distance)
    expected_4 = True
    assert result_4 == expected_4

    # Edge case 2 - zero threshold distance
    pos9 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    pos10 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    distance = zero_val
    result_5 = spherical_collision(pos9, pos10, distance)
    expected_5 = True
    assert result_5 == expected_5

    # Edge case 3 - negative coordinates
    pos11 = jnp.array([-1.0, -2.0, -3.0], dtype=FLOAT_DTYPE)
    pos12 = jnp.array([-1.5, -2.5, -3.5], dtype=FLOAT_DTYPE)
    distance = one_val
    result_6 = spherical_collision(pos11, pos12, distance)
    expected_6 = True
    assert result_6 == expected_6

    # Edge case 4 - very small distance close to EPS
    pos13 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    pos14 = jnp.array([EPS / 2, 0.0, 0.0], dtype=FLOAT_DTYPE)
    distance = jnp.array(EPS, dtype=FLOAT_DTYPE)
    result_7 = spherical_collision(pos13, pos14, distance)
    expected_7 = True
    assert result_7 == expected_7

    # Test with vmap
    positions_1 = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [5.0, 3.0, 1.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    positions_2 = jnp.array(
        [
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 1.5, 1.5],
            [5.0, 3.0, 1.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    distances = jnp.array([1.0, 1.0, 1.0, 0.1], dtype=FLOAT_DTYPE)

    spherical_collision_vmap = jax.vmap(spherical_collision)
    vmap_results = spherical_collision_vmap(positions_1, positions_2, distances)

    expected_vmap = jnp.array([False, True, True, True], dtype=bool)
    assert jnp.array_equal(vmap_results, expected_vmap)


def test_spherical_collision_comprehensive(jit_mode: str) -> None:
    """Comprehensive spherical collision detection tests."""
    # Performance edge case 1 - very large distances
    pos_origin = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    pos_far = jnp.array([1000.0, 1000.0, 1000.0], dtype=FLOAT_DTYPE)
    large_threshold = jnp.array(2000.0, dtype=FLOAT_DTYPE)
    result_1 = spherical_collision(pos_origin, pos_far, large_threshold)
    assert result_1 == jnp.array(True)  # Should collide with large threshold

    small_threshold = jnp.array(1500.0, dtype=FLOAT_DTYPE)
    result_2 = spherical_collision(pos_origin, pos_far, small_threshold)
    assert result_2 == jnp.array(False)  # Should not collide with smaller threshold

    # Performance edge case 2 - micro-precision collisions
    pos_micro1 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    pos_micro2 = jnp.array([EPS * 0.9, 0.0, 0.0], dtype=FLOAT_DTYPE)
    micro_threshold = jnp.array(EPS, dtype=FLOAT_DTYPE)
    result_3 = spherical_collision(pos_micro1, pos_micro2, micro_threshold)
    assert result_3 == jnp.array(True)  # Should detect micro collision

    # 3D collision scenario 1 - diagonal distances
    pos_3d1 = jnp.array([1.0, 2.0, 3.0], dtype=FLOAT_DTYPE)
    pos_3d2 = jnp.array([4.0, 6.0, 8.0], dtype=FLOAT_DTYPE)
    # Distance = sqrt((4-1)² + (6-2)² + (8-3)²) = sqrt(9 + 16 + 25) = sqrt(50) ≈ 7.07
    diagonal_threshold = jnp.array(8.0, dtype=FLOAT_DTYPE)
    result_4 = spherical_collision(pos_3d1, pos_3d2, diagonal_threshold)
    assert result_4 == jnp.array(True)

    tight_threshold = jnp.array(7.0, dtype=FLOAT_DTYPE)
    result_5 = spherical_collision(pos_3d1, pos_3d2, tight_threshold)
    assert result_5 == jnp.array(False)

    # 3D collision scenario 2 - complex 3D geometry
    center = jnp.array([5.0, 5.0, 5.0], dtype=FLOAT_DTYPE)
    # Points arranged in a 3D cross pattern
    cross_points = jnp.array(
        [
            [8.0, 5.0, 5.0],  # +X
            [2.0, 5.0, 5.0],  # -X
            [5.0, 8.0, 5.0],  # +Y
            [5.0, 2.0, 5.0],  # -Y
            [5.0, 5.0, 8.0],  # +Z
            [5.0, 5.0, 2.0],  # -Z
        ],
        dtype=FLOAT_DTYPE,
    )

    cross_threshold = jnp.array(3.5, dtype=FLOAT_DTYPE)  # Distance is 3.0
    collision_vmap = jax.vmap(lambda p: spherical_collision(center, p, cross_threshold))
    cross_results = collision_vmap(cross_points)
    assert jnp.all(cross_results)  # All should collide

    # Boundary value testing 1 - just inside threshold
    pos_boundary1 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    pos_boundary2 = jnp.array([3.0, 4.0, 0.0], dtype=FLOAT_DTYPE)  # Distance = 5.0
    just_inside = jnp.array(5.000001, dtype=FLOAT_DTYPE)
    result_6 = spherical_collision(pos_boundary1, pos_boundary2, just_inside)
    assert result_6 == jnp.array(True)

    # Boundary value testing 2 - just outside threshold
    just_outside = jnp.array(4.999999, dtype=FLOAT_DTYPE)
    result_7 = spherical_collision(pos_boundary1, pos_boundary2, just_outside)
    assert result_7 == jnp.array(False)

    # Stress testing - multiple collision detections with various scales
    scales = jnp.array([0.1, 1.0, 10.0, 100.0], dtype=FLOAT_DTYPE)
    base_pos1 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    base_pos2 = jnp.array([1.0, 0.0, 0.0], dtype=FLOAT_DTYPE)

    # Test at different scales
    stress_results = jax.vmap(
        lambda scale: spherical_collision(
            base_pos1 * scale,
            base_pos2 * scale,
            scale * jnp.array(1.5, dtype=FLOAT_DTYPE),
        )
    )(scales)
    assert jnp.all(stress_results)  # Should all collide at 1.5x threshold

    # Test non-collision at different scales
    no_collision_results = jax.vmap(
        lambda scale: spherical_collision(
            base_pos1 * scale,
            base_pos2 * scale,
            scale * jnp.array(0.5, dtype=FLOAT_DTYPE),
        )
    )(scales)
    assert jnp.all(~no_collision_results)  # Should all NOT collide at 0.5x threshold


def test_interpolate_nearest(jit_mode: str) -> None:
    """Test nearest-neighbor interpolation."""
    # Common values
    zero_val = jnp.array(0.0, dtype=FLOAT_DTYPE)
    one_val = jnp.array(1.0, dtype=FLOAT_DTYPE)
    two_val = jnp.array(2.0, dtype=FLOAT_DTYPE)
    half_val = jnp.array(0.5, dtype=FLOAT_DTYPE)

    # Create test heightmap
    heightmap = jnp.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ],
        dtype=FLOAT_DTYPE,
    )

    # Standard case 1 - exact integer coordinates
    result_1 = interpolate_nearest(heightmap, one_val, one_val)
    expected_1 = 0.5
    assert jnp.isclose(result_1, expected_1, atol=EPS)

    # Standard case 2 - corner position
    result_2 = interpolate_nearest(heightmap, zero_val, zero_val)
    expected_2 = 0.1
    assert jnp.isclose(result_2, expected_2, atol=EPS)

    # Standard case 3 - another corner
    result_3 = interpolate_nearest(heightmap, two_val, two_val)
    expected_3 = 0.9
    assert jnp.isclose(result_3, expected_3, atol=EPS)

    # Edge case 1 - fractional coordinates round to nearest
    u_14 = jnp.array(1.4, dtype=FLOAT_DTYPE)
    v_16 = jnp.array(1.6, dtype=FLOAT_DTYPE)
    result_4 = interpolate_nearest(heightmap, u_14, v_16)
    expected_4 = 0.6  # rounds to (1, 2)
    assert jnp.isclose(result_4, expected_4, atol=EPS)

    # Edge case 2 - fractional coordinates round down
    u_13 = jnp.array(1.3, dtype=FLOAT_DTYPE)
    result_5 = interpolate_nearest(heightmap, u_13, u_13)
    expected_5 = 0.5  # rounds to (1, 1)
    assert jnp.isclose(result_5, expected_5, atol=EPS)

    # Edge case 3 - coordinates at exactly 0.5
    result_6 = interpolate_nearest(heightmap, half_val, half_val)
    assert result_6 in [0.1, 0.5]

    # Test with vmap
    us = jnp.array([0.0, 1.0, 2.0, 1.4], dtype=FLOAT_DTYPE)
    vs = jnp.array([0.0, 1.0, 2.0, 1.6], dtype=FLOAT_DTYPE)

    interpolate_nearest_vmap = jax.vmap(
        lambda u, v: interpolate_nearest(heightmap, u, v)
    )
    vmap_results = interpolate_nearest_vmap(us, vs)

    expected_vmap = jnp.array([0.1, 0.5, 0.9, 0.6], dtype=FLOAT_DTYPE)
    assert jnp.allclose(vmap_results, expected_vmap, atol=EPS)


def test_interpolate_bilinear(jit_mode: str) -> None:
    """Test bilinear interpolation."""
    # Common values
    zero_val = jnp.array(0.0, dtype=FLOAT_DTYPE)
    one_val = jnp.array(1.0, dtype=FLOAT_DTYPE)
    half_val = jnp.array(0.5, dtype=FLOAT_DTYPE)
    neg_one_val = jnp.array(-1.0, dtype=FLOAT_DTYPE)
    ten_val = jnp.array(10.0, dtype=FLOAT_DTYPE)

    # Create test heightmap
    heightmap = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=FLOAT_DTYPE,
    )

    # Standard case 1 - exact integer coordinates
    result_1 = interpolate_bilinear(heightmap, one_val, one_val)
    expected_1 = 1.0
    assert jnp.isclose(result_1, expected_1, atol=EPS)

    # Standard case 2 - corner position
    result_2 = interpolate_bilinear(heightmap, zero_val, zero_val)
    expected_2 = 0.0
    assert jnp.isclose(result_2, expected_2, atol=EPS)

    # Standard case 3 - midpoint interpolation
    result_3 = interpolate_bilinear(heightmap, half_val, half_val)
    expected_3 = 0.25  # average of corners: (0+0+0+1)/4
    assert jnp.isclose(result_3, expected_3, atol=EPS)

    # Edge case 1 - coordinates beyond bounds get clamped
    result_4 = interpolate_bilinear(heightmap, neg_one_val, neg_one_val)
    expected_4 = 0.0  # clamped to (0, 0)
    assert jnp.isclose(result_4, expected_4, atol=EPS)

    # Edge case 2 - coordinates beyond upper bounds get clamped
    result_5 = interpolate_bilinear(heightmap, ten_val, ten_val)
    expected_5 = 0.0  # clamped to (2, 2)
    assert jnp.isclose(result_5, expected_5, atol=EPS)

    # Edge case 3 - partial out of bounds
    u_25 = jnp.array(2.5, dtype=FLOAT_DTYPE)
    result_6 = interpolate_bilinear(heightmap, u_25, one_val)
    expected_6 = 0.0  # u clamped to 2.0
    assert jnp.isclose(result_6, expected_6, atol=EPS)

    # Standard case 4 - gradient heightmap
    gradient_map = jnp.array(
        [
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
        ],
        dtype=FLOAT_DTYPE,
    )

    result_7 = interpolate_bilinear(gradient_map, zero_val, half_val)
    expected_7 = 0.25  # interpolation between 0.0 and 0.5
    assert jnp.isclose(result_7, expected_7, atol=EPS)

    # Standard case 5 - check interpolation weights
    unit_map = jnp.ones((3, 3), dtype=FLOAT_DTYPE)
    u_15 = jnp.array(1.5, dtype=FLOAT_DTYPE)
    result_8 = interpolate_bilinear(unit_map, u_15, u_15)
    expected_8 = 1.0  # all corners are 1, so result should be 1
    assert jnp.isclose(result_8, expected_8, atol=EPS)

    # Test with vmap
    test_map = jnp.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=FLOAT_DTYPE,
    )

    us = jnp.array([0.0, 1.0, 0.5, 1.5], dtype=FLOAT_DTYPE)
    vs = jnp.array([0.0, 1.0, 0.5, 1.5], dtype=FLOAT_DTYPE)

    interpolate_bilinear_vmap = jax.vmap(
        lambda u, v: interpolate_bilinear(test_map, u, v)
    )
    vmap_results = interpolate_bilinear_vmap(us, vs)

    # Calculate expected values manually
    expected_0 = 1.0  # exact corner
    expected_1 = 5.0  # exact center
    expected_2 = 3.0  # interpolation: (1+2+4+5)/4 = 12/4 = 3.0
    expected_3 = 7.0  # interpolation: (5+6+8+9)/4 = 28/4 = 7.0

    expected_vmap = jnp.array(
        [expected_0, expected_1, expected_2, expected_3], dtype=FLOAT_DTYPE
    )
    assert jnp.allclose(vmap_results, expected_vmap, atol=EPS)


def test_interpolation_edge_cases(jit_mode: str) -> None:
    """Comprehensive interpolation edge case testing."""
    # Common values
    half_val = jnp.array(0.5, dtype=FLOAT_DTYPE)

    # Edge boundary testing for nearest interpolation
    edge_map = jnp.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        dtype=FLOAT_DTYPE,
    )

    # Test all four corners with nearest
    corner_coords = jnp.array(
        [
            [0.0, 0.0],  # Top-left
            [0.0, 3.0],  # Top-right
            [2.0, 0.0],  # Bottom-left
            [2.0, 3.0],  # Bottom-right
        ],
        dtype=FLOAT_DTYPE,
    )

    nearest_corner_results = jax.vmap(
        lambda coords: interpolate_nearest(edge_map, coords[0], coords[1])
    )(corner_coords)
    expected_corners = jnp.array([1.0, 4.0, 9.0, 12.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(nearest_corner_results, expected_corners, atol=EPS)

    # Test edge midpoints with nearest
    edge_coords = jnp.array(
        [
            [0.0, 1.5],  # Top edge
            [2.0, 1.5],  # Bottom edge
            [1.0, 0.0],  # Left edge
            [1.0, 3.0],  # Right edge
        ],
        dtype=FLOAT_DTYPE,
    )

    nearest_edge_results = jax.vmap(
        lambda coords: interpolate_nearest(edge_map, coords[0], coords[1])
    )(edge_coords)
    # Results depend on rounding, should be valid edge values
    assert jnp.all(jnp.isin(nearest_edge_results, edge_map.flatten()))

    # Comprehensive bilinear testing with gradient map
    gradient_h = jnp.array(
        [
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, 0.25, 0.5, 0.75, 1.0],
        ],
        dtype=FLOAT_DTYPE,
    )

    # Test horizontal interpolation (should match gradient exactly)
    h_coords = jnp.array([1.0, 0.0, 1.0, 2.0, 1.0, 4.0], dtype=FLOAT_DTYPE).reshape(
        -1, 2
    )
    h_results = jax.vmap(
        lambda coords: interpolate_bilinear(gradient_h, coords[0], coords[1])
    )(h_coords)
    h_expected = jnp.array([0.0, 0.5, 1.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(h_results, h_expected, atol=EPS)

    # Vertical gradient map
    gradient_v = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
        ],
        dtype=FLOAT_DTYPE,
    )

    # Test vertical interpolation
    v_coords = jnp.array([0.0, 1.0, 0.5, 1.0, 2.0, 1.0], dtype=FLOAT_DTYPE).reshape(
        -1, 2
    )
    v_results = jax.vmap(
        lambda coords: interpolate_bilinear(gradient_v, coords[0], coords[1])
    )(v_coords)
    v_expected = jnp.array([0.0, 0.25, 1.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(v_results, v_expected, atol=EPS)

    # Test interpolation accuracy with known values
    precise_map = jnp.array(
        [
            [1.0, 3.0],
            [7.0, 9.0],
        ],
        dtype=FLOAT_DTYPE,
    )

    # Center point should be average of all corners: (1+3+7+9)/4 = 5.0
    center_result = interpolate_bilinear(precise_map, half_val, half_val)
    assert jnp.isclose(center_result, 5.0, atol=EPS)

    # Quarter points
    quarter_coords = jnp.array(
        [
            [0.25, 0.25],  # Closer to top-left
            [0.75, 0.75],  # Closer to bottom-right
            [0.25, 0.75],  # Top-right quadrant
            [0.75, 0.25],  # Bottom-left quadrant
        ],
        dtype=FLOAT_DTYPE,
    )

    quarter_results = jax.vmap(
        lambda coords: interpolate_bilinear(precise_map, coords[0], coords[1])
    )(quarter_coords)

    # Expected values computed manually for bilinear interpolation
    # At (0.25, 0.25): weighted towards corner [0,0] = 1.0
    # At (0.75, 0.75): weighted towards corner [1,1] = 9.0
    # Should form expected pattern
    assert quarter_results.shape == (4,)
    assert jnp.all(quarter_results >= 1.0)
    assert jnp.all(quarter_results <= 9.0)


def test_heightmap_variations(jit_mode: str) -> None:
    """Test interpolation with different heightmap sizes and patterns."""
    # Common values
    zero_val = jnp.array(0.0, dtype=FLOAT_DTYPE)
    half_val = jnp.array(0.5, dtype=FLOAT_DTYPE)

    # Test with 1x1 heightmap (degenerate case)
    tiny_map = jnp.array([[0.42]], dtype=FLOAT_DTYPE)
    result_tiny_nearest = interpolate_nearest(tiny_map, zero_val, zero_val)
    result_tiny_bilinear = interpolate_bilinear(tiny_map, zero_val, zero_val)
    assert jnp.isclose(result_tiny_nearest, 0.42, atol=EPS)
    assert jnp.isclose(result_tiny_bilinear, 0.42, atol=EPS)

    # Test with 2x2 heightmap
    small_map = jnp.array(
        [
            [0.1, 0.3],
            [0.7, 0.9],
        ],
        dtype=FLOAT_DTYPE,
    )

    # Test all corners and center
    small_coords = jnp.array(
        [
            [0.0, 0.0],  # 0.1
            [0.0, 1.0],  # 0.3
            [1.0, 0.0],  # 0.7
            [1.0, 1.0],  # 0.9
            [0.5, 0.5],  # center: (0.1+0.3+0.7+0.9)/4 = 0.5
        ],
        dtype=FLOAT_DTYPE,
    )

    small_results = jax.vmap(
        lambda coords: interpolate_bilinear(small_map, coords[0], coords[1])
    )(small_coords)
    small_expected = jnp.array([0.1, 0.3, 0.7, 0.9, 0.5], dtype=FLOAT_DTYPE)
    assert jnp.allclose(small_results, small_expected, atol=EPS)

    # Test with 5x5 heightmap - checkerboard pattern
    checkerboard = jnp.array(
        [
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0],
        ],
        dtype=FLOAT_DTYPE,
    )

    # Test interpolation at mid-points between checkerboard squares
    check_coords = jnp.array(
        [
            [0.5, 0.5],  # Between alternating values, should be 0.5
            [1.5, 1.5],  # Between alternating values, should be 0.5
            [2.0, 2.0],  # Exact center, should be 1.0
        ],
        dtype=FLOAT_DTYPE,
    )

    check_results = jax.vmap(
        lambda coords: interpolate_bilinear(checkerboard, coords[0], coords[1])
    )(check_coords)
    check_expected = jnp.array([0.5, 0.5, 1.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(check_results, check_expected, atol=EPS)

    # Test with extreme values heightmap
    extreme_map = jnp.array(
        [
            [0.0, 1000.0],
            [-500.0, 0.5],
        ],
        dtype=FLOAT_DTYPE,
    )

    # Test that interpolation handles extreme values correctly
    extreme_result = interpolate_bilinear(extreme_map, half_val, half_val)
    extreme_expected = (0.0 + 1000.0 + (-500.0) + 0.5) / 4.0  # 125.125
    assert jnp.isclose(extreme_result, extreme_expected, atol=EPS)

    # Test numerical stability with very small differences
    precision_map = jnp.array(
        [
            [1.0, 1.0 + EPS],
            [1.0 + 2 * EPS, 1.0 + 3 * EPS],
        ],
        dtype=FLOAT_DTYPE,
    )

    precision_result = interpolate_bilinear(precision_map, half_val, half_val)
    precision_expected = 1.0 + 1.5 * EPS
    assert jnp.isclose(precision_result, precision_expected, atol=EPS)


def test_calculate_height_diff(jit_mode: str) -> None:
    """Test height difference calculation."""
    # Standard case 1 - position at center of map
    heightmap = jnp.array(
        [
            [0.0, 0.5, 1.0],
            [0.25, 0.5, 0.75],
            [0.0, 0.5, 1.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    position_1 = jnp.array([0.0, 0.0, 5.0], dtype=FLOAT_DTYPE)
    map_size_1 = jnp.array(20.0, dtype=FLOAT_DTYPE)
    terrain_height_1 = jnp.array(10.0, dtype=FLOAT_DTYPE)
    use_bilinear_1 = jnp.array(True, dtype=bool)
    result_1 = calculate_height_diff(
        heightmap, position_1, map_size_1, terrain_height_1, use_bilinear_1
    )
    expected_1 = jnp.array(5.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_1, expected_1, atol=EPS)

    # Standard case 2 - position at corner (lower terrain)
    position_2 = jnp.array([-10.0, -10.0, 5.0], dtype=FLOAT_DTYPE)
    result_2 = calculate_height_diff(
        heightmap, position_2, map_size_1, terrain_height_1, use_bilinear_1
    )
    expected_2 = jnp.array(-5.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_2, expected_2, atol=EPS)

    # Standard case 3 - position at high terrain corner
    position_3 = jnp.array([-10.0, 10.0, 5.0], dtype=FLOAT_DTYPE)
    result_3 = calculate_height_diff(
        heightmap, position_3, map_size_1, terrain_height_1, use_bilinear_1
    )
    expected_3 = jnp.array(15.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_3, expected_3, atol=EPS)

    # Standard case 4 - entity at ground level
    position_4 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_4 = calculate_height_diff(
        heightmap, position_4, map_size_1, terrain_height_1, use_bilinear_1
    )
    expected_4 = jnp.array(0.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_4, expected_4, atol=EPS)

    # Standard case 5 - different map size
    position_5 = jnp.array([0.0, 0.0, 10.0], dtype=FLOAT_DTYPE)
    map_size_5 = jnp.array(50.0, dtype=FLOAT_DTYPE)
    result_5 = calculate_height_diff(
        heightmap, position_5, map_size_5, terrain_height_1, use_bilinear_1
    )
    expected_5 = jnp.array(10.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_5, expected_5, atol=EPS)

    # Edge case 1 - out of bounds position
    position_oob = jnp.array([100.0, 100.0, 7.0], dtype=FLOAT_DTYPE)
    result_6 = calculate_height_diff(
        heightmap, position_oob, map_size_1, terrain_height_1, use_bilinear_1
    )
    expected_6 = jnp.array(7.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_6, expected_6, atol=EPS)

    # Edge case 2 - negative out of bounds
    position_neg_oob = jnp.array([-100.0, -100.0, 3.0], dtype=FLOAT_DTYPE)
    result_7 = calculate_height_diff(
        heightmap, position_neg_oob, map_size_1, terrain_height_1, use_bilinear_1
    )
    expected_7 = jnp.array(3.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_7, expected_7, atol=EPS)

    # Edge case 3 - test with nearest neighbor interpolation
    position_nearest = jnp.array([2.5, 2.5, 8.0], dtype=FLOAT_DTYPE)
    use_bilinear_false = jnp.array(False, dtype=bool)
    result_8 = calculate_height_diff(
        heightmap, position_nearest, map_size_1, terrain_height_1, use_bilinear_false
    )
    min_val = jnp.array(-2.0, dtype=FLOAT_DTYPE)
    max_val = jnp.array(18.0, dtype=FLOAT_DTYPE)
    assert result_8 >= min_val and result_8 <= max_val

    # Test with vmap
    positions = jnp.array(
        [
            [0.0, 0.0, 5.0],
            [-10.0, -10.0, 5.0],
            [-10.0, 10.0, 5.0],
            [100.0, 100.0, 7.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    map_sizes = jnp.array([20.0, 20.0, 20.0, 20.0], dtype=FLOAT_DTYPE)
    terrain_heights = jnp.array([10.0, 10.0, 10.0, 10.0], dtype=FLOAT_DTYPE)
    use_bilinears = jnp.array([True, True, True, True], dtype=bool)

    calculate_height_diff_vmap = jax.vmap(calculate_height_diff)
    vmap_results = calculate_height_diff_vmap(
        jnp.tile(heightmap, (4, 1, 1)),
        positions,
        map_sizes,
        terrain_heights,
        use_bilinears,
    )

    expected_vmap = jnp.array([5.0, -5.0, 15.0, 7.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(vmap_results, expected_vmap, atol=EPS)


def test_calculate_terrain_collision(jit_mode: str) -> None:
    """Test terrain collision detection."""
    # Standard case 1 - position above terrain (no collision)
    heightmap = jnp.array(
        [
            [0.0, 0.5, 1.0],
            [0.25, 0.5, 0.75],
            [0.0, 0.5, 1.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    position_1 = jnp.array([0.0, 0.0, -5.0], dtype=FLOAT_DTYPE)
    map_size_1 = jnp.array(20.0, dtype=FLOAT_DTYPE)
    terrain_height_1 = jnp.array(10.0, dtype=FLOAT_DTYPE)
    use_bilinear_1 = jnp.array(True, dtype=bool)
    result_1 = calculate_terrain_collision(
        heightmap, position_1, map_size_1, terrain_height_1, use_bilinear_1
    )
    expected_1 = jnp.array(False, dtype=bool)
    assert result_1 == expected_1

    # Standard case 2 - position below terrain (collision)
    position_2 = jnp.array([0.0, 0.0, 5.0], dtype=FLOAT_DTYPE)
    result_2 = calculate_terrain_collision(
        heightmap, position_2, map_size_1, terrain_height_1, use_bilinear_1
    )
    expected_2 = jnp.array(True, dtype=bool)
    assert result_2 == expected_2

    # Standard case 3 - position exactly at terrain surface
    position_3 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_3 = calculate_terrain_collision(
        heightmap, position_3, map_size_1, terrain_height_1, use_bilinear_1
    )
    expected_3 = jnp.array(True, dtype=bool)
    assert result_3 == expected_3

    # Standard case 4 - test with nearest neighbor interpolation
    position_4 = jnp.array([2.5, 2.5, 0.0], dtype=FLOAT_DTYPE)
    use_bilinear_4 = jnp.array(False, dtype=bool)
    result_4 = calculate_terrain_collision(
        heightmap, position_4, map_size_1, terrain_height_1, use_bilinear_4
    )
    assert isinstance(result_4, (bool, jnp.ndarray))

    # Edge case 1 - position at low terrain corner
    position_low_corner = jnp.array([-10.0, -10.0, -15.0], dtype=FLOAT_DTYPE)
    result_5 = calculate_terrain_collision(
        heightmap, position_low_corner, map_size_1, terrain_height_1, use_bilinear_1
    )
    expected_5 = jnp.array(False, dtype=bool)
    assert result_5 == expected_5

    # Edge case 2 - position at high terrain corner
    position_high_corner = jnp.array([-10.0, 10.0, 15.0], dtype=FLOAT_DTYPE)
    result_6 = calculate_terrain_collision(
        heightmap, position_high_corner, map_size_1, terrain_height_1, use_bilinear_1
    )
    expected_6 = jnp.array(True, dtype=bool)
    assert result_6 == expected_6

    # Edge case 3 - out of bounds above terrain
    position_oob_above = jnp.array([100.0, 100.0, -5.0], dtype=FLOAT_DTYPE)
    result_7 = calculate_terrain_collision(
        heightmap, position_oob_above, map_size_1, terrain_height_1, use_bilinear_1
    )
    expected_7 = jnp.array(False, dtype=bool)
    assert result_7 == expected_7

    # Edge case 4 - out of bounds below terrain
    position_oob_below = jnp.array([100.0, 100.0, 5.0], dtype=FLOAT_DTYPE)
    result_8 = calculate_terrain_collision(
        heightmap, position_oob_below, map_size_1, terrain_height_1, use_bilinear_1
    )
    expected_8 = jnp.array(True, dtype=bool)
    assert result_8 == expected_8

    # Test with vmap
    positions = jnp.array(
        [
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
            [-10.0, -10.0, -15.0],
            [-10.0, 10.0, 15.0],
            [100.0, 100.0, -5.0],
            [100.0, 100.0, 5.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    map_sizes = jnp.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0], dtype=FLOAT_DTYPE)
    terrain_heights = jnp.array(
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=FLOAT_DTYPE
    )
    use_bilinears = jnp.array([True, True, True, True, True, True, True], dtype=bool)

    calculate_terrain_collision_vmap = jax.vmap(calculate_terrain_collision)
    vmap_results = calculate_terrain_collision_vmap(
        jnp.tile(heightmap, (7, 1, 1)),
        positions,
        map_sizes,
        terrain_heights,
        use_bilinears,
    )

    expected_vmap = jnp.array([False, True, True, False, True, False, True], dtype=bool)
    assert jnp.array_equal(vmap_results, expected_vmap)


def test_height_diff_comprehensive(jit_mode: str) -> None:
    """Comprehensive tests for height difference calculation."""
    # Test with complex terrain heightmap
    complex_heightmap = jnp.array(
        [
            [0.0, 0.2, 0.5, 0.8, 1.0],
            [0.1, 0.3, 0.5, 0.7, 0.9],
            [0.2, 0.4, 0.5, 0.6, 0.8],
            [0.3, 0.5, 0.5, 0.5, 0.7],
            [0.4, 0.6, 0.5, 0.4, 0.6],
        ],
        dtype=FLOAT_DTYPE,
    )
    map_size_complex = jnp.array(50.0, dtype=FLOAT_DTYPE)
    terrain_height_complex = jnp.array(100.0, dtype=FLOAT_DTYPE)

    # Test multiple positions with bilinear interpolation
    test_positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [-25.0, -25.0, 0.0],
            [25.0, 25.0, 0.0],
            [-12.5, 0.0, 50.0],
            [12.5, 0.0, -50.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    use_bilinear_true = jnp.array(True, dtype=bool)
    bilinear_results = jax.vmap(
        lambda pos: calculate_height_diff(
            complex_heightmap,
            pos,
            map_size_complex,
            terrain_height_complex,
            use_bilinear_true,
        )
    )(test_positions)

    # Test with nearest neighbor interpolation
    use_bilinear_false = jnp.array(False, dtype=bool)
    nearest_results = jax.vmap(
        lambda pos: calculate_height_diff(
            complex_heightmap,
            pos,
            map_size_complex,
            terrain_height_complex,
            use_bilinear_false,
        )
    )(test_positions)

    assert bilinear_results.shape == (5,)
    assert nearest_results.shape == (5,)
    assert jnp.all(jnp.isfinite(bilinear_results))
    assert jnp.all(jnp.isfinite(nearest_results))

    # Test with extreme terrain heights
    position_extreme = jnp.array([-25.0, -25.0, 10.0], dtype=FLOAT_DTYPE)
    tiny_terrain = jnp.array(0.1, dtype=FLOAT_DTYPE)
    huge_terrain = jnp.array(1000.0, dtype=FLOAT_DTYPE)
    tiny_result = calculate_height_diff(
        complex_heightmap,
        position_extreme,
        map_size_complex,
        tiny_terrain,
        use_bilinear_true,
    )
    huge_result = calculate_height_diff(
        complex_heightmap,
        position_extreme,
        map_size_complex,
        huge_terrain,
        use_bilinear_true,
    )
    assert jnp.abs(huge_result) > jnp.abs(tiny_result)

    # Test boundary behavior with varying map sizes
    boundary_position = jnp.array([5.0, 5.0, 10.0], dtype=FLOAT_DTYPE)
    map_sizes = jnp.array([10.0, 50.0, 100.0, 500.0], dtype=FLOAT_DTYPE)
    size_results = jax.vmap(
        lambda size: calculate_height_diff(
            complex_heightmap,
            boundary_position,
            size,
            terrain_height_complex,
            use_bilinear_true,
        )
    )(map_sizes)
    assert size_results.shape == (4,)
    assert jnp.all(jnp.isfinite(size_results))

    # Test numerical stability near boundaries
    eps_position = jnp.array([24.99999, 24.99999, 0.0], dtype=FLOAT_DTYPE)
    eps_result = calculate_height_diff(
        complex_heightmap,
        eps_position,
        map_size_complex,
        terrain_height_complex,
        use_bilinear_true,
    )
    assert jnp.isfinite(eps_result)

    # Test with uniform heightmap
    uniform_map = jnp.ones((3, 3), dtype=FLOAT_DTYPE) * jnp.array(
        0.7, dtype=FLOAT_DTYPE
    )
    uniform_position = jnp.array([0.0, 0.0, 25.0], dtype=FLOAT_DTYPE)
    uniform_result = calculate_height_diff(
        uniform_map,
        uniform_position,
        map_size_complex,
        terrain_height_complex,
        use_bilinear_true,
    )
    expected_uniform = jnp.array(65.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(uniform_result, expected_uniform, atol=EPS)


def test_spatial_performance(jit_mode: str) -> None:
    """Test large-scale vectorized spatial operations."""
    # Common values
    one_val = jnp.array(1.0, dtype=FLOAT_DTYPE)
    five_val = jnp.array(5.0, dtype=FLOAT_DTYPE)

    # Large-scale collision detection test
    n_points = 100
    # Generate random positions in a sphere
    key = jax.random.PRNGKey(42)
    positions_1 = jax.random.normal(key, (n_points, 3), dtype=FLOAT_DTYPE) * five_val
    positions_2 = positions_1 + jax.random.normal(
        jax.random.split(key)[0], (n_points, 3), dtype=FLOAT_DTYPE
    )

    # Test with various thresholds
    thresholds = jnp.array([0.5, 1.0, 2.0, 5.0], dtype=FLOAT_DTYPE)

    # Vectorized collision detection across all combinations
    collision_results = jax.vmap(
        lambda thresh: jax.vmap(lambda p1, p2: spherical_collision(p1, p2, thresh))(
            positions_1, positions_2
        )
    )(thresholds)

    assert collision_results.shape == (4, n_points)
    # Should have more collisions with larger thresholds
    collision_counts = jnp.sum(collision_results, axis=1)
    # Verify monotonic increase (larger threshold -> more collisions)
    assert jnp.all(collision_counts[1:] >= collision_counts[:-1])

    # Large-scale interpolation performance test
    # Create a large heightmap
    map_size = 20
    large_heightmap = jnp.sin(
        jnp.linspace(0, 2 * jnp.pi, map_size * map_size, dtype=FLOAT_DTYPE)
    ).reshape(map_size, map_size)

    # Generate systematic coordinate grid
    u_coords = jnp.linspace(0.0, map_size - 1, 50, dtype=FLOAT_DTYPE)
    v_coords = jnp.linspace(0.0, map_size - 1, 50, dtype=FLOAT_DTYPE)
    u_grid, v_grid = jnp.meshgrid(u_coords, v_coords)
    coords_flat = jnp.stack([u_grid.flatten(), v_grid.flatten()], axis=1)

    # Test both interpolation methods on the same coordinates
    nearest_results = jax.vmap(
        lambda coords: interpolate_nearest(large_heightmap, coords[0], coords[1])
    )(coords_flat)

    bilinear_results = jax.vmap(
        lambda coords: interpolate_bilinear(large_heightmap, coords[0], coords[1])
    )(coords_flat)

    assert nearest_results.shape == (2500,)  # 50x50 grid
    assert bilinear_results.shape == (2500,)

    # Test coordinate sweeps - systematic sampling
    sweep_u = jnp.linspace(0.0, 2.0, 21, dtype=FLOAT_DTYPE)  # 3x3 heightmap coordinates
    sweep_v = jnp.linspace(0.0, 2.0, 21, dtype=FLOAT_DTYPE)

    simple_map = jnp.array(
        [
            [0.0, 0.5, 1.0],
            [0.5, 1.0, 0.5],
            [1.0, 0.5, 0.0],
        ],
        dtype=FLOAT_DTYPE,
    )

    # Test horizontal sweep (fixed v)
    h_sweep_results = jax.vmap(lambda u: interpolate_bilinear(simple_map, u, one_val))(
        sweep_u
    )

    # Test vertical sweep (fixed u)
    v_sweep_results = jax.vmap(lambda v: interpolate_bilinear(simple_map, one_val, v))(
        sweep_v
    )

    assert h_sweep_results.shape == (21,)
    assert v_sweep_results.shape == (21,)

    # Values should be bounded by heightmap range
    assert jnp.all(h_sweep_results >= 0.0)
    assert jnp.all(h_sweep_results <= 1.0)
    assert jnp.all(v_sweep_results >= 0.0)
    assert jnp.all(v_sweep_results <= 1.0)

    # Test diagonal sweep
    diagonal_coords = jnp.linspace(0.0, 2.0, 21, dtype=FLOAT_DTYPE)
    diagonal_results = jax.vmap(
        lambda coord: interpolate_bilinear(simple_map, coord, coord)
    )(diagonal_coords)

    assert diagonal_results.shape == (21,)
    # Should follow the diagonal pattern from the heightmap
    assert diagonal_results[0] == simple_map[0, 0]  # Start at corner
    assert diagonal_results[-1] == simple_map[2, 2]  # End at opposite corner

    # Test performance with different sampling densities
    densities = [5, 10, 20, 40]
    performance_results = []

    for density in densities:
        test_coords = jnp.linspace(0.0, 2.0, density, dtype=FLOAT_DTYPE)
        coord_grid = jnp.stack(jnp.meshgrid(test_coords, test_coords), axis=-1).reshape(
            -1, 2
        )

        # Time interpolation (in practice, just verify it works)
        density_results = jax.vmap(
            lambda coords: interpolate_bilinear(simple_map, coords[0], coords[1])
        )(coord_grid)

        performance_results.append(density_results.shape[0])
        assert density_results.shape == (density * density,)

    # Verify we tested increasing densities
    expected_sizes = [d * d for d in densities]
    assert performance_results == expected_sizes
