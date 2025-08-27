"""Tests for primitives module."""

import jax
import jax.numpy as jnp

from nimbus.core.primitives import (
    EPS,
    FLOAT_DTYPE,
    norm_2,
    norm_3,
    norm_4,
)


def test_norm_2(jit_mode: str) -> None:
    """Test norm_2 computation."""
    # Standard case 1 - unit vector along x-axis
    v1 = jnp.array([1.0, 0.0], dtype=FLOAT_DTYPE)
    result_1 = norm_2(v1)
    expected_1 = jnp.array(1.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_1, expected_1, atol=EPS)

    # Standard case 2 - unit vector along y-axis
    v2 = jnp.array([0.0, 1.0], dtype=FLOAT_DTYPE)
    result_2 = norm_2(v2)
    expected_2 = jnp.array(1.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_2, expected_2, atol=EPS)

    # Standard case 3 - 3-4-5 triangle
    v3 = jnp.array([3.0, 4.0], dtype=FLOAT_DTYPE)
    result_3 = norm_2(v3)
    expected_3 = jnp.array(5.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_3, expected_3, atol=EPS)

    # Edge case 1 - zero vector
    v4 = jnp.array([0.0, 0.0], dtype=FLOAT_DTYPE)
    result_4 = norm_2(v4)
    zero_scalar = jnp.array(0.0, dtype=FLOAT_DTYPE)
    assert result_4 == zero_scalar

    # Edge case 2 - negative components
    v5 = jnp.array([-3.0, -4.0], dtype=FLOAT_DTYPE)
    result_5 = norm_2(v5)
    expected_5 = jnp.array(5.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_5, expected_5, atol=EPS)

    # Edge case 3 - very small values near EPS
    v6 = jnp.array([EPS, EPS], dtype=FLOAT_DTYPE)
    result_6 = norm_2(v6)
    zero_scalar = jnp.array(0.0, dtype=FLOAT_DTYPE)
    assert result_6 > zero_scalar
    expected_6 = jnp.sqrt(jnp.array(2.0, dtype=FLOAT_DTYPE)) * EPS
    assert jnp.isclose(result_6, expected_6, atol=EPS)

    # Edge case 4 - large values
    large = jnp.array(1e10, dtype=FLOAT_DTYPE)
    v7 = jnp.array([large, 0.0], dtype=FLOAT_DTYPE)
    result_7 = norm_2(v7)
    expected_7 = large
    rtol_val = jnp.array(1e-6, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_7, expected_7, rtol=rtol_val)

    # Test with vmap
    vectors = jnp.array(
        [[1.0, 0.0], [0.0, 1.0], [3.0, 4.0], [-3.0, -4.0], [0.0, 0.0]],
        dtype=FLOAT_DTYPE,
    )

    norm_2_vmap = jax.vmap(norm_2)
    vmap_results = norm_2_vmap(vectors)

    expected_vmap = jnp.array([1.0, 1.0, 5.0, 5.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(vmap_results, expected_vmap, atol=EPS)


def test_norm_3(jit_mode: str) -> None:
    """Test norm_3 computation."""
    # Standard case 1 - unit vector along x-axis
    v1 = jnp.array([1.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_1 = norm_3(v1)
    expected_1 = jnp.array(1.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_1, expected_1, atol=EPS)

    # Standard case 2 - unit vector along diagonal
    v2 = jnp.array([1.0, 1.0, 1.0], dtype=FLOAT_DTYPE)
    result_2 = norm_3(v2)
    expected_2 = jnp.sqrt(jnp.array(3.0, dtype=FLOAT_DTYPE))
    assert jnp.isclose(result_2, expected_2, atol=EPS)

    # Standard case 3 - 2-3-6 gives 7
    v3 = jnp.array([2.0, 3.0, 6.0], dtype=FLOAT_DTYPE)
    result_3 = norm_3(v3)
    expected_3 = jnp.array(7.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_3, expected_3, atol=EPS)

    # Edge case 1 - zero vector
    v4 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_4 = norm_3(v4)
    zero_scalar = jnp.array(0.0, dtype=FLOAT_DTYPE)
    assert result_4 == zero_scalar

    # Edge case 2 - negative components
    v5 = jnp.array([-1.0, -1.0, -1.0], dtype=FLOAT_DTYPE)
    result_5 = norm_3(v5)
    expected_5 = jnp.sqrt(jnp.array(3.0, dtype=FLOAT_DTYPE))
    assert jnp.isclose(result_5, expected_5, atol=EPS)

    # Edge case 3 - mixed positive/negative components
    v6 = jnp.array([1.0, -1.0, 1.0], dtype=FLOAT_DTYPE)
    result_6 = norm_3(v6)
    expected_6 = jnp.sqrt(jnp.array(3.0, dtype=FLOAT_DTYPE))
    assert jnp.isclose(result_6, expected_6, atol=EPS)

    # Edge case 4 - very small values
    v7 = jnp.array([EPS, EPS, EPS], dtype=FLOAT_DTYPE)
    result_7 = norm_3(v7)
    zero_scalar = jnp.array(0.0, dtype=FLOAT_DTYPE)
    assert result_7 > zero_scalar
    expected_7 = jnp.sqrt(jnp.array(3.0, dtype=FLOAT_DTYPE)) * EPS
    assert jnp.isclose(result_7, expected_7, atol=EPS)

    # Edge case 5 - large values to check overflow
    large = jnp.array(1e10, dtype=FLOAT_DTYPE)
    v8 = jnp.array([large, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_8 = norm_3(v8)
    expected_8 = large
    rtol_val = jnp.array(1e-6, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_8, expected_8, rtol=rtol_val)

    # Test with vmap
    vectors = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 3.0, 6.0],
        ],
        dtype=FLOAT_DTYPE,
    )

    norm_3_vmap = jax.vmap(norm_3)
    vmap_results = norm_3_vmap(vectors)

    expected_vmap = jnp.array([1.0, 1.0, 1.0, jnp.sqrt(3.0), 7.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(vmap_results, expected_vmap, atol=EPS)


def test_norm_4(jit_mode: str) -> None:
    """Test norm_4 computation."""
    # Standard case 1 - identity quaternion
    q1 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_1 = norm_4(q1)
    expected_1 = jnp.array(1.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_1, expected_1, atol=EPS)

    # Standard case 2 - unit quaternion with equal components
    q2 = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=FLOAT_DTYPE)
    result_2 = norm_4(q2)
    expected_2 = jnp.array(1.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_2, expected_2, atol=EPS)

    # Standard case 3 - scaled quaternion
    q3 = jnp.array([3.0, 4.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_3 = norm_4(q3)
    expected_3 = jnp.array(5.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_3, expected_3, atol=EPS)

    # Edge case 1 - zero quaternion
    q4 = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_4 = norm_4(q4)
    zero_scalar = jnp.array(0.0, dtype=FLOAT_DTYPE)
    assert result_4 == zero_scalar

    # Edge case 2 - negative scalar component
    q5 = jnp.array([-1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_5 = norm_4(q5)
    expected_5 = jnp.array(1.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_5, expected_5, atol=EPS)

    # Edge case 3 - all negative components
    q6 = jnp.array([-0.5, -0.5, -0.5, -0.5], dtype=FLOAT_DTYPE)
    result_6 = norm_4(q6)
    expected_6 = jnp.array(1.0, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_6, expected_6, atol=EPS)

    # Edge case 4 - very small quaternion
    q7 = jnp.array([EPS, EPS, EPS, EPS], dtype=FLOAT_DTYPE)
    result_7 = norm_4(q7)
    zero_scalar = jnp.array(0.0, dtype=FLOAT_DTYPE)
    assert result_7 > zero_scalar
    expected_7 = jnp.array(2.0, dtype=FLOAT_DTYPE) * EPS
    assert jnp.isclose(result_7, expected_7, atol=EPS)

    # Edge case 5 - large values
    large = jnp.array(1e10, dtype=FLOAT_DTYPE)
    q8 = jnp.array([large, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_8 = norm_4(q8)
    expected_8 = large
    rtol_val = jnp.array(1e-6, dtype=FLOAT_DTYPE)
    assert jnp.isclose(result_8, expected_8, rtol=rtol_val)

    # Test with vmap
    quaternions = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=FLOAT_DTYPE,
    )

    norm_4_vmap = jax.vmap(norm_4)
    vmap_results = norm_4_vmap(quaternions)

    expected_vmap = jnp.array([1.0, 1.0, 5.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(vmap_results, expected_vmap, atol=EPS)
