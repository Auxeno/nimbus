"""Tests for quaternion module."""

import jax
import jax.numpy as jnp

from nimbus.core.primitives import EPS, FLOAT_DTYPE
from nimbus.core.quaternion import (
    canonicalize,
    conjugate,
    derivative,
    from_axis_angle,
    from_euler_zyx,
    inverse,
    multiply,
    normalize,
    rotate_vector,
    slerp,
    to_euler_zyx,
    to_rotation_matrix,
)

pi = jnp.array(jnp.pi, dtype=FLOAT_DTYPE)


def quaternions_close(q1, q2, atol=EPS):
    """Check if two quaternions are equivalent rotations (q or -q)."""
    return jnp.allclose(q1, q2, atol=atol) or jnp.allclose(q1, -q2, atol=atol)


def test_canonicalize(jit_mode: str) -> None:
    """Test quaternion canonicalisation."""
    # Standard case 1 - positive w should remain unchanged
    q1 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_1 = canonicalize(q1)
    assert jnp.allclose(result_1, q1, atol=EPS)

    # Standard case 2 - positive w with other components
    q2 = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=FLOAT_DTYPE)
    result_2 = canonicalize(q2)
    assert jnp.allclose(result_2, q2, atol=EPS)

    # Standard case 3 - negative w should flip sign
    q3 = jnp.array([-1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_3 = canonicalize(q3)
    expected_3 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_3, expected_3, atol=EPS)

    # Edge case 1 - negative w with other components flips all
    q4 = jnp.array([-0.5, 0.5, 0.5, 0.5], dtype=FLOAT_DTYPE)
    result_4 = canonicalize(q4)
    expected_4 = jnp.array([0.5, -0.5, -0.5, -0.5], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_4, expected_4, atol=EPS)

    # Edge case 2 - w exactly zero remains unchanged
    q5 = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_5 = canonicalize(q5)
    assert jnp.allclose(result_5, q5, atol=EPS)

    # Edge case 3 - very small negative w
    q6 = jnp.array([-EPS, 1.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_6 = canonicalize(q6)
    expected_6 = jnp.array([EPS, -1.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_6, expected_6, atol=EPS)

    # Test with vmap
    quaternions = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5, 0.5],
        ],
        dtype=FLOAT_DTYPE,
    )

    canonicalize_vmap = jax.vmap(canonicalize)
    vmap_results = canonicalize_vmap(quaternions)

    expected_vmap = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5, -0.5],
        ],
        dtype=FLOAT_DTYPE,
    )
    assert jnp.allclose(vmap_results, expected_vmap, atol=EPS)


def test_normalize(jit_mode: str) -> None:
    """Test quaternion normalisation."""
    # Standard case 1 - already normalised quaternion
    q1 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_1 = normalize(q1)
    assert jnp.allclose(result_1, q1, atol=EPS)

    # Standard case 2 - unnormalised quaternion
    q2 = jnp.array([2.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_2 = normalize(q2)
    expected_2 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_2, expected_2, atol=EPS)

    # Standard case 3 - general quaternion
    q3 = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=FLOAT_DTYPE)
    result_3 = normalize(q3)
    expected_3 = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_3, expected_3, atol=EPS)

    # Edge case 1 - zero quaternion returns identity
    q4 = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_4 = normalize(q4)
    expected_4 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_4, expected_4, atol=EPS)

    # Edge case 2 - very small quaternion returns identity
    q5 = jnp.array([EPS / 2, EPS / 2, EPS / 2, EPS / 2], dtype=FLOAT_DTYPE)
    result_5 = normalize(q5)
    expected_5 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_5, expected_5, atol=EPS)

    # Edge case 3 - negative w gets canonicalized
    q6 = jnp.array([-1.0, -1.0, -1.0, -1.0], dtype=FLOAT_DTYPE)
    result_6 = normalize(q6)
    expected_6 = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_6, expected_6, atol=EPS)

    # Edge case 4 - large values
    q7 = jnp.array([1e10, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_7 = normalize(q7)
    expected_7 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_7, expected_7, atol=EPS)

    # Test with vmap
    quaternions = jnp.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0],
        ],
        dtype=FLOAT_DTYPE,
    )

    normalize_vmap = jax.vmap(normalize)
    vmap_results = normalize_vmap(quaternions)

    expected_vmap = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=FLOAT_DTYPE,
    )
    assert jnp.allclose(vmap_results, expected_vmap, atol=EPS)


def test_multiply(jit_mode: str) -> None:
    """Test quaternion multiplication."""
    # Standard case 1 - identity quaternion multiplication
    q1 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    q2 = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=FLOAT_DTYPE)
    result_1 = multiply(q1, q2)
    assert jnp.allclose(result_1, q2, atol=EPS)

    # Standard case 2 - 90 degree rotations about axes
    qx = jnp.array([jnp.cos(pi / 4), jnp.sin(pi / 4), 0.0, 0.0], dtype=FLOAT_DTYPE)
    qy = jnp.array([jnp.cos(pi / 4), 0.0, jnp.sin(pi / 4), 0.0], dtype=FLOAT_DTYPE)
    result_2 = multiply(qx, qy)

    # With Hamilton product (apply qy then qx)
    expected = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_2, expected, atol=EPS)

    # Standard case 3 - inverse multiplication gives identity
    q3 = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=FLOAT_DTYPE)
    q3_inv = conjugate(q3)
    result_3 = multiply(q3, q3_inv)
    expected_3 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_3, expected_3, atol=EPS)

    # Edge case 1 - multiplication with zero quaternion
    q4 = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    q5 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_4 = multiply(q4, q5)
    expected_4 = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_4, expected_4, atol=EPS)

    # Edge case 2 - non-commutativity check
    qa = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    qb = jnp.array([0.0, 0.0, 1.0, 0.0], dtype=FLOAT_DTYPE)
    result_5a = multiply(qa, qb)
    result_5b = multiply(qb, qa)
    assert not jnp.allclose(result_5a, result_5b, atol=EPS)

    # Test with vmap
    q1s = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    q2s = jnp.array(
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5, -0.5],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=FLOAT_DTYPE,
    )

    multiply_vmap = jax.vmap(multiply)
    vmap_results = multiply_vmap(q1s, q2s)

    # Verify results have expected properties
    assert vmap_results.shape == (3, 4)
    # First multiplication: identity * q = q
    assert jnp.allclose(vmap_results[0], q2s[0], atol=EPS)
    # Second multiplication: q * q_conjugate = identity
    expected_1 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(vmap_results[1], expected_1, atol=EPS)


def test_conjugate(jit_mode: str) -> None:
    """Test quaternion conjugation."""
    # Standard case 1 - identity quaternion
    q1 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_1 = conjugate(q1)
    expected_1 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_1, expected_1, atol=EPS)

    # Standard case 2 - general quaternion
    q2 = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=FLOAT_DTYPE)
    result_2 = conjugate(q2)
    expected_2 = jnp.array([0.5, -0.5, -0.5, -0.5], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_2, expected_2, atol=EPS)

    # Standard case 3 - pure imaginary quaternion
    q3 = jnp.array([0.0, 1.0, 2.0, 3.0], dtype=FLOAT_DTYPE)
    result_3 = conjugate(q3)
    expected_3 = jnp.array([0.0, -1.0, -2.0, -3.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_3, expected_3, atol=EPS)

    # Edge case 1 - negative scalar component
    q4 = jnp.array([-1.0, 1.0, 1.0, 1.0], dtype=FLOAT_DTYPE)
    result_4 = conjugate(q4)
    expected_4 = jnp.array([-1.0, -1.0, -1.0, -1.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_4, expected_4, atol=EPS)

    # Edge case 2 - zero quaternion
    q5 = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_5 = conjugate(q5)
    expected_5 = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_5, expected_5, atol=EPS)

    # Test with vmap
    quaternions = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [0.0, 1.0, 2.0, 3.0],
            [-1.0, 1.0, 1.0, 1.0],
        ],
        dtype=FLOAT_DTYPE,
    )

    conjugate_vmap = jax.vmap(conjugate)
    vmap_results = conjugate_vmap(quaternions)

    expected_vmap = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, -0.5, -0.5, -0.5],
            [0.0, -1.0, -2.0, -3.0],
            [-1.0, -1.0, -1.0, -1.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    assert jnp.allclose(vmap_results, expected_vmap, atol=EPS)


def test_inverse(jit_mode: str) -> None:
    """Test quaternion inversion."""
    # Standard case 1 - identity quaternion
    q1 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_1 = inverse(q1)
    expected_1 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_1, expected_1, atol=EPS)

    # Standard case 2 - unit quaternion (inverse = conjugate)
    q2 = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=FLOAT_DTYPE)
    result_2 = inverse(q2)
    expected_2 = conjugate(q2)
    assert jnp.allclose(result_2, expected_2, atol=EPS)

    # Standard case 3 - non-unit quaternion
    q3 = jnp.array([2.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_3 = inverse(q3)
    expected_3 = jnp.array([0.5, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)  # corrected
    assert jnp.allclose(result_3, expected_3, atol=EPS)

    # Edge case 1 - zero quaternion returns identity
    q4 = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_4 = inverse(q4)
    expected_4 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_4, expected_4, atol=EPS)

    # Edge case 2 - very small quaternion returns identity
    q5 = jnp.array([EPS / 2, EPS / 2, EPS / 2, EPS / 2], dtype=FLOAT_DTYPE)
    result_5 = inverse(q5)
    expected_5 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_5, expected_5, atol=EPS)

    # Edge case 3 - verify q * q^(-1) = identity
    q6 = jnp.array([2.0, 3.0, 4.0, 5.0], dtype=FLOAT_DTYPE)
    q6_inv = inverse(q6)
    result_6 = multiply(q6, q6_inv)
    expected_6 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_6, expected_6, atol=1e-6)

    # Test with vmap
    quaternions = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=FLOAT_DTYPE,
    )

    inverse_vmap = jax.vmap(inverse)
    vmap_results = inverse_vmap(quaternions)

    # Verify each inverse
    expected_0 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(vmap_results[0], expected_0, atol=EPS)
    expected_1 = jnp.array([0.5, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)  # corrected
    assert jnp.allclose(vmap_results[1], expected_1, atol=EPS)
    assert jnp.allclose(vmap_results[2], conjugate(quaternions[2]), atol=EPS)
    expected_3 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(vmap_results[3], expected_3, atol=EPS)


def test_rotate_vector(jit_mode: str) -> None:
    """Test vector rotation by quaternion."""
    # Standard case 1 - no rotation (identity quaternion)
    v1 = jnp.array([1.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    q1 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_1 = rotate_vector(v1, q1)
    assert jnp.allclose(result_1, v1, atol=EPS)

    # Standard case 2 - 90 degree rotation about z-axis
    v2 = jnp.array([1.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    q2 = jnp.array([jnp.cos(pi / 4), 0.0, 0.0, jnp.sin(pi / 4)], dtype=FLOAT_DTYPE)
    result_2 = rotate_vector(v2, q2)
    expected_2 = jnp.array([0.0, 1.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_2, expected_2, atol=EPS)

    # Standard case 3 - 180 degree rotation about x-axis
    v3 = jnp.array([0.0, 1.0, 0.0], dtype=FLOAT_DTYPE)
    q3 = jnp.array([jnp.cos(pi / 2), jnp.sin(pi / 2), 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_3 = rotate_vector(v3, q3)
    expected_3 = jnp.array([0.0, -1.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_3, expected_3, atol=1e-6)

    # Edge case 1 - zero vector remains zero
    v4 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    q4 = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=FLOAT_DTYPE)
    result_4 = rotate_vector(v4, q4)
    expected_4 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_4, expected_4, atol=EPS)

    # Edge case 2 - verify rotation preserves vector magnitude
    v5 = jnp.array([3.0, 4.0, 5.0], dtype=FLOAT_DTYPE)
    q5 = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=FLOAT_DTYPE)
    result_5 = rotate_vector(v5, q5)
    mag_before = jnp.linalg.norm(v5)
    mag_after = jnp.linalg.norm(result_5)
    assert jnp.isclose(mag_before, mag_after, atol=EPS)

    # Test with vmap
    vectors = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    quaternion = jnp.array(
        [jnp.cos(pi / 4), 0.0, 0.0, jnp.sin(pi / 4)], dtype=FLOAT_DTYPE
    )

    rotate_vector_vmap = jax.vmap(lambda v: rotate_vector(v, quaternion))
    vmap_results = rotate_vector_vmap(vectors)

    # 90 degree rotation about z-axis: x->y, y->-x, z->z
    expected_vmap = jnp.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    assert jnp.allclose(vmap_results, expected_vmap, atol=EPS)


def test_derivative(jit_mode: str) -> None:
    """Test quaternion derivative computation."""
    # Standard case 1 - no angular velocity
    q1 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    omega1 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_1 = derivative(q1, omega1)
    expected_1 = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_1, expected_1, atol=EPS)

    # Standard case 2 - rotation about z-axis
    q2 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    omega2 = jnp.array([0.0, 0.0, 2.0], dtype=FLOAT_DTYPE)
    result_2 = derivative(q2, omega2)
    expected_2 = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_2, expected_2, atol=EPS)

    # Standard case 3 - general angular velocity
    q3 = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=FLOAT_DTYPE)
    omega3 = jnp.array([1.0, 2.0, 3.0], dtype=FLOAT_DTYPE)
    result_3 = derivative(q3, omega3)
    # Verify it's perpendicular to original quaternion
    dot_product = jnp.dot(q3, result_3)
    assert jnp.isclose(
        dot_product,
        jnp.array(0.0, dtype=FLOAT_DTYPE),
        atol=EPS,
    )

    # Edge case 1 - very large angular velocity
    q4 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    omega4 = jnp.array([100.0, 200.0, 300.0], dtype=FLOAT_DTYPE)
    result_4 = derivative(q4, omega4)
    expected_4 = jnp.array([0.0, 50.0, 100.0, 150.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_4, expected_4, atol=EPS)

    # Edge case 2 - zero quaternion
    q5 = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    omega5 = jnp.array([1.0, 1.0, 1.0], dtype=FLOAT_DTYPE)
    result_5 = derivative(q5, omega5)
    expected_5 = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_5, expected_5, atol=EPS)

    # Test with vmap
    quaternions = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    omegas = jnp.array(
        [
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=FLOAT_DTYPE,
    )

    derivative_vmap = jax.vmap(derivative)
    vmap_results = derivative_vmap(quaternions, omegas)

    assert vmap_results.shape == (3, 4)
    expected_0 = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(vmap_results[0], expected_0, atol=EPS)
    expected_2 = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(vmap_results[2], expected_2, atol=EPS)


def test_slerp(jit_mode: str) -> None:
    """Test spherical linear interpolation."""
    # Standard case 1 - interpolation endpoints
    q1 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    q2 = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    zero_t = jnp.array(0.0, dtype=FLOAT_DTYPE)
    one_t = jnp.array(1.0, dtype=FLOAT_DTYPE)
    result_1a = slerp(q1, q2, zero_t)
    assert jnp.allclose(result_1a, q1, atol=EPS)
    result_1b = slerp(q1, q2, one_t)
    assert jnp.allclose(result_1b, q2, atol=EPS)

    # Standard case 2 - midpoint interpolation
    q3 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    q4 = jnp.array([jnp.cos(pi / 4), 0.0, 0.0, jnp.sin(pi / 4)], dtype=FLOAT_DTYPE)
    half_t = jnp.array(0.5, dtype=FLOAT_DTYPE)
    result_2 = slerp(q3, q4, half_t)
    expected = jnp.array(
        [jnp.cos(pi / 8), 0.0, 0.0, jnp.sin(pi / 8)], dtype=FLOAT_DTYPE
    )
    assert quaternions_close(result_2, expected, atol=EPS)

    # Standard case 3 - verify unit quaternion result
    qa = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    qb = jnp.array([0.0, 0.0, 1.0, 0.0], dtype=FLOAT_DTYPE)
    quarter_t = jnp.array(0.25, dtype=FLOAT_DTYPE)
    result_3 = slerp(qa, qb, quarter_t)
    assert jnp.isclose(
        jnp.linalg.norm(result_3),
        jnp.array(1.0, dtype=FLOAT_DTYPE),
        atol=EPS,
    )

    # Edge case 1 - identical quaternions
    q5 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    q6 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_4 = slerp(q5, q6, half_t)
    assert quaternions_close(result_4, q5, atol=EPS)

    # Edge case 2 - opposite quaternions
    q7 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    q8 = jnp.array([-1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_5 = slerp(q7, q8, half_t)
    assert jnp.isclose(
        jnp.linalg.norm(result_5),
        jnp.array(1.0, dtype=FLOAT_DTYPE),
        atol=EPS,
    )

    # Edge case 3 - very small angle
    q9 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    q10 = jnp.array([1.0 - EPS / 2, EPS / 2, 0.0, 0.0], dtype=FLOAT_DTYPE)
    q10 = normalize(q10)
    result_6 = slerp(q9, q10, half_t)
    assert jnp.isclose(
        jnp.linalg.norm(result_6),
        jnp.array(1.0, dtype=FLOAT_DTYPE),
        atol=EPS,
    )

    # Test with vmap
    q1s = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    q2s = jnp.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [jnp.cos(pi / 4), 0.0, 0.0, jnp.sin(pi / 4)],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    ts = jnp.array([0.0, 0.5, 1.0], dtype=FLOAT_DTYPE)

    slerp_vmap = jax.vmap(slerp)
    vmap_results = slerp_vmap(q1s, q2s, ts)

    assert vmap_results.shape == (3, 4)
    # All results should be unit quaternions
    norms = jnp.linalg.norm(vmap_results, axis=1)
    assert jnp.allclose(norms, jnp.ones(3, dtype=FLOAT_DTYPE), atol=EPS)


def test_from_axis_angle(jit_mode: str) -> None:
    """Test quaternion creation from axis-angle."""
    # Standard case 1 - rotation about x-axis
    axis1 = jnp.array([1.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    angle1 = pi / jnp.array(2.0, dtype=FLOAT_DTYPE)
    result_1 = from_axis_angle(axis1, angle1)
    expected_1 = jnp.array(
        [jnp.cos(pi / 4), jnp.sin(pi / 4), 0.0, 0.0], dtype=FLOAT_DTYPE
    )
    assert quaternions_close(result_1, expected_1, atol=EPS)

    # Standard case 2 - rotation about arbitrary axis
    axis2 = jnp.array([1.0, 1.0, 1.0], dtype=FLOAT_DTYPE)
    two_val = jnp.array(2.0, dtype=FLOAT_DTYPE)
    three_val = jnp.array(3.0, dtype=FLOAT_DTYPE)
    angle2 = two_val * pi / three_val
    result_2 = from_axis_angle(axis2, angle2)
    assert jnp.isclose(
        jnp.linalg.norm(result_2),
        jnp.array(1.0, dtype=FLOAT_DTYPE),
        atol=EPS,
    )

    # Standard case 3 - zero angle rotation
    axis3 = jnp.array([0.0, 0.0, 1.0], dtype=FLOAT_DTYPE)
    angle3 = jnp.array(0.0, dtype=FLOAT_DTYPE)
    result_3 = from_axis_angle(axis3, angle3)
    expected_3 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_3, expected_3, atol=EPS)

    # Edge case 1 - zero axis defaults to x-axis
    axis4 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    angle4 = pi / jnp.array(2.0, dtype=FLOAT_DTYPE)
    result_4 = from_axis_angle(axis4, angle4)
    expected_4 = jnp.array(
        [jnp.cos(pi / 4), jnp.sin(pi / 4), 0.0, 0.0], dtype=FLOAT_DTYPE
    )
    assert quaternions_close(result_4, expected_4, atol=EPS)

    # Edge case 2 - very small axis magnitude
    axis5 = jnp.array([EPS / 2, EPS / 2, EPS / 2], dtype=FLOAT_DTYPE)
    angle5 = pi
    result_5 = from_axis_angle(axis5, angle5)
    expected_5 = jnp.array(
        [jnp.cos(pi / 2), jnp.sin(pi / 2), 0.0, 0.0], dtype=FLOAT_DTYPE
    )
    assert quaternions_close(result_5, expected_5, atol=EPS)

    # Edge case 3 - unnormalised axis
    axis6 = jnp.array([3.0, 4.0, 0.0], dtype=FLOAT_DTYPE)
    angle6 = pi
    result_6 = from_axis_angle(axis6, angle6)
    assert jnp.isclose(
        jnp.linalg.norm(result_6),
        jnp.array(1.0, dtype=FLOAT_DTYPE),
        atol=EPS,
    )

    # Edge case 4 - full rotation
    axis7 = jnp.array([0.0, 0.0, 1.0], dtype=FLOAT_DTYPE)
    angle7 = two_val * pi
    result_7 = from_axis_angle(axis7, angle7)
    expected_7 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(jnp.abs(result_7), expected_7, atol=1e-6)

    # Test with vmap
    axes = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    angles = jnp.array([pi / 2, pi / 3, pi / 4], dtype=FLOAT_DTYPE)

    from_axis_angle_vmap = jax.vmap(from_axis_angle)
    vmap_results = from_axis_angle_vmap(axes, angles)

    assert vmap_results.shape == (3, 4)
    # All results should be unit quaternions
    norms = jnp.linalg.norm(vmap_results, axis=1)
    assert jnp.allclose(norms, jnp.ones(3, dtype=FLOAT_DTYPE), atol=EPS)


def test_from_euler_zyx(jit_mode: str) -> None:
    """Test quaternion creation from Euler angles."""
    # Standard case 1 - single axis rotations
    zero_angle = jnp.array(0.0, dtype=FLOAT_DTYPE)
    result_1 = from_euler_zyx(
        pi / jnp.array(2.0, dtype=FLOAT_DTYPE), zero_angle, zero_angle
    )
    expected_1 = jnp.array(
        [jnp.cos(pi / 4), 0.0, 0.0, jnp.sin(pi / 4)], dtype=FLOAT_DTYPE
    )
    assert quaternions_close(result_1, expected_1, atol=EPS)

    # Standard case 2 - combined rotations
    four_val = jnp.array(4.0, dtype=FLOAT_DTYPE)
    pi_quarter = pi / four_val
    result_2 = from_euler_zyx(pi_quarter, pi_quarter, pi_quarter)
    assert jnp.isclose(
        jnp.linalg.norm(result_2),
        jnp.array(1.0, dtype=FLOAT_DTYPE),
        atol=EPS,
    )

    # Standard case 3 - zero rotations
    result_3 = from_euler_zyx(zero_angle, zero_angle, zero_angle)
    expected_3 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_3, expected_3, atol=EPS)

    # Edge case 1 - gimbal lock
    result_4 = from_euler_zyx(
        zero_angle, pi / jnp.array(2.0, dtype=FLOAT_DTYPE), zero_angle
    )
    assert jnp.isclose(jnp.linalg.norm(result_4), 1.0, atol=EPS)

    # Edge case 2 - large angles
    two_pi = jnp.array(2.0, dtype=FLOAT_DTYPE) * pi
    result_5 = from_euler_zyx(two_pi, two_pi, two_pi)
    assert jnp.isclose(
        jnp.linalg.norm(result_5),
        jnp.array(1.0, dtype=FLOAT_DTYPE),
        atol=EPS,
    )

    # Edge case 3 - negative angles
    neg_pi_half = -pi / jnp.array(2.0, dtype=FLOAT_DTYPE)
    neg_pi_third = -pi / jnp.array(3.0, dtype=FLOAT_DTYPE)
    neg_pi_quarter = -pi / four_val
    result_6 = from_euler_zyx(neg_pi_half, neg_pi_third, neg_pi_quarter)
    assert jnp.isclose(
        jnp.linalg.norm(result_6),
        jnp.array(1.0, dtype=FLOAT_DTYPE),
        atol=EPS,
    )

    # Test with vmap
    yaws = jnp.array([0.0, pi / 4, pi / 2], dtype=FLOAT_DTYPE)
    pitches = jnp.array([0.0, pi / 6, pi / 3], dtype=FLOAT_DTYPE)
    rolls = jnp.array([0.0, pi / 8, pi / 4], dtype=FLOAT_DTYPE)

    from_euler_zyx_vmap = jax.vmap(from_euler_zyx)
    vmap_results = from_euler_zyx_vmap(yaws, pitches, rolls)

    assert vmap_results.shape == (3, 4)
    # All results should be unit quaternions
    norms = jnp.linalg.norm(vmap_results, axis=1)
    assert jnp.allclose(norms, jnp.ones(3, dtype=FLOAT_DTYPE), atol=EPS)


def test_to_euler_zyx(jit_mode: str) -> None:
    """Test conversion from quaternion to Euler angles."""
    # Standard case 1 - identity quaternion
    q1 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_1 = to_euler_zyx(q1)
    expected_1 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_1, expected_1, atol=EPS)

    # Standard case 2 - round trip conversion
    angles_in = jnp.array([pi / 6, pi / 8, pi / 4], dtype=FLOAT_DTYPE)
    q2 = from_euler_zyx(angles_in[0], angles_in[1], angles_in[2])
    result_2 = to_euler_zyx(q2)
    assert jnp.allclose(result_2, angles_in, atol=EPS)

    # Standard case 3 - pure yaw rotation
    q3 = jnp.array([jnp.cos(pi / 4), 0.0, 0.0, jnp.sin(pi / 4)], dtype=FLOAT_DTYPE)
    result_3 = to_euler_zyx(q3)
    expected_3 = jnp.array([pi / 2, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_3, expected_3, atol=EPS)

    # Edge case 1 - gimbal lock
    small_val = jnp.array(1e-6, dtype=FLOAT_DTYPE)
    q4 = from_euler_zyx(
        jnp.array(0.3, dtype=FLOAT_DTYPE),
        pi / jnp.array(2.0, dtype=FLOAT_DTYPE) - small_val,
        jnp.array(0.5, dtype=FLOAT_DTYPE),
    )
    result_4 = to_euler_zyx(q4)
    assert jnp.isclose(
        result_4[1], pi / jnp.array(2.0, dtype=FLOAT_DTYPE) - small_val, atol=EPS
    )

    # Edge case 2 - quaternion with negative w
    q5 = jnp.array([-0.5, 0.5, 0.5, 0.5], dtype=FLOAT_DTYPE)
    result_5 = to_euler_zyx(q5)
    assert jnp.all(jnp.abs(result_5) <= pi)

    # Edge case 3 - near-zero quaternion components
    q6 = jnp.array([1.0, EPS, EPS, EPS], dtype=FLOAT_DTYPE)
    q6 = normalize(q6)
    result_6 = to_euler_zyx(q6)
    expected_6 = jnp.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_6, expected_6, atol=1e-6)

    # Test with vmap
    quaternions = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [jnp.cos(pi / 4), 0.0, 0.0, jnp.sin(pi / 4)],
            [jnp.cos(pi / 8), jnp.sin(pi / 8), 0.0, 0.0],
        ],
        dtype=FLOAT_DTYPE,
    )

    to_euler_zyx_vmap = jax.vmap(to_euler_zyx)
    vmap_results = to_euler_zyx_vmap(quaternions)

    assert vmap_results.shape == (3, 3)
    # Verify results are valid angles
    assert jnp.all(jnp.abs(vmap_results) <= pi)


def test_to_rotation_matrix(jit_mode: str) -> None:
    """Test conversion from quaternion to rotation matrix."""
    # Standard case 1 - identity quaternion
    q1 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_1 = to_rotation_matrix(q1)
    expected_1 = jnp.eye(3, dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_1, expected_1, atol=EPS)

    # Standard case 2 - 90° rotation about z-axis
    q2 = jnp.array([jnp.cos(pi / 4), 0.0, 0.0, jnp.sin(pi / 4)], dtype=FLOAT_DTYPE)
    result_2 = to_rotation_matrix(q2)
    expected_2 = jnp.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_2, expected_2, atol=1e-6)

    # Standard case 3 - general rotation
    q3 = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=FLOAT_DTYPE)
    result_3 = to_rotation_matrix(q3)
    # Verify orthogonality
    product = result_3 @ result_3.T
    assert jnp.allclose(product, jnp.eye(3, dtype=FLOAT_DTYPE), atol=EPS)
    # Verify determinant is 1
    det = jnp.linalg.det(result_3)
    assert jnp.isclose(det, jnp.array(1.0, dtype=FLOAT_DTYPE), atol=EPS)

    # Edge case 1 - negative w quaternion
    q4 = jnp.array([-1.0, 0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_4 = to_rotation_matrix(q4)
    expected_4 = jnp.eye(3, dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_4, expected_4, atol=EPS)

    # Edge case 2 - pure imaginary quaternion
    q5 = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    result_5 = to_rotation_matrix(q5)
    expected_5 = jnp.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=FLOAT_DTYPE)
    assert jnp.allclose(result_5, expected_5, atol=EPS)

    # Edge case 3 - verify magnitude preservation
    q6 = jnp.array([0.6, 0.4, 0.3, 0.6], dtype=FLOAT_DTYPE)
    q6 = normalize(q6)
    R = to_rotation_matrix(q6)
    v = jnp.array([1.0, 2.0, 3.0], dtype=FLOAT_DTYPE)
    v_rotated = R @ v
    assert jnp.isclose(
        jnp.linalg.norm(v),
        jnp.linalg.norm(v_rotated),
        atol=EPS,
    )

    # Test with vmap
    quaternions = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [jnp.cos(pi / 4), jnp.sin(pi / 4), 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=FLOAT_DTYPE,
    )

    to_rotation_matrix_vmap = jax.vmap(to_rotation_matrix)
    vmap_results = to_rotation_matrix_vmap(quaternions)

    assert vmap_results.shape == (3, 3, 3)
    # All should be valid rotation matrices
    for i in range(3):
        R = vmap_results[i]
        # Check orthogonality
        assert jnp.allclose(
            R @ R.T,
            jnp.eye(3, dtype=FLOAT_DTYPE),
            atol=EPS,
        )
        # Check determinant
        assert jnp.isclose(
            jnp.linalg.det(R),
            jnp.array(1.0, dtype=FLOAT_DTYPE),
            atol=EPS,
        )
