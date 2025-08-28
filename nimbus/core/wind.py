"""
Wind and turbulence generation using Ornstein-Uhlenbeck process.

All quantities use SI units and NED world-frame unless noted.
"""

import jax
import jax.numpy as jnp
from chex import PRNGKey

from .primitives import FLOAT_DTYPE, FloatScalar, Vector3


def step_ornstein_uhlenbeck(
    key: PRNGKey,
    current: Vector3,
    mean_reversion_rate: FloatScalar,
    stationary_rms: FloatScalar,
    dt: FloatScalar,
    scale_factors: Vector3,
) -> Vector3:
    """
    Update a vector state using the exact discrete-time Ornstein-Uhlenbeck process.

    Parameters
    ----------
    key: PRNGKey
        JAX random key for noise generation.
    current: Vector3
        Current state vector [m/s].
    mean_reversion_rate: FloatScalar
        OU rate theta = 1/tau, where tau is time constant [1/s].
    stationary_rms: FloatScalar
        Target stationary RMS for unscaled components [m/s].
    dt: FloatScalar
        Time step [s].
    scale_factors: Vector3
        Per-axis multipliers for the stationary RMS (unitless).

    Returns
    -------
    current: Vector3
        Updated state vector [m/s].

    Notes
    -----
    Uses exact discrete-time OU: x_{t+dt} = exp(-theta*dt) * x_t + noise,
    where noise amplitude ensures stationary std = stationary_rms.
    For anisotropic turbulence use scale_factors=[1, 1, vertical_scale].
    """
    theta = mean_reversion_rate
    decay = jnp.exp(-theta * dt)

    # Exact discrete-time noise ensures stationary std equals stationary_rms
    base_std = stationary_rms * jnp.sqrt(1.0 - decay**2)

    # Independent gaussian noise for each component
    eps = jax.random.normal(key, (3,), dtype=FLOAT_DTYPE)

    # Per-axis scaling applied to noise only
    noise = eps * base_std * scale_factors

    # OU update: exponential decay plus additive noise
    return current * decay + noise


def calculate_wind(
    key: PRNGKey,
    gust: Vector3,
    gust_intensity: FloatScalar,
    gust_duration: FloatScalar,
    vertical_damping: FloatScalar,
    dt: FloatScalar,
) -> Vector3:
    """
    Evolve turbulent gust velocity using vector Ornstein-Uhlenbeck process.

    Parameters
    ----------
    key: PRNGKey
        JAX random key for noise generation.
    gust: Vector3
        Current gust velocity in NED [m/s].
    gust_intensity: FloatScalar
        Stationary RMS of horizontal gusts [m/s].
    gust_duration: FloatScalar
        Time constant tau for temporal correlation [s].
    vertical_damping: FloatScalar
        Vertical RMS scale relative to horizontal (unitless).
    dt: FloatScalar
        Time step [s].

    Returns
    -------
    gust: Vector3
        Updated gust velocity in NED [m/s].

    Notes
    -----
    Horizontal components use full gust_intensity RMS.
    Vertical component uses vertical_damping * gust_intensity RMS.
    Temporal correlation preserves both magnitude and direction.
    """
    # Mean reversion rate from time constant
    theta = 1.0 / gust_duration

    # Isotropic horizontal, reduced vertical turbulence
    scale_factors = jnp.array([1.0, 1.0, vertical_damping], dtype=FLOAT_DTYPE)

    return step_ornstein_uhlenbeck(
        key=key,
        current=gust,
        mean_reversion_rate=jnp.array(theta, dtype=FLOAT_DTYPE),
        stationary_rms=jnp.array(gust_intensity, dtype=FLOAT_DTYPE),
        dt=dt,
        scale_factors=scale_factors,
    )
