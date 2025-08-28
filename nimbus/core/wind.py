"""Wind and turbulence generation using Ornstein-Uhlenbeck process."""

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
    Update a vector state using the exact discrete-time OU process.

    This function uses the *stationary RMS* parametrization:
      x_{t+dt} = exp(-theta*dt) * x_t + (rms * sqrt(1 - exp(-2*theta*dt))) * eps,
    where eps ~ N(0, I). If different RMS per-axis are desired, pass them via
    `scale_factors` (e.g., [1, 1, vertical_scale]) and they will multiply the
    base RMS for the noise term only.

    Parameters
    ----------
    key : PRNGKey
        JAX random key for noise generation.
    current : Vector3
        Current state vector (e.g., gust velocity in NED) [m/s].
    mean_reversion_rate : FloatScalar
        OU rate theta = 1 / tau, where tau is the time constant [1/s].
    stationary_rms : FloatScalar
        Target stationary RMS (standard deviation) for the *unscaled* components [m/s].
    dt : FloatScalar
        Time step [s].
    scale_factors : Vector3
        Per-axis multipliers for the stationary RMS (unitless). For isotropic
        horizontal with reduced vertical, use [1.0, 1.0, vertical_scale].

    Returns
    -------
    Vector3
        Updated state vector.
    """
    theta = mean_reversion_rate
    decay = jnp.exp(-theta * dt)

    # Exact discrete-time noise standard deviation so that stationary std = stationary_rms.
    base_std = stationary_rms * jnp.sqrt(1.0 - decay**2)

    # One 3-vector draw (independent components)
    eps = jax.random.normal(key, (3,), dtype=FLOAT_DTYPE)

    # Apply per-axis scaling to the *noise* (do not scale the state itself)
    noise = eps * base_std * scale_factors

    # OU update: exponential decay + additive noise
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
    Evolve the gust (zero-mean turbulence) as a vector OU process.

    Notes
    -----
    - `gust_intensity` is interpreted as the *stationary RMS* of the horizontal
      gust components [m/s]. Vertical RMS is `vertical_damping * gust_intensity`.
    - Horizontal isotropy emerges by using the same RMS for north/east and not
      re-randomizing direction each step (no angle sampling). This preserves
      temporal correlation in both magnitude and direction, reducing jitter.

    Parameters
    ----------
    key : PRNGKey
        JAX random key for noise generation.
    gust : Vector3
        Current gust velocity in NED [m/s].
    gust_intensity : FloatScalar
        Stationary RMS of horizontal gusts [m/s].
    gust_duration : FloatScalar
        OU time constant tau [s].
    vertical_damping : FloatScalar
        Vertical RMS scale relative to horizontal (unitless).
    dt : FloatScalar
        Time step [s].

    Returns
    -------
    Vector3
        Updated gust velocity in NED [m/s].
    """
    theta = 1.0 / gust_duration

    # Per-axis RMS scaling: isotropic horizontal, reduced vertical
    scale_factors = jnp.array([1.0, 1.0, vertical_damping], dtype=FLOAT_DTYPE)

    return step_ornstein_uhlenbeck(
        key=key,
        current=gust,
        mean_reversion_rate=jnp.array(theta, dtype=FLOAT_DTYPE),
        stationary_rms=jnp.array(gust_intensity, dtype=FLOAT_DTYPE),
        dt=dt,
        scale_factors=scale_factors,
    )
