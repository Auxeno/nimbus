"""Configuration Dataclasses for aircraft, physics and map parameters."""

from dataclasses import dataclass
from functools import partial

import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PhysicsConfig:
    """Configuration for physical constants."""

    gravity: float = 9.80665
    """Standard gravitational acceleration [m/s^2]."""

    rho_0: float = 1.225
    """Air density at sea level [kg/m^3]."""

    rho_decay: float = 5500.0
    """Height at which air density halves [m]."""


@partial(
    jax.tree_util.register_dataclass,
    data_fields=(
        "base_scale",
        "persistence",
        "lacunarity",
        "mountain_gain",
        "bump_gain",
    ),
    meta_fields=(
        "resolution",
        "octaves",
        "padding",
    ),
)
@dataclass(frozen=True)
class TerrainConfig:
    """Configuration for terrain heightmap generation."""

    resolution: int = 256
    """Size of square heightmap in pixels (width = height = size)."""

    base_scale: float = 0.012
    """Frequency scale of the base noise field."""

    octaves: int = 5
    """Number of noise layers (fractal Brownian motion octaves)."""

    persistence: float = 0.5
    """Amplitude reduction per octave (controls roughness)."""

    lacunarity: float = 2.0
    """Frequency growth per octave (controls detail level)."""

    mountain_gain: float = 1.0
    """Relative gain for large-scale terrain features (low octaves)."""

    bump_gain: float = 1.0
    """Relative gain for fine-scale features (high octaves)."""

    padding: int = 0
    """Extra constant border padding around the map (0.5 height)."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MapConfig:
    """Configuration for world map scaling and terrain."""

    terrain: TerrainConfig = TerrainConfig()
    """Terrain generation parameters."""

    size: float = 10_000.0
    """World map width/length square [m]."""

    terrain_height: float = 1500.0
    """Maximum terrain elevation [m]."""

    use_bilinear: bool = True
    """Use bilinear or nearest neighbour sampling for terrain height."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PDControllerConfig:
    """Configuration for PD controller."""

    kp: float
    """Proportional gain."""

    kd: float
    """Derivative gain."""

    max_correction: float
    """Maximum control correction magnitude."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AircraftConfig:
    """Configuration for aircraft physical and aerodynamic parameters."""

    mass: float = 10_000.0
    """Aircraft mass [kg]."""

    surface_areas: tuple[float, float, float] = (10.0, 20.0, 30.0)
    """Reference wing, tailplane, and fin surface areas [m^2]."""

    max_thrust: float = 200_000.0
    """Maximum available engine thrust [N]."""

    max_attack_angle: float = 18.0
    """Maximum angle of attack before stall [degrees]."""

    max_sideslip_angle: float = 20.0
    """Maximum sideslip angle before stall [degrees]."""

    coef_drag: float = 0.2
    """Baseline drag coefficient."""

    coef_lift: float = 10.0
    """Lift curve slope coefficient."""

    coef_sideslip: float = 5.0
    """Sideslip slop coefficient."""

    coef_rot_damping: float = 4.5
    """Rotational damping coefficient."""

    coefs_torque: tuple[float, float, float] = (20.0, 6.0, 1.0)
    """Control torque coefficients for roll, pitch, and yaw."""

    g_limit_max: float = 9.0
    """Maximum positive G-force limit."""

    g_limit_min: float = -3.0
    """Minimum negative G-force limit."""

    g_limiter_controller_config: PDControllerConfig = PDControllerConfig(
        kp=2.0, kd=0.2, max_correction=1.2
    )
    """PD controller configuration for G-limiter."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RouteConfig:
    """Configuration for waypoint route."""

    radius: float = 50.0
    """Waypoint radius [m]."""

    loop: bool = True
    """Whether too loop back to beginning once route has been finished."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for simulation."""

    aircraft: AircraftConfig = AircraftConfig()
    """Config for simulation aircraft."""

    physics: PhysicsConfig = PhysicsConfig()
    """Config for simulation physics."""

    map: MapConfig = MapConfig()
    """Config for simulation map."""

    route: RouteConfig = RouteConfig()
    """Config for waypoint route."""

    dt: float = 1 / 60
    """Fixed time step delta [s]."""
