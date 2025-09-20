"""Nimbus - JAX-based flight simulation framework."""

import jax
import jax.numpy as jnp

from nimbus.core.config import (
    AircraftConfig,
    MapConfig,
    PhysicsConfig,
    PIDControllerConfig,
    RouteConfig,
    SimulationConfig,
    TerrainConfig,
    WindConfig,
)
from nimbus.core.primitives import Matrix
from nimbus.core.scenario import (
    Fixed,
    InitialConditions,
    Uniform,
    generate_route,
    generate_scenario,
    generate_simulation,
    generate_terrain_map,
)
from nimbus.core.simulation import (
    update_controls,
    step,
    step_aircraft_euler,
    step_aircraft_rk4,
)
from nimbus.core.state import (
    Aircraft,
    Body,
    Controls,
    Meta,
    PIDControllerState,
    Route,
    Simulation,
    Wind,
)
from nimbus.core.terrain import generate_heightmap


# Convenience functions
def quick_scenario(seed: int = 42) -> tuple[Simulation, Matrix, Route]:
    """Generate a default scenario without managing PRNG keys."""
    key = jax.random.PRNGKey(seed)
    return generate_scenario(key, InitialConditions.showcase(), TerrainConfig())


def quick_terrain(seed: int = 42, style: str = "default") -> Matrix:
    """Generate terrain heightmap with preset styles."""
    key = jax.random.PRNGKey(seed)

    styles = {
        "flat": TerrainConfig(mountain_gain=0.1, bump_gain=0.1),
        "mountains": TerrainConfig(mountain_gain=2.0, bump_gain=0.3),
        "rough": TerrainConfig(mountain_gain=1.0, bump_gain=2.0),
        "default": TerrainConfig(),
    }

    config = styles.get(style, TerrainConfig())
    return generate_terrain_map(key, config)


def simple_conditions(
    altitude: float = 500,
    speed: float = 150,
    heading: float = 0,
) -> InitialConditions:
    """Create InitialConditions with simple parameters."""
    heading_rad = jnp.deg2rad(heading)
    vn = speed * jnp.cos(heading_rad)
    ve = speed * jnp.sin(heading_rad)

    return InitialConditions(
        position=(Fixed(0.0), Fixed(0.0), Fixed(-float(altitude))),
        velocity=(Fixed(float(vn)), Fixed(float(ve)), Fixed(0.0)),
        orientation_euler=(Fixed(float(heading)), Fixed(0.0), Fixed(0.0)),
        angular_velocity=(Fixed(0.0), Fixed(0.0), Fixed(0.0)),
        wind_speed=Fixed(0.0),
        wind_direction=Fixed(0.0),
        waypoints=(
            (Fixed(1000.0), Fixed(0.0), Fixed(-float(altitude))),
            (Fixed(2000.0), Fixed(0.0), Fixed(-float(altitude))),
            (Fixed(3000.0), Fixed(0.0), Fixed(-float(altitude))),
            (Fixed(4000.0), Fixed(0.0), Fixed(-float(altitude))),
        ),
    )


__all__ = [
    # Configuration classes
    "AircraftConfig",
    "MapConfig",
    "PhysicsConfig",
    "PIDControllerConfig",
    "RouteConfig",
    "SimulationConfig",
    "TerrainConfig",
    "WindConfig",
    # Scenario generation
    "Fixed",
    "Uniform",
    "InitialConditions",
    "generate_scenario",
    "generate_simulation",
    "generate_route",
    "generate_terrain_map",
    # Core simulation
    "step",
    "update_controls",
    "step_aircraft_euler",
    "step_aircraft_rk4",
    # State classes
    "Simulation",
    "Aircraft",
    "Body",
    "Controls",
    "Wind",
    "Route",
    "Meta",
    "PIDControllerState",
    # Terrain
    "generate_heightmap",
    # Convenience functions
    "quick_scenario",
    "quick_terrain",
    "simple_conditions",
]
