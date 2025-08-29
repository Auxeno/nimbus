"""
Ursina runtime module.
"""

from ursina import AmbientLight, DirectionalLight, Sky, Texture, Ursina, scene

from ...core.config import SimulationConfig
from ...core.primitives import Matrix
from ...core.state import Route, Simulation
from .config import UrsinaConfig
from .entities import (
    AircraftEntity,
    Coordinator,
    FollowCamera,
    TerrainSurface,
    WaypointEntity,
)
from .utils import generate_gradient_image, hex_to_rgba

default_simulation_config = SimulationConfig()
default_ursina_config = UrsinaConfig()


class UrsinaRuntime:
    def __init__(
        self,
        simulation: Simulation,
        heightmap: Matrix,
        route: Route,
        simulation_config: SimulationConfig = default_simulation_config,
        ursina_config: UrsinaConfig = default_ursina_config,
    ) -> None:
        self.simulation = simulation
        self.heightmap = heightmap
        self.route = route
        self.simulation_config = simulation_config
        self.ursina_config = ursina_config

    def _build_scene(self, use_contour_texture: bool = True) -> None:
        TerrainSurface(
            heightmap=self.heightmap,
            simulation_config=self.simulation_config,
            ursina_config=self.ursina_config,
            use_contour_texture=use_contour_texture,
        )

        Sky(
            texture=Texture(
                generate_gradient_image(
                    width=2000,
                    height=500,
                    top_color=self.ursina_config.sky_top_color,
                    bottom_color=self.ursina_config.sky_bottom_color,
                    mid_pos=self.ursina_config.sky_gradient_midpoint,
                )
            )
        )

        AmbientLight(color=hex_to_rgba(self.ursina_config.light_color))
        DirectionalLight(shadows=True, rotation=(0, -0.8, -0.866))

        scene.fog_color = hex_to_rgba(self.ursina_config.fog_color)
        scene.fog_density = self.ursina_config.fog_density

        aircraft = AircraftEntity(
            simulation=self.simulation,
            simulation_config=self.simulation_config,
            ursina_config=self.ursina_config,
        )

        waypoints = [
            WaypointEntity(
                position=self.route.positions[idx],
                simulation_config=self.simulation_config,
                ursina_config=self.ursina_config,
            )
            for idx in range(self.route.positions.shape[0])
        ]

        FollowCamera(target=aircraft)

        Coordinator(
            aircraft_entity=aircraft,
            waypoint_entities=waypoints,
            simulation=self.simulation,
            heightmap=self.heightmap,
            route=self.route,
            config=self.simulation_config,
            ursina_config=self.ursina_config,
        )

    def run(self) -> None:
        app = Ursina(vsync=True, title="nimbus", development_mode=False)
        self._build_scene()
        app.run()
