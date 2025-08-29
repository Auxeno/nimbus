"""
Module defining Ursina environment entities.
"""

import time
from dataclasses import replace
from math import cos, radians, sin

import jax
import jax.numpy as jnp
import numpy as np
from ursina import (
    Color,
    Entity,
    Mesh,
    Quat,
    Terrain,
    Text,
    Texture,
    Vec3,
    application,
    camera,
    distance,
    lerp,
    mouse,
    scene,
)
from ursina.color import rgba, white
from ursina.input_handler import held_keys
from ursina.shaders import lit_with_shadows_shader, unlit_shader

from ...core.config import SimulationConfig
from ...core.interface import calculate_g_force
from ...core.primitives import FLOAT_DTYPE, Matrix, Vector3
from ...core.simulation import set_controls, step
from ...core.state import Aircraft, Controls, Route, Simulation
from .config import UrsinaConfig
from .utils import (
    compute_flight_metrics,
    convert_scale,
    generate_blank_texture,
    generate_terrain_contour_image,
    hex_to_rgba,
    local_asset_folder,
    ned_quat_to_eun_quat,
    ned_to_eun,
)

# Apply lighting shader to all entities
Entity.default_shader = lit_with_shadows_shader  # type: ignore


# JIT compile functions used by simulation
step = jax.jit(step)
calculate_g_force = jax.jit(calculate_g_force)
set_controls = jax.jit(set_controls)
compute_flight_metrics = jax.jit(compute_flight_metrics)


class FollowCamera(Entity):
    def __init__(self, target: Entity) -> None:
        super().__init__(position=target.position)
        camera.parent = self
        camera.fov = 100
        mouse.locked = False

        self.body_offset = Vec3(0.0, 0.0, -6.0)
        self.world_offset = Vec3(0.0, 7.0, 0.0)
        self.target = target

        # Zoom parameters
        self.target_fov = camera.fov
        self.zoom_speed = 1.0
        self.zoom_smoothing = 4
        self.min_fov = 30
        self.max_fov = 140

    def input(self, key: str) -> None:
        if key == "scroll up":
            # Zoom in (decrease FOV)
            self.target_fov -= self.zoom_speed * (abs(self.target_fov) * 0.1)
            self.target_fov = max(self.target_fov, self.min_fov)

        elif key == "scroll down":
            # Zoom out (increase FOV)
            self.target_fov += self.zoom_speed * (abs(self.target_fov) * 0.1)
            self.target_fov = min(self.target_fov, self.max_fov)

    def update(self) -> None:
        # Horizontal (xz) direction that points straight out of the nose
        fwd = self.target.forward
        fwd_xz = Vec3(fwd.x, 0.0, fwd.z).normalized()  # discard y component

        # Rebuild the right vector in xz to stay orthogonal
        right_xz = fwd_xz.cross(Vec3(0.0, 1.0, 0.0)).normalized()

        # Body-frame offset using the flattened axes
        relative_offset = fwd_xz * self.body_offset.z + right_xz * self.body_offset.x

        desired_pos = self.target.position + relative_offset + self.world_offset

        self.position = lerp(self.position, desired_pos, 8 * time.dt)  # type: ignore
        self.look_at(self.target)
        self.rotation_z = 0.0

        # Smooth zoom interpolation
        camera.fov = lerp(camera.fov, self.target_fov, time.dt * self.zoom_smoothing)  # type: ignore


class InputHandler(Entity):
    def __init__(self) -> None:
        super().__init__(visible=False, ignore_paused=True)
        self.controls = Controls.default()
        self.paused_text = Text(
            "PAUSED",
            origin=(0, 0),
            scale=2.0,
            enabled=False,
        )

    def update(self) -> None:
        self.controls = Controls.default()

        if held_keys["q"]:
            self.controls = replace(self.controls, aileron=jnp.array(-1.0, dtype=FLOAT_DTYPE))
        if held_keys["e"]:
            self.controls = replace(self.controls, aileron=jnp.array(1.0, dtype=FLOAT_DTYPE))
        if held_keys["w"]:
            self.controls = replace(self.controls, elevator=jnp.array(-1.0, dtype=FLOAT_DTYPE))
        if held_keys["s"]:
            self.controls = replace(self.controls, elevator=jnp.array(1.0, dtype=FLOAT_DTYPE))
        if held_keys["a"]:
            self.controls = replace(self.controls, rudder=jnp.array(-1.0, dtype=FLOAT_DTYPE))
        if held_keys["d"]:
            self.controls = replace(self.controls, rudder=jnp.array(1.0, dtype=FLOAT_DTYPE))

    def input(self, key: str) -> None:
        if key == "p":
            application.paused = not application.paused
            self.paused_text.enabled = application.paused
            # Save replay when pausing if recorder is available
            if application.paused and self.recorder and self.recorder.frames:
                print("Paused - saving replay to disk...")
                if self.recorder.save_to_disk():
                    print("Replay saved successfully!")

    def get_inputs(self) -> Controls:
        return self.controls


class TextUI(Entity):
    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(parent=camera.ui)
        self.config = config

        # Flight indicators
        self.flight_info_text = Text(
            text="",
            origin=(0, -1),
            position=(-0.0, -0.45),
            color=hex_to_rgba("#FFFFFFDD"),
            scale=0.8,
            parent=camera.ui,
        )

    def update_ui(self, simulation_state: Simulation, heightmap: Matrix) -> None:
        # Flight metrics
        alpha, speed, g_force, heading, agl, vs = compute_flight_metrics(
            simulation_state.aircraft,
            simulation_state.wind,
            heightmap,
            self.config.aircraft,
            self.config.physics,
            self.config.map,
        )
        self.flight_info_text.text = (
            f"α {alpha:4.1f}°\t" + f"S {speed:4.0f} \t" + f"AGL {-agl:4.0f}\n"
            f"G {g_force:4.1f} \t" + f"H  {heading:03.0f} \t" + f"VS  {vs:+4.0f}"
        )


class IconUI(Entity):
    def __init__(self, config: SimulationConfig, ursina_config: UrsinaConfig) -> None:
        super().__init__(parent=camera.ui, visible=False)
        self.config = config
        self.ursina_config = ursina_config

        # Store base color from config
        self.base_color = hex_to_rgba(ursina_config.icon_color)

        # Alpha multipliers for on/off screen
        self.onscreen_alpha_multiplier = hex_to_rgba("#FFFFFF22")
        self.offscreen_alpha_multiplier = hex_to_rgba("#FFFFFFFF")

        # Distance-based fade parameters [m]
        self.start_fade_distance = 1500.0
        self.end_fade_distance = 1000.0

        # Screen boundary limits for icon clamping
        self.screen_limit = Vec3(0.74, 0.45, 0)

        with local_asset_folder():
            # Main waypoint ring icon
            self.icon_entity = Entity(
                model="circle.glb",
                color=self.base_color,
                scale=convert_scale(0.02),
                parent=camera.ui,
                visible=False,
            )

            # Smaller offscreen indicator ring
            self.offscreen_indicator = Entity(
                model="circle.glb",
                color=self.base_color,
                scale=convert_scale(0.01),
                parent=camera.ui,
                visible=False,
            )

            self.distance_text = Text(
                text="",
                origin=(0, 0),
                position=(0.0, 2.0),
                scale=35.0,
                color=self.base_color,
                parent=self.icon_entity,
                visible=True,
            )

    def update_waypoint_ui(
        self, aircraft: Aircraft, route: Route, waypoint_entity: Entity | None
    ) -> None:
        # Hide if no waypoints or all visited
        if waypoint_entity is None or jnp.all(route.visited):
            self.icon_entity.visible = False
            self.offscreen_indicator.visible = False
            return

        current_idx = int(route.current_idx)
        waypoint_pos_ned = route.positions[current_idx]

        delta = waypoint_pos_ned - aircraft.body.position
        distance_m = float(jnp.linalg.norm(delta))
        distance_km = distance_m / 1000.0

        self.distance_text.text = f"{distance_km:.1f} km"

        screen_pos = waypoint_entity.screen_position

        on_screen = (
            -self.screen_limit.x <= screen_pos.x <= self.screen_limit.x
            and -self.screen_limit.y <= screen_pos.y <= self.screen_limit.y
        )

        if on_screen:
            # Position icon at waypoint screen position with reduced alpha
            self.icon_entity.position = Vec3(screen_pos.x, screen_pos.y, 0)
            self.icon_entity.visible = True
            self.offscreen_indicator.visible = False

            # Apply reduced alpha when on-screen (original behavior for icon)
            reduced_color = rgba(
                self.base_color.r * self.onscreen_alpha_multiplier.r,
                self.base_color.g * self.onscreen_alpha_multiplier.g,
                self.base_color.b * self.onscreen_alpha_multiplier.b,
                self.base_color.a * self.onscreen_alpha_multiplier.a,
            )
            self.icon_entity.color = reduced_color

            # Calculate distance-based alpha fade for text only
            if distance_m <= self.end_fade_distance:
                # Within fade range - alpha goes to 0 at end_fade_distance
                alpha = 0.0
            elif distance_m >= self.start_fade_distance:
                # Beyond fade range - use base color (no fade)
                alpha = 1.0
            else:
                # Interpolate alpha between start and end fade distances
                fade_factor = (distance_m - self.end_fade_distance) / (
                    self.start_fade_distance - self.end_fade_distance
                )
                alpha = fade_factor

            # Apply the calculated alpha to text
            self.distance_text.color = self.base_color * rgba(1.0, 1.0, 1.0, alpha)
        else:
            # Clamp to screen edge with full alpha
            direction = Vec3(screen_pos.x, screen_pos.y, 0).normalized()
            edge_x = direction.x * self.screen_limit.x
            edge_y = direction.y * self.screen_limit.y

            self.icon_entity.position = Vec3(edge_x, edge_y, 0)
            self.icon_entity.visible = True

            # Apply full alpha when off-screen
            self.icon_entity.color = self.base_color
            self.distance_text.color = self.base_color

            # Position offscreen indicator slightly inside
            self.offscreen_indicator.position = self.icon_entity.position * 0.9
            self.offscreen_indicator.visible = True
            self.offscreen_indicator.color = self.base_color


class Ribbon(Mesh):
    def __init__(
        self,
        width: float = 1.0,
        color: Color = white,
        **kwargs,
    ) -> None:
        super().__init__(mode="triangle", **kwargs)
        self.set_two_sided(True)
        self.width = width
        self.color = color

        self.path: list[Vec3] = []
        self.rolls: list[float] = []
        self.segment_alphas: list[float] = []

        # Mesh buffers
        self.vertices: list[Vec3] = []
        self.triangles: list[int] = []
        self.colors: list[Color] = []

    def generate(self) -> None:
        half_width = self.width * 0.5

        # Clear previous mesh
        self.vertices.clear()
        self.triangles.clear()
        self.colors.clear()

        if len(self.path) < 2:
            super().generate()
            return

        # Build left/right offsets for each path point
        left_points: list[Vec3] = []
        right_points: list[Vec3] = []

        for i, pos in enumerate(self.path):
            # Forward direction from neighbors
            if i < len(self.path) - 1:
                fwd = (self.path[i + 1] - pos).normalized()  # type: ignore
            else:
                fwd = (pos - self.path[i - 1]).normalized()  # type: ignore

            # Lateral normal in XZ, then apply roll
            normal = Vec3(-fwd.z, 0, fwd.x)
            roll = self.rolls[i] if i < len(self.rolls) else 0.0
            if roll:
                c = cos(roll)
                s = sin(roll)
                normal = Vec3(normal.x * c - normal.z * s, 0, normal.x * s + normal.z * c)

            normal *= half_width
            left_points.append(pos + normal)
            right_points.append(pos - normal)

        # Use provided external alphas, fallback to 1.0 if lengths mismatch
        if len(self.segment_alphas) == len(self.path):
            alphas = self.segment_alphas
        else:
            alphas = [1.0] * len(self.path)

        # Build quad strips (two triangles per segment)
        for i in range(len(left_points) - 1):
            l1, r1 = left_points[i], right_points[i]
            l2, r2 = left_points[i + 1], right_points[i + 1]

            base = len(self.vertices)
            self.vertices += [l1, r1, l2, r2]
            self.triangles += [base, base + 1, base + 2, base + 1, base + 3, base + 2]

            a = alphas[i]
            col = rgba(self.color.r, self.color.g, self.color.b, a)
            self.colors += [col, col, col, col]

        super().generate()


class Trail(Entity):
    def __init__(
        self,
        parent: Entity,
        ursina_config: UrsinaConfig,
        **kwargs,
    ) -> None:
        super().__init__(parent=parent, shader=unlit_shader, **kwargs)  # type: ignore
        self.set_two_sided(True)

        self.ursina_config = ursina_config

        self.ribbon = Ribbon(
            width=ursina_config.trail_width,
            color=hex_to_rgba(ursina_config.trail_color),
        )
        self.renderer = Entity(model=self.ribbon, parent=scene)

        self.default_alpha = ursina_config.trail_default_alpha
        self._t = 0.0

        # (timestamp, base_alpha) per point
        self.segments: list[tuple[float, float]] = []

        # Mirror visibility with renderer
        self.on_enable = self.renderer.enable
        self.on_disable = self.renderer.disable

    def add_segment(self, alpha: float | None = None) -> None:
        pos = self.world_position
        roll = radians(getattr(self.parent, "rotation_z", 0.0))

        # Append point + roll
        self.ribbon.path.append(pos)
        self.ribbon.rolls.append(roll)

        # Record time + base alpha (G may adjust default_alpha externally)
        ts = time.time()
        self.segments.append((ts, self.default_alpha if alpha is None else alpha))

    def update(self) -> None:
        # Rate limit updates
        self._t += time.dt  # type: ignore
        if self._t < self.ursina_config.trail_update_interval:
            return
        self._t = 0.0

        # Add new point if moved enough
        if (
            not self.ribbon.path
            or distance(self.world_position, self.ribbon.path[-1])
            > self.ursina_config.trail_min_spacing
        ):
            self.add_segment()

        now = time.time()

        # Age out old points beyond fade_duration
        while self.segments and (
            now - self.segments[0][0] > self.ursina_config.trail_fade_duration
        ):
            self.segments.pop(0)
            self.ribbon.path.pop(0)
            self.ribbon.rolls.pop(0)

        # Enforce max length (points, hence quads = points-1)
        if len(self.segments) > self.ursina_config.trail_segments:
            excess = len(self.segments) - self.ursina_config.trail_segments
            del self.segments[:excess]
            del self.ribbon.path[:excess]
            del self.ribbon.rolls[:excess]

        # Recompute per-point alpha (external control for Ribbon)
        self.ribbon.segment_alphas = []
        for ts, base_a in self.segments:
            age = now - ts
            fade = 1.0 - (age / self.ursina_config.trail_fade_duration)
            self.ribbon.segment_alphas.append(float(np.clip(base_a * fade, 0.0, 1.0)))

        # Rebuild mesh
        self.ribbon.generate()


class WingtipsTrail(Entity):
    def __init__(
        self,
        parent: Entity,
        model_scale: float,
        ursina_config: UrsinaConfig,
        **kwargs,
    ) -> None:
        super().__init__(parent=parent, **kwargs)
        self.trails: list[Trail] = []

        wingtip_offset = Vec3(*ursina_config.wingtip_offset)

        for direction in (-1.0, +1.0):
            offset = Vec3(direction, 1, 1) * wingtip_offset
            anchor = Entity(parent=self, position=(offset / model_scale), visible=False)

            trail = Trail(parent=anchor, ursina_config=ursina_config)
            self.trails.append(trail)

    def update(self) -> None:
        rz = getattr(self.parent, "rotation_z", 0.0)
        for trail in self.trails:
            trail.parent.rotation_z = rz

    def set_current_g_force(
        self,
        g_force: float,
        g_min: float = 3.5,
        g_max: float = 8.0,
    ) -> None:
        g_norm = float(np.clip((g_force - g_min) / (g_max - g_min), 0.0, 1.0))
        target_alpha = 0.9 * g_norm
        for trail in self.trails:
            trail.default_alpha = target_alpha


class TerrainSurface:
    def __init__(
        self,
        heightmap: Matrix,
        simulation_config: SimulationConfig,
        ursina_config: UrsinaConfig,
        use_contour_texture: bool = False,
    ) -> None:
        # Transpose height map and normalise to range [-1, +1]
        height_map = 2 * (heightmap.T - 0.5)

        # Multiply by 255 (Ursina expects uint8 image), scale by terrain height
        height_values = 255 * simulation_config.map.terrain_height * height_map

        # Generate contour texture if requested
        texture = None
        if use_contour_texture:
            contour_img = generate_terrain_contour_image(
                heightmap=np.flipud(np.array(heightmap)),
                resolution=ursina_config.contour_texture_resolution,
                contour_interval=ursina_config.contour_interval,
                terrain_height=simulation_config.map.terrain_height,
                terrain_color=ursina_config.terrain_color,
                contour_color=ursina_config.contour_color,
            )
            texture = Texture(contour_img)

        # Terrain entity with heightmap mesh
        terrain_color = (
            hex_to_rgba(ursina_config.terrain_color) if not use_contour_texture else white
        )
        Entity(
            model=Terrain(height_values=height_values, skip=0),
            scale=Vec3(simulation_config.map.size, 1.0, simulation_config.map.size),
            color=terrain_color,
            texture=texture,  # type: ignore
            shader=unlit_shader,  # type: ignore
        )
        # Build normal map for lighting (can disable for faster loading)
        # terrain.model.generate_normals()  # type: ignore

        # Add flat planes that surround central terrain (3x3 grid)
        blank_texture = generate_blank_texture(resolution=1, color=ursina_config.terrain_color)
        for dx in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dz == 0:
                    continue  # skip the center tile (terrain)
                Entity(
                    model="plane",
                    scale=Vec3(simulation_config.map.size, 1.0, simulation_config.map.size),
                    position=Vec3(
                        dx * simulation_config.map.size,
                        0.0,
                        dz * simulation_config.map.size,
                    ),
                    texture=Texture(blank_texture),  # type: ignore
                    shader=unlit_shader,  # type: ignore
                )


class AircraftEntity(Entity):
    def __init__(
        self,
        simulation: Simulation,
        simulation_config: SimulationConfig,
        ursina_config: UrsinaConfig,
    ) -> None:
        color = hex_to_rgba(ursina_config.aircraft_color)
        model_scale = ursina_config.aircraft_model_scale

        with local_asset_folder():
            super().__init__(
                model=ursina_config.aircraft_model_path,
                scale=convert_scale(model_scale),
                texture=Texture(
                    generate_blank_texture(resolution=1, color=ursina_config.aircraft_color)
                ),  # type: ignore
                color=color,
            )
        self.flipped_faces_setter(False)

        self.simulation_config = simulation_config

        # Add wingtip trails
        self.wingtip_trail = WingtipsTrail(
            parent=self,
            model_scale=model_scale,
            ursina_config=ursina_config,
        )

        self.update_entity(simulation)

    def update_entity(self, simulation: Simulation):
        if simulation.aircraft.meta.active:
            self.position = ned_to_eun(simulation.aircraft.body.position)
            orientation_enu = ned_quat_to_eun_quat(simulation.aircraft.body.orientation)
            self.quaternion_setter(Quat(*orientation_enu))

            # Update wingtip trails
            g_force_vec = calculate_g_force(
                simulation.aircraft,
                simulation.wind,
                self.simulation_config.aircraft,
                self.simulation_config.physics,
            )
            g_force = abs(float(-g_force_vec[2]))
            self.wingtip_trail.set_current_g_force(g_force)


class WaypointEntity(Entity):
    def __init__(
        self,
        position: Vector3,
        simulation_config: SimulationConfig,
        ursina_config: UrsinaConfig,
    ) -> None:
        radius = simulation_config.route.radius

        with local_asset_folder():
            super().__init__(
                model="sphere.glb",
                scale=convert_scale(radius),
                color=hex_to_rgba(ursina_config.waypoint_color_unvisited),
                position=Vec3(*ned_to_eun(position)),
            )

        self.color_current = hex_to_rgba(ursina_config.waypoint_color_current)
        self.color_next = hex_to_rgba(ursina_config.waypoint_color_next)
        self.color_unvisited = hex_to_rgba(ursina_config.waypoint_color_unvisited)
        self.color_visited = hex_to_rgba(ursina_config.waypoint_color_visited)

    def update_entity(self, is_current: bool, is_next: bool, is_visited: bool) -> None:
        if is_visited:
            self.color = self.color_visited
        elif is_current:
            self.color = self.color_current
        elif is_next:
            self.color = self.color_next
        else:
            self.color = self.color_unvisited


class Coordinator(Entity):
    def __init__(
        self,
        aircraft_entity: AircraftEntity,
        waypoint_entities: list[WaypointEntity],
        simulation: Simulation,
        heightmap: Matrix,
        route: Route,
        config: SimulationConfig,
        ursina_config: UrsinaConfig,
    ) -> None:
        super().__init__(visible=False)
        self.input_handler = InputHandler()
        self.text_ui = TextUI(config=config)
        self.icon_ui = IconUI(config=config, ursina_config=ursina_config)
        self.aircraft_entity = aircraft_entity
        self.waypoint_entities = waypoint_entities
        self.simulation = simulation
        self.heightmap = heightmap
        self.route = route
        self.config = config
        self.ursina_config = ursina_config

    def _update_entities(self) -> None:
        current_waypoint_idx = int(self.route.current_idx)
        next_waypoint_idx = (current_waypoint_idx + 1) % len(self.waypoint_entities)
        for entity_idx, waypoint in enumerate(self.waypoint_entities):
            is_current = entity_idx == current_waypoint_idx
            is_next = entity_idx == next_waypoint_idx
            is_visited = bool(self.route.visited[entity_idx])
            waypoint.update_entity(is_current, is_next, is_visited)

        self.aircraft_entity.update_entity(self.simulation)

        self.text_ui.update_ui(self.simulation, self.heightmap)

        # Update waypoint UI - pass current waypoint entity
        current_waypoint_entity = None
        if 0 <= current_waypoint_idx < len(self.waypoint_entities):
            current_waypoint_entity = self.waypoint_entities[current_waypoint_idx]
        self.icon_ui.update_waypoint_ui(
            self.simulation.aircraft, self.route, current_waypoint_entity
        )

    def update(self) -> None:
        controls = self.input_handler.get_inputs()
        self.simulation = set_controls(self.simulation, controls)
        key = jax.random.PRNGKey(int(time.time() * 1000000) % 2**32)
        self.simulation, self.route = step(
            key, self.simulation, self.heightmap, self.route, self.config
        )
        self._update_entities()
