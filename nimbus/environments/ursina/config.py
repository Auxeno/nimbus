"""
Configuration for Ursina runtime.
"""

from dataclasses import dataclass


@dataclass
class UrsinaConfig:
    white = "#FFFFFF"
    light_gray = "#EEEEEE"

    color_1 = "#98C1D9"
    color_2 = "#2F3E9E"
    color_3 = "#293542"
    color_4 = "#1E2A38"

    alpha_30 = "55"
    alpha_50 = "88"
    alpha_75 = "CC"
    alpha_90 = "DD"

    aircraft_color: str = color_1
    aircraft_model_path: str = "vampire.glb"
    aircraft_model_scale: float = 1.0

    terrain_color: str = color_4
    contour_color: str = color_3
    contour_interval: float = 100.0
    contour_texture_resolution: int = 8192

    waypoint_color_current: str = color_1 + alpha_75
    waypoint_color_next: str = color_2 + alpha_75
    waypoint_color_unvisited: str = color_2 + alpha_30
    waypoint_color_visited: str = color_4 + alpha_30

    sky_top_color: str = color_4
    sky_bottom_color: str = color_2
    sky_gradient_midpoint: float = 0.8

    light_color: str = light_gray + alpha_50

    fog_color: str = white
    fog_density: float = 0.00004

    trail_color: str = color_1
    trail_width: float = 0.4
    trail_fade_duration: float = 6.0
    trail_min_spacing: float = 0.5
    trail_update_interval: float = 0.02
    trail_default_alpha: float = 0.7
    trail_segments: int = 150

    icon_color: str = white + alpha_90

    wingtip_offset: tuple[float, float, float] = (5.5, 0.0, 0.8)
