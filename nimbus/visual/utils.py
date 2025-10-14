"""
Utility module for Ursina-specific transforms and asset handling.
"""

from contextlib import contextmanager
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw
from skimage import measure, transform
from ursina import Color, Entity, Vec3, application
from ursina.color import rgba

from ..core import physics, quaternion, spatial
from ..core.config import AircraftConfig, MapConfig, PhysicsConfig
from ..core.interface import calculate_g_force
from ..core.primitives import (
    FLOAT_DTYPE,
    FloatScalar,
    Matrix,
    Quaternion,
    Vector,
    norm_3,
)
from ..core.state import Aircraft, Wind


def ned_to_eun(position_ned: Vector) -> Vector:
    """Convert from NED (North-East-Down) to EUN (East-Up-North)."""
    n, e, d = position_ned
    return jnp.array([e, -d, n], dtype=FLOAT_DTYPE)


def ned_quat_to_eun_quat(q_ned: Quaternion) -> Quaternion:
    """Convert a quaternion from NED (simulation) to EUN (Ursina)."""
    q_transform = jnp.array([0.5, -0.5, 0.5, 0.5], dtype=FLOAT_DTYPE)
    return quaternion.multiply(
        quaternion.multiply(q_transform, q_ned), quaternion.conjugate(q_transform)
    )


def set_alpha(entity: Entity, alpha: float) -> None:
    """Sets the alpha channel of an entity's colour."""
    entity.color[3] = alpha


def convert_scale(scale: float) -> Vec3:
    """Convert a scale float to an Ursina Vec3 to fix IDE type errors."""
    return Vec3(scale, scale, scale)


@contextmanager
def local_asset_folder():
    """
    Context-manager that temporarily sets `application.asset_folder`
    (where Ursina looks for models/textures).

    Usage:
    ------
        with local_asset_folder():
            Entity(model="my_model.glb")
    """
    local_folder = Path(__file__).resolve().parent.parent.parent / "assets"
    previous = application.asset_folder
    application.asset_folder = local_folder
    try:
        yield
    finally:
        application.asset_folder = previous


def hex_to_rgb(hex_color: str, as_tuple: bool = False) -> tuple | np.ndarray:
    """
    Convert '#RRGGBB' or '#RGB' (or without '#') to RGB values.

    Parameters
    ----------
    hex_color : str
        Hex color string
    as_tuple : bool
        If True, return (R,G,B) tuple of ints [0,255].
        If False, return numpy array of float32 values.

    Returns
    -------
    tuple or np.ndarray
        RGB values either as tuple of ints or numpy array
    """
    s = hex_color.lstrip("#")
    if len(s) == 3:  # expand short form, e.g. "0f8" -> "00ff88"
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        raise ValueError(f"Invalid hex color: {hex_color!r}")
    rgb = tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))
    if as_tuple:
        return rgb
    return np.array(rgb, dtype=np.float32)


def hex_to_rgba(hex_color: str) -> Color:
    """
    Convert hex color string to ursina.color.rgba with normalized values.

    Parameters
    ----------
    hex_color : str
        Hex color string in format '#RRGGBB' or '#RRGGBBAA' (with or without '#')
        If 6 characters, alpha defaults to 1.0 (fully opaque)
        If 8 characters, alpha is extracted from the last 2 characters

    Returns
    -------
    ursina.color.rgba
        RGBA color with values normalized between 0 and 1
    """
    s = hex_color.lstrip("#")

    # Handle 6-character hex (RGB)
    if len(s) == 6:
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
        a = 1.0
    # Handle 8-character hex (RGBA)
    elif len(s) == 8:
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
        a = int(s[6:8], 16) / 255.0
    else:
        raise ValueError(f"Invalid hex color: {hex_color!r}. Expected 6 or 8 characters.")

    return rgba(r, g, b, a)


def generate_gradient_image(
    width: int,
    height: int,
    top_color: str,
    bottom_color: str,
    mid_pos: float = 0.5,
) -> Image.Image:
    """
    Generate a vertical RGBA gradient image from top_color to bottom_color.
    `mid_pos` in (0,1) is where the 50/50 blend occurs (0=top, 1=bottom).
    Returns a PIL.Image compatible with `Texture(img)` in Ursina.
    """
    # Parse colors
    top_rgb = hex_to_rgb(top_color, as_tuple=False)
    bot_rgb = hex_to_rgb(bottom_color, as_tuple=False)

    # Clamp mid_pos to avoid division by zero
    mid_pos = float(np.clip(mid_pos, 1e-6, 1 - 1e-6))

    # Normalized vertical coordinate [0,1] for each row
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)

    # Piecewise-linear remap so that f(mid_pos) = 0.5
    t = np.empty_like(y)
    top_mask = y <= mid_pos
    bot_mask = ~top_mask
    t[top_mask] = 0.5 * (y[top_mask] / mid_pos)
    t[bot_mask] = 0.5 + 0.5 * ((y[bot_mask] - mid_pos) / (1.0 - mid_pos))

    # Interpolate per row: color = (1 - t) * top + t * bottom
    row_colors = (1.0 - t[:, None]) * top_rgb[None, :] + t[:, None] * bot_rgb[None, :]  # type: ignore
    row_colors = np.clip(row_colors, 0, 255).astype(np.uint8)  # (H, 3)

    # Expand to full image width and add full alpha channel
    img_rgb = np.repeat(row_colors[:, None, :], width, axis=1)  # (H, W, 3)
    alpha = np.full((height, width, 1), 255, dtype=np.uint8)  # (H, W, 1)
    img_rgba = np.concatenate([img_rgb, alpha], axis=-1)  # (H, W, 4)

    return Image.fromarray(img_rgba, mode="RGBA")


def generate_terrain_contour_image(
    heightmap: np.ndarray,
    resolution: int = 8192,
    contour_interval: float = 100.0,
    terrain_height: float = 5000.0,
    terrain_color: str = "#1E2A38",
    contour_color: str = "#000000",
    contour_width: int = 1,
) -> Image.Image:
    """
    Generate a terrain texture with contour lines from a heightmap.

    Parameters
    ----------
    heightmap : np.ndarray
        2D array of normalized height values in range [0, 1]
    resolution : int
        Output image resolution (square image)
    contour_interval : float
        Height interval between contour lines in meters
    terrain_height : float
        Maximum terrain height in meters (for scaling heightmap values)
    terrain_color : str
        Base color for terrain (hex format)
    contour_color : str
        Color for contour lines (hex format)
    contour_width : int
        Width of contour lines in pixels

    Returns
    -------
    Image.Image
        PIL Image with terrain and contour lines, compatible with Ursina Texture
    """

    terrain_rgb = hex_to_rgb(terrain_color, as_tuple=True)
    contour_rgb = hex_to_rgb(contour_color, as_tuple=True)

    # Resize heightmap to target resolution
    h, w = heightmap.shape
    size = min(h, w)
    heightmap = heightmap[:size, :size]
    # Use skimage.transform.resize instead of scipy.ndimage.zoom
    heightmap_resized = transform.resize(
        heightmap, (resolution, resolution), order=1, anti_aliasing=False
    )
    heightmap_resized = heightmap_resized[:resolution, :resolution]

    # Convert to elevation in meters
    # Match the terrain mesh transformation: [0,1] -> [-0.5, 0.5] * 2 = [-1, 1] then scaled
    # This matches: height_map = 2 * (heightmap.T - 0.5)
    elevation_map = 2 * (heightmap_resized - 0.5) * terrain_height

    # Create base terrain image with shading
    brightness_variation = 0.3
    min_brightness = 1.0 - brightness_variation / 2
    max_brightness = 1.0 + brightness_variation / 2
    brightness = min_brightness + heightmap_resized * (max_brightness - min_brightness)

    img_array = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    for i in range(3):
        img_array[:, :, i] = np.clip(terrain_rgb[i] * brightness, 0, 255).astype(np.uint8)

    # Create PIL image
    img = Image.fromarray(img_array, mode="RGB")
    draw = ImageDraw.Draw(img)

    # Calculate contour levels
    min_elevation = np.min(elevation_map)
    max_elevation = np.max(elevation_map)
    min_contour = np.ceil(min_elevation / contour_interval) * contour_interval
    max_contour = np.floor(max_elevation / contour_interval) * contour_interval
    contour_levels = np.arange(min_contour, max_contour + contour_interval, contour_interval)

    # Generate contours using skimage's marching squares
    for level in contour_levels:
        contours = measure.find_contours(elevation_map, level)

        for contour in contours:
            # Convert contour to pixel coordinates and draw
            points = [(int(x), int(y)) for y, x in contour]

            if len(points) > 1:
                # Draw the contour line
                for i in range(len(points) - 1):
                    draw.line(
                        [points[i], points[i + 1]],
                        fill=contour_rgb,  # type: ignore
                        width=contour_width,
                    )

    img = img.convert("RGBA")

    return img


def generate_blank_texture(
    resolution: int = 128,
    color: str = "#1E2A38",
) -> Image.Image:
    """Generate a blank texture with a specified color."""
    rgb = hex_to_rgb(color, as_tuple=True)
    # Create RGBA image with full alpha
    img_array = np.zeros((resolution, resolution, 4), dtype=np.uint8)
    img_array[:, :, 0] = rgb[0]
    img_array[:, :, 1] = rgb[1]
    img_array[:, :, 2] = rgb[2]
    img_array[:, :, 3] = 255  # Full alpha

    return Image.fromarray(img_array, mode="RGBA")


def compute_flight_metrics(
    aircraft: Aircraft,
    wind: Wind,
    heightmap: Matrix,
    aircraft_config: AircraftConfig,
    physics_config: PhysicsConfig,
    map_config: MapConfig,
) -> tuple[
    FloatScalar,
    FloatScalar,
    FloatScalar,
    FloatScalar,
    FloatScalar,
    FloatScalar,
    FloatScalar,
    FloatScalar,
    FloatScalar,
]:
    # Mach number
    vel = aircraft.body.velocity
    speed = norm_3(vel)

    # Angle of attack
    alpha = jnp.rad2deg(
        physics.calculate_angle_of_attack(aircraft.body.velocity, aircraft.body.orientation)
    )

    # Angle of sideslip (beta)
    beta = jnp.rad2deg(
        physics.calculate_angle_of_sideslip(aircraft.body.velocity, aircraft.body.orientation)
    )

    # G-force
    g = -calculate_g_force(aircraft, wind, aircraft_config, physics_config)[2]

    # Heading
    yaw, pitch, roll = quaternion.to_euler_zyx(aircraft.body.orientation)
    roll_deg = jnp.rad2deg(roll)
    pitch_deg = jnp.rad2deg(pitch)
    yaw_deg = jnp.mod(jnp.rad2deg(yaw) + 180.0, 360.0) - 180.0

    # AGL (height above ground level)
    agl = spatial.calculate_height_diff(
        heightmap,
        aircraft.body.position,
        jnp.array(map_config.size, dtype=FLOAT_DTYPE),
        jnp.array(map_config.terrain_height, dtype=FLOAT_DTYPE),
        use_bilinear=jnp.array(True, dtype=bool),
    )

    # VS (vertical speed) - negative of down component since positive means climbing
    vs = -aircraft.body.velocity[2]

    return alpha, beta, speed, g, agl, vs, roll_deg, pitch_deg, yaw_deg
