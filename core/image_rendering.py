from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from PLIFMix.algorithms.mixing import shift_image
from PLIFMix.algorithms.visualization import blacked_jet_colormap, blue_white_red_colormap
from PLIFMix.core.image_coordinates import quarter_turns


Ellipse = tuple[int, int] | tuple[int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class OverlayOptions:
    center: tuple[int, int] = (0, 0)
    ellipse: Ellipse | None = None
    crosshair: bool = False
    line_width: int = 1
    quad: np.ndarray | None = None
    special_points: object | None = None


@dataclass(frozen=True, slots=True)
class RawRenderOptions:
    gain: float
    auto_gain: bool
    rotate: float
    shift: int
    denoise: float
    pseudo: bool
    value_scale: float
    value_range: tuple[float, float] | None
    zero_threshold_8bit: float = 0
    overlays: OverlayOptions = field(default_factory=OverlayOptions)


@dataclass(frozen=True, slots=True)
class ComputedRenderOptions:
    gain: float = 1
    colormap: str = "jet"
    rotate: float = 0
    denoise: float = 0
    rotation_center: tuple[int, int] | None = None
    overlays: OverlayOptions = field(default_factory=OverlayOptions)
    retain_base_display: bool = True


@dataclass(frozen=True, slots=True)
class RenderResult:
    image: np.ndarray
    display: np.ndarray
    computed_gain: float | None = None
    value_range: tuple[float, float] | None = None
    base_display: np.ndarray | None = None
    special_points: object | None = None


def render_raw_image(source: np.ndarray, options: RawRenderOptions) -> RenderResult:
    image = source
    value_range = options.value_range
    if options.zero_threshold_8bit < 0:
        raise ValueError("去噪阈值不能小于0")
    if options.zero_threshold_8bit > 0:
        image = image.copy()
        image[image <= options.zero_threshold_8bit * 256 / options.value_scale] = 0
        value_range = None
    if options.denoise > 0:
        if image is source:
            image = image.copy()
        image[image < np.max(image) * options.denoise] = 0
        value_range = None
    if options.rotate:
        image = rotate_quarter_turns(image, options.rotate)
        value_range = None
    if options.shift:
        image = shift_image(image, options.shift)
        value_range = None

    computed_gain = None
    gain = options.gain
    if options.auto_gain:
        _, maximum = finite_range(image)
        computed_gain = 65535.0 / (maximum * options.value_scale) if maximum > 0 else None
        gain = computed_gain if computed_gain is not None else 0.0
    scaled = scale_gray_for_display(image, gain, options.value_scale)
    colormap = "jet" if options.pseudo else "none"
    display = gray_to_bgr(scaled, colormap)
    draw_overlays_on(display, options.overlays)
    return RenderResult(image=image, display=display, computed_gain=computed_gain, value_range=value_range)


def render_computed_image(source: np.ndarray, options: ComputedRenderOptions) -> RenderResult:
    image = source
    special_points = options.overlays.special_points
    if options.rotate:
        original_shape = image.shape[:2]
        image = rotate_quarter_turns(image, options.rotate)
        special_points = rotate_special_points_quarter_turns(
            special_points, options.rotate, original_shape
        )

    if image.ndim == 2:
        shown = scale_computed_gray_for_display(image, options.gain, options.denoise)
        display = gray_to_bgr(shown, options.colormap)
    else:
        display = np.clip(image, 0, 255).astype(np.uint8)

    base_display = display.copy() if options.retain_base_display else None
    overlays = OverlayOptions(
        center=options.overlays.center,
        ellipse=options.overlays.ellipse,
        crosshair=options.overlays.crosshair,
        line_width=options.overlays.line_width,
        quad=options.overlays.quad,
        special_points=special_points,
    )
    draw_overlays_on(display, overlays)
    return RenderResult(
        image=image,
        display=display,
        base_display=base_display,
        special_points=special_points,
    )


def gray_to_bgr(gray: np.ndarray, colormap: str = "jet") -> np.ndarray:
    if colormap == "bwr":
        return cv2.applyColorMap(gray, blue_white_red_colormap())
    if colormap == "none":
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(gray, blacked_jet_colormap())


def draw_overlays_on(
    display: np.ndarray,
    overlays: OverlayOptions,
) -> np.ndarray:
    """Draw overlays in place and return the same display array."""
    if overlays.special_points is not None and len(overlays.special_points):
        try:
            y, x = overlays.special_points
            display[y, x] = [255, 255, 255]
        except (IndexError, TypeError, ValueError):
            pass
    if overlays.crosshair:
        cv2.line(
            display,
            (overlays.center[0], 0),
            (overlays.center[0], display.shape[0]),
            (0, 0, 255),
            overlays.line_width,
        )
        cv2.line(
            display,
            (0, overlays.center[1]),
            (display.shape[1], overlays.center[1]),
            (0, 0, 255),
            overlays.line_width,
        )
    if overlays.ellipse:
        if len(overlays.ellipse) == 2:
            cv2.ellipse(
                display,
                overlays.center,
                overlays.ellipse,
                0,
                0,
                360,
                (0, 0, 255),
                overlays.line_width,
            )
        else:
            x, y, major, minor, ellipse_width = overlays.ellipse
            cv2.ellipse(display, (x, y), (major, minor), 0, 0, 360, (0, 0, 255), ellipse_width)
    if overlays.quad is not None:
        cv2.polylines(display, [overlays.quad], True, (0, 0, 255), overlays.line_width)
    return display


def copy_display_for_save(
    base_display: np.ndarray | None,
    include_crosshair: bool,
    include_ellipse: bool,
    crosshair: tuple[int, int, int] | None = None,
    ellipse: Ellipse | None = None,
    *,
    last_crosshair: tuple[int, int, int] | None = None,
    last_ellipse: Ellipse | None = None,
    last_special_points=None,
) -> np.ndarray | None:
    if base_display is None:
        return None
    image = base_display.copy()
    selected_crosshair = crosshair if include_crosshair else None
    selected_ellipse = ellipse if include_ellipse else None
    if selected_crosshair is None and include_crosshair:
        selected_crosshair = last_crosshair
    if selected_ellipse is None and include_ellipse:
        selected_ellipse = last_ellipse
    overlays = OverlayOptions(
        center=(selected_crosshair[0], selected_crosshair[1]) if selected_crosshair else (0, 0),
        ellipse=selected_ellipse,
        crosshair=selected_crosshair is not None,
        line_width=selected_crosshair[2] if selected_crosshair else 1,
        special_points=last_special_points,
    )
    return draw_overlays_on(image, overlays)


def rotate_quarter_turns(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    turns = quarter_turns(angle_degrees)
    if turns == 1:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if turns == 2:
        return cv2.rotate(image, cv2.ROTATE_180)
    if turns == 3:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def rotation_matrix_about_point(
    center: tuple[float, float],
    angle_degrees: float,
) -> np.ndarray:
    """Return the source-to-display matrix of a rotation about a point."""
    return cv2.getRotationMatrix2D((float(center[0]), float(center[1])), float(angle_degrees), 1.0)


def rotate_image_about_point(
    image: np.ndarray,
    center: tuple[float, float],
    angle_degrees: float,
) -> np.ndarray:
    """Rotate an image about a point, keeping its shape and zeroing the outside.

    The angle follows the OpenCV convention: a positive value turns the shown
    image counter-clockwise relative to the source. Shape is preserved, so the
    corners outside the rotated field are filled with zero instead of growing
    the canvas. A zero angle returns the input array unchanged.
    """
    if not angle_degrees:
        return image
    matrix = rotation_matrix_about_point(center, angle_degrees)
    return cv2.warpAffine(
        image,
        matrix,
        (int(image.shape[1]), int(image.shape[0])),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def rotate_points_about_point(
    points,
    center: tuple[float, float],
    angle_degrees: float,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Map (row, column) overlay points with the same matrix as the image.

    Out-of-frame results are dropped instead of clipped to the border, so an
    overlay never grows a false edge along the image boundary.
    """
    if points is None or len(points) != 2:
        empty = np.array([], dtype=np.int32)
        return empty, empty
    rows = np.asarray(points[0])
    columns = np.asarray(points[1])
    if rows.size == 0 or columns.size == 0:
        empty = np.array([], dtype=np.int32)
        return empty, empty
    if not angle_degrees:
        return rows.astype(np.int32, copy=False), columns.astype(np.int32, copy=False)

    matrix = rotation_matrix_about_point(center, angle_degrees)
    source_columns = columns.astype(np.float64, copy=False)
    source_rows = rows.astype(np.float64, copy=False)
    rotated_columns = matrix[0, 0] * source_columns + matrix[0, 1] * source_rows + matrix[0, 2]
    rotated_rows = matrix[1, 0] * source_columns + matrix[1, 1] * source_rows + matrix[1, 2]
    new_rows = np.rint(rotated_rows).astype(np.int32)
    new_columns = np.rint(rotated_columns).astype(np.int32)
    height, width = int(shape[0]), int(shape[1])
    valid = (
        (new_rows >= 0)
        & (new_rows < height)
        & (new_columns >= 0)
        & (new_columns < width)
    )
    return new_rows[valid], new_columns[valid]


def rotate_special_points_quarter_turns(special_points, angle_degrees: float, shape: tuple[int, int]):
    if special_points is None or len(special_points) != 2:
        return special_points
    turns = quarter_turns(angle_degrees)
    y_points = np.asarray(special_points[0])
    x_points = np.asarray(special_points[1])
    height, width = shape
    if turns == 1:
        rotated_x = height - 1 - y_points
        rotated_y = x_points
    elif turns == 2:
        rotated_x = width - 1 - x_points
        rotated_y = height - 1 - y_points
    elif turns == 3:
        rotated_x = y_points
        rotated_y = width - 1 - x_points
    else:
        return special_points
    return np.asarray(rotated_y), np.asarray(rotated_x)


def scale_gray_for_display(image: np.ndarray, gain: float, value_scale: float = 1.0) -> np.ndarray:
    scaled = image.astype(np.float32, copy=False) * np.float32(value_scale * gain / 256)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def scale_computed_gray_for_display(image: np.ndarray, gain: float, denoise: float) -> np.ndarray:
    shown = image.copy()
    if denoise > 0:
        shown[shown < np.max(shown) * denoise] = 0
    if gain == 0:
        maximum = float(np.nanmax(shown)) if shown.size else 0
        gain = 65535 / maximum if maximum else 1
    shown = shown * gain
    if shown.dtype != np.uint8:
        shown = shown / 256
    return np.clip(shown, 0, 255).astype(np.uint8)


def finite_range(image: np.ndarray) -> tuple[float, float]:
    if image.size == 0:
        return 0.0, 0.0
    return float(np.nanmin(image)), float(np.nanmax(image))


def quad_values(image: np.ndarray, quad):
    if quad is None:
        return image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [quad], 255)
    return image[mask > 0]
