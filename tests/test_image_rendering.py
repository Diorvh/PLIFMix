from __future__ import annotations

from dataclasses import replace
from math import isclose, isfinite
import unittest

import cv2
import numpy as np

from PLIFMix.algorithms.mixing import shift_image
from PLIFMix.algorithms.visualization import blacked_jet_colormap, blue_white_red_colormap
from PLIFMix.core.image_coordinates import (
    ImageCoordinateMapping as CoreImageCoordinateMapping,
    quarter_turns as core_quarter_turns,
)
from PLIFMix.core.image_rendering import (
    ComputedRenderOptions,
    OverlayOptions,
    RawRenderOptions,
    copy_display_for_save,
    draw_overlays_on,
    render_computed_image,
    render_raw_image,
)
from PLIFMix.ui.coordinate_mapping import (
    ImageCoordinateMapping as UiImageCoordinateMapping,
    quarter_turns as ui_quarter_turns,
)
from PLIFMix.ui.components.viewer_panel import _draw_overlays_on as viewer_draw_overlays_on


class ImageRenderingTests(unittest.TestCase):
    def test_ui_coordinate_module_reexports_core_implementation(self) -> None:
        self.assertIs(UiImageCoordinateMapping, CoreImageCoordinateMapping)
        self.assertIs(ui_quarter_turns, core_quarter_turns)

    def test_raw_renderer_matches_legacy_for_8bit_and_16bit_inputs(self) -> None:
        cases = [
            (
                np.array(
                    [[0, 2, 5, 10], [20, 40, 80, 120], [3, 7, 11, 15]],
                    dtype=np.uint8,
                ),
                RawRenderOptions(
                    gain=1.75,
                    auto_gain=False,
                    rotate=90,
                    shift=1,
                    denoise=0.2,
                    pseudo=True,
                    value_scale=256.0,
                    value_range=(0.0, 120.0),
                    zero_threshold_8bit=2.0,
                    overlays=OverlayOptions(center=(1, 1), crosshair=True, line_width=1),
                ),
            ),
            (
                np.array(
                    [[0, 256, 1024], [2048, 4096, 8192], [16384, 32768, 65535]],
                    dtype=np.uint16,
                ),
                RawRenderOptions(
                    gain=1.0,
                    auto_gain=True,
                    rotate=270,
                    shift=-1,
                    denoise=0.1,
                    pseudo=False,
                    value_scale=1.0,
                    value_range=(0.0, 65535.0),
                    zero_threshold_8bit=1.0,
                    overlays=OverlayOptions(
                        center=(1, 1),
                        ellipse=(1, 1),
                        crosshair=False,
                        line_width=2,
                    ),
                ),
            ),
        ]

        for source, options in cases:
            with self.subTest(dtype=source.dtype, rotate=options.rotate, shift=options.shift):
                expected = _legacy_raw_render(source, options)
                actual = render_raw_image(source, options)
                _assert_render_result_equal(self, actual, expected)

    def test_computed_gray_renderer_matches_legacy_dtype_and_colormap_semantics(self) -> None:
        sources = [
            np.array(
                [[0.0, 64.0, 128.0], [256.0, 1024.0, 4096.0]],
                dtype=np.float64,
            ),
            np.array([[0, 32, 128], [200, 240, 255]], dtype=np.uint8),
        ]
        for source in sources:
            for colormap in ("jet", "bwr", "none"):
                options = ComputedRenderOptions(
                    gain=0 if source.dtype != np.uint8 else 1,
                    colormap=colormap,
                    rotate=180 if source.dtype != np.uint8 else 0,
                    denoise=0.1 if source.dtype != np.uint8 else 0,
                    rotation_center=(99, 101),
                    overlays=OverlayOptions(
                        center=(1, 0),
                        ellipse=(1, 0, 1, 1, 1),
                        crosshair=True,
                        line_width=1,
                        special_points=(np.array([0, 1]), np.array([0, 2])),
                    ),
                    retain_base_display=True,
                )
                with self.subTest(dtype=source.dtype, colormap=colormap):
                    expected = _legacy_computed_render(source, options)
                    actual = render_computed_image(source, options)
                    _assert_render_result_equal(self, actual, expected)

    def test_computed_color_renderer_preserves_channel_order_and_rotation(self) -> None:
        source = np.array(
            [
                [[-5, 10, 20], [30, 40, 260], [70, 80, 90]],
                [[100, 110, 120], [130, 140, 150], [160, 170, 180]],
            ],
            dtype=np.int16,
        )
        options = ComputedRenderOptions(
            gain=7,
            colormap="bwr",
            rotate=90,
            denoise=0.9,
            overlays=OverlayOptions(
                special_points=(np.array([0, 1]), np.array([0, 2])),
            ),
            retain_base_display=False,
        )

        expected = _legacy_computed_render(source, options)
        actual = render_computed_image(source, options)
        _assert_render_result_equal(self, actual, expected)
        self.assertEqual(actual.display.shape, (3, 2, 3))

    def test_overlay_and_save_helpers_match_legacy_and_keep_base_unchanged(self) -> None:
        base = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
        overlays = OverlayOptions(
            center=(1, 0),
            ellipse=(1, 0, 1, 1, 1),
            crosshair=True,
            line_width=1,
            quad=np.array([[0, 0], [3, 0], [3, 1]], dtype=np.int32),
            special_points=(np.array([1]), np.array([2])),
        )
        expected_draw = base.copy()
        _legacy_draw_overlays(expected_draw, overlays)
        actual_draw = base.copy()
        returned = draw_overlays_on(actual_draw, overlays)
        np.testing.assert_array_equal(actual_draw, expected_draw)
        self.assertIs(returned, actual_draw)

        compatibility_draw = base.copy()
        viewer_draw_overlays_on(
            compatibility_draw,
            overlays.center,
            overlays.ellipse,
            overlays.crosshair,
            overlays.line_width,
            overlays.quad,
            overlays.special_points,
        )
        np.testing.assert_array_equal(compatibility_draw, expected_draw)

        expected_saved = _legacy_copy_for_save(
            base,
            include_crosshair=True,
            include_ellipse=False,
            crosshair=None,
            ellipse=None,
            last_crosshair=(2, 1, 1),
            last_ellipse=(1, 1, 1, 1, 1),
            last_special_points=overlays.special_points,
        )
        actual_saved = copy_display_for_save(
            base,
            True,
            False,
            last_crosshair=(2, 1, 1),
            last_ellipse=(1, 1, 1, 1, 1),
            last_special_points=overlays.special_points,
        )
        np.testing.assert_array_equal(actual_saved, expected_saved)
        np.testing.assert_array_equal(base, np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3))


def _legacy_raw_render(source: np.ndarray, options: RawRenderOptions):
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
        image = _legacy_rotate_quarter_turns(image, options.rotate)
        value_range = None
    if options.shift:
        image = shift_image(image, options.shift)
        value_range = None

    computed_gain = None
    gain = options.gain
    if options.auto_gain:
        _, maximum = _legacy_finite_range(image)
        computed_gain = 65535.0 / (maximum * options.value_scale) if maximum > 0 else None
        gain = computed_gain if computed_gain is not None else 0.0
    scaled = _legacy_scale_gray_for_display(image, gain, options.value_scale)
    display = (
        cv2.applyColorMap(scaled, blacked_jet_colormap())
        if options.pseudo
        else cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
    )
    _legacy_draw_overlays(display, options.overlays)
    return image, display, computed_gain, value_range, None, None


def _legacy_computed_render(source: np.ndarray, options: ComputedRenderOptions):
    image = source
    special_points = options.overlays.special_points
    if options.rotate:
        original_shape = image.shape[:2]
        image = _legacy_rotate_quarter_turns(image, options.rotate)
        special_points = _legacy_rotate_special_points(special_points, options.rotate, original_shape)

    if image.ndim == 2:
        shown = image.copy()
        if options.denoise > 0:
            shown[shown < np.max(shown) * options.denoise] = 0
        gain = options.gain
        if gain == 0:
            maximum = float(np.nanmax(shown)) if shown.size else 0
            gain = 65535 / maximum if maximum else 1
        shown = shown * gain
        if shown.dtype != np.uint8:
            shown = shown / 256
        shown = np.clip(shown, 0, 255).astype(np.uint8)
        if options.colormap == "bwr":
            display = cv2.applyColorMap(shown, blue_white_red_colormap())
        elif options.colormap == "none":
            display = cv2.cvtColor(shown, cv2.COLOR_GRAY2BGR)
        else:
            display = cv2.applyColorMap(shown, blacked_jet_colormap())
    else:
        display = np.clip(image, 0, 255).astype(np.uint8)

    base_display = display.copy() if options.retain_base_display else None
    overlays = replace(options.overlays, special_points=special_points)
    _legacy_draw_overlays(display, overlays)
    return image, display, None, None, base_display, special_points


def _legacy_rotate_quarter_turns(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    angle = float(angle_degrees)
    if not isfinite(angle):
        raise ValueError("旋转角度必须是有限数值")
    turns = int(round(angle / 90))
    if not isclose(angle, turns * 90, rel_tol=1e-5, abs_tol=1e-9):
        raise ValueError("目前只支持90°的整数倍旋转")
    turns %= 4
    if turns == 1:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if turns == 2:
        return cv2.rotate(image, cv2.ROTATE_180)
    if turns == 3:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _legacy_rotate_special_points(special_points, angle_degrees: float, shape: tuple[int, int]):
    if special_points is None or len(special_points) != 2:
        return special_points
    angle = float(angle_degrees)
    turns = int(round(angle / 90)) % 4
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


def _legacy_scale_gray_for_display(image: np.ndarray, gain: float, value_scale: float = 1.0) -> np.ndarray:
    scaled = image.astype(np.float32, copy=False) * np.float32(value_scale * gain / 256)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _legacy_finite_range(image: np.ndarray) -> tuple[float, float]:
    if image.size == 0:
        return 0.0, 0.0
    return float(np.nanmin(image)), float(np.nanmax(image))


def _legacy_draw_overlays(display: np.ndarray, overlays: OverlayOptions) -> None:
    special_points = overlays.special_points
    if special_points is not None and len(special_points):
        try:
            y, x = special_points
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
            cv2.ellipse(
                display,
                (x, y),
                (major, minor),
                0,
                0,
                360,
                (0, 0, 255),
                ellipse_width,
            )
    if overlays.quad is not None:
        cv2.polylines(display, [overlays.quad], True, (0, 0, 255), overlays.line_width)


def _legacy_copy_for_save(
    base_display: np.ndarray | None,
    include_crosshair: bool,
    include_ellipse: bool,
    crosshair,
    ellipse,
    *,
    last_crosshair,
    last_ellipse,
    last_special_points,
):
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
    _legacy_draw_overlays(image, overlays)
    return image


def _assert_render_result_equal(test_case: unittest.TestCase, actual, expected) -> None:
    np.testing.assert_array_equal(actual.image, expected[0])
    np.testing.assert_array_equal(actual.display, expected[1])
    test_case.assertEqual(actual.computed_gain, expected[2])
    test_case.assertEqual(actual.value_range, expected[3])
    if expected[4] is None:
        test_case.assertIsNone(actual.base_display)
    else:
        np.testing.assert_array_equal(actual.base_display, expected[4])
    if expected[5] is None:
        test_case.assertIsNone(actual.special_points)
    else:
        for actual_points, expected_points in zip(actual.special_points, expected[5]):
            np.testing.assert_array_equal(actual_points, expected_points)


if __name__ == "__main__":
    unittest.main()
