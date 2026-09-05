from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from PLIFv2.algorithms.mixing import shift_image
from PLIFv2.ui.coordinate_mapping import ImageCoordinateMapping
from PLIFv2.ui.pages.image_player_page import ImagePlayerPage


class _Value:
    def __init__(self, value) -> None:
        self.value = value

    def get(self):
        return self.value

    def set(self, value) -> None:
        self.value = value


class _PixelPlot:
    def __init__(self) -> None:
        self.series_args = None
        self.errors: list[str] = []

    def set_loading(self, _x: int, _y: int) -> None:
        pass

    def set_series(self, *args) -> None:
        self.series_args = args

    def set_error(self, message: str) -> None:
        self.errors.append(message)


class PlayerCoordinateMappingTests(unittest.TestCase):
    def test_non_square_quarter_turns_match_display_pixels(self) -> None:
        source = np.arange(6, dtype=np.int64).reshape(2, 3)

        for angle, turns in ((90, -1), (180, 2), (270, 1)):
            mapping = ImageCoordinateMapping.from_parameters(source.shape, angle)
            displayed = np.rot90(source, turns)
            self.assertEqual(mapping.display_shape, displayed.shape)
            for display_y in range(displayed.shape[0]):
                for display_x in range(displayed.shape[1]):
                    source_point = mapping.display_to_source(display_x, display_y)
                    self.assertIsNotNone(source_point)
                    source_x, source_y = source_point
                    self.assertEqual(displayed[display_y, display_x], source[source_y, source_x])
                    self.assertEqual(
                        mapping.source_to_display(source_x, source_y),
                        (display_x, display_y),
                    )

    def test_positive_and_negative_shift_match_viewer_padding(self) -> None:
        source = np.arange(6, dtype=np.int64).reshape(2, 3)

        for shift in (1, -1):
            mapping = ImageCoordinateMapping.from_parameters(source.shape, 90, shift)
            rotated = np.rot90(source, -1)
            displayed = shift_image(rotated, shift)
            for display_y in range(displayed.shape[0]):
                for display_x in range(displayed.shape[1]):
                    source_point = mapping.display_to_source(display_x, display_y)
                    expected = 0 if source_point is None else source[source_point[1], source_point[0]]
                    self.assertEqual(displayed[display_y, display_x], expected)

            padding_x = displayed.shape[1] - 1 if shift > 0 else 0
            self.assertIsNone(mapping.display_to_source(padding_x, 0))

    def test_player_series_maps_display_point_and_preserves_signal_values(self) -> None:
        frames = [
            np.array([[1, 4], [2, 8]], dtype=np.uint8),
            np.array([[256, 1024], [512, 2048]], dtype=np.uint16),
        ]
        mapping = ImageCoordinateMapping.from_parameters(frames[0].shape, 0, 0)
        page = _make_player(frames, mapping, x=1, y=0, zero_threshold=1, relative_denoise=0.5)
        page.gain = _Value("1000")

        page._request_manual_pixel_series()

        self.assertEqual(page.pixel_plot.errors, [])
        self.assertIsNotNone(page.pixel_plot.series_args)
        values = page.pixel_plot.series_args[2]
        np.testing.assert_array_equal(values, np.array([1024.0, 1024.0]))

    def test_player_series_returns_zero_for_shift_padding(self) -> None:
        frames = [
            np.full((2, 3), 7, dtype=np.uint8),
            np.full((2, 3), 1024, dtype=np.uint16),
        ]
        for shift, x in ((1, 2), (-1, 0)):
            mapping = ImageCoordinateMapping.from_parameters(frames[0].shape, 0, shift)
            page = _make_player(frames, mapping, x=x, y=0)

            page._request_manual_pixel_series()

            self.assertEqual(page.pixel_plot.errors, [])
            self.assertIsNotNone(page.pixel_plot.series_args)
            np.testing.assert_array_equal(page.pixel_plot.series_args[2], np.zeros(2))


def _make_player(
    frames: list[np.ndarray],
    mapping: ImageCoordinateMapping,
    *,
    x: int,
    y: int,
    zero_threshold: float = 0,
    relative_denoise: float = 0,
) -> ImagePlayerPage:
    page = ImagePlayerPage.__new__(ImagePlayerPage)
    page.files = [Path(f"{index}.npy") for index in range(len(frames))]
    page._frame_cache = frames
    page._cache_ranges = [(float(np.min(frame)), float(np.max(frame))) for frame in frames]
    page._plot_index_set = set(range(len(frames)))
    page.pixel_x = _Value(str(x))
    page.pixel_y = _Value(str(y))
    page.denoise_threshold = _Value(str(zero_threshold))
    page.relative_denoise = _Value(str(relative_denoise))
    page.pixel_top_k = _Value("10")
    page.pixel_plot = _PixelPlot()
    page.panel = SimpleNamespace(coordinate_mapping=mapping)
    return page


if __name__ == "__main__":
    unittest.main()
