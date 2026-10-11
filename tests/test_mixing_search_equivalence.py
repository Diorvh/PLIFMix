from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from PLIFMix.algorithms.mixing import calc_result, generate_mask_coords, shift_image
from PLIFMix.core.experiment_images import ExperimentImageTriplet
from PLIFMix.domain.settings import ManualMixingSettings, ShiftSearchSettings
from PLIFMix.services import mixing_service as mixing_service_module
from PLIFMix.services.mixing_service import MixingService


class _ArrayTripletCache:
    def __init__(self, images: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        self.images = images

    def load(self, folder, f_exp_no, o_exp_no, fo_exp_no) -> ExperimentImageTriplet:
        return ExperimentImageTriplet(*(image.copy() for image in self.images))


def _make_images() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_values, x_values = np.indices((9, 13), dtype=np.float64)
    f_img = 2.0 + 80.0 * np.exp(-((x_values - 4.0) ** 2 + (y_values - 4.0) ** 2) / 8.0)
    o_img = 3.0 + 65.0 * np.exp(-((x_values - 8.0) ** 2 + (y_values - 4.5) ** 2) / 7.0)
    fo_img = 1.1 * f_img + 0.9 * o_img + ((x_values + 2.0 * y_values) % 4.0) * 0.25
    return f_img, o_img, fo_img


def _manual_settings(use_ellipse_region: bool) -> ManualMixingSettings:
    return ManualMixingSettings(
        image_folder=Path("manual-images"),
        f_exp_no="1",
        o_exp_no="2",
        fo_exp_no="3",
        f_shift=1,
        o_shift=-1,
        center_xfo=6,
        center_yfo=4,
        major_axis=4,
        minor_axis=3,
        global_of=1.35,
        denoise_ratio=0.35,
        use_ellipse_region=use_ellipse_region,
    )


def _search_settings(process_count: int = 1) -> ShiftSearchSettings:
    return ShiftSearchSettings(
        image_folder=Path("search-images"),
        f_exp_no="unused-f",
        o_exp_no="unused-o",
        fo_exp_no="unused-fo",
        f_min=-1,
        f_max=2,
        o_min=-2,
        o_max=1,
        process_count=process_count,
    )


def _legacy_manual_outputs(
    images: tuple[np.ndarray, np.ndarray, np.ndarray],
    settings: ManualMixingSettings,
):
    f_img, o_img, fo_img = images
    f_img = shift_image(f_img, settings.f_shift)
    o_img = shift_image(o_img, settings.o_shift)
    fo_img = fo_img.copy()

    mask = (f_img < (np.max(f_img) * settings.denoise_ratio)) & (
        o_img < (np.max(o_img) * settings.denoise_ratio)
    )
    f_img[mask] = 0
    o_img[mask] = 0
    fo_img[mask] = 0

    raw_f_img = f_img.copy()
    raw_o_img = o_img.copy()
    raw_fo_img = fo_img.copy()

    if settings.use_ellipse_region:
        coords = generate_mask_coords(
            settings.center_xfo,
            settings.center_yfo,
            settings.major_axis,
            settings.minor_axis,
            f_img.shape,
        )
    else:
        ys, xs = np.indices(f_img.shape)
        coords = xs.ravel(), ys.ravel()
    global_mix_ratio = settings.global_of / (1 + settings.global_of)
    _, raw_me_o_fo, raw_me_f_fo, raw_me_o_f = calc_result(
        raw_f_img,
        raw_o_img,
        raw_fo_img,
        coords,
        global_mix_ratio,
    )

    record_of = float(np.sum(o_img) / np.sum(f_img)) if np.sum(f_img) else 0.0
    if record_of:
        o_img = o_img * settings.global_of / record_of
    combined = f_img + o_img
    fo_sum = np.sum(fo_img)
    if fo_sum:
        fo_img = fo_img / fo_sum * np.sum(combined)

    abs_value, me_o_fo, me_f_fo, me_o_f = calc_result(f_img, o_img, fo_img, coords, global_mix_ratio)
    display_images, mix_curve = mixing_service_module._build_display_outputs(
        f_img,
        o_img,
        fo_img,
        settings.global_of,
        settings.center_yfo,
        global_mix_ratio,
    )
    raw_display_images, raw_mix_curve = mixing_service_module._build_display_outputs(
        raw_f_img,
        raw_o_img,
        raw_fo_img,
        settings.global_of,
        settings.center_yfo,
        global_mix_ratio,
    )
    summary = (
        abs_value,
        me_o_fo,
        me_f_fo,
        me_o_f,
        (me_o_fo + me_f_fo + me_o_f) / 3,
        record_of,
        raw_me_o_fo,
        raw_me_f_fo,
        raw_me_o_f,
        (raw_me_o_fo + raw_me_f_fo + raw_me_o_f) / 3,
    )
    return summary, display_images, mix_curve, raw_display_images, raw_mix_curve


class MixingSearchEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.images = _make_images()
        self.service = MixingService(_ArrayTripletCache(self.images))

    def test_manual_display_matches_head_manual_baseline(self) -> None:
        for use_ellipse_region in (True, False):
            with self.subTest(use_ellipse_region=use_ellipse_region):
                settings = _manual_settings(use_ellipse_region)
                expected = _legacy_manual_outputs(self.images, settings)
                actual = self.service.run_manual_display(settings)

                actual_summary = tuple(vars(actual.summary).values())
                np.testing.assert_array_equal(actual_summary, expected[0])
                for actual_image, expected_image in zip(actual.images, expected[1]):
                    self.assertEqual(actual_image.name, expected_image.name)
                    self.assertEqual(actual_image.colormap, expected_image.colormap)
                    np.testing.assert_array_equal(actual_image.image, expected_image.image)
                for actual_curve, expected_curve in zip(actual.mix_curve, expected[2]):
                    np.testing.assert_array_equal(actual_curve, expected_curve)
                for actual_image, expected_image in zip(actual.raw_images, expected[3]):
                    self.assertEqual(actual_image.name, expected_image.name)
                    self.assertEqual(actual_image.colormap, expected_image.colormap)
                    np.testing.assert_array_equal(actual_image.image, expected_image.image)
                for actual_curve, expected_curve in zip(actual.raw_mix_curve, expected[4]):
                    np.testing.assert_array_equal(actual_curve, expected_curve)

    def test_each_search_cell_matches_calibrated_manual_metrics_for_both_rois(self) -> None:
        for use_ellipse_region in (True, False):
            with self.subTest(use_ellipse_region=use_ellipse_region):
                manual = _manual_settings(use_ellipse_region)
                search_result = self.service.run_manual_shift_search(manual, _search_settings())

                self.assertEqual(search_result.f_shifts, [-1, 0, 1, 2])
                self.assertEqual(search_result.o_shifts, [-2, -1, 0, 1])
                for row, f_shift in enumerate(search_result.f_shifts):
                    for column, o_shift in enumerate(search_result.o_shifts):
                        candidate = replace(manual, f_shift=f_shift, o_shift=o_shift)
                        summary = self.service.run_manual_display(candidate).summary
                        expected = np.array(
                            [summary.abs_value, summary.me_o_fo, summary.me_f_fo, summary.me_o_f],
                            dtype=np.float64,
                        )
                        actual = np.array(
                            [
                                search_result.abs_matrix[row, column],
                                search_result.me1_matrix[row, column],
                                search_result.me2_matrix[row, column],
                                search_result.me3_matrix[row, column],
                            ],
                            dtype=np.float64,
                        )
                        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

                self.assertEqual(search_result.abs_matrix.shape, (4, 4))
                self.assertEqual(search_result.me_ptp_matrix.shape, (4, 4))
                self.assertEqual(search_result.me1_matrix.shape, (4, 4))
                self.assertEqual(search_result.me2_matrix.shape, (4, 4))
                self.assertEqual(search_result.me3_matrix.shape, (4, 4))

    def test_manual_search_parallel_matches_single_and_skips_display_builder(self) -> None:
        manual = _manual_settings(True)
        single = self.service.run_manual_shift_search(manual, _search_settings(process_count=1))
        with patch.object(
            mixing_service_module,
            "_build_display_outputs",
            side_effect=AssertionError("search must not build display outputs"),
        ):
            parallel = self.service.run_manual_shift_search(manual, _search_settings(process_count=2))

        for field in ("abs_matrix", "me_ptp_matrix", "me1_matrix", "me2_matrix", "me3_matrix"):
            np.testing.assert_array_equal(getattr(parallel, field), getattr(single, field))


if __name__ == "__main__":
    unittest.main()
