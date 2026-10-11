from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from PLIFMix.algorithms.mixing import extrapolate_fitted_center_line
from PLIFMix.core.experiment_images import ExperimentImageTriplet
from PLIFMix.core.image_rendering import rotate_image_about_point, rotate_points_about_point
from PLIFMix.domain.settings import DistributionSettings
from PLIFMix.services import distribution_service as distribution_service_module
from PLIFMix.services.distribution_presentation import synthesize_distribution_image
from PLIFMix.services.distribution_service import DistributionFitService


_EMPTY_POINTS = (np.array([], dtype=np.int32), np.array([], dtype=np.int32))


class RotationGeometryTests(unittest.TestCase):
    """The rotation view must land on the same frame the centre line was fitted in."""

    def test_rotation_straightens_a_tilted_ridge_line(self) -> None:
        shape = (140, 120)
        center_x, center_y = 60, 70
        angle = 25.0
        # a -> 0 keeps the fitted centre line straight, so the ridge tilts by `angle`.
        rows, columns = extrapolate_fitted_center_line(
            shape, 1e-9, 20.0, center_x, center_y, angle
        )
        self.assertGreater(rows.size, 10)

        rotated_rows, rotated_columns = rotate_points_about_point(
            (rows, columns), (center_x, center_y), -angle, shape
        )

        self.assertGreater(rotated_rows.size, 10)
        self.assertLess(float(np.max(np.abs(rotated_columns - center_x))), 3.0)
        self.assertGreater(float(np.ptp(rotated_rows)), 10.0)

    def test_rotation_is_not_applied_in_the_wrong_direction(self) -> None:
        shape = (140, 120)
        center_x, center_y = 60, 70
        angle = 25.0
        rows, columns = extrapolate_fitted_center_line(
            shape, 1e-9, 20.0, center_x, center_y, angle
        )

        wrong_rows, wrong_columns = rotate_points_about_point(
            (rows, columns), (center_x, center_y), angle, shape
        )

        self.assertGreater(float(np.max(np.abs(wrong_columns - center_x))), 10.0)

    def test_zero_angle_returns_the_same_array(self) -> None:
        image = np.arange(12, dtype=np.float64).reshape(3, 4)
        self.assertIs(rotate_image_about_point(image, (1.0, 1.0), 0), image)

    def test_quarter_turn_moves_points_with_the_image(self) -> None:
        image = np.zeros((40, 60), dtype=np.float64)
        image[10, 20] = 5.0
        center = (30.0, 20.0)

        rotated = rotate_image_about_point(image, center, -90.0)
        rows, columns = rotate_points_about_point(
            (np.array([10]), np.array([20])), center, -90.0, image.shape
        )

        self.assertEqual(rows.size, 1)
        self.assertAlmostEqual(float(rotated[rows[0], columns[0]]), 5.0, places=6)
        self.assertAlmostEqual(float(rotated[10, 20]), 0.0, places=6)

    def test_points_outside_the_frame_are_dropped(self) -> None:
        rows, columns = rotate_points_about_point(
            (np.array([1]), np.array([1])), (0.0, 0.0), 180.0, (10, 10)
        )
        self.assertEqual(rows.size, 0)
        self.assertEqual(columns.size, 0)


class DistributionRotationDisplayTests(unittest.TestCase):
    def _synthesize(self, norm_img, *, image_mode, rotate_angle=0.0, point_sets=None):
        return synthesize_distribution_image(
            display_name="f",
            image_mode=image_mode,
            show_bent=False,
            need_symmetric_restore=False,
            norm_img=norm_img,
            source_total=1.0,
            image_shape=norm_img.shape,
            center_x=10,
            center_y=10,
            rotate_angle=rotate_angle,
            fitted_point=_EMPTY_POINTS,
            symmetric_fitted_point=_EMPTY_POINTS,
            bent_img=None,
            surface_prediction=None,
            point_sets=[] if point_sets is None else point_sets,
            map_center_line_points=lambda *args, **kwargs: _EMPTY_POINTS,
            unstraighten_center_line_image=lambda image, points, center_x: image,
        )

    def setUp(self) -> None:
        self.norm_img = np.linspace(0.0, 1.0, 400, dtype=np.float64).reshape(20, 20)

    def test_rotation_view_keeps_the_source_array_untouched(self) -> None:
        expected = self.norm_img.copy()
        self._synthesize(self.norm_img, image_mode="rot_img", rotate_angle=17.0)
        np.testing.assert_array_equal(self.norm_img, expected)

    def test_zero_angle_matches_the_original_view(self) -> None:
        original = self._synthesize(self.norm_img, image_mode="ori_img")
        rotated = self._synthesize(self.norm_img, image_mode="rot_img", rotate_angle=0.0)
        np.testing.assert_allclose(rotated.image, original.image)

    def test_rotation_view_actually_rotates(self) -> None:
        original = self._synthesize(self.norm_img, image_mode="ori_img")
        rotated = self._synthesize(self.norm_img, image_mode="rot_img", rotate_angle=30.0)
        self.assertFalse(np.allclose(rotated.image, original.image))

    def test_rotation_view_rotates_overlay_points(self) -> None:
        points = (np.array([5, 6, 7], dtype=np.int32), np.array([5, 6, 7], dtype=np.int32))
        original = self._synthesize(self.norm_img, image_mode="ori_img", point_sets=[points])
        rotated = self._synthesize(
            self.norm_img, image_mode="rot_img", rotate_angle=30.0, point_sets=[points]
        )
        np.testing.assert_array_equal(original.special_points[0], points[0])
        self.assertFalse(np.array_equal(rotated.special_points[0], points[0]))


class _StubTripletCache:
    def __init__(self) -> None:
        self.base = np.arange(1, 26, dtype=np.float64).reshape(5, 5)

    def load(self, folder, f_exp_no, o_exp_no, fo_exp_no) -> ExperimentImageTriplet:
        return ExperimentImageTriplet(
            self.base.copy(),
            self.base.copy() + 30,
            self.base.copy() + 60,
        )


class DistributionRotationServiceTests(unittest.TestCase):
    """The rotation view is display only: it must not feed the fit."""

    def setUp(self) -> None:
        self._patchers = [
            patch.object(distribution_service_module, "fit_center_line", side_effect=self._fake_center_line),
            patch.object(
                distribution_service_module,
                "sample_horizontal_profile",
                side_effect=self._fake_horizontal_profile,
            ),
            patch.object(distribution_service_module, "func_fit_3", side_effect=self._fake_curve_fit),
            patch.object(
                distribution_service_module,
                "extrapolate_fitted_center_line",
                return_value=(np.array([2], dtype=np.int32), np.array([2], dtype=np.int32)),
            ),
        ]
        for patcher in self._patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.service = DistributionFitService(_StubTripletCache())
        self.settings = DistributionSettings(
            image_folder=Path("test-images"),
            f_exp_no="1",
            o_exp_no="2",
            fo_exp_no="3",
            center_xf=2,
            center_yf=2,
            center_xo=2,
            center_yo=2,
            center_xfo=2,
            center_yfo=2,
            q_center_up_f=0,
            q_center_down_f=4,
            q_center_up_o=0,
            q_center_down_o=4,
            q_center_up_fo=0,
            q_center_down_fo=4,
            denoise_ratio=0.0,
            max_k=30,
            rotate_angle_f=12,
            rotate_angle_o=7,
            rotate_angle_fo=3,
        )

    @staticmethod
    def _fake_center_line(image, q_center_up, q_center_down, max_k, center_x, center_y, rotate_angle):
        points = (np.array([2], dtype=np.int32), np.array([2], dtype=np.int32))
        return 1.0, 2.0, 0.9, points, points, np.array([0.0])

    @staticmethod
    def _fake_horizontal_profile(image, center_x, center_y, rotate_angle):
        return np.array([-1.0, 0.0, 1.0]), np.array([0.2, 0.4, 0.2])

    @staticmethod
    def _fake_curve_fit(x_values, y_values, fit_func):
        x_values = np.asarray(x_values, dtype=np.float64)
        return 1.0, 2.0, 3.0, 0.8, np.full(x_values.shape, 0.5, dtype=np.float64)

    def test_rotation_view_keeps_the_fit_numbers(self) -> None:
        original = self.service.run(self.settings, image_mode="ori_img", show_bent=False)
        rotated = self.service.run(self.settings, image_mode="rot_img", show_bent=False)

        self.assertEqual(rotated.numeric_results, original.numeric_results)
        self.assertEqual(rotated.fit_text, original.fit_text)
        self.assertEqual(rotated.summary_rows, original.summary_rows)

    def test_rotation_view_never_straightens(self) -> None:
        with patch.object(distribution_service_module, "straighten_center_line_image") as straighten:
            self.service.run(self.settings, image_mode="rot_img", show_bent=True)
        straighten.assert_not_called()

    def test_rotation_view_does_not_start_a_surface_fit(self) -> None:
        with patch.object(self.service, "_fit_surface_models_batch") as fit_batch:
            result = self.service.run(self.settings, image_mode="rot_img", show_bent=False)
        fit_batch.assert_not_called()
        self.assertTrue(all(item.student_scale_x is None for item in result.numeric_results))

    def test_rejects_an_unknown_image_mode(self) -> None:
        with self.assertRaises(ValueError):
            self.service.run(self.settings, image_mode="spin_img", show_bent=False)


if __name__ == "__main__":
    unittest.main()
