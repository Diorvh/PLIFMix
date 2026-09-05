from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np

from PLIFv2.core.experiment_images import ExperimentImageTriplet
from PLIFv2.domain.settings import DistributionSettings
from PLIFv2.services import distribution_service as distribution_service_module
from PLIFv2.services.distribution_service import DistributionFitService
from PLIFv2.services.distribution_workers import SurfaceFitParameters


class _FreshTripletCache:
    def __init__(self) -> None:
        base = np.arange(1, 26, dtype=np.float64).reshape(5, 5)
        self.images = {
            "1": base,
            "2": base + 30,
            "3": base + 60,
        }

    def load(self, folder, f_exp_no, o_exp_no, fo_exp_no) -> ExperimentImageTriplet:
        return ExperimentImageTriplet(
            self.images[str(f_exp_no)].copy(),
            self.images[str(o_exp_no)].copy(),
            self.images[str(fo_exp_no)].copy(),
        )


class DistributionSurfaceCacheTests(unittest.TestCase):
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
            patch.object(
                distribution_service_module,
                "straighten_center_line_image",
                side_effect=lambda image, fitted_point, center_x: image.copy(),
            ),
            patch.object(
                distribution_service_module,
                "unstraighten_center_line_image",
                side_effect=lambda image, fitted_point, center_x: image.copy(),
            ),
        ]
        for patcher in self._patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

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

    def _new_service(self):
        cache = _FreshTripletCache()
        service = DistributionFitService(cache)
        calls: list[tuple[str, ...]] = []

        def fit_batch(tasks):
            calls.append(tuple(task.image_name for task in tasks))
            results = []
            for task in tasks:
                value = {"f_img": 1.0, "o_img": 2.0, "fo_img": 3.0}[task.image_name]
                student = SurfaceFitParameters(value, value + 0.1, 1.5, None, 0.95)
                compact = SurfaceFitParameters(value + 0.2, value + 0.3, 1.6, 4.0, 0.94)
                results.append(
                    distribution_service_module._SurfaceModelFits(
                        student,
                        compact,
                        task.target.copy(),
                        None,
                        None,
                        (),
                    )
                )
            return results

        service._fit_surface_models_batch = Mock(side_effect=fit_batch)
        settings = DistributionSettings(
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
        )
        return cache, service, settings, calls

    @staticmethod
    def _fit(service, settings):
        return service.run(settings, image_mode="fit_img", show_bent=False)

    def test_single_image_changes_only_submit_that_image_and_keep_one_latest_entry(self) -> None:
        cache, service, settings, calls = self._new_service()
        self._fit(service, settings)

        changed = deepcopy(settings)
        changed.f_exp_no = "4"
        cache.images["4"] = cache.images["1"] + 100
        changed.center_xf += 1
        changed.f_shift = 1
        self._fit(service, changed)
        self.assertEqual(calls, [("f_img", "o_img", "fo_img"), ("f_img",)])

        self._fit(service, settings)
        self.assertEqual(calls[-1], ("f_img",))
        self.assertEqual(set(service._surface_fit_cache), {"f_img", "o_img", "fo_img"})

    def test_image_specific_parameters_only_invalidate_their_slot(self) -> None:
        cases = (
            ("f_img", "center_xf", 3),
            ("f_img", "center_yf", 3),
            ("f_img", "q_center_up_f", 1),
            ("f_img", "q_center_down_f", 3),
            ("f_img", "rotate_angle_f", 5),
            ("f_img", "f_shift", 1),
            ("o_img", "center_xo", 3),
            ("o_img", "center_yo", 3),
            ("o_img", "q_center_up_o", 1),
            ("o_img", "q_center_down_o", 3),
            ("o_img", "rotate_angle_o", 5),
            ("o_img", "o_shift", 1),
            ("fo_img", "center_xfo", 3),
            ("fo_img", "center_yfo", 3),
            ("fo_img", "q_center_up_fo", 1),
            ("fo_img", "q_center_down_fo", 3),
            ("fo_img", "rotate_angle_fo", 5),
        )
        for image_name, field, value in cases:
            with self.subTest(image_name=image_name, field=field):
                _cache, service, settings, calls = self._new_service()
                self._fit(service, settings)
                changed = deepcopy(settings)
                setattr(changed, field, value)
                self._fit(service, changed)
                self.assertEqual(calls[-1], (image_name,))

    def test_experiment_number_change_only_invalidates_that_image(self) -> None:
        cases = (
            ("f_exp_no", "4", "f_img"),
            ("o_exp_no", "5", "o_img"),
            ("fo_exp_no", "6", "fo_img"),
        )
        for field, number, image_name in cases:
            with self.subTest(field=field):
                cache, service, settings, calls = self._new_service()
                self._fit(service, settings)
                changed = deepcopy(settings)
                setattr(changed, field, number)
                cache.images[number] = cache.images[str(getattr(settings, field))] + 100
                self._fit(service, changed)
                self.assertEqual(calls[-1], (image_name,))

    def test_source_update_only_invalidates_changed_image_and_reuses_it_unchanged(self) -> None:
        cache, service, settings, calls = self._new_service()
        self._fit(service, settings)

        cache.images["2"][0, 0] += 1
        self._fit(service, settings)
        self.assertEqual(calls, [("f_img", "o_img", "fo_img"), ("o_img",)])

        self._fit(service, settings)
        self.assertEqual(calls, [("f_img", "o_img", "fo_img"), ("o_img",)])

    def test_global_denoise_or_max_k_invalidates_all_three_slots(self) -> None:
        for field, value in (("denoise_ratio", 0.1), ("max_k", 31)):
            with self.subTest(field=field):
                _cache, service, settings, calls = self._new_service()
                self._fit(service, settings)
                changed = deepcopy(settings)
                setattr(changed, field, value)
                self._fit(service, changed)
                self.assertEqual(calls[-1], ("f_img", "o_img", "fo_img"))

    def test_display_modes_overlays_and_gain_reuse_surface_fits(self) -> None:
        _cache, service, settings, calls = self._new_service()
        self._fit(service, settings)

        changed = deepcopy(settings)
        changed.gain = 7
        service.run(changed, annotations=frozenset({"center_line"}), image_mode="ori_img", show_bent=False)
        service.run(changed, annotations=frozenset({"fitted_center_line"}), image_mode="delta_img", show_bent=True)
        service.run(changed, annotations=frozenset(), image_mode="fit_img", show_bent=True)
        self.assertEqual(calls, [("f_img", "o_img", "fo_img")])

    def test_original_mode_shows_cached_results_per_image_after_one_image_misses(self) -> None:
        cache, service, settings, calls = self._new_service()
        self._fit(service, settings)

        changed = deepcopy(settings)
        changed.center_xf += 1
        result = service.run(changed, image_mode="ori_img", show_bent=False)

        self.assertEqual(calls, [("f_img", "o_img", "fo_img")])
        self.assertEqual(result.summary_rows[0].student_equation, "未执行")
        self.assertEqual(result.summary_rows[0].student_scale_x, "-")
        self.assertEqual(result.summary_rows[1].student_scale_x, "2.0000")
        self.assertEqual(result.summary_rows[2].student_scale_x, "3.0000")
        self.assertIsNone(result.fit_rows[0][14])
        self.assertEqual(result.fit_rows[1][14], 2.0)
        self.assertEqual(result.fit_rows[2][14], 3.0)


if __name__ == "__main__":
    unittest.main()
