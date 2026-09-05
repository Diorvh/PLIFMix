from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from PLIFv2.algorithms import distribution


def _reference_prediction(parameters: np.ndarray, fit_data, compact: bool) -> np.ndarray | None:
    scale_x, scale_y, exponent_minus_one, *compact_parameters = np.exp(parameters)
    exponent = 1.0 + exponent_minus_one
    radius_squared = (
        fit_data.pixel_x_squared / (scale_x * scale_x)
        + fit_data.pixel_y_squared / (scale_y * scale_y)
    )
    if compact:
        cutoff_radius = compact_parameters[0]
        cutoff = 1.0 - radius_squared / (cutoff_radius * cutoff_radius)
        valid = cutoff > 0
        weights = np.zeros_like(radius_squared)
        weights[valid] = (
            np.power(1.0 + radius_squared[valid], -exponent)
            * np.power(cutoff[valid], 2.0)
        )
    else:
        weights = np.power(1.0 + radius_squared, -exponent)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        return None
    prediction = weights / total
    return prediction if np.all(np.isfinite(prediction)) else None


def _reference_objective(parameters: np.ndarray, fit_data, compact: bool) -> float:
    prediction = _reference_prediction(parameters, fit_data, compact)
    if prediction is None:
        return np.inf
    residual = fit_data.target - prediction
    return float(np.sum(residual * residual) / fit_data.ss_total)


class DistributionWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = np.array(
            [
                [1.0, 2.0, 4.0, 2.0],
                [2.5, 5.0, 8.0, 3.0],
                [1.5, 3.0, 6.0, 2.25],
            ],
            dtype=np.float64,
        )
        self.pixel_x = np.array([-1.8, -0.5, 0.7, 1.8], dtype=np.float64)
        self.pixel_y = np.array([-1.2, 0.0, 1.2], dtype=np.float64)
        self.fit_data = distribution.prepare_surface_fit_data(
            self.target,
            self.pixel_x,
            self.pixel_y,
            1.2,
            0.95,
            2.5,
        )

    def _run_with_recorded_objectives(
        self,
        fit_function,
        parameters: list[np.ndarray],
        compact: bool,
        fit_data=None,
        **fit_kwargs,
    ):
        if fit_data is None:
            fit_data = self.fit_data
        objectives: list[float] = []
        predictions: list[np.ndarray | None] = []
        captured: dict[str, np.ndarray] = {}
        final_parameters = parameters[-1].copy()

        def fake_minimize(objective, initial, method, bounds, options):
            captured["initial"] = initial.copy()
            self.assertEqual(method, "L-BFGS-B")
            for parameter_set in parameters:
                objectives.append(objective(parameter_set))
            final_objective = objective(final_parameters)
            return SimpleNamespace(
                x=final_parameters.copy(),
                success=True,
                fun=final_objective,
                message="test result",
            )

        original_normalize = distribution._normalize_prediction

        def recording_normalize(weights, prediction=None):
            value = original_normalize(weights, prediction)
            predictions.append(None if value is None else value.copy(order="K"))
            return value

        with (
            patch.object(distribution.optimize, "minimize", side_effect=fake_minimize),
            patch.object(distribution, "_normalize_prediction", side_effect=recording_normalize),
        ):
            result = fit_function(fit_data, **fit_kwargs)
        expected_objectives = [
            _reference_objective(parameter_set, fit_data, compact)
            for parameter_set in parameters
        ]
        return result, objectives, expected_objectives, captured["initial"], predictions

    def test_pure_fixed_predictions_objectives_and_final_fit_match_reference(self) -> None:
        parameters = [
            np.log(np.array([1.3, 0.8, 2.3 - 1.0], dtype=np.float64)),
            np.log(np.array([1.9, 1.4, 4.1 - 1.0], dtype=np.float64)),
        ]
        result, objectives, expected_objectives, _, predictions = self._run_with_recorded_objectives(
            distribution.fit_elliptical_student_prepared,
            parameters,
            compact=False,
        )

        np.testing.assert_array_equal(objectives, expected_objectives)
        for observed, parameter_set in zip(predictions[: len(parameters)], parameters):
            np.testing.assert_array_equal(
                observed,
                _reference_prediction(parameter_set, self.fit_data, compact=False),
            )
        expected_prediction = _reference_prediction(parameters[-1], self.fit_data, compact=False)
        np.testing.assert_array_equal(result.prediction, expected_prediction)
        expected_residual = self.fit_data.target - expected_prediction
        expected_r_squared = float(
            1.0 - np.sum(expected_residual * expected_residual) / self.fit_data.ss_total
        )
        self.assertEqual(result.r_squared, expected_r_squared)

    def test_compact_fixed_predictions_objectives_and_final_fit_match_reference(self) -> None:
        parameters = [
            np.log(np.array([1.3, 0.8, 2.3 - 1.0, 2.5], dtype=np.float64)),
            np.log(np.array([1.9, 1.4, 4.1 - 1.0, 4.0], dtype=np.float64)),
        ]
        result, objectives, expected_objectives, _, predictions = self._run_with_recorded_objectives(
            distribution.fit_compact_elliptical_student_prepared,
            parameters,
            compact=True,
        )

        np.testing.assert_array_equal(objectives, expected_objectives)
        for observed, parameter_set in zip(predictions[: len(parameters)], parameters):
            np.testing.assert_array_equal(
                observed,
                _reference_prediction(parameter_set, self.fit_data, compact=True),
            )
        expected_prediction = _reference_prediction(parameters[-1], self.fit_data, compact=True)
        np.testing.assert_array_equal(result.prediction, expected_prediction)
        expected_residual = self.fit_data.target - expected_prediction
        expected_r_squared = float(
            1.0 - np.sum(expected_residual * expected_residual) / self.fit_data.ss_total
        )
        self.assertEqual(result.r_squared, expected_r_squared)

    def test_compact_keeps_explicit_pure_fit_values_as_initial_values(self) -> None:
        pure_initial = (1.65, 1.05, 3.4)
        _, _, _, captured_initial, _ = self._run_with_recorded_objectives(
            distribution.fit_compact_elliptical_student_prepared,
            [np.log(np.array([1.4, 0.9, 2.6 - 1.0, 3.0], dtype=np.float64))],
            compact=True,
            initial_scale_x=pure_initial[0],
            initial_scale_y=pure_initial[1],
            initial_exponent=pure_initial[2],
        )
        np.testing.assert_array_equal(
            captured_initial[:3],
            np.log(np.array([pure_initial[0], pure_initial[1], pure_initial[2] - 1.0])),
        )

    def test_optimize_surface_returns_prediction_snapshot(self) -> None:
        parameters = np.log(np.array([1.5, 1.1, 2.8 - 1.0], dtype=np.float64))
        later_parameters = np.log(np.array([2.0, 0.75, 4.5 - 1.0], dtype=np.float64))
        prediction_scratch = np.empty_like(self.fit_data.target)
        objective_values: list[float] = []

        def predict(parameter_set: np.ndarray) -> np.ndarray:
            prediction_scratch[...] = _reference_prediction(parameter_set, self.fit_data, compact=False)
            return prediction_scratch

        def fake_minimize(objective, initial, method, bounds, options):
            objective_values.append(objective(parameters))
            final_objective = objective(later_parameters)
            return SimpleNamespace(
                x=later_parameters.copy(),
                success=True,
                fun=final_objective,
                message="test result",
            )

        original_target = self.fit_data.target.copy()
        with patch.object(distribution.optimize, "minimize", side_effect=fake_minimize):
            _, prediction = distribution._optimize_surface(
                parameters,
                ((-10.0, 10.0),) * 3,
                predict,
                self.fit_data,
            )

        predict(parameters)
        np.testing.assert_array_equal(
            prediction,
            _reference_prediction(later_parameters, self.fit_data, compact=False),
        )
        np.testing.assert_array_equal(self.fit_data.target, original_target)
        self.assertEqual(objective_values[0], _reference_objective(parameters, self.fit_data, compact=False))

    def test_layout_variants_match_reference(self) -> None:
        storage = np.arange(1.0, 25.0, dtype=np.float64).reshape(6, 4)
        targets = [
            np.ascontiguousarray(self.target),
            np.asfortranarray(self.target),
            storage[::2],
        ]
        grid_x = np.broadcast_to(self.pixel_x, self.target.shape)
        grid_y = np.broadcast_to(self.pixel_y[:, None], self.target.shape)
        coordinates = [
            (self.pixel_x, self.pixel_y),
            (np.ascontiguousarray(grid_x), np.ascontiguousarray(grid_y)),
            (np.asfortranarray(grid_x), np.asfortranarray(grid_y)),
        ]
        parameters = {
            False: np.log(np.array([1.3, 0.8, 2.3 - 1.0], dtype=np.float64)),
            True: np.log(np.array([1.3, 0.8, 2.3 - 1.0, 3.0], dtype=np.float64)),
        }

        for target in targets:
            for pixel_x, pixel_y in coordinates:
                fit_data = distribution.prepare_surface_fit_data(
                    target,
                    pixel_x,
                    pixel_y,
                    1.2,
                    0.95,
                    2.5,
                )
                for compact in (False, True):
                    fit_function = (
                        distribution.fit_compact_elliptical_student_prepared
                        if compact
                        else distribution.fit_elliptical_student_prepared
                    )
                    result, objectives, expected_objectives, _, predictions = (
                        self._run_with_recorded_objectives(
                            fit_function,
                            [parameters[compact]],
                            compact,
                            fit_data=fit_data,
                        )
                    )
                    expected_prediction = _reference_prediction(
                        parameters[compact], fit_data, compact
                    )
                    np.testing.assert_array_equal(predictions[0], expected_prediction)
                    np.testing.assert_array_equal(result.prediction, expected_prediction)
                    self.assertEqual(objectives, expected_objectives)
                    expected_residual = fit_data.target - expected_prediction
                    expected_r_squared = float(
                        1.0
                        - np.sum(expected_residual * expected_residual) / fit_data.ss_total
                    )
                    self.assertEqual(result.r_squared, expected_r_squared)
                    self.assertEqual(
                        result.prediction.flags.f_contiguous,
                        pixel_x.ndim == 2
                        and pixel_x.flags.f_contiguous
                        and pixel_y.flags.f_contiguous,
                    )

    def test_compact_large_grid_support_changes_match_reference(self) -> None:
        pixel_x = np.linspace(-6.0, 6.0, 47, dtype=np.float64)
        pixel_y = np.linspace(-4.0, 4.0, 31, dtype=np.float64)
        row_signal = np.linspace(1.0, 2.0, 31, dtype=np.float64)
        column_signal = np.linspace(0.5, 3.0, 47, dtype=np.float64)
        target = row_signal[:, None] + column_signal[None, :]
        fit_data = distribution.prepare_surface_fit_data(
            target,
            pixel_x,
            pixel_y,
            2.4,
            1.8,
            2.6,
        )
        parameters = [
            np.log(np.array([2.4, 1.8, 2.6 - 1.0, 10.0], dtype=np.float64)),
            np.log(np.array([2.4, 1.8, 2.6 - 1.0, 2.75], dtype=np.float64)),
            np.log(np.array([2.4, 1.8, 2.6 - 1.0, 12.0], dtype=np.float64)),
        ]

        result, objectives, expected_objectives, _, predictions = (
            self._run_with_recorded_objectives(
                distribution.fit_compact_elliptical_student_prepared,
                parameters,
                compact=True,
                fit_data=fit_data,
            )
        )

        self.assertGreater(parameters[0][-1], parameters[1][-1])
        self.assertGreater(parameters[2][-1], parameters[1][-1])
        self.assertEqual(objectives, expected_objectives)
        for observed, parameter_set in zip(predictions[: len(parameters)], parameters):
            expected_prediction = _reference_prediction(parameter_set, fit_data, compact=True)
            np.testing.assert_array_equal(observed, expected_prediction)
            self.assertGreater(np.count_nonzero(expected_prediction > 0), 0)
        small_prediction = _reference_prediction(parameters[1], fit_data, compact=True)
        self.assertGreater(np.count_nonzero(small_prediction == 0), 0)
        np.testing.assert_array_equal(
            result.prediction,
            _reference_prediction(parameters[-1], fit_data, compact=True),
        )


if __name__ == "__main__":
    unittest.main()
