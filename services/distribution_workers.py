from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from PLIFMix.algorithms.distribution import (
    DistributionModelFitError,
    fit_compact_elliptical_student_prepared,
    fit_elliptical_student_prepared,
    prepare_surface_fit_data,
)


@dataclass(frozen=True)
class SurfaceFitTask:
    target: np.ndarray
    pixel_x: np.ndarray
    pixel_y: np.ndarray
    initial_scale_x: float
    initial_scale_y: float
    initial_exponent: float
    image_name: str


@dataclass(frozen=True)
class SurfaceFitParameters:
    scale_x: float
    scale_y: float
    exponent: float
    cutoff_radius: float | None
    r_squared: float


@dataclass(frozen=True)
class SurfaceFitWorkerResult:
    student_fit: SurfaceFitParameters | None
    compact_fit: SurfaceFitParameters | None
    selected_prediction: np.ndarray | None
    student_error: str | None
    compact_error: str | None
    warnings: tuple[str, ...]


def fit_surface_models_worker(task: SurfaceFitTask) -> SurfaceFitWorkerResult:
    """Fit one image's two surface models in a process-pool worker."""
    try:
        fit_data = prepare_surface_fit_data(
            task.target,
            task.pixel_x,
            task.pixel_y,
            task.initial_scale_x,
            task.initial_scale_y,
            task.initial_exponent,
        )
    except DistributionModelFitError as exc:
        error = str(exc)
        return SurfaceFitWorkerResult(
            None,
            None,
            None,
            error,
            error,
            (
                f"{task.image_name} 纯椭圆 Student 拟合失败：{error}",
                f"{task.image_name} 紧支撑 Student 拟合失败：{error}",
            ),
        )

    student_fit: SurfaceFitParameters | None = None
    compact_fit: SurfaceFitParameters | None = None
    student_error: str | None = None
    compact_error: str | None = None
    student_prediction: np.ndarray | None = None
    compact_prediction: np.ndarray | None = None
    warnings: list[str] = []

    try:
        fitted = fit_elliptical_student_prepared(fit_data)
        student_prediction = fitted.prediction
        student_fit = SurfaceFitParameters(
            fitted.scale_x,
            fitted.scale_y,
            fitted.exponent,
            fitted.cutoff_radius,
            fitted.r_squared,
        )
    except DistributionModelFitError as exc:
        student_error = str(exc)
        warnings.append(f"{task.image_name} 纯椭圆 Student 拟合失败：{student_error}")

    compact_initial = student_fit
    try:
        fitted = fit_compact_elliptical_student_prepared(
            fit_data,
            compact_initial.scale_x if compact_initial is not None else task.initial_scale_x,
            compact_initial.scale_y if compact_initial is not None else task.initial_scale_y,
            compact_initial.exponent if compact_initial is not None else task.initial_exponent,
        )
        compact_prediction = fitted.prediction
        compact_fit = SurfaceFitParameters(
            fitted.scale_x,
            fitted.scale_y,
            fitted.exponent,
            fitted.cutoff_radius,
            fitted.r_squared,
        )
    except DistributionModelFitError as exc:
        compact_error = str(exc)
        warnings.append(f"{task.image_name} 紧支撑 Student 拟合失败：{compact_error}")

    selected_prediction = compact_prediction if compact_fit is not None else student_prediction

    return SurfaceFitWorkerResult(
        student_fit,
        compact_fit,
        selected_prediction,
        student_error,
        compact_error,
        tuple(warnings),
    )
