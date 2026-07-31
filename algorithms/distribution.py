from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize
from scipy.ndimage import gaussian_filter


_MIN_RIDGE_POINTS = 12
_MAX_ABS_ANGLE = 35.0
_MIN_STUDENT_EXPONENT = 1.0001
_MAX_STUDENT_EXPONENT = 100.0
_MIN_STUDENT_SCALE = 0.25
_COMPACT_CUTOFF_POWER = 2.0


@dataclass(frozen=True)
class DistributionRegistrationEstimate:
    center_x: int
    center_y: int
    angle_degrees: int
    symmetry_error_pixels: float
    ridge_point_count: int
    used_fallback: bool


@dataclass(frozen=True)
class EllipticalStudentFit:
    scale_x: float
    scale_y: float
    exponent: float
    cutoff_radius: float | None
    r_squared: float
    prediction: np.ndarray


class DistributionModelFitError(RuntimeError):
    pass


@dataclass(frozen=True)
class _AngleCandidate:
    angle_degrees: int
    score: float


def fit_elliptical_student(
    target: np.ndarray,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    initial_scale_x: float,
    initial_scale_y: float,
    initial_exponent: float,
) -> EllipticalStudentFit:
    """Fit a normalized three-parameter elliptical Pearson VII surface."""
    fit_data = prepare_surface_fit_data(
        target,
        pixel_x,
        pixel_y,
        initial_scale_x,
        initial_scale_y,
        initial_exponent,
    )
    return fit_elliptical_student_prepared(fit_data)


def fit_elliptical_student_prepared(fit_data: _SurfaceFitData) -> EllipticalStudentFit:
    """Fit the pure model using an already prepared surface workspace."""
    initial = np.log(
        [
            fit_data.initial_scale_x,
            fit_data.initial_scale_y,
            fit_data.initial_exponent - 1.0,
        ]
    )
    bounds = (
        (np.log(_MIN_STUDENT_SCALE), np.log(fit_data.maximum_scale_x)),
        (np.log(_MIN_STUDENT_SCALE), np.log(fit_data.maximum_scale_y)),
        (
            np.log(_MIN_STUDENT_EXPONENT - 1.0),
            np.log(_MAX_STUDENT_EXPONENT - 1.0),
        ),
    )

    def predict(parameters: np.ndarray) -> np.ndarray | None:
        scale_x, scale_y, exponent_minus_one = np.exp(parameters)
        exponent = 1.0 + exponent_minus_one
        radius_squared = (
            fit_data.pixel_x_squared / (scale_x * scale_x)
            + fit_data.pixel_y_squared / (scale_y * scale_y)
        )
        weights = np.power(1.0 + radius_squared, -exponent)
        return _normalize_prediction(weights)

    result, prediction = _optimize_surface(initial, bounds, predict, fit_data)
    scale_x, scale_y, exponent_minus_one = np.exp(result.x)
    return EllipticalStudentFit(
        scale_x=float(scale_x),
        scale_y=float(scale_y),
        exponent=float(1.0 + exponent_minus_one),
        cutoff_radius=None,
        r_squared=_surface_r_squared(fit_data.target, prediction, fit_data.ss_total),
        prediction=prediction,
    )


def fit_compact_elliptical_student(
    target: np.ndarray,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    initial_scale_x: float,
    initial_scale_y: float,
    initial_exponent: float,
) -> EllipticalStudentFit:
    """Fit an elliptical Pearson VII surface with a smooth finite cutoff."""
    fit_data = prepare_surface_fit_data(
        target,
        pixel_x,
        pixel_y,
        initial_scale_x,
        initial_scale_y,
        initial_exponent,
    )
    return fit_compact_elliptical_student_prepared(fit_data)


def fit_compact_elliptical_student_prepared(
    fit_data: _SurfaceFitData,
    initial_scale_x: float | None = None,
    initial_scale_y: float | None = None,
    initial_exponent: float | None = None,
) -> EllipticalStudentFit:
    """Fit the compact model using an already prepared surface workspace."""
    compact_scale_x = _bounded_initial(
        fit_data.initial_scale_x if initial_scale_x is None else initial_scale_x,
        _MIN_STUDENT_SCALE,
        fit_data.maximum_scale_x,
    )
    compact_scale_y = _bounded_initial(
        fit_data.initial_scale_y if initial_scale_y is None else initial_scale_y,
        _MIN_STUDENT_SCALE,
        fit_data.maximum_scale_y,
    )
    compact_exponent = _bounded_initial(
        fit_data.initial_exponent if initial_exponent is None else initial_exponent,
        _MIN_STUDENT_EXPONENT,
        _MAX_STUDENT_EXPONENT,
    )
    initial_radius_squared = (
        fit_data.pixel_x_squared / (compact_scale_x**2)
        + fit_data.pixel_y_squared / (compact_scale_y**2)
    )
    initial_cutoff = max(
        0.5,
        _weighted_radius_quantile(initial_radius_squared, fit_data.target, 0.995) * 1.05,
    )
    maximum_cutoff = max(
        initial_cutoff * 4.0,
        float(np.sqrt(np.max(initial_radius_squared))) * 2.0,
        2.0,
    )
    initial = np.log(
        [
            compact_scale_x,
            compact_scale_y,
            compact_exponent - 1.0,
            initial_cutoff,
        ]
    )
    bounds = (
        (np.log(_MIN_STUDENT_SCALE), np.log(fit_data.maximum_scale_x)),
        (np.log(_MIN_STUDENT_SCALE), np.log(fit_data.maximum_scale_y)),
        (
            np.log(_MIN_STUDENT_EXPONENT - 1.0),
            np.log(_MAX_STUDENT_EXPONENT - 1.0),
        ),
        (np.log(0.25), np.log(maximum_cutoff)),
    )

    def predict(parameters: np.ndarray) -> np.ndarray | None:
        scale_x, scale_y, exponent_minus_one, cutoff_radius = np.exp(parameters)
        exponent = 1.0 + exponent_minus_one
        radius_squared = (
            fit_data.pixel_x_squared / (scale_x * scale_x)
            + fit_data.pixel_y_squared / (scale_y * scale_y)
        )
        cutoff = 1.0 - radius_squared / (cutoff_radius * cutoff_radius)
        valid = cutoff > 0
        weights = np.zeros_like(radius_squared)
        weights[valid] = (
            np.power(1.0 + radius_squared[valid], -exponent)
            * np.power(cutoff[valid], _COMPACT_CUTOFF_POWER)
        )
        return _normalize_prediction(weights)

    result, prediction = _optimize_surface(initial, bounds, predict, fit_data)
    scale_x, scale_y, exponent_minus_one, cutoff_radius = np.exp(result.x)
    return EllipticalStudentFit(
        scale_x=float(scale_x),
        scale_y=float(scale_y),
        exponent=float(1.0 + exponent_minus_one),
        cutoff_radius=float(cutoff_radius),
        r_squared=_surface_r_squared(fit_data.target, prediction, fit_data.ss_total),
        prediction=prediction,
    )


@dataclass(frozen=True)
class _SurfaceFitData:
    target: np.ndarray
    pixel_x_squared: np.ndarray
    pixel_y_squared: np.ndarray
    initial_scale_x: float
    initial_scale_y: float
    initial_exponent: float
    maximum_scale_x: float
    maximum_scale_y: float
    ss_total: float


def prepare_surface_fit_data(
    target: np.ndarray,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    initial_scale_x: float,
    initial_scale_y: float,
    initial_exponent: float,
) -> _SurfaceFitData:
    normalized = np.asarray(target, dtype=np.float64)
    x_values = np.asarray(pixel_x, dtype=np.float64)
    y_values = np.asarray(pixel_y, dtype=np.float64)
    if normalized.ndim != 2:
        raise DistributionModelFitError("二维拟合图像必须是二维数组。")
    if x_values.ndim == 1 and y_values.ndim == 1:
        if x_values.size != normalized.shape[1] or y_values.size != normalized.shape[0]:
            raise DistributionModelFitError("一维坐标轴长度必须与二维拟合图像的宽高一致。")
        pixel_x_squared = (x_values * x_values)[np.newaxis, :]
        pixel_y_squared = (y_values * y_values)[:, np.newaxis]
    elif x_values.shape == normalized.shape and y_values.shape == normalized.shape:
        pixel_x_squared = x_values * x_values
        pixel_y_squared = y_values * y_values
    else:
        raise DistributionModelFitError("二维拟合图像和坐标网格必须是形状相同的二维数组，或使用两个一维坐标轴。")
    if not np.all(np.isfinite(normalized)) or np.any(normalized < 0):
        raise DistributionModelFitError("二维拟合图像包含负值或非有限值。")
    total = float(np.sum(normalized))
    if total <= 0:
        raise DistributionModelFitError("二维拟合图像总量必须大于 0。")
    normalized = normalized / total
    ss_total = float(np.sum((normalized - np.mean(normalized)) ** 2))
    if ss_total <= 0:
        raise DistributionModelFitError("二维拟合图像没有可拟合的强度变化。")

    maximum_scale_x = max(float(np.max(np.abs(x_values))) * 4.0, 4.0)
    maximum_scale_y = max(float(np.max(np.abs(y_values))) * 4.0, 4.0)
    scale_x = _bounded_initial(initial_scale_x, _MIN_STUDENT_SCALE, maximum_scale_x)
    scale_y = _bounded_initial(initial_scale_y, _MIN_STUDENT_SCALE, maximum_scale_y)
    exponent = _bounded_initial(
        initial_exponent,
        _MIN_STUDENT_EXPONENT,
        _MAX_STUDENT_EXPONENT,
    )
    return _SurfaceFitData(
        target=normalized,
        pixel_x_squared=pixel_x_squared,
        pixel_y_squared=pixel_y_squared,
        initial_scale_x=scale_x,
        initial_scale_y=scale_y,
        initial_exponent=exponent,
        maximum_scale_x=maximum_scale_x,
        maximum_scale_y=maximum_scale_y,
        ss_total=ss_total,
    )


def _optimize_surface(
    initial: np.ndarray,
    bounds: tuple[tuple[float, float], ...],
    predict,
    fit_data: _SurfaceFitData,
) -> tuple[optimize.OptimizeResult, np.ndarray]:
    def objective(parameters: np.ndarray) -> float:
        prediction = predict(parameters)
        if prediction is None:
            return np.inf
        residual = fit_data.target - prediction
        return float(np.sum(residual * residual) / fit_data.ss_total)

    result = optimize.minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40},
    )
    prediction = predict(result.x)
    if not result.success or not np.isfinite(result.fun) or prediction is None:
        raise DistributionModelFitError(f"二维 Student 拟合未收敛：{result.message}")
    return result, prediction


def _normalize_prediction(weights: np.ndarray) -> np.ndarray | None:
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        return None
    prediction = weights / total
    return prediction if np.all(np.isfinite(prediction)) else None


def _surface_r_squared(target: np.ndarray, prediction: np.ndarray, ss_total: float) -> float:
    residual = target - prediction
    return float(1.0 - np.sum(residual * residual) / ss_total)


def _weighted_radius_quantile(radius_squared: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    positive = weights > 0
    radii = np.asarray(radius_squared[positive], dtype=np.float64)
    positive_weights = np.asarray(weights[positive], dtype=np.float64)
    order = np.argsort(radii)
    cumulative = np.cumsum(positive_weights[order])
    cutoff = quantile * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, cutoff, side="left")), radii.size - 1)
    return float(np.sqrt(radii[order[index]]))


def _bounded_initial(value: float, lower: float, upper: float) -> float:
    if not np.isfinite(value):
        return float(np.sqrt(lower * upper))
    return float(np.clip(abs(value), lower, upper))


def estimate_distribution_registration(
    image: np.ndarray,
    up_limit: int,
    down_limit: int,
    max_k: int,
) -> DistributionRegistrationEstimate:
    """Estimate the geometric vertex and clockwise rotation of a C-shaped ridge."""
    source = np.asarray(image, dtype=np.float64)
    if source.ndim != 2 or source.size == 0:
        raise ValueError("自动定位需要非空二维图像。")
    if not np.all(np.isfinite(source)):
        raise ValueError("自动定位图像包含非有限数值。")
    if float(np.max(source)) <= 0:
        raise ValueError("自动定位图像没有正强度信号。")

    smoothed = gaussian_filter(source, sigma=(1.5, 2.5), mode="nearest")
    row_start, row_stop = _effective_row_bounds(source.shape[0], up_limit, down_limit)
    peak_y, peak_x = np.unravel_index(int(np.argmax(smoothed)), smoothed.shape)
    peak_x = int(peak_x)
    peak_y = int(peak_y)

    ridge_y, ridge_x, ridge_weights = _extract_ridge(
        smoothed,
        row_start,
        row_stop,
        max_k,
    )
    if ridge_y.size < _MIN_RIDGE_POINTS:
        return _fallback_estimate(peak_x, peak_y, ridge_y, ridge_x, ridge_weights)

    u0 = peak_y - ridge_y
    v0 = peak_x - ridge_x
    candidate = _find_fixed_origin_angle(
        u0,
        v0,
        ridge_weights,
    )
    if candidate is None:
        return _fallback_estimate(
            peak_x,
            peak_y,
            ridge_y,
            ridge_x,
            ridge_weights,
        )

    return DistributionRegistrationEstimate(
        center_x=peak_x,
        center_y=peak_y,
        angle_degrees=candidate.angle_degrees,
        symmetry_error_pixels=candidate.score,
        ridge_point_count=int(ridge_y.size),
        used_fallback=False,
    )


def _effective_row_bounds(height: int, up_limit: int, down_limit: int) -> tuple[int, int]:
    start = int(np.clip(up_limit, 0, height - 1))
    stop = int(np.clip(down_limit, 0, height - 1))
    if stop - start + 1 < _MIN_RIDGE_POINTS:
        return 0, height - 1
    return start, stop


def _extract_ridge(
    smoothed: np.ndarray,
    row_start: int,
    row_stop: int,
    max_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = smoothed[row_start : row_stop + 1]
    row_peaks = np.max(rows, axis=1)
    global_peak = float(np.max(row_peaks))
    valid = row_peaks >= global_peak * 0.03
    valid_rows = np.flatnonzero(valid) + row_start
    half_window = max(3, min(smoothed.shape[1] // 10, max(1, int(max_k)) // 2))

    ridge_x: list[float] = []
    ridge_y: list[float] = []
    weights: list[float] = []
    for y in valid_rows:
        row = smoothed[y]
        peak_x = int(np.argmax(row))
        left = max(0, peak_x - half_window)
        right = min(row.size, peak_x + half_window + 1)
        local_values = row[left:right]
        local_total = float(np.sum(local_values))
        if local_total <= 0:
            continue
        local_x = np.arange(left, right, dtype=np.float64)
        ridge_x.append(float(np.sum(local_x * local_values) / local_total))
        ridge_y.append(float(y))
        weights.append(float(np.sqrt(row[peak_x] / global_peak)))
    return (
        np.asarray(ridge_y, dtype=np.float64),
        np.asarray(ridge_x, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )


def _principal_direction_angle(
    u: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray,
) -> float:
    if u.size < 2:
        return 0.0
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        return 0.0
    coordinates = np.column_stack((u, v))
    center = np.sum(coordinates * weights[:, None], axis=0) / total_weight
    centered = coordinates - center
    covariance = (centered * weights[:, None]).T @ centered / total_weight
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    if direction[0] < 0:
        direction = -direction
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    return float(np.clip(angle, -_MAX_ABS_ANGLE, _MAX_ABS_ANGLE))


def _find_fixed_origin_angle(
    u0: np.ndarray,
    v0: np.ndarray,
    weights: np.ndarray,
) -> _AngleCandidate | None:
    best: _AngleCandidate | None = None
    minimum_radius = 5.0
    for angle in range(-int(_MAX_ABS_ANGLE), int(_MAX_ABS_ANGLE) + 1):
        theta = np.radians(angle)
        u = u0 * np.cos(theta) + v0 * np.sin(theta)
        v = -u0 * np.sin(theta) + v0 * np.cos(theta)

        positive = u > minimum_radius
        negative = u < -minimum_radius
        if np.count_nonzero(positive) < _MIN_RIDGE_POINTS or np.count_nonzero(negative) < _MIN_RIDGE_POINTS:
            continue

        positive_radius = u[positive]
        negative_radius = -u[negative]
        positive_order = np.argsort(positive_radius)
        negative_order = np.argsort(negative_radius)
        positive_radius = positive_radius[positive_order]
        negative_radius = negative_radius[negative_order]
        common_radius = float(min(positive_radius[-1], negative_radius[-1]))
        if common_radius <= minimum_radius:
            continue

        sample_count = max(30, min(200, int(np.ceil(common_radius - minimum_radius)) + 1))
        sample_radius = np.linspace(minimum_radius, common_radius, sample_count)
        positive_values = np.interp(sample_radius, positive_radius, v[positive][positive_order])
        negative_values = np.interp(sample_radius, negative_radius, v[negative][negative_order])
        positive_weights = np.interp(sample_radius, positive_radius, weights[positive][positive_order])
        negative_weights = np.interp(sample_radius, negative_radius, weights[negative][negative_order])
        pair_weights = np.sqrt(positive_weights * negative_weights)
        total_weight = float(np.sum(pair_weights))
        if total_weight <= 0:
            continue

        residual = positive_values - negative_values
        score = float(np.sqrt(np.sum(pair_weights * residual**2) / total_weight))
        candidate = _AngleCandidate(
            angle_degrees=angle,
            score=score,
        )
        if best is None or candidate.score < best.score:
            best = candidate
    return best


def _fallback_estimate(
    peak_x: int,
    peak_y: int,
    ridge_y: np.ndarray,
    ridge_x: np.ndarray,
    ridge_weights: np.ndarray,
    principal_angle: float | None = None,
) -> DistributionRegistrationEstimate:
    if principal_angle is None and ridge_y.size >= 2:
        principal_angle = _principal_direction_angle(
            peak_y - ridge_y,
            peak_x - ridge_x,
            ridge_weights,
        )
    return DistributionRegistrationEstimate(
        center_x=int(peak_x),
        center_y=int(peak_y),
        angle_degrees=int(np.rint(principal_angle or 0.0)),
        symmetry_error_pixels=float("nan"),
        ridge_point_count=int(ridge_y.size),
        used_fallback=True,
    )
