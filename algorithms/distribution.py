from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


_MIN_RIDGE_POINTS = 12
_MAX_ABS_ANGLE = 35.0


@dataclass(frozen=True)
class DistributionRegistrationEstimate:
    center_x: int
    center_y: int
    angle_degrees: int
    symmetry_error_pixels: float
    ridge_point_count: int
    used_fallback: bool


@dataclass(frozen=True)
class _AngleCandidate:
    angle_degrees: int
    score: float


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
