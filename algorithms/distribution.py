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
    fit_r_squared: float
    ridge_point_count: int
    used_fallback: bool


@dataclass(frozen=True)
class _QuadraticCandidate:
    angle_degrees: int
    vertex_u: float
    vertex_v: float
    r_squared: float
    curvature_excursion: float
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
    selected = smoothed[row_start : row_stop + 1]
    peak_local_y, peak_x = np.unravel_index(int(np.argmax(selected)), selected.shape)
    peak_y = int(peak_local_y + row_start)

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
    principal_angle = _principal_direction_angle(u0, v0, ridge_weights)
    candidate = _find_quadratic_candidate(
        u0,
        v0,
        ridge_weights,
        principal_angle,
    )
    if candidate is None:
        return _fallback_estimate(
            peak_x,
            peak_y,
            ridge_y,
            ridge_x,
            ridge_weights,
            principal_angle,
        )

    theta = np.radians(candidate.angle_degrees)
    vertex_u0 = candidate.vertex_u * np.cos(theta) - candidate.vertex_v * np.sin(theta)
    vertex_v0 = candidate.vertex_u * np.sin(theta) + candidate.vertex_v * np.cos(theta)
    center_x = float(peak_x - vertex_v0)
    center_y = float(peak_y - vertex_u0)
    height, width = source.shape
    vertex_in_image = -0.5 <= center_x <= width - 0.5 and -0.5 <= center_y <= height - 0.5
    reliable = (
        vertex_in_image
        and candidate.r_squared >= 0.35
        and candidate.curvature_excursion >= 1.0
    )
    if not reliable:
        return _fallback_estimate(
            peak_x,
            peak_y,
            ridge_y,
            ridge_x,
            ridge_weights,
            principal_angle,
        )

    return DistributionRegistrationEstimate(
        center_x=int(np.clip(np.rint(center_x), 0, width - 1)),
        center_y=int(np.clip(np.rint(center_y), 0, height - 1)),
        angle_degrees=int(np.rint(candidate.angle_degrees)),
        fit_r_squared=candidate.r_squared,
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


def _find_quadratic_candidate(
    u0: np.ndarray,
    v0: np.ndarray,
    weights: np.ndarray,
    principal_angle: float,
) -> _QuadraticCandidate | None:
    best: _QuadraticCandidate | None = None
    for angle in range(-int(_MAX_ABS_ANGLE), int(_MAX_ABS_ANGLE) + 1):
        theta = np.radians(angle)
        u = u0 * np.cos(theta) + v0 * np.sin(theta)
        v = -u0 * np.sin(theta) + v0 * np.cos(theta)
        coefficients, valid = _robust_quadratic_fit(u, v, weights)
        if coefficients is None or np.count_nonzero(valid) < _MIN_RIDGE_POINTS:
            continue
        c2, c1, c0 = coefficients
        valid_u = u[valid]
        valid_v = v[valid]
        valid_weights = weights[valid]
        span = float(np.ptp(valid_u))
        if span <= 0 or abs(c2) < 1e-10:
            continue
        vertex_u = float(-c1 / (2 * c2))
        vertex_v = float(c2 * vertex_u**2 + c1 * vertex_u + c0)
        predicted = c2 * valid_u**2 + c1 * valid_u + c0
        residual = valid_v - predicted
        weighted_mean = float(np.average(valid_v, weights=valid_weights))
        ss_res = float(np.sum(valid_weights * residual**2))
        ss_tot = float(np.sum(valid_weights * (valid_v - weighted_mean) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(ss_res / np.sum(valid_weights)))
        scale = float(np.sqrt(ss_tot / np.sum(valid_weights))) if ss_tot > 0 else 1.0
        lower = float(np.min(valid_u))
        upper = float(np.max(valid_u))
        outside_span = max(lower - vertex_u, vertex_u - upper, 0.0) / span
        direction_penalty = abs(angle - principal_angle) / (2 * _MAX_ABS_ANGLE)
        score = rmse / max(scale, 1e-6) + 2.0 * outside_span + 0.02 * direction_penalty
        candidate = _QuadraticCandidate(
            angle_degrees=angle,
            vertex_u=vertex_u,
            vertex_v=vertex_v,
            r_squared=float(r_squared),
            curvature_excursion=float(abs(c2) * span**2),
            score=score,
        )
        if best is None or candidate.score < best.score:
            best = candidate
    return best


def _robust_quadratic_fit(
    u: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray]:
    valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(weights) & (weights > 0)
    coefficients: np.ndarray | None = None
    for _ in range(4):
        if np.count_nonzero(valid) < _MIN_RIDGE_POINTS:
            return None, valid
        fit_u = u[valid]
        fit_v = v[valid]
        fit_weights = weights[valid]
        center = float(np.average(fit_u, weights=fit_weights))
        scale = float(np.sqrt(np.average((fit_u - center) ** 2, weights=fit_weights)))
        if scale <= 1e-9:
            return None, valid
        normalized_u = (fit_u - center) / scale
        design = np.column_stack((normalized_u**2, normalized_u, np.ones_like(normalized_u)))
        weighted_design = design * np.sqrt(fit_weights)[:, None]
        weighted_values = fit_v * np.sqrt(fit_weights)
        normalized_coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)
        d2, d1, d0 = normalized_coefficients
        coefficients = np.array(
            [
                d2 / scale**2,
                d1 / scale - 2 * d2 * center / scale**2,
                d0 - d1 * center / scale + d2 * center**2 / scale**2,
            ],
            dtype=np.float64,
        )
        residual = v - np.polyval(coefficients, u)
        center_residual = float(np.median(residual[valid]))
        mad = float(np.median(np.abs(residual[valid] - center_residual)))
        if mad <= 1e-9:
            break
        new_valid = valid & (np.abs(residual - center_residual) <= 3.5 * 1.4826 * mad)
        if np.array_equal(new_valid, valid):
            break
        valid = new_valid
    return coefficients, valid


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
        fit_r_squared=0.0,
        ridge_point_count=int(ridge_y.size),
        used_fallback=True,
    )
