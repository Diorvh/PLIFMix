from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Iterable

import cv2
import numpy as np


class FitConvergenceError(RuntimeError):
    pass


@dataclass
class ShiftSearchResult:
    f_shifts: list[int]
    o_shifts: list[int]
    abs_matrix: np.ndarray
    me_ptp_matrix: np.ndarray
    me1_matrix: np.ndarray
    me2_matrix: np.ndarray
    me3_matrix: np.ndarray

    @property
    def best_abs(self) -> tuple[int, int, float]:
        row, col = np.unravel_index(self.abs_matrix.argmin(), self.abs_matrix.shape)
        return self.f_shifts[row], self.o_shifts[col], float(self.abs_matrix[row, col])

    @property
    def best_me_ptp(self) -> tuple[int, int, float]:
        row, col = np.unravel_index(self.me_ptp_matrix.argmin(), self.me_ptp_matrix.shape)
        return self.f_shifts[row], self.o_shifts[col], float(self.me_ptp_matrix[row, col])


def shift_image(image: np.ndarray, shift: int) -> np.ndarray:
    shifted = np.zeros_like(image)
    if shift > 0:
        shifted[:, :-shift] = image[:, shift:]
    elif shift < 0:
        shifted[:, -shift:] = image[:, :shift]
    else:
        shifted[:, :] = image[:, :]
    return shifted


def generate_mask_coords(cx: int, cy: int, major: int, minor: int, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (major, minor), 0, 0, 360, (255,), -1)
    ys, xs = np.where(mask > 0)
    return xs, ys


def calc_flowrate_contour(image: np.ndarray, thres_contour: float) -> tuple[float, set[tuple[int, int]]]:
    values = image.flatten()
    sorted_values = np.sort(values)[::-1]
    prefix_sum = np.cumsum(sorted_values)
    if prefix_sum.size == 0 or prefix_sum[-1] <= 0:
        return 0.0, set()
    index = np.searchsorted(prefix_sum, thres_contour * prefix_sum[-1])
    index = min(index, len(sorted_values) - 1)
    contour_value = float(sorted_values[index])

    above = image > contour_value
    below = image < contour_value
    boundary = np.zeros(image.shape, dtype=bool)

    horizontal_crossing = (above[:, :-1] & below[:, 1:]) | (below[:, :-1] & above[:, 1:])
    boundary[:, :-1] |= horizontal_crossing
    boundary[:, 1:] |= horizontal_crossing

    vertical_crossing = (above[:-1, :] & below[1:, :]) | (below[:-1, :] & above[1:, :])
    boundary[:-1, :] |= vertical_crossing
    boundary[1:, :] |= vertical_crossing

    rows, columns = np.nonzero(boundary)
    return contour_value, set(zip(rows.tolist(), columns.tolist()))


def rotate_points_clockwise(x: np.ndarray, y: np.ndarray, angle_degrees: float) -> tuple[np.ndarray, np.ndarray]:
    theta = np.radians(angle_degrees)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return x * cos_theta + y * sin_theta, -x * sin_theta + y * cos_theta


def fit_center_line(
    image: np.ndarray,
    up_limit: int,
    down_limit: int,
    max_k: int,
    cx: int,
    cy: int,
    rotate_angle: float,
) -> tuple[float, float, float, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], np.ndarray]:
    up_limit = max(0, up_limit)
    down_limit = min(image.shape[0] - 1, down_limit)
    max_k = min(max(1, max_k), image.shape[1])
    if up_limit > down_limit:
        raise FitConvergenceError("中心线上界必须小于或等于下界。")

    source_rows = image[up_limit : down_limit + 1]
    partition_index = source_rows.shape[1] - max_k
    top_k_indices = np.argpartition(source_rows, partition_index, axis=1)[:, -max_k:]
    top_k_values = np.take_along_axis(source_rows, top_k_indices, axis=1)
    valid_rows = np.all(top_k_values != 0, axis=1)
    py_arr = np.arange(up_limit, down_limit + 1, dtype=np.int32)[valid_rows]
    px_arr = np.mean(top_k_indices[valid_rows], axis=1)
    if py_arr.size < 3:
        raise FitConvergenceError(f"中心线有效数据行不足，至少需要 3 行，当前为 {py_arr.size} 行。")

    point_x = -(py_arr - cy)
    point_y = -(px_arr - cx)
    point_x, point_y = rotate_points_clockwise(point_x, point_y, rotate_angle)

    def fit_function(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * np.sqrt(x**2 + b**2) - a * b

    a, b, r_squared, y_pred = func_fit_2(point_x, point_y, fit_function)
    origin = (py_arr.astype(np.int32), np.round(px_arr).astype(np.int32))
    fitted = extrapolate_fitted_center_line(image.shape, a, b, cx, cy, rotate_angle)
    return a, b, r_squared, origin, fitted, point_x


def extrapolate_fitted_center_line(
    shape: tuple[int, int],
    a: float,
    b: float,
    cx: int,
    cy: int,
    rotate_angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the fitted centre line over the full image rather than only its source rows."""
    height, width = shape
    extent = int(np.ceil(np.hypot(height, width)))
    sample_count = max(height, width) * 4 + 1
    fit_axis = np.linspace(-extent, extent, sample_count)
    fit_offset = a * np.sqrt(fit_axis**2 + b**2) - a * b
    vertical_offset, horizontal_offset = rotate_points_clockwise(fit_axis, fit_offset, -rotate_angle)
    rows = np.rint(cy - vertical_offset).astype(np.int32)
    columns = np.rint(cx - horizontal_offset).astype(np.int32)
    valid = (0 <= rows) & (rows < height) & (0 <= columns) & (columns < width)
    if not np.any(valid):
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    rows = rows[valid]
    columns = columns[valid]
    encoded = rows.astype(np.int64) * width + columns
    _, unique_indices = np.unique(encoded, return_index=True)
    return rows[unique_indices], columns[unique_indices]


def straighten_center_line_image(
    image: np.ndarray,
    fitted_point: tuple[np.ndarray, np.ndarray],
    center_x: int,
) -> np.ndarray:
    """Straighten rows with conservative subpixel redistribution."""
    rows, columns = fitted_point
    height, width = image.shape
    if len(rows) == 0:
        return np.zeros(image.shape, dtype=np.float64)

    row_sums = np.bincount(rows, weights=columns, minlength=height)
    row_counts = np.bincount(rows, minlength=height)
    known_rows = np.flatnonzero(row_counts)
    centre_columns = np.full(height, center_x, dtype=np.float64)
    centre_columns[known_rows] = row_sums[known_rows] / row_counts[known_rows]
    if len(known_rows) > 1:
        centre_columns = np.interp(np.arange(height), known_rows, centre_columns[known_rows])

    source = np.asarray(image, dtype=np.float64)
    straightened = np.zeros(source.shape, dtype=np.float64)
    shifts = center_x - centre_columns
    for row_index, shift in enumerate(shifts):
        integer_shift = int(np.floor(shift))
        fraction = float(shift - integer_shift)
        _redistribute_shifted_row(
            source[row_index],
            straightened[row_index],
            integer_shift,
            1.0 - fraction,
            row_index,
        )
        _redistribute_shifted_row(
            source[row_index],
            straightened[row_index],
            integer_shift + 1,
            fraction,
            row_index,
        )

    source_total = float(np.sum(source))
    straightened_total = float(np.sum(straightened))
    absolute_total = float(np.sum(np.abs(source)))
    tolerance = np.finfo(np.float64).eps * max(1.0, absolute_total) * 32
    if not np.isclose(straightened_total, source_total, rtol=0.0, atol=tolerance):
        raise RuntimeError(
            "亚像素掰直后强度和发生变化："
            f"原图={source_total:.16g}，掰直后={straightened_total:.16g}"
        )
    return straightened


def _redistribute_shifted_row(
    source: np.ndarray,
    destination: np.ndarray,
    offset: int,
    weight: float,
    row_index: int,
) -> None:
    if weight <= np.finfo(np.float64).eps:
        return
    width = source.size
    source_start = max(0, -offset)
    source_stop = min(width, width - offset)
    if source_start >= source_stop:
        if np.any(source != 0):
            raise ValueError(f"第 {row_index} 行亚像素平移会使非零强度移出图像边界。")
        return
    if (
        (source_start > 0 and np.any(source[:source_start] != 0))
        or (source_stop < width and np.any(source[source_stop:] != 0))
    ):
        raise ValueError(f"第 {row_index} 行亚像素平移会使非零强度移出图像边界。")
    destination_start = source_start + offset
    destination_stop = source_stop + offset
    destination[destination_start:destination_stop] += source[source_start:source_stop] * weight


def func_fit_2(x: np.ndarray, y: np.ndarray, fit_function):
    """Fit two parameters with a 2-pixel Huber transition for ridge outliers."""
    valid = np.isfinite(x) & np.isfinite(y)
    fit_x = np.asarray(x, dtype=np.float64)[valid]
    fit_y = np.asarray(y, dtype=np.float64)[valid]
    if fit_x.size < 3:
        raise FitConvergenceError("二参数拟合至少需要 3 个有效数据点。")
    try:
        from scipy.optimize import least_squares

        initial_b = max(float(np.std(fit_x)), 1.0)
        fit_result = least_squares(
            lambda parameters: fit_function(fit_x, *parameters) - fit_y,
            x0=[0.1, initial_b],
            bounds=([-np.inf, 1e-9], [np.inf, np.inf]),
            loss="huber",
            f_scale=2.0,
            max_nfev=20000,
        )
        if not fit_result.success or not np.all(np.isfinite(fit_result.x)):
            raise RuntimeError(fit_result.message)
        a_fit, b_fit = fit_result.x
    except Exception as exc:
        raise FitConvergenceError(f"二参数 Huber 曲线拟合未收敛: {exc}") from exc
    y_fit = fit_function(x, a_fit, b_fit)
    if not np.all(np.isfinite(y_fit)):
        raise FitConvergenceError("二参数拟合产生了非有限预测值。")
    valid_prediction = np.asarray(y_fit)[valid]
    ss_res = np.sum((fit_y - valid_prediction) ** 2)
    ss_tot = np.sum((fit_y - np.mean(fit_y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return float(a_fit), float(b_fit), float(r_squared), y_fit


def func_fit_3(x: np.ndarray, y: np.ndarray, fit_function):
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_values) & np.isfinite(y_values) & (y_values > 0)
    if np.count_nonzero(mask) < 4:
        raise FitConvergenceError("三参数拟合至少需要 4 个正的有效数据点。")
    try:
        from scipy.optimize import curve_fit

        fit_x = x_values[mask]
        fit_y = y_values[mask]
        initial_b = 1.0 / max(float(np.std(fit_x)), 1.0)
        popt, _ = curve_fit(
            fit_function,
            fit_x,
            fit_y,
            p0=[float(np.max(fit_y)), initial_b, 1.0],
            bounds=([0.0, 1e-12, 1e-9], [np.inf, np.inf, np.inf]),
            maxfev=30000,
        )
        a_fit, b_fit, c_fit = popt
    except Exception as exc:
        raise FitConvergenceError(f"三参数曲线拟合未收敛: {exc}") from exc
    y_fit = fit_function(x, a_fit, b_fit, c_fit)
    if not np.all(np.isfinite(y_fit)):
        raise FitConvergenceError("三参数拟合产生了非有限预测值。")
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return float(a_fit), float(b_fit), float(c_fit), float(r_squared), y_fit


def calc_mixing_efficiency(source_values: np.ndarray, total_values: np.ndarray, global_mix_ratio: float) -> tuple[float, np.ndarray]:
    valid = total_values > 0
    mass_ratio = np.zeros_like(total_values, dtype=np.float64)
    mixing_diff = np.zeros_like(total_values, dtype=np.float64)
    total_sum = np.sum(total_values)
    if total_sum <= 0:
        return 0.0, mixing_diff
    mass_ratio[valid] = total_values[valid] / total_sum
    mixing_diff[valid] = global_mix_ratio - source_values[valid] / total_values[valid]
    denom = np.where(mixing_diff > 0, global_mix_ratio, global_mix_ratio - 1)
    result_term = np.divide(mixing_diff, denom, out=np.zeros_like(mixing_diff), where=denom != 0) * mass_ratio
    return float(1 - np.sum(result_term)), result_term


def calc_result(
    f_img: np.ndarray,
    o_img: np.ndarray,
    fo_img: np.ndarray,
    coords_me: tuple[np.ndarray, np.ndarray],
    global_mix_ratio: float,
) -> tuple[float, float, float, float]:
    xs, ys = coords_me
    f_values = f_img[ys, xs].astype(np.float64)
    o_values = o_img[ys, xs].astype(np.float64)
    fo_values = fo_img[ys, xs].astype(np.float64)
    me1, _ = calc_mixing_efficiency(o_values, fo_values, global_mix_ratio)
    me2, _ = calc_mixing_efficiency(f_values, fo_values, 1 - global_mix_ratio)
    me3, _ = calc_mixing_efficiency(o_values, f_values + o_values, global_mix_ratio)
    diff_sum_abs = float(np.sum(np.abs(f_values + o_values - fo_values)))
    return diff_sum_abs, me1, me2, me3


def calculate_shift_search(
    images: tuple[np.ndarray, np.ndarray, np.ndarray],
    f_range: tuple[int, int],
    o_range: tuple[int, int],
    ellipse: tuple[int, int, int, int],
    global_mix_ratio: float,
    process_count: int | None = None,
) -> ShiftSearchResult:
    coords_me = generate_mask_coords(*ellipse, images[0].shape)
    f_shifts = list(range(*f_range))
    o_shifts = list(range(*o_range))
    workers = min(process_count or cpu_count(), max(1, len(f_shifts)))
    if workers <= 1:
        tasks = [(f_shift, o_shifts, images, coords_me, global_mix_ratio) for f_shift in f_shifts]
        rows = [_process_f_shift(task) for task in tasks]
    else:
        with Pool(
            processes=workers,
            initializer=_initialize_shift_worker,
            initargs=(images, coords_me, global_mix_ratio),
        ) as pool:
            rows = pool.map(_process_cached_f_shift, [(f_shift, o_shifts) for f_shift in f_shifts])

    abs_matrix = np.zeros((len(f_shifts), len(o_shifts)))
    me1_matrix = np.zeros_like(abs_matrix)
    me2_matrix = np.zeros_like(abs_matrix)
    me3_matrix = np.zeros_like(abs_matrix)
    for i, row in enumerate(rows):
        for j, (abs_val, me1, me2, me3) in enumerate(row):
            abs_matrix[i, j] = abs_val
            me1_matrix[i, j] = me1
            me2_matrix[i, j] = me2
            me3_matrix[i, j] = me3
    me_ptp_matrix = np.ptp(np.stack([me1_matrix, me2_matrix, me3_matrix], axis=2), axis=2)
    return ShiftSearchResult(f_shifts, o_shifts, abs_matrix, me_ptp_matrix, me1_matrix, me2_matrix, me3_matrix)


def _process_f_shift(args: tuple[int, Iterable[int], tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], float]):
    f_shift, o_shifts, images, coords_me, global_mix_ratio = args
    f_img, o_img, fo_img = images
    shifted_f = shift_image(f_img, f_shift)
    return [
        calc_result(shifted_f, shift_image(o_img, o_shift), fo_img, coords_me, global_mix_ratio)
        for o_shift in o_shifts
    ]


_SHIFT_WORKER_IMAGES: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
_SHIFT_WORKER_COORDS: tuple[np.ndarray, np.ndarray] | None = None
_SHIFT_WORKER_RATIO = 0.0


def _initialize_shift_worker(
    images: tuple[np.ndarray, np.ndarray, np.ndarray],
    coords_me: tuple[np.ndarray, np.ndarray],
    global_mix_ratio: float,
) -> None:
    global _SHIFT_WORKER_IMAGES, _SHIFT_WORKER_COORDS, _SHIFT_WORKER_RATIO
    _SHIFT_WORKER_IMAGES = images
    _SHIFT_WORKER_COORDS = coords_me
    _SHIFT_WORKER_RATIO = global_mix_ratio


def _process_cached_f_shift(args: tuple[int, Iterable[int]]):
    if _SHIFT_WORKER_IMAGES is None or _SHIFT_WORKER_COORDS is None:
        raise RuntimeError("平移搜索子进程尚未初始化。")
    f_shift, o_shifts = args
    return _process_f_shift(
        (f_shift, o_shifts, _SHIFT_WORKER_IMAGES, _SHIFT_WORKER_COORDS, _SHIFT_WORKER_RATIO)
    )
