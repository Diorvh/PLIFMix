from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Iterable

import cv2
import numpy as np


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

    height, width = image.shape
    boundary_points: set[tuple[int, int]] = set()
    for y in range(height):
        for x in range(width):
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if (image[y, x] - contour_value) * (image[ny, nx] - contour_value) < 0:
                        boundary_points.add((y, x))
                        boundary_points.add((ny, nx))
    return contour_value, boundary_points


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
    max_k = max(1, max_k)
    if up_limit > down_limit:
        empty = (np.array([], dtype=np.int32), np.array([], dtype=np.int32))
        return 0.0, 0.0, 0.0, empty, empty, np.array([])

    source_rows = image[up_limit : down_limit + 1]
    top_k_indices = np.argsort(source_rows, axis=1)[:, -max_k:]
    top_k_values = np.take_along_axis(source_rows, top_k_indices, axis=1)
    valid_rows = np.all(top_k_values != 0, axis=1)
    py_arr = np.arange(up_limit, down_limit + 1, dtype=np.int32)[valid_rows]
    px_arr = np.mean(top_k_indices[valid_rows], axis=1)
    if py_arr.size == 0:
        empty = (np.array([], dtype=np.int32), np.array([], dtype=np.int32))
        return 0.0, 0.0, 0.0, empty, empty, np.array([])

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
    """Shift every image row to the fitted centre line using the full extrapolated curve."""
    rows, columns = fitted_point
    height, width = image.shape
    if len(rows) == 0:
        return np.zeros_like(image)

    row_sums = np.bincount(rows, weights=columns, minlength=height)
    row_counts = np.bincount(rows, minlength=height)
    known_rows = np.flatnonzero(row_counts)
    centre_columns = np.full(height, center_x, dtype=np.float64)
    centre_columns[known_rows] = row_sums[known_rows] / row_counts[known_rows]
    if len(known_rows) > 1:
        centre_columns = np.interp(np.arange(height), known_rows, centre_columns[known_rows])

    destination_x = np.broadcast_to(np.arange(width), (height, width))
    source_x = np.rint(destination_x - (center_x - centre_columns[:, None])).astype(np.int32)
    valid = (0 <= source_x) & (source_x < width)
    row_indices = np.broadcast_to(np.arange(height)[:, None], source_x.shape)
    straightened = np.zeros_like(image)
    straightened[valid] = image[row_indices[valid], source_x[valid]]
    return straightened


def func_fit_2(x: np.ndarray, y: np.ndarray, fit_function):
    try:
        from scipy.optimize import curve_fit

        popt, _ = curve_fit(fit_function, x, y, p0=[1, 1])
        a_fit, b_fit = popt
    except Exception:
        a_fit, b_fit = 0.0, 0.0
    y_fit = fit_function(x, a_fit, b_fit)
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return float(a_fit), float(b_fit), float(r_squared), y_fit


def func_fit_3(x: np.ndarray, y: np.ndarray, fit_function):
    mask = y > 0
    try:
        from scipy.optimize import curve_fit

        popt, _ = curve_fit(fit_function, x[mask], y[mask], p0=[1, 1, 1])
        a_fit, b_fit, c_fit = popt
    except Exception:
        a_fit, b_fit, c_fit = 0.0, 0.0, 0.0
    y_fit = fit_function(x, a_fit, b_fit, c_fit)
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
    tasks = [(f_shift, o_shifts, images, coords_me, global_mix_ratio) for f_shift in f_shifts]
    if workers <= 1:
        rows = [_process_f_shift(task) for task in tasks]
    else:
        with Pool(processes=workers) as pool:
            rows = pool.map(_process_f_shift, tasks)

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
