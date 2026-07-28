from __future__ import annotations

from pathlib import Path

import numpy as np

from PLIFv2.core.image_io import read_image, save_float_image


def apply_correction_matrix(
    image_path: str | Path,
    matrix: np.ndarray,
    output_folder: str | Path,
) -> dict[str, Path]:
    image = read_image(image_path).astype(np.float64)
    if image.shape != matrix.shape:
        raise ValueError(
            f"图像与修正矩阵尺寸不一致: image={image.shape}, matrix={matrix.shape}, file={Path(image_path).name}"
        )
    corrected = image * matrix
    corrected = np.clip(corrected, 0, None)
    return save_float_image(corrected, output_folder, Path(image_path).stem)


def fit_linear_gain_matrix(image_stack: np.ndarray, x_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x_values.astype(np.float64)
    if image_stack.ndim != 3 or image_stack.shape[2] != x.size:
        raise ValueError(
            f"标定图数量与自变量数量不一致: images={image_stack.shape}, x_values={x.size}"
        )
    denominator = float(np.dot(x, x))
    if denominator <= 0:
        raise ValueError("标定自变量平方和必须大于 0。")
    sum_xy = np.tensordot(image_stack.astype(np.float64), x, axes=([2], [0]))
    k_matrix = sum_xy / denominator
    maximum = float(np.nanmax(k_matrix))
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("标定图拟合结果没有有效的正斜率。")
    norm = k_matrix / maximum
    correction = np.divide(1, norm, out=np.zeros_like(norm), where=norm > 0)
    return k_matrix, correction
