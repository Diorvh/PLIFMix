from __future__ import annotations

from pathlib import Path

import numpy as np

from PLIFv2.core.image_io import read_image, save_float_image


def apply_correction_matrix(
    image_path: str | Path,
    matrix: np.ndarray,
    output_folder: str | Path,
    operation: str = "multiply",
) -> dict[str, Path]:
    image = read_image(image_path).astype(np.float64)
    if operation == "multiply":
        corrected = image * matrix
    elif operation == "subtract":
        corrected = image - matrix
    else:
        raise ValueError(f"Unsupported correction operation: {operation}")
    corrected = np.clip(corrected, 0, None)
    return save_float_image(corrected, output_folder, Path(image_path).stem)


def fit_linear_gain_matrix(image_stack: np.ndarray, x_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x_values.astype(np.float64)
    xtx_inv = np.linalg.inv(x.reshape(-1, 1).T @ x.reshape(-1, 1))
    sum_xy = np.tensordot(image_stack.astype(np.float64), x, axes=([2], [0]))
    k_matrix = xtx_inv[0, 0] * sum_xy
    norm = k_matrix / np.nanmax(k_matrix)
    correction = np.where(norm > 0, 1 / norm, 0)
    return k_matrix, correction

