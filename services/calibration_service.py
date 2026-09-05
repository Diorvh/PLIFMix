from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from PLIFv2.algorithms.intensifier import fit_linear_gain_matrix
from PLIFv2.core.image_io import extract_plif_signal, list_image_files, read_image, save_float_image
from PLIFv2.core.paths import path_is_set
from PLIFv2.core.result_snapshot import ResultSnapshot
from PLIFv2.domain.settings import CalibrationSettings


@dataclass
class CalibrationAnalysisResult:
    files: list[Path]
    x_values: np.ndarray
    image_stack: np.ndarray
    k_matrix: np.ndarray
    correction_matrix: np.ndarray
    error_map: np.ndarray
    snapshot: ResultSnapshot | None = None

    def pixel_stats(self, x: int, y: int) -> dict[str, object]:
        values = self.image_stack[y, x, :].astype(np.float64)
        k = float(self.k_matrix[y, x])
        predicted = k * self.x_values
        ss_res = np.sum((values - predicted) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = float(1 - ss_res / ss_tot) if ss_tot else 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            errors = np.divide(predicted - values, values, out=np.zeros_like(values), where=values != 0)
        max_index = int(np.argmax(np.abs(errors)))
        return {
            "values": values,
            "predicted": predicted,
            "k": k,
            "r_squared": r_squared,
            "max_error": float(np.abs(errors[max_index])),
            "max_error_x": float(self.x_values[max_index]),
        }

    def region_average_error(self, radius: int) -> float:
        height, width = self.error_map.shape
        y, x = np.ogrid[:height, :width]
        mask = (x - width / 2) ** 2 + (y - height / 2) ** 2 <= radius**2
        return float(np.nanmean(self.error_map[mask]))


@dataclass
class MatrixComparisonResult:
    file_count: int
    max_relative_ptp: float
    max_index: tuple[int, int]
    relative_ptp: np.ndarray


@dataclass
class StabilityResult:
    frame_count: int
    values: np.ndarray
    average: float
    value_range: float
    relative_range: float
    relative_std: float


class CalibrationService:
    def analyze(self, settings: CalibrationSettings) -> CalibrationAnalysisResult:
        folder = Path(settings.source_folder)
        files = list_image_files(folder, (".png", ".tif", ".tiff", ".bmp", ".npy"))
        if not files:
            raise ValueError(f"No calibration images found: {folder}")
        x_values = np.array([_parse_numeric_name(path.stem) for path in files], dtype=np.float64)
        stack = np.dstack([_single_channel(read_image(path)) for path in files])
        k_matrix, correction = fit_linear_gain_matrix(stack, x_values)
        predicted = k_matrix[..., None] * x_values
        with np.errstate(divide="ignore", invalid="ignore"):
            errors = np.abs(np.divide(predicted - stack, stack, out=np.zeros_like(predicted), where=stack != 0))
        error_map = np.nanmax(np.nan_to_num(errors), axis=2)
        return CalibrationAnalysisResult(files, x_values, stack, k_matrix, correction, error_map)

    def save_correction(self, settings: CalibrationSettings, result: CalibrationAnalysisResult) -> dict[str, Path]:
        output = Path(settings.output_folder) if path_is_set(settings.output_folder) else Path(settings.source_folder).parent
        return save_float_image(result.correction_matrix, output, Path(settings.source_folder).name)

    def compare_matrices(self, settings: CalibrationSettings) -> MatrixComparisonResult:
        folder = Path(settings.compare_folder)
        names = [name.strip() for name in settings.compare_files.split(",") if name.strip()]
        files = [folder / name for name in names] if names else sorted(folder.glob("*.npy"))
        if len(files) < 2:
            raise ValueError("Select at least two correction matrices.")
        matrices = [np.load(path).astype(np.float64) for path in files]
        if len({matrix.shape for matrix in matrices}) != 1:
            raise ValueError("All matrices must have the same shape.")
        stack = np.stack(matrices)
        maximum = np.max(stack, axis=0)
        minimum = np.min(stack, axis=0)
        y, x = np.ogrid[:maximum.shape[0], :maximum.shape[1]]
        mask = (x - settings.compare_center_x) ** 2 + (y - settings.compare_center_y) ** 2 <= settings.compare_radius**2
        relative = np.zeros_like(maximum)
        np.divide(maximum - minimum, maximum, out=relative, where=(maximum != 0) & mask)
        index = tuple(int(value) for value in np.unravel_index(np.argmax(relative), relative.shape))
        return MatrixComparisonResult(len(files), float(np.max(relative)), index, relative)

    def single_image_matrix(self, settings: CalibrationSettings) -> dict[str, Path]:
        image_path = Path(settings.single_image_path)
        image = _single_channel(read_image(image_path)).astype(np.float64)
        maximum = float(np.max(image))
        if maximum <= 0:
            raise ValueError("The selected image has no positive signal.")
        normalized = image / maximum
        correction = np.divide(1, normalized, out=np.zeros_like(normalized), where=normalized > 0)
        output = Path(settings.output_folder) if path_is_set(settings.output_folder) else image_path.parent
        return save_float_image(correction, output, f"{image_path.stem}_correct_matrix")

    def pixel_stability(self, folder: Path, x: int, y: int, start: int = 0, end: int | None = None) -> StabilityResult:
        files = list_image_files(folder, (".png", ".tif", ".tiff", ".bmp", ".npy"))[start:end]
        values = []
        for path in files:
            image = _single_channel(read_image(path))
            values.append(float(image[y, x]))
        array = np.array(values)
        average = float(np.mean(array)) if array.size else 0.0
        value_range = float(np.ptp(array)) if array.size else 0.0
        std = float(np.std(array)) if array.size else 0.0
        return StabilityResult(len(files), array, average, value_range, value_range / average if average else 0.0, std / average if average else 0.0)


def _single_channel(image: np.ndarray) -> np.ndarray:
    return extract_plif_signal(image)


def _parse_numeric_name(stem: str) -> int:
    cleaned = stem[:-2] if stem.lower().endswith("ns") else stem
    digits = "".join(character for character in cleaned if character.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse calibration x value from: {stem}")
    return int(digits)
