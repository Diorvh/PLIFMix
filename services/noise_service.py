from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from PLIFv2.core.image_io import list_image_files, read_image
from PLIFv2.domain.settings import NoiseSettings


@dataclass
class NoiseAnalysisResult:
    frame_indices: np.ndarray
    max_values: np.ndarray

    @property
    def maximum(self) -> float:
        return float(np.max(self.max_values)) if self.max_values.size else 0.0

    @property
    def average(self) -> float:
        return float(np.mean(self.max_values)) if self.max_values.size else 0.0


class NoiseAnalysisService:
    def run(self, settings: NoiseSettings) -> NoiseAnalysisResult:
        files = list_image_files(settings.source_folder, (".png", ".tif", ".tiff", ".bmp", ".npy"))
        selected = files[settings.frame_start : settings.frame_end]
        if not selected:
            raise ValueError("No frames selected for noise analysis.")
        workers = min(max(1, settings.process_count), len(selected))
        if workers == 1:
            values = [_image_max(path) for path in selected]
        else:
            with Pool(workers) as pool:
                values = pool.map(_image_max, selected)
        indices = np.arange(settings.frame_start, settings.frame_start + len(selected))
        return NoiseAnalysisResult(indices, np.array(values))


def _image_max(path: Path) -> float:
    image = read_image(path).astype(np.float64)
    if image.ndim == 3:
        image = np.dot(image[:, :, :3], [0.114, 0.587, 0.299])
    return float(np.max(image))
