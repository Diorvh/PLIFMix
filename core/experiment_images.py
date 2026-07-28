from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from PLIFv2.core.image_io import read_image


@dataclass(frozen=True)
class ExperimentImageTriplet:
    f: np.ndarray
    o: np.ndarray
    fo: np.ndarray

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.f, self.o, self.fo


def load_experiment_triplet(
    folder: str | Path,
    f_exp_no: str,
    o_exp_no: str,
    fo_exp_no: str,
) -> ExperimentImageTriplet:
    root = Path(folder)
    images = tuple(
        _load_scalar_image(experiment_npy_path(root, number), label)
        for number, label in ((f_exp_no, "F"), (o_exp_no, "O"), (fo_exp_no, "FO"))
    )
    shapes = {image.shape for image in images}
    if len(shapes) != 1:
        shape_text = ", ".join(f"{label}={image.shape}" for label, image in zip(("F", "O", "FO"), images))
        raise ValueError(f"F、O、FO 图像尺寸不一致: {shape_text}")
    return ExperimentImageTriplet(*images)


def experiment_npy_path(folder: str | Path, experiment_number: str) -> Path:
    number = str(experiment_number).strip()
    if not number:
        raise ValueError("需要输入 F、O 和 FO 图像编号。")
    stem = number if number.startswith("实验编号") else f"实验编号{number}"
    return Path(folder) / f"{stem}.npy"


def _load_scalar_image(path: Path, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{label} 图像不存在: {path}")
    image = read_image(path)
    if image is None:
        raise ValueError(f"{label} 图像无法读取: {path}")
    if image.ndim != 2:
        raise ValueError(f"{label} 图像必须是二维 NPY 矩阵，实际形状为 {image.shape}: {path}")
    image = image.astype(np.float64, copy=False)
    if not np.all(np.isfinite(image)):
        raise ValueError(f"{label} 图像包含 NaN 或无穷值: {path}")
    return image
