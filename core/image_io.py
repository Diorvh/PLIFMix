from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def read_images(paths: Iterable[str | Path]) -> list[np.ndarray]:
    return [read_image(path) for path in paths]


def write_tif(path: str | Path, image: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".tif", image)
    if not ok:
        raise ValueError(f"Failed to encode tif: {path}")
    encoded.tofile(str(path))
    return path


def save_float_image(image: np.ndarray, folder: str | Path, name: str) -> dict[str, Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    npy_path = folder / f"{name}.npy"
    np.save(npy_path, image)

    tif_path = folder / f"{name}.tif"
    clipped = np.clip(image, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    write_tif(tif_path, clipped)

    max_value = float(np.nanmax(image)) if image.size else 0
    if max_value > 0:
        gain = int(65535 / max_value)
        if gain == 0:
            gain = round(65535 / max_value, 1)
        gain_path = folder / f"{name}-GAIN{gain}.tif"
        write_tif(gain_path, np.clip(image * gain, 0, 65535).astype(np.uint16))
    else:
        gain_path = tif_path
    return {"npy": npy_path, "tif": tif_path, "gain_tif": gain_path}


def to_float_single_channel(
    image: np.ndarray,
    green_channel: bool = True,
    zero_threshold: float | None = None,
) -> np.ndarray:
    source_dtype = image.dtype
    if image.ndim == 3:
        if green_channel:
            image = image[:, :, 1]
        else:
            image = np.dot(image[:, :, :3], [0.114, 0.587, 0.299])
    image = image.astype(np.float64)
    if zero_threshold is not None:
        image[image <= zero_threshold] = 0
    if source_dtype == np.uint8:
        image *= 256
    return image


def list_image_files(folder: str | Path, suffixes: tuple[str, ...]) -> list[Path]:
    folder = Path(folder)
    suffix_set = {suffix.lower() for suffix in suffixes}
    files = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in suffix_set]
    return sorted(files, key=_natural_key)


def _natural_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    return (int(stem), stem) if stem.isdigit() else (10**12, stem)
