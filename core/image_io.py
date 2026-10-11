from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from PLIFMix.core.safe_io import atomic_save_npy, atomic_write_bytes


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
    return atomic_write_bytes(path, encoded.tobytes(), "write TIF image")


def write_png(path: str | Path, image: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Failed to encode png: {path}")
    return atomic_write_bytes(path, encoded.tobytes(), "write PNG image")


def save_float_image(image: np.ndarray, folder: str | Path, name: str) -> dict[str, Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    npy_path = folder / f"{name}.npy"
    atomic_save_npy(npy_path, image)

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


def extract_plif_signal(image: np.ndarray) -> np.ndarray:
    """Return the PLIF signal plane without changing its numeric scale."""
    return image[:, :, 1] if image.ndim == 3 else image


def to_16bit_plif_signal(image: np.ndarray) -> np.ndarray:
    """Return the green PLIF signal in the viewer's 16-bit value domain."""
    signal = extract_plif_signal(image)
    if signal.dtype == np.uint8:
        return signal.astype(np.uint16) * 256
    if signal.dtype == np.uint16:
        return signal
    return signal.astype(np.float64)


def to_float_single_channel(
    image: np.ndarray,
    green_channel: bool = True,
    zero_threshold: float | None = None,
) -> np.ndarray:
    source_dtype = image.dtype
    if image.ndim == 3:
        image = extract_plif_signal(image) if green_channel else np.dot(image[:, :, :3], [0.114, 0.587, 0.299])
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
