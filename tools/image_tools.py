from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from PLIFv2.core.image_io import read_image, write_tif


def export_gif(
    folder: Path,
    output_path: Path,
    start: int = 0,
    end: int | None = None,
    gain: float = 1,
    duration_ms: int = 50,
    max_width: int = 400,
) -> Path:
    files = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".tif", ".tiff", ".bmp", ".npy"})[start:end]
    if not files:
        raise ValueError("No frames selected for GIF export.")
    images: list[Image.Image] = []
    for path in files:
        array = read_image(path)
        if array.ndim == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        if array.dtype == np.uint16:
            array = array / 256
        array = np.clip(array * gain, 0, 255).astype(np.uint8)
        image = Image.fromarray(array)
        if image.width > max_width:
            ratio = max_width / image.width
            image = image.resize((max_width, int(image.height * ratio)), Image.Resampling.LANCZOS)
        images.append(image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    return output_path


def create_colorbar(
    height: int,
    width: int,
    scale_min: float,
    scale_max: float,
    padding_top: int = 20,
    padding_bottom: int = 20,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    inner_height = height - padding_top - padding_bottom
    gradient = np.linspace(255, 0, inner_height, dtype=np.uint8)
    gradient = np.tile(gradient, (width, 1)).T
    bar = cv2.applyColorMap(gradient, colormap)
    padded = np.full((height, width, 3), 255, dtype=np.uint8)
    padded[padding_top : padding_top + inner_height] = bar
    output = np.full((height, width + 130, 3), 255, dtype=np.uint8)
    output[:, :width] = padded
    for index in range(6):
        y = int(padding_top + inner_height - index * inner_height / 5)
        cv2.line(output, (width, y), (width + 6, y), (0, 0, 0), 2)
        label = scale_min + (scale_max - scale_min) * index / 5
        cv2.putText(output, f"{label:.2g}", (width + 12, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return output


def save_colorbar(path: Path, image: np.ndarray) -> Path:
    return write_tif(path, image)


def top_k_values(array: np.ndarray, k: int) -> list[tuple[float, tuple[int, ...]]]:
    flat = array.ravel()
    k = min(max(1, k), flat.size)
    indices = np.argpartition(flat, -k)[-k:]
    indices = indices[np.argsort(flat[indices])[::-1]]
    return [(float(flat[index]), tuple(int(value) for value in np.unravel_index(index, array.shape))) for index in indices]


def transform_npy(input_path: Path, output_path: Path, operation: str, value: float = 1) -> Path:
    array = np.load(input_path).astype(np.float64)
    if operation == "multiply":
        result = array * value
    elif operation == "add":
        result = array + value
    elif operation == "normalize":
        maximum = float(np.max(array))
        result = array / maximum if maximum else array
    elif operation == "reciprocal":
        result = np.divide(1, array, out=np.zeros_like(array), where=array != 0)
    else:
        raise ValueError(f"Unsupported NPY operation: {operation}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, result)
    return output_path
