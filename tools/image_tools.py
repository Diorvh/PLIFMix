from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from PLIFv2.algorithms.visualization import blacked_jet_colormap
from PLIFv2.core.image_io import extract_plif_signal, list_image_files, read_image, write_tif
from PLIFv2.core.safe_io import atomic_save_npy, atomic_write


def export_gif(
    folder: Path,
    output_path: Path,
    start: int = 0,
    end: int | None = None,
    gain: float = 1,
    duration_ms: int = 50,
    max_width: int = 400,
) -> Path:
    files = _select_gif_files(folder, start, end)
    if not files:
        raise ValueError("No frames selected for GIF export.")
    images: list[Image.Image] = []
    for path in files:
        array = extract_plif_signal(read_image(path))
        if array.dtype == np.uint16:
            array = array / 256
        array = np.clip(array * gain, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(array, blacked_jet_colormap())
        image = Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
        if image.width > max_width:
            ratio = max_width / image.width
            image = image.resize((max_width, int(image.height * ratio)), Image.Resampling.LANCZOS)
        images.append(image)
    try:
        return atomic_write(
            output_path,
            lambda temporary: images[0].save(
                temporary,
                format="GIF",
                save_all=True,
                append_images=images[1:],
                duration=duration_ms,
                loop=0,
            ),
            "export GIF",
        )
    finally:
        for image in images:
            image.close()


def _select_gif_files(folder: Path, start: int, end: int | None) -> list[Path]:
    files = list_image_files(folder, (".png", ".tif", ".tiff", ".bmp", ".npy"))
    if not files:
        return []
    if start < 0 or (end is not None and end < start):
        raise ValueError("GIF编号范围无效")

    numbered: dict[int, Path] = {}
    for path in files:
        if path.stem.isdigit():
            numbered.setdefault(int(path.stem), path)
    if numbered:
        last = max(numbered) if end is None else end
        return [numbered[number] for number in sorted(numbered) if start <= number <= last]

    # Keep positional slicing for legacy folders whose filenames are not numbers.
    return files[start:] if end is None else files[start:end + 1]


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
    return atomic_save_npy(output_path, result)
