from __future__ import annotations

from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Sequence

import numpy as np

from PLIFv2.core.image_io import read_image, to_float_single_channel


def average_image_chunk(args: tuple[list[Path], float]) -> tuple[np.ndarray | None, int]:
    paths, denoise_threshold = args
    accum = None
    count = 0
    for path in paths:
        image = to_float_single_channel(
            read_image(path),
            green_channel=True,
            zero_threshold=denoise_threshold,
        )
        accum = image if accum is None else accum + image
        count += 1
    return accum, count


def average_images(
    paths: Sequence[Path],
    process_count: int | None = None,
    denoise_threshold: float = 2,
) -> np.ndarray:
    if not paths:
        raise ValueError("No image files selected for averaging.")
    workers = min(process_count or cpu_count(), len(paths))
    if workers <= 1:
        accum, count = average_image_chunk((list(paths), denoise_threshold))
        if accum is None or count == 0:
            raise ValueError("No readable images were averaged.")
        return accum / count

    chunk_size = max(1, len(paths) // workers + 1)
    chunks = [list(paths[i : i + chunk_size]) for i in range(0, len(paths), chunk_size)]
    with Pool(processes=workers) as pool:
        results = pool.map(
            average_image_chunk,
            [(chunk, denoise_threshold) for chunk in chunks],
        )

    total = None
    count = 0
    for accum, chunk_count in results:
        if accum is None:
            continue
        total = accum if total is None else total + accum
        count += chunk_count
    if total is None or count == 0:
        raise ValueError("No readable images were averaged.")
    return total / count
