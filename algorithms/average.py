from __future__ import annotations

from math import ceil
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Sequence

import numpy as np

from PLIFv2.core.image_io import read_image, to_float_single_channel


class AverageSetError(RuntimeError):
    def __init__(self, set_index: int, path: Path, message: str):
        self.set_index = set_index
        self.path = Path(path)
        self.message = message
        super().__init__(set_index, str(path), message)

    def __str__(self) -> str:
        return f"{self.path.name}: {self.message}"


def average_image_chunk(args: tuple[list[Path], float]) -> tuple[np.ndarray | None, int]:
    paths, denoise_threshold = args
    accum = None
    count = 0
    for path in paths:
        try:
            image = to_float_single_channel(
                read_image(path),
                green_channel=True,
                zero_threshold=denoise_threshold,
            )
        except Exception as exc:
            raise RuntimeError(f"Unable to read image: {path.name}; {exc}") from exc
        accum = image if accum is None else accum + image
        count += 1
    return accum, count


def split_average_tasks(
    image_sets: Sequence[Sequence[Path]],
    process_count: int | None = None,
) -> list[tuple[int, list[Path]]]:
    """Split image sets into balanced, contiguous tasks for one shared pool."""
    sets = [list(paths) for paths in image_sets]
    if not sets or any(not paths for paths in sets):
        raise ValueError("No image files selected for averaging.")

    requested_workers = _requested_workers(process_count)
    set_count = len(sets)
    parts_per_set = 1 if set_count >= requested_workers else ceil((requested_workers + 1) / set_count)
    tasks: list[tuple[int, list[Path]]] = []
    for set_index, paths in enumerate(sets):
        chunk_count = min(parts_per_set, len(paths))
        base_size, extra = divmod(len(paths), chunk_count)
        start = 0
        for chunk_index in range(chunk_count):
            stop = start + base_size + (1 if chunk_index < extra else 0)
            tasks.append((set_index, paths[start:stop]))
            start = stop
    return tasks


def average_image_sets(
    image_sets: Sequence[Sequence[Path]],
    process_count: int | None = None,
    denoise_threshold: float = 2,
) -> list[np.ndarray]:
    """Average multiple image sets through one multiprocessing pool."""
    sets = [list(paths) for paths in image_sets]
    tasks = split_average_tasks(sets, process_count)
    workers = min(_requested_workers(process_count), len(tasks))
    task_args = [(set_index, chunk, denoise_threshold) for set_index, chunk in tasks]
    if workers <= 1:
        results = [_average_set_chunk(args) for args in task_args]
    else:
        with Pool(processes=workers) as pool:
            results = pool.map(_average_set_chunk, task_args)

    totals: list[np.ndarray | None] = [None] * len(sets)
    counts = [0] * len(sets)
    for set_index, accum, chunk_count in results:
        if accum is None or chunk_count == 0:
            continue
        totals[set_index] = accum if totals[set_index] is None else totals[set_index] + accum
        counts[set_index] += chunk_count

    averages: list[np.ndarray] = []
    for total, count in zip(totals, counts):
        if total is None or count == 0:
            set_index = len(averages)
            raise AverageSetError(set_index, sets[set_index][0], "No readable images were averaged.")
        averages.append(total / count)
    return averages


def average_images(
    paths: Sequence[Path],
    process_count: int | None = None,
    denoise_threshold: float = 2,
) -> np.ndarray:
    return average_image_sets([paths], process_count, denoise_threshold)[0]


def _average_set_chunk(args: tuple[int, list[Path], float]) -> tuple[int, np.ndarray | None, int]:
    set_index, paths, denoise_threshold = args
    try:
        accum, count = average_image_chunk((paths, denoise_threshold))
    except Exception as exc:
        raise AverageSetError(set_index, paths[0], str(exc)) from exc
    return set_index, accum, count


def _requested_workers(process_count: int | None) -> int:
    if process_count is not None and process_count < 1:
        raise ValueError("Process count must be at least 1.")
    return process_count or cpu_count()
