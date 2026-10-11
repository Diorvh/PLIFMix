from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from PLIFMix.core.image_io import extract_plif_signal, read_image


FramePath = str | Path
FrameRange = tuple[float, float]
IndexedFramePath = tuple[int, Path]
CachedFrame = np.ndarray | None
FrameBatchItem = tuple[int, CachedFrame, FrameRange | None]
FrameBatchResult = tuple[list[FrameBatchItem], int]


@dataclass
class FrameCacheResult:
    cache: list[CachedFrame]
    ranges: list[FrameRange | None]
    loaded_count: int
    failed_count: int
    cache_bytes: int
    elapsed_seconds: float


def split_frame_cache_tasks(
    paths: Sequence[FramePath], worker_count: int
) -> tuple[tuple[IndexedFramePath, ...], ...]:
    """Split paths into continuous, balanced batches for cache workers."""
    normalized_paths = tuple(Path(path) for path in paths)
    if worker_count < 1:
        raise ValueError("worker_count must be at least 1")
    if not normalized_paths:
        return ()

    actual_worker_count = min(worker_count, len(normalized_paths))
    batch_size, remainder = divmod(len(normalized_paths), actual_worker_count)
    indexed_paths = tuple(enumerate(normalized_paths))
    batches: list[tuple[IndexedFramePath, ...]] = []
    start_index = 0
    for batch_index in range(actual_worker_count):
        stop_index = start_index + batch_size + (1 if batch_index < remainder else 0)
        batches.append(indexed_paths[start_index:stop_index])
        start_index = stop_index
    return tuple(batches)


def read_raw_frame(path: FramePath) -> np.ndarray:
    """Read one image and retain the player's raw PLIF signal semantics."""
    image = extract_plif_signal(read_image(path))
    # Keep a compact, owning buffer instead of retaining a three-channel view.
    return np.ascontiguousarray(image)


def read_frame_batch(indexed_paths: tuple[IndexedFramePath, ...]) -> FrameBatchResult:
    """Read a cache batch without importing any GUI modules."""
    frames: list[FrameBatchItem] = []
    failed = 0
    for frame_index, path in indexed_paths:
        try:
            frame = read_raw_frame(path)
            frames.append((frame_index, frame, (float(np.min(frame)), float(np.max(frame)))))
        except Exception:
            frames.append((frame_index, None, None))
            failed += 1
    return frames, failed


def _failed_batch(indexed_paths: tuple[IndexedFramePath, ...]) -> FrameBatchResult:
    return ([(frame_index, None, None) for frame_index, _path in indexed_paths], len(indexed_paths))


def load_frame_cache(paths: Iterable[FramePath], worker_count: int) -> FrameCacheResult:
    """Synchronously load and merge the player's raw frame cache."""
    cache_started = time.perf_counter()
    normalized_paths = tuple(Path(path) for path in paths)
    total = len(normalized_paths)
    batches = split_frame_cache_tasks(normalized_paths, worker_count)

    cache: list[CachedFrame] = [None] * total
    ranges: list[FrameRange | None] = [None] * total
    loaded_count = 0
    failed_count = 0
    cache_bytes = 0

    if len(batches) == 1:
        results = [read_frame_batch(batches[0])]
    elif batches:
        results: list[FrameBatchResult] = []
        with ProcessPoolExecutor(
            max_workers=len(batches),
            mp_context=mp.get_context("spawn"),
        ) as executor:
            future_batches = {
                executor.submit(read_frame_batch, indexed_paths): indexed_paths
                for indexed_paths in batches
            }
            for future in as_completed(future_batches):
                indexed_paths = future_batches[future]
                try:
                    results.append(future.result())
                except Exception:
                    results.append(_failed_batch(indexed_paths))
    else:
        results = []

    for frames, failed in results:
        failed_count += failed
        for frame_index, frame, value_range in frames:
            cache[frame_index] = frame
            ranges[frame_index] = value_range
            if frame is not None:
                loaded_count += 1
                cache_bytes += frame.nbytes

    return FrameCacheResult(
        cache=cache,
        ranges=ranges,
        loaded_count=loaded_count,
        failed_count=failed_count,
        cache_bytes=cache_bytes,
        elapsed_seconds=time.perf_counter() - cache_started,
    )
