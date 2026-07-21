from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import uuid

from PLIFv2.algorithms.average import average_images
from PLIFv2.core.image_io import list_image_files, save_float_image
from PLIFv2.core.paths import path_is_set
from PLIFv2.domain.settings import AverageSettings


@dataclass
class AverageResult:
    selected_count: int
    outputs: dict[str, Path]
    items: list["AverageItem"]


@dataclass
class AverageItem:
    source_folder: Path
    selected_count: int
    outputs: dict[str, Path]


class AverageImageService:
    def run(self, settings: AverageSettings) -> AverageResult:
        source_root = Path(settings.source_folder)
        if not source_root.exists():
            raise FileNotFoundError(f"Source folder does not exist: {source_root}")
        experiments = _resolve_experiment_folders(source_root, settings.experiment_numbers)

        # Experiments are intentionally processed one by one. Multiprocessing is
        # used only to split the selected image range inside each experiment.
        items = [self._run_folder(folder, save_name, settings) for folder, save_name in experiments]
        return AverageResult(
            selected_count=sum(item.selected_count for item in items),
            outputs=items[0].outputs if len(items) == 1 else {},
            items=items,
        )

    def _run_folder(self, folder: Path, save_name: str, settings: AverageSettings) -> AverageItem:
        files = list_image_files(folder, (".png", ".tif", ".tiff", ".bmp"))
        if not files:
            raise ValueError(f"No image files found in: {folder}")
        files = _ensure_numbered_sequence(files)
        selected = _select_files(files, settings)
        if not selected:
            raise ValueError(f"No files selected in {folder}. Check frame selection.")
        output_folder = Path(settings.output_folder) if path_is_set(settings.output_folder) else folder.parent
        avg = average_images(selected, settings.process_count, settings.denoise_threshold)
        outputs = save_float_image(avg, output_folder, save_name)
        return AverageItem(folder, len(selected), outputs)


def _select_files(files: list[Path], settings: AverageSettings) -> list[Path]:
    if settings.use_frame_segments:
        if not settings.frame_segments.strip():
            raise ValueError("Non-contiguous interval mode is selected, but no intervals were entered.")
        selected: list[Path] = []
        segments = re.split(r"[,，;；]+", settings.frame_segments)
        for segment in segments:
            try:
                start_text, end_text = segment.strip().split(":", 1)
                start, end = int(start_text), int(end_text)
            except ValueError as exc:
                raise ValueError(f"Invalid interval '{segment}'. Use start:end,start:end.") from exc
            if start < 0 or end <= start or end > len(files):
                raise ValueError(f"Interval {start}:{end} is outside the available frame range 0:{len(files)}.")
            selected.extend(files[start:end])
        return selected
    if settings.sample_frequency <= 0:
        raise ValueError("Sample frequency must be greater than zero.")
    start = int(settings.time_start * settings.sample_frequency)
    end = int(settings.time_end * settings.sample_frequency)
    if start < 0 or end <= start or end > len(files):
        raise ValueError(f"Time range selects frames {start}:{end}, outside the available range 0:{len(files)}.")
    return files[start:end]


def _resolve_experiment_folders(source_root: Path, experiment_numbers: str) -> list[tuple[Path, str]]:
    tokens = [token for token in re.split(r"[,，;；\s]+", experiment_numbers.strip()) if token]
    if not tokens:
        return [(source_root, source_root.name)]
    resolved: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for token in tokens:
        name = token if token.startswith("实验编号") else f"实验编号{token}"
        if name in seen:
            continue
        seen.add(name)
        folder = source_root if source_root.name == name else source_root / name
        if not folder.exists():
            raise FileNotFoundError(f"Experiment folder does not exist: {folder}")
        resolved.append((folder, name))
    return resolved


def _ensure_numbered_sequence(files: list[Path]) -> list[Path]:
    if all(path.stem == f"{index:04d}" for index, path in enumerate(files)):
        return files
    token = uuid.uuid4().hex
    temporary: list[tuple[Path, Path]] = []
    for index, path in enumerate(files):
        temp = path.with_name(f".plifv2_{token}_{index}{path.suffix}")
        path.rename(temp)
        temporary.append((temp, path))
    renamed: list[Path] = []
    for index, (temp, original) in enumerate(temporary):
        final = original.with_name(f"{index:04d}{original.suffix.upper()}")
        temp.rename(final)
        renamed.append(final)
    return list_image_files(files[0].parent, (".png", ".tif", ".tiff", ".bmp"))
