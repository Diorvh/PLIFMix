from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time
import uuid

from PLIFv2.algorithms.average import AverageSetError, average_image_sets
from PLIFv2.core.image_io import list_image_files, save_float_image
from PLIFv2.core.paths import path_is_set
from PLIFv2.domain.settings import AverageSettings, IntensifierAverageSettings


@dataclass
class AverageResult:
    selected_count: int
    outputs: dict[str, Path]
    items: list["AverageItem"]
    calculation_seconds: float


@dataclass
class AverageItem:
    source_folder: Path
    selected_count: int
    outputs: dict[str, Path]


@dataclass
class _PreparedAverage:
    source_folder: Path
    save_name: str
    selected: list[Path]
    output_folder: Path


class AverageExperimentError(RuntimeError):
    def __init__(self, experiment_name: str, stage: str, message: str):
        self.experiment_name = experiment_name
        self.stage = stage
        self.message = message
        super().__init__(experiment_name, stage, message)

    def __str__(self) -> str:
        return f"{self.experiment_name} {self.stage}失败: {self.message}"


class AverageImageService:
    def run(self, settings: AverageSettings) -> AverageResult:
        source_root = Path(settings.source_folder)
        if not source_root.exists():
            raise FileNotFoundError(f"Source folder does not exist: {source_root}")
        experiment_numbers = settings.experiment_numbers if settings.use_experiment_numbers else ""
        experiments = _resolve_experiment_folders(source_root, experiment_numbers)
        prepared: list[_PreparedAverage] = []
        for folder, save_name in experiments:
            try:
                prepared.append(self._prepare_folder(folder, save_name, settings))
            except Exception as exc:
                raise AverageExperimentError(save_name, "准备图像", str(exc)) from exc

        return self._calculate_and_save(prepared, settings)

    def run_intensifier_average(self, settings: IntensifierAverageSettings) -> AverageResult:
        """Average calibration images from every Gain/exposure folder pair."""
        source_root = Path(settings.source_folder)
        if not source_root.exists():
            raise FileNotFoundError(f"Source folder does not exist: {source_root}")
        if not path_is_set(settings.output_folder):
            raise ValueError("Select an average-image output folder for intensifier calibration.")
        output_folder = Path(settings.output_folder)

        gain_folders = sorted(
            path
            for path in source_root.iterdir()
            if path.is_dir() and re.fullmatch(r"gain\d+", path.name, flags=re.IGNORECASE)
        )
        if not gain_folders:
            raise ValueError(f"No Gainxx calibration folders found in: {source_root}")

        prepared: list[_PreparedAverage] = []
        for gain_folder in gain_folders:
            exposure_folders = sorted(path for path in gain_folder.iterdir() if path.is_dir())
            if not exposure_folders:
                raise AverageExperimentError(gain_folder.name, "准备图像", "No exposure folders found.")
            for exposure_folder in exposure_folders:
                try:
                    prepared.append(
                        self._prepare_folder(
                            exposure_folder,
                            exposure_folder.name,
                            settings,
                            output_folder=output_folder / gain_folder.name,
                            select_all=True,
                        )
                    )
                except Exception as exc:
                    name = f"{gain_folder.name}/{exposure_folder.name}"
                    raise AverageExperimentError(name, "准备图像", str(exc)) from exc

        return self._calculate_and_save(prepared, settings)

    def _calculate_and_save(
        self,
        prepared: list[_PreparedAverage],
        settings: AverageSettings | IntensifierAverageSettings,
    ) -> AverageResult:

        calculation_start = time.perf_counter()
        try:
            averages = average_image_sets(
                [item.selected for item in prepared],
                settings.best_process_count,
                settings.denoise_threshold,
            )
        except AverageSetError as exc:
            raise AverageExperimentError(prepared[exc.set_index].save_name, "图像计算", str(exc)) from exc
        calculation_seconds = time.perf_counter() - calculation_start

        items: list[AverageItem] = []
        for item, average in zip(prepared, averages):
            try:
                outputs = save_float_image(average, item.output_folder, item.save_name)
            except Exception as exc:
                raise AverageExperimentError(item.save_name, "保存结果", str(exc)) from exc
            items.append(AverageItem(item.source_folder, len(item.selected), outputs))
        return AverageResult(
            selected_count=sum(item.selected_count for item in items),
            outputs=items[0].outputs if len(items) == 1 else {},
            items=items,
            calculation_seconds=calculation_seconds,
        )

    def _prepare_folder(
        self,
        folder: Path,
        save_name: str,
        settings: AverageSettings | IntensifierAverageSettings,
        output_folder: Path | None = None,
        select_all: bool = False,
    ) -> _PreparedAverage:
        files = list_image_files(folder, (".png", ".tif", ".tiff", ".bmp"))
        if not files:
            raise ValueError(f"No image files found in: {folder}")
        files = _ensure_numbered_sequence(files)
        if select_all:
            selected = files
        elif isinstance(settings, AverageSettings):
            selected = _select_files(files, settings)
        else:
            raise TypeError("Time-range selection requires AverageSettings.")
        if not selected:
            raise ValueError(f"No files selected in {folder}. Check frame selection.")
        target_folder = output_folder or (Path(settings.output_folder) if path_is_set(settings.output_folder) else folder.parent)
        return _PreparedAverage(folder, save_name, selected, target_folder)


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
    renamed: list[tuple[Path, Path]] = []
    try:
        for index, path in enumerate(files):
            temp = path.with_name(f".plifv2_{token}_{index}{path.suffix}")
            path.rename(temp)
            temporary.append((temp, path))
        for index, (temp, original) in enumerate(temporary):
            final = original.with_name(f"{index:04d}{original.suffix.upper()}")
            temp.rename(final)
            renamed.append((final, original))
    except Exception:
        for final, original in reversed(renamed):
            if final.exists():
                final.rename(original)
        for temp, original in reversed(temporary):
            if temp.exists():
                temp.rename(original)
        raise
    return list_image_files(files[0].parent, (".png", ".tif", ".tiff", ".bmp"))
