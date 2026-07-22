from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from PLIFv2.core.image_io import read_image, save_float_image
from PLIFv2.core.paths import path_is_set
from PLIFv2.domain.settings import DavisSettings


@dataclass
class DavisImportResult:
    tif_count: int
    npy_count: int
    output_folder: Path


@dataclass
class DavisPrepareResult:
    copied_count: int
    output_folder: Path


class DavisService:
    def prepare_input_folder(self, settings: DavisSettings) -> DavisPrepareResult:
        source = Path(settings.corrected_folder)
        output = Path(settings.davis_input_folder)
        if not source.exists():
            raise FileNotFoundError(f"Corrected image folder does not exist: {source}")
        if not path_is_set(output):
            output = source.parent / f"{source.name}-Davis输入"
        output.mkdir(parents=True, exist_ok=True)
        copied = 0
        for path in source.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"} or "GAIN" not in path.name.upper():
                continue
            destination = output / path.name
            if destination.exists() and destination.resolve() != path.resolve():
                raise FileExistsError(f"Duplicate Davis input file name: {destination.name}")
            if destination.resolve() != path.resolve():
                shutil.copy2(path, destination)
            copied += 1
        return DavisPrepareResult(copied, output)

    def import_export_folder(self, settings: DavisSettings) -> DavisImportResult:
        source = Path(settings.davis_export_folder)
        if not source.exists():
            raise FileNotFoundError(f"Davis export folder does not exist: {source}")
        output = Path(settings.output_folder) if path_is_set(settings.output_folder) else source
        output.mkdir(parents=True, exist_ok=True)

        tif_count = 0
        npy_count = 0
        source_by_experiment: dict[str, Path] = {}
        for path in sorted(source.iterdir(), key=lambda item: item.name):
            if path.is_dir():
                path = _first_file(path)
                if path is None:
                    continue
            if not path.is_file():
                continue
            exp_name = _extract_experiment_name(path.name)
            if not exp_name:
                continue
            previous = source_by_experiment.get(exp_name)
            if previous is not None and previous.resolve() != path.resolve():
                raise FileExistsError(f"Duplicate Davis output for {exp_name}: {previous.name}, {path.name}")
            source_by_experiment[exp_name] = path

        for exp_name, path in source_by_experiment.items():
            tif_path = output / f"{exp_name}.tif"
            if path.resolve() != tif_path.resolve():
                shutil.copy2(path, tif_path)
            tif_count += 1
            if settings.restore_gain:
                gain = _extract_gain(tif_path.name)
                if gain:
                    image = read_image(tif_path).astype(np.float64) / gain
                    save_float_image(image, output, exp_name.split("-")[0])
                    npy_count += 1
        return DavisImportResult(tif_count=tif_count, npy_count=npy_count, output_folder=output)


def _first_file(folder: Path) -> Path | None:
    for item in sorted(folder.iterdir(), key=lambda item: item.name):
        if item.is_file():
            return item
    return None


def _extract_experiment_name(filename: str) -> str:
    for part in filename.split("_"):
        if "实验编号" in part:
            return Path(part).stem
    match = re.search(r"(实验编号\d+(?:-GAIN[\d.]+)?)", filename)
    return match.group(1) if match else ""


def _extract_gain(filename: str) -> float | None:
    match = re.search(r"GAIN([\d.]+)", filename)
    return float(match.group(1)) if match else None
