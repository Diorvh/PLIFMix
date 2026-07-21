from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from threading import get_ident
from typing import Any, TypeVar, get_args, get_origin

from PLIFv2.domain.settings import AppSettings, dataclass_to_jsonable

T = TypeVar("T")


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "runtime" / "settings.json"


def save_settings(settings: AppSettings, path: Path | None = None) -> Path:
    config_path = path or default_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = config_path.with_name(f".{config_path.name}.{get_ident()}.tmp")
    temporary_path.write_text(
        json.dumps(dataclass_to_jsonable(settings), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(config_path)
    return config_path


def load_settings(path: Path | None = None) -> AppSettings:
    config_path = path or default_config_path()
    if not config_path.exists():
        return AppSettings()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    _migrate_average_settings(data)
    return _from_dict(AppSettings, data)


def _migrate_average_settings(data: dict[str, Any]) -> None:
    average = data.get("average")
    if not isinstance(average, dict):
        return
    if "denoise_threshold" not in average and "zero_threshold" in average:
        average["denoise_threshold"] = average["zero_threshold"]
    average.setdefault("experiment_numbers", "")
    average.setdefault("use_frame_segments", bool(str(average.get("frame_segments", "")).strip()))
    if average.get("selection_mode", "frame").lower() == "frame":
        frequency = float(average.get("sample_frequency", 400) or 400)
        average["time_start"] = float(average.get("frame_start", 0)) / frequency
        average["time_end"] = float(average.get("frame_end", 0)) / frequency


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    obj = cls()  # type: ignore[call-arg]
    for field_info in fields(obj):
        if field_info.name not in data:
            continue
        current = getattr(obj, field_info.name)
        value = data[field_info.name]
        if is_dataclass(current):
            setattr(obj, field_info.name, _from_dict(type(current), value))
        else:
            setattr(obj, field_info.name, _coerce_value(field_info.type, value, current))
    return obj


def _coerce_value(type_hint: Any, value: Any, current: Any) -> Any:
    if isinstance(current, Path):
        return Path(value)
    if isinstance(current, tuple):
        return tuple(value)
    origin = get_origin(type_hint)
    args = get_args(type_hint)
    if origin is tuple and args:
        return tuple(value)
    return value
