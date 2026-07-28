from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from threading import get_ident
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

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
    _migrate_distribution_settings(data)
    _migrate_ui_state(data, config_path)
    return _from_dict(AppSettings, data)


def _migrate_average_settings(data: dict[str, Any]) -> None:
    average = data.get("average")
    if not isinstance(average, dict):
        return
    if "denoise_threshold" not in average and "zero_threshold" in average:
        average["denoise_threshold"] = average["zero_threshold"]
    if "best_process_count" not in average and "process_count" in average:
        average["best_process_count"] = average["process_count"]
    average.setdefault("experiment_numbers", "")
    average.setdefault("use_frame_segments", bool(str(average.get("frame_segments", "")).strip()))
    if average.get("selection_mode", "frame").lower() == "frame":
        frequency = float(average.get("sample_frequency", 400) or 400)
        average["time_start"] = float(average.get("frame_start", 0)) / frequency
        average["time_end"] = float(average.get("frame_end", 0)) / frequency


def _migrate_distribution_settings(data: dict[str, Any]) -> None:
    distribution = data.get("distribution")
    if not isinstance(distribution, dict):
        return
    for suffix in ("f", "o", "fo"):
        distribution.setdefault(f"q_center_up_{suffix}", distribution.get("q_center_up", 0))
        distribution.setdefault(f"q_center_down_{suffix}", distribution.get("q_center_down", 0))


def _migrate_ui_state(data: dict[str, Any], config_path: Path) -> None:
    if isinstance(data.get("ui"), dict):
        return
    legacy_path = config_path.with_name("ui_state.json")
    if not legacy_path.exists():
        data["ui"] = {"app": {}, "pages": {}}
        return
    try:
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        legacy = {}
    data["ui"] = legacy if isinstance(legacy, dict) else {"app": {}, "pages": {}}


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    obj = cls()  # type: ignore[call-arg]
    type_hints = get_type_hints(type(obj))
    for field_info in fields(obj):
        if field_info.name not in data:
            continue
        current = getattr(obj, field_info.name)
        value = data[field_info.name]
        if is_dataclass(current):
            setattr(obj, field_info.name, _from_dict(type(current), value))
        else:
            field_type = type_hints.get(field_info.name, field_info.type)
            setattr(obj, field_info.name, _coerce_value(field_type, value, current))
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
    if type_hint is int:
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"整数设置不能使用小数值: {value}")
        return int(value)
    if type_hint is float:
        return float(value)
    if type_hint is str:
        return str(value)
    if type_hint is bool:
        return bool(value)
    return value
