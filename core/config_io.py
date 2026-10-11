from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

from PLIFMix.core.safe_io import atomic_write_text
from PLIFMix.core.paths import bundled_sample_folder, path_is_set
from PLIFMix.domain.settings import SETTINGS_SCHEMA_VERSION, AppSettings, dataclass_to_jsonable

T = TypeVar("T")

_SAMPLE_OUTPUT_PATHS = {
    "manual": {"output_folder": Path("outputs/sample/mixing")},
    "distribution": {
        "output_folder": Path("outputs/sample/distribution"),
        "image_save_folder": Path("outputs/sample/images"),
    },
}


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "runtime" / "settings.json"


def save_settings(settings: AppSettings, path: Path | None = None) -> Path:
    config_path = path or default_config_path()
    settings.schema_version = SETTINGS_SCHEMA_VERSION
    payload = dataclass_to_jsonable(settings)
    payload["schema_version"] = SETTINGS_SCHEMA_VERSION
    if path is None:
        _make_sample_paths_portable(payload)
    return atomic_write_text(
        config_path,
        json.dumps(payload, ensure_ascii=False, indent=2),
    )


def load_settings(path: Path | None = None) -> AppSettings:
    config_path = path or default_config_path()
    if not config_path.exists():
        settings = AppSettings()
        return _with_sample_defaults(settings, fill_missing=path is None)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("设置文件必须是 JSON 对象")
    schema_version = _read_schema_version(data)
    if schema_version > SETTINGS_SCHEMA_VERSION:
        raise ValueError(
            f"设置文件版本 {schema_version} 高于当前支持版本 {SETTINGS_SCHEMA_VERSION}"
        )
    if schema_version < SETTINGS_SCHEMA_VERSION:
        _migrate_legacy_settings(data, config_path)
    elif isinstance(data.get("ui"), dict):
        data["ui"] = _normalise_ui_state(data["ui"])
    data["schema_version"] = SETTINGS_SCHEMA_VERSION
    settings = _from_dict(AppSettings, data)
    return _with_sample_defaults(settings, fill_missing=path is None)


def _with_sample_defaults(settings: AppSettings, *, fill_missing: bool) -> AppSettings:
    sample_folder = bundled_sample_folder()
    if not all((sample_folder / f"实验编号{number}.npy").is_file() for number in (67, 57, 45)):
        return settings
    defaults = None
    using_sample = False
    for section, output_paths in _SAMPLE_OUTPUT_PATHS.items():
        current = getattr(settings, section)
        if not path_is_set(current.image_folder):
            if not fill_missing:
                continue
            if defaults is None:
                defaults = json.loads((sample_folder / "parameters.json").read_text(encoding="utf-8"))
            replacement = _from_dict(type(current), defaults[section])
            for name in (*output_paths, "fit_result_path"):
                value = getattr(current, name, Path())
                if path_is_set(value):
                    setattr(replacement, name, value)
            current = replacement
            setattr(settings, section, current)
            current.image_folder = sample_folder
        elif current.image_folder == Path(sample_folder.name):
            current.image_folder = sample_folder
        if current.image_folder != sample_folder:
            continue
        using_sample = True
        for name, relative_path in output_paths.items():
            value = getattr(current, name)
            if not path_is_set(value) or value == relative_path:
                setattr(current, name, sample_folder.parent / relative_path)
    if using_sample and fill_missing:
        settings.ui.app.setdefault("active_page", "4 手动调参")
    return settings


def _make_sample_paths_portable(payload: dict[str, Any]) -> None:
    sample_folder = bundled_sample_folder()
    for section, output_paths in _SAMPLE_OUTPUT_PATHS.items():
        values = payload[section]
        if Path(values["image_folder"]) not in {sample_folder, Path(sample_folder.name)}:
            continue
        if Path(values["image_folder"]) == sample_folder:
            values["image_folder"] = sample_folder.name
        for name, relative_path in output_paths.items():
            if Path(values[name]) == sample_folder.parent / relative_path:
                values[name] = relative_path.as_posix()


def _read_schema_version(data: dict[str, Any]) -> int:
    value = data.get("schema_version", 0)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("设置文件 schema_version 必须是整数")
    if value < 0:
        raise ValueError("设置文件 schema_version 不能为负数")
    return value


def _migrate_legacy_settings(data: dict[str, Any], config_path: Path) -> None:
    _migrate_average_settings(data)
    _migrate_distribution_settings(data)
    _migrate_ui_state(data, config_path)


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
    selection_mode = str(average.get("selection_mode", "frame") or "frame").lower()
    if selection_mode != "frame":
        return
    try:
        frequency = float(average.get("sample_frequency", 400) or 400)
    except (TypeError, ValueError):
        return
    if frequency == 0:
        return
    for time_name, frame_name in (("time_start", "frame_start"), ("time_end", "frame_end")):
        if time_name in average:
            continue
        try:
            frame = float(average.get(frame_name, 0))
        except (TypeError, ValueError):
            continue
        average[time_name] = frame / frequency


def _migrate_distribution_settings(data: dict[str, Any]) -> None:
    distribution = data.get("distribution")
    if not isinstance(distribution, dict):
        return
    for suffix in ("f", "o", "fo"):
        distribution.setdefault(f"q_center_up_{suffix}", distribution.get("q_center_up", 0))
        distribution.setdefault(f"q_center_down_{suffix}", distribution.get("q_center_down", 0))


def _migrate_ui_state(data: dict[str, Any], config_path: Path) -> None:
    if isinstance(data.get("ui"), dict):
        data["ui"] = _normalise_ui_state(data["ui"])
        return
    legacy_path = config_path.with_name("ui_state.json")
    if not legacy_path.exists():
        data["ui"] = _empty_ui_state()
        return
    try:
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        legacy = {}
    data["ui"] = _normalise_ui_state(legacy)


def _empty_ui_state() -> dict[str, Any]:
    return {"app": {}, "pages": {}}


def _normalise_ui_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return _empty_ui_state()
    app = value.get("app")
    pages = value.get("pages")
    return {
        "app": app if isinstance(app, dict) else {},
        "pages": pages if isinstance(pages, dict) else {},
    }


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
