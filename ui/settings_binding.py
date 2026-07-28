from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, get_type_hints


def set_dataclass_text_field(settings: Any, name: str, raw: str) -> None:
    field_type = _field_types(type(settings)).get(name, str)
    try:
        if field_type is Path:
            value = Path(raw) if raw else Path()
        elif field_type is int:
            value = int(raw)
        elif field_type is float:
            value = float(raw)
        else:
            value = raw
    except ValueError as exc:
        if name.startswith("rotate_angle_"):
            raise ValueError("旋转角度必须是整数") from exc
        expected = "整数" if field_type is int else "数值"
        raise ValueError(f"参数 {name} 必须是{expected}") from exc
    setattr(settings, name, value)


@lru_cache(maxsize=None)
def _field_types(settings_type: type) -> dict[str, Any]:
    return get_type_hints(settings_type)
