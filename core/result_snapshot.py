from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable


_PRESENTATION_FIELDS = frozenset({
    "preset_path", "save_name", "output_folder", "fit_result_path",
    "gain", "auto_gain", "zoom", "line_width", "display_vmax", "analysis_radius",
})
_EXPORT_FIELDS = frozenset({"save_name", "output_folder", "fit_result_path"})


def _signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


@dataclass(frozen=True)
class ResultSnapshot:
    """Immutable provenance; presentation-only changes do not invalidate a fit."""

    settings_type: type
    values: tuple[tuple[str, Any], ...]
    sources: tuple[tuple[Path, tuple[int, int] | None], ...]

    @classmethod
    def capture(cls, settings: Any, paths: Iterable[Path]) -> ResultSnapshot:
        values = tuple((field.name, getattr(settings, field.name)) for field in fields(settings))
        sources = tuple((path.resolve(), _signature(path)) for path in paths)
        return cls(type(settings), values, sources)

    def require_current(self, settings: Any) -> None:
        if type(settings) is not self.settings_type:
            raise ValueError("结果与当前页面不匹配，请重新计算。")
        changed = [name for name, value in self.values
                   if name not in _PRESENTATION_FIELDS and getattr(settings, name) != value]
        if changed:
            raise ValueError("计算参数或实验编号已改变，请等待或重新计算后再保存。")
        if any(_signature(path) != signature for path, signature in self.sources):
            raise ValueError("源图像文件已改变，请重新计算后再保存。")

    def export_settings(self, current: Any) -> Any:
        self.require_current(current)
        values = dict(self.values)
        for name in _EXPORT_FIELDS & values.keys():
            values[name] = getattr(current, name)
        return self.settings_type(**values)


def require_current_result(result: Any, settings: Any) -> ResultSnapshot:
    snapshot = getattr(result, "snapshot", None)
    if snapshot is None:
        raise ValueError("没有与当前参数关联的计算结果，请先计算。")
    snapshot.require_current(settings)
    return snapshot
