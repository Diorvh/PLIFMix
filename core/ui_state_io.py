from __future__ import annotations

from collections.abc import Iterator, Mapping
import tkinter as tk
from tkinter import ttk
from typing import Any

from PLIFv2.domain.settings import UiStateSettings


class UIStateStore:
    """Capture semantic UI values in the application's single settings document."""

    def __init__(self, state: UiStateSettings):
        self.data = state

    def restore_page(self, page_key: str, root: tk.Misc) -> None:
        values = self.data.pages.get(page_key, {})
        if not isinstance(values, dict):
            values = {}
        if getattr(root, "persist_widget_values", True):
            semantic_values = _semantic_values(values)
            if semantic_values:
                _restore_named_state(root, semantic_values)
            else:
                matched = _restore_legacy_widget_state(root, values)
                if matched:
                    migrated = _capture_named_state(root)
                    if migrated:
                        values = {
                            key: value
                            for key, value in values.items()
                            if not _is_legacy_widget_key(key)
                        }
                        values.update(migrated)
                        self.data.pages[page_key] = values
        restore_custom = getattr(root, "restore_ui_state", None)
        if callable(restore_custom):
            restore_custom(values.get("__custom__", {}))

    def capture_page(self, page_key: str, root: tk.Misc) -> None:
        values: dict[str, Any] = {}
        if getattr(root, "persist_widget_values", True):
            values.update(_capture_named_state(root))
        export_custom = getattr(root, "export_ui_state", None)
        if callable(export_custom):
            values["__custom__"] = export_custom()
        self.data.pages[page_key] = values

    def get_app_value(self, key: str, default: Any = None) -> Any:
        return self.data.app.get(key, default)

    def set_app_value(self, key: str, value: Any) -> None:
        self.data.app[key] = value


def _walk_widgets(root: tk.Misc):
    for child in root.winfo_children():
        yield child
        yield from _walk_widgets(child)


def _relative_widget_path(root: tk.Misc, widget: tk.Misc) -> str:
    prefix = f"{root._w}."
    return widget._w[len(prefix) :] if widget._w.startswith(prefix) else widget._w


def _capture_named_state(root: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for key, kind, value in _walk_named_state(root):
        captured = _capture_named_value(kind, value)
        if captured is not None:
            state[key] = captured
    return state


def _restore_named_state(root: Any, state: Mapping[str, Any]) -> None:
    current = {key: (kind, value) for key, kind, value in _walk_named_state(root)}
    converted: list[tuple[str, Any, Any]] = []
    for key, saved in state.items():
        target = current.get(key)
        if target is None:
            continue
        kind, value = target
        try:
            converted_value = _coerce_named_value(kind, value, saved)
        except (TypeError, ValueError, OverflowError, tk.TclError):
            continue
        converted.append((kind, value, converted_value))
    for kind, target, value in converted:
        try:
            _restore_named_value(kind, target, value)
        except (TypeError, ValueError, OverflowError, tk.TclError):
            pass


def _walk_named_state(
    value: Any,
    prefix: str = "",
    visited: set[int] | None = None,
) -> Iterator[tuple[str, str, Any]]:
    if visited is None:
        visited = set()
    if isinstance(value, tk.Variable):
        if prefix:
            yield prefix, "variable", value
        return

    if isinstance(value, ttk.Notebook):
        if prefix:
            try:
                selected = value.select()
                if selected:
                    yield prefix, "notebook", value
            except tk.TclError:
                pass
    elif isinstance(value, tk.Scale):
        if prefix:
            yield prefix, "scale", value

    object_id = id(value)
    if object_id in visited:
        return
    visited.add(object_id)

    if isinstance(value, Mapping):
        for name in sorted(value, key=str):
            child = value[name]
            child_prefix = _join_name(prefix, str(name))
            yield from _walk_named_state(child, child_prefix, visited)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_named_state(child, f"{prefix}[{index}]", visited)
        return
    if callable(value) or isinstance(value, type):
        return

    attributes = getattr(value, "__dict__", None)
    if not isinstance(attributes, dict):
        return
    if not _is_named_container(value):
        return
    for name in sorted(attributes):
        if _skip_named_attribute(name):
            continue
        child_prefix = _join_name(prefix, name)
        yield from _walk_named_state(attributes[name], child_prefix, visited)


def _is_named_container(value: Any) -> bool:
    if isinstance(value, tk.Misc):
        return True
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        if "vars" in attributes or any(isinstance(item, tk.Variable) for item in attributes.values()):
            return True
    module = type(value).__module__
    return module.startswith("PLIFv2.ui") or module == "types"


def _skip_named_attribute(name: str) -> bool:
    return name.startswith("_") or name in {
        "children",
        "master",
        "tk",
        "_w",
        "pathName",
        "_tclCommands",
        "_last_child_ids",
        "_name",
    }


def _join_name(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _capture_named_value(kind: str, value: Any) -> Any | None:
    try:
        if kind == "variable":
            return {"kind": "variable", "value": value.get()}
        if kind == "notebook":
            selected = value.select()
            return {"kind": "notebook", "value": value.index(selected)} if selected else None
        if kind == "scale":
            return {"kind": "scale", "value": value.get()}
    except (TypeError, ValueError, tk.TclError):
        return None
    return None


def _coerce_named_value(kind: str, target: Any, saved: Any) -> Any:
    if isinstance(saved, dict):
        if "value" not in saved:
            raise ValueError("缺少 UI 状态值")
        value = saved["value"]
    else:
        value = saved
    if kind == "variable":
        if isinstance(target, tk.BooleanVar):
            return _coerce_bool(value)
        if isinstance(target, tk.IntVar):
            if isinstance(value, float) and not value.is_integer():
                raise ValueError(f"整数 UI 状态不能使用小数值: {value}")
            return int(value)
        if isinstance(target, tk.DoubleVar):
            return float(value)
        return str(value)
    if kind == "notebook":
        return int(value)
    if kind == "scale":
        return float(value)
    raise ValueError(f"未知的 UI 状态类型: {kind}")


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on", "是"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", "否", ""}:
            return False
    raise ValueError(f"无法转换 UI 布尔状态: {value}")


def _restore_named_value(kind: str, target: Any, value: Any) -> None:
    if kind == "variable":
        target.set(value)
    elif kind == "notebook":
        target.select(value)
    elif kind == "scale":
        target.set(value)


def _semantic_values(values: Mapping[str, Any]) -> dict[str, Any]:
    for marker in ("__variables__", "variables"):
        nested = values.get(marker)
        if isinstance(nested, dict):
            return nested
    return {
        key: value
        for key, value in values.items()
        if isinstance(key, str)
        and key != "__custom__"
        and not _is_legacy_widget_key(key)
    }


def _is_legacy_widget_key(key: Any) -> bool:
    return isinstance(key, str) and (key.startswith("!") or key.startswith("."))


def _restore_legacy_widget_state(root: tk.Misc, values: Mapping[str, Any]) -> int:
    legacy_keys = {key for key in values if _is_legacy_widget_key(key)}
    if not legacy_keys:
        return 0
    matched = 0
    for widget in _walk_widgets(root):
        key = _relative_widget_path(root, widget)
        if key in legacy_keys:
            _restore_widget(widget, values[key])
            matched += 1
    return matched


def _capture_widget(widget: tk.Misc) -> Any | None:
    if isinstance(widget, ttk.Notebook):
        selected = widget.select()
        return {"kind": "notebook", "value": widget.index(selected)} if selected else None
    if isinstance(widget, (tk.Entry, tk.Spinbox)) or widget.winfo_class() in {"TEntry", "TCombobox", "TSpinbox"}:
        return {"kind": "text", "value": widget.get()}
    if isinstance(widget, (tk.Checkbutton, tk.Radiobutton)) or widget.winfo_class() in {"TCheckbutton", "TRadiobutton"}:
        variable = str(widget.cget("variable"))
        return {"kind": "variable", "value": widget.getvar(variable)} if variable else None
    if isinstance(widget, tk.Scale) or widget.winfo_class() == "TScale":
        return {"kind": "scale", "value": widget.get()}
    return None


def _restore_widget(widget: tk.Misc, state: Any) -> None:
    if not isinstance(state, dict) or "value" not in state:
        return
    kind = state.get("kind")
    value = state["value"]
    try:
        if kind == "text":
            variable = str(widget.cget("textvariable"))
            if variable:
                widget.setvar(variable, value)
            elif hasattr(widget, "set"):
                widget.set(value)
            else:
                widget.delete(0, tk.END)
                widget.insert(0, value)
        elif kind == "variable":
            variable = str(widget.cget("variable"))
            if variable:
                widget.setvar(variable, value)
        elif kind == "scale":
            widget.set(value)
        elif kind == "notebook":
            widget.select(int(value))
    except (tk.TclError, ValueError):
        pass
