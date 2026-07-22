from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any


def default_ui_state_path() -> Path:
    return Path(__file__).resolve().parents[1] / "runtime" / "ui_state.json"


class UIStateStore:
    def __init__(self, path: Path | None = None):
        self.path = path or default_ui_state_path()
        self.data = self._load()

    def restore_page(self, page_key: str, root: tk.Misc) -> None:
        values = self.data.get("pages", {}).get(page_key, {})
        for widget in _walk_widgets(root):
            key = _relative_widget_path(root, widget)
            if key in values:
                _restore_widget(widget, values[key])
        restore_custom = getattr(root, "restore_ui_state", None)
        if callable(restore_custom):
            restore_custom(values.get("__custom__", {}))

    def capture_page(self, page_key: str, root: tk.Misc) -> None:
        pages = self.data.setdefault("pages", {})
        values: dict[str, Any] = {}
        for widget in _walk_widgets(root):
            value = _capture_widget(widget)
            if value is not None:
                values[_relative_widget_path(root, widget)] = value
        export_custom = getattr(root, "export_ui_state", None)
        if callable(export_custom):
            values["__custom__"] = export_custom()
        pages[page_key] = values

    def get_app_value(self, key: str, default: Any = None) -> Any:
        return self.data.get("app", {}).get(key, default)

    def set_app_value(self, key: str, value: Any) -> None:
        self.data.setdefault("app", {})[key] = value

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)
        return self.path

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"pages": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"pages": {}}
        return data if isinstance(data, dict) else {"pages": {}}


def _walk_widgets(root: tk.Misc):
    for child in root.winfo_children():
        yield child
        yield from _walk_widgets(child)


def _relative_widget_path(root: tk.Misc, widget: tk.Misc) -> str:
    prefix = f"{root._w}."
    return widget._w[len(prefix) :] if widget._w.startswith(prefix) else widget._w


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
