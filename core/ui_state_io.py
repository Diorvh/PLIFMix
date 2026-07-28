from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from PLIFv2.domain.settings import UiStateSettings


class UIStateStore:
    """Capture widget values in the application's single settings document."""

    def __init__(self, state: UiStateSettings):
        self.data = state

    def restore_page(self, page_key: str, root: tk.Misc) -> None:
        values = self.data.pages.get(page_key, {})
        if getattr(root, "persist_widget_values", True):
            for widget in _walk_widgets(root):
                key = _relative_widget_path(root, widget)
                if key in values:
                    _restore_widget(widget, values[key])
        restore_custom = getattr(root, "restore_ui_state", None)
        if callable(restore_custom):
            restore_custom(values.get("__custom__", {}))

    def capture_page(self, page_key: str, root: tk.Misc) -> None:
        values: dict[str, Any] = {}
        if getattr(root, "persist_widget_values", True):
            for widget in _walk_widgets(root):
                value = _capture_widget(widget)
                if value is not None:
                    values[_relative_widget_path(root, widget)] = value
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
