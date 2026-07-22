from __future__ import annotations

import tkinter as tk
from queue import Empty, Queue
from threading import Thread
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any, Callable


@dataclass
class FormField:
    name: str
    label: str
    kind: type = str
    width: int = 18
    browse: str | None = None


class BoundForm(ttk.Frame):
    def __init__(self, master: tk.Misc, model: Any, fields: list[FormField], columns: int = 2):
        super().__init__(master)
        self.model = model
        self.fields = fields
        self.vars: dict[str, tk.StringVar] = {}
        for index, field in enumerate(fields):
            row = index // columns
            col = (index % columns) * 3
            ttk.Label(self, text=field.label).grid(row=row, column=col, sticky="e", padx=4, pady=3)
            var = tk.StringVar(value=_stringify(getattr(model, field.name)))
            self.vars[field.name] = var
            ttk.Entry(self, textvariable=var, width=field.width).grid(row=row, column=col + 1, sticky="w", padx=4, pady=3)
            if field.browse:
                command = self._browse_dir if field.browse == "dir" else self._browse_file
                ttk.Button(self, text="...", width=3, command=lambda name=field.name, cmd=command: cmd(name)).grid(
                    row=row, column=col + 2, sticky="w"
                )

    def save_to_model(self) -> None:
        for field in self.fields:
            raw = self.vars[field.name].get()
            setattr(self.model, field.name, _coerce(raw, field.kind))

    def load_from_model(self) -> None:
        for field in self.fields:
            self.vars[field.name].set(_stringify(getattr(self.model, field.name)))

    def _browse_dir(self, name: str) -> None:
        path = filedialog.askdirectory()
        if path:
            self.vars[name].set(path)

    def _browse_file(self, name: str) -> None:
        path = filedialog.askopenfilename()
        if path:
            self.vars[name].set(path)


class LogBox(ttk.Frame):
    def __init__(self, master: tk.Misc, height: int = 8):
        super().__init__(master)
        self.text = tk.Text(self, height=height, wrap="word")
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(self, command=self.text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.configure(yscrollcommand=scroll.set)

    def write(self, message: str) -> None:
        self.text.insert(tk.END, message.rstrip() + "\n")
        self.text.see(tk.END)

    def clear(self) -> None:
        self.text.delete("1.0", tk.END)


def run_guarded(log: LogBox, action: Callable[[], str]) -> None:
    try:
        log.write(action())
    except Exception as exc:
        log.write(f"错误: {exc}")


def run_in_background(
    owner: tk.Misc,
    log: LogBox,
    work: Callable[[], Any],
    on_success: Callable[[Any], None] | None = None,
    start_message: str = "处理中...",
) -> None:
    log.write(start_message)
    results: Queue[tuple[str, Any]] = Queue()

    def task() -> None:
        try:
            result = work()
        except Exception as exc:
            results.put(("error", exc))
        else:
            results.put(("success", result))

    def poll() -> None:
        try:
            status, value = results.get_nowait()
        except Empty:
            owner.after(50, poll)
            return
        if status == "error":
            log.write(f"错误: {value}")
        elif on_success is not None:
            on_success(value)

    Thread(target=task, daemon=True).start()
    owner.after(50, poll)


class LatestTaskRunner:
    """Run at most one calculation and retain only the newest pending request."""

    def __init__(self, owner: tk.Misc, log, delay_ms: int = 80):
        self.owner = owner
        self.log = log
        self.delay_ms = delay_ms
        self._running = False
        self._pending: tuple[Callable[[], Any], Callable[[Any], None], str] | None = None
        self._delay_job: str | None = None

    def submit(self, work: Callable[[], Any], on_success: Callable[[Any], None], start_message: str) -> None:
        self._pending = (work, on_success, start_message)
        if self._running:
            return
        if self._delay_job is not None:
            self.owner.after_cancel(self._delay_job)
        self._delay_job = self.owner.after(self.delay_ms, self._start_pending)

    def _start_pending(self) -> None:
        self._delay_job = None
        if self._running or self._pending is None:
            return
        work, on_success, start_message = self._pending
        self._pending = None
        self._running = True
        self.log.write(start_message)
        results: Queue[tuple[str, Any]] = Queue()

        def task() -> None:
            try:
                results.put(("success", work()))
            except Exception as exc:
                results.put(("error", exc))

        def poll() -> None:
            try:
                status, value = results.get_nowait()
            except Empty:
                self.owner.after(50, poll)
                return
            self._running = False
            if status == "error":
                self.log.write(f"错误: {value}")
            else:
                on_success(value)
            if self._pending is not None:
                self._start_pending()

        Thread(target=task, daemon=True).start()
        self.owner.after(50, poll)


def _stringify(value: Any) -> str:
    if isinstance(value, Path):
        return "" if str(value) == "." else str(value)
    return str(value)


def _coerce(value: str, kind: type) -> Any:
    if kind is Path:
        return Path(value) if value else Path()
    if kind is int:
        return int(value)
    if kind is float:
        return float(value)
    if kind is bool:
        return value.lower() in {"1", "true", "yes", "y", "是"}
    return value
