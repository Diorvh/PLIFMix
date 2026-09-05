from __future__ import annotations

from queue import Empty, Queue
from threading import Thread
import tkinter as tk
from typing import Any, Callable
from weakref import WeakSet


_RUNNERS: WeakSet = WeakSet()


def _save_ui_settings(owner) -> None:
    state = getattr(owner.winfo_toplevel(), "state_model", None)
    if state is not None:
        state.save()


class TaskRunner:
    """One worker per owner, with explicit pending-request and shutdown policies."""

    def __init__(self, owner, log, *, latest_only: bool, delay_ms: int = 0):
        self.owner = owner
        self.log = log
        self.latest_only = latest_only
        self.delay_ms = delay_ms
        self._pending = None
        self._job = None
        self._thread: Thread | None = None
        self._running = False
        self._closed = False
        _RUNNERS.add(self)
        owner.bind("<Destroy>", self._on_destroy, add="+")

    @property
    def busy(self) -> bool:
        return self._running or self._pending is not None

    @property
    def worker_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def submit(self, work: Callable[[], Any], on_success, start_message: str) -> bool:
        if self._closed or getattr(self.owner.winfo_toplevel(), "_closing", False):
            return False
        if self.busy and not self.latest_only:
            self.log.write("当前任务尚未完成，请勿重复启动。")
            return False
        _save_ui_settings(self.owner)
        self._pending = (work, on_success, start_message)
        if not self._running:
            self._cancel_job()
            self._job = self.owner.after(self.delay_ms, self._start_pending)
        return True

    def _start_pending(self) -> None:
        self._job = None
        if self._closed or self._running or self._pending is None:
            return
        work, on_success, message = self._pending
        self._pending = None
        self._running = True
        self.log.write(message)
        results: Queue = Queue()

        def work_once() -> None:
            try:
                results.put((True, work()))
            except Exception as exc:
                results.put((False, exc))

        def poll() -> None:
            self._job = None
            if self._closed:
                return
            try:
                success, value = results.get_nowait()
            except Empty:
                self._job = self.owner.after(50, poll)
                return
            self._running = False
            try:
                _save_ui_settings(self.owner)
                if success and on_success is not None:
                    on_success(value)
                elif not success:
                    self.log.write(f"错误: {value}")
            except Exception as exc:
                self.log.write(f"结果显示失败: {exc}")
            finally:
                if self._pending is not None and not self._closed:
                    self._start_pending()

        self._thread = Thread(target=work_once, daemon=True)
        self._thread.start()
        self._job = self.owner.after(50, poll)

    def _cancel_job(self) -> None:
        if self._job is not None:
            try:
                self.owner.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None

    def close(self) -> None:
        self._closed = True
        self._pending = None
        self._cancel_job()

    def _on_destroy(self, event) -> None:
        if event.widget is self.owner:
            self.close()


class LatestTaskRunner(TaskRunner):
    def __init__(self, owner, log, delay_ms: int = 80):
        super().__init__(owner, log, latest_only=True, delay_ms=delay_ms)


def close_task_runners(root) -> tuple[TaskRunner, ...]:
    selected = []
    for runner in _RUNNERS:
        try:
            if runner.owner.winfo_toplevel() is root:
                selected.append(runner)
        except tk.TclError:
            runner.close()
    runners = tuple(selected)
    for runner in runners:
        runner.close()
    return runners
