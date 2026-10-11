from __future__ import annotations

from types import SimpleNamespace
from weakref import WeakKeyDictionary
import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.text import Text

from PLIFMix.ui.localization import language_for


class DebouncedFigureCanvasTkAgg(FigureCanvasTkAgg):
    """TkAgg canvas that redraws once after resize activity settles."""

    def __init__(self, figure=None, master=None, resize_delay_ms: int = 150):
        self.resize_delay_ms = resize_delay_ms
        self._resize_job: str | None = None
        self._pending_size: tuple[int, int] | None = None
        self._draw_pending = True
        self._active = True
        self._text_sources = WeakKeyDictionary()
        super().__init__(figure, master)
        widget = self.get_tk_widget()
        self._language = language_for(widget)
        self._language.register(self)
        widget._plif_mpl_canvas = self  # type: ignore[attr-defined]
        widget.bind("<Map>", self._handle_map, add="+")

    def resize(self, event) -> None:
        width, height = int(event.width), int(event.height)
        if width <= 1 or height <= 1:
            return
        self._pending_size = (width, height)
        self._cancel_resize_job()
        self._resize_job = self.get_tk_widget().after(self.resize_delay_ms, self._apply_pending_resize)

    def draw_idle(self) -> None:
        self.request_draw()

    def draw(self) -> None:
        self._translate_figure_text()
        super().draw()

    def _translate_figure_text(self) -> None:
        for artist in self.figure.findobj(match=Text):
            current = artist.get_text()
            source, displayed = self._text_sources.get(artist, (current, current))
            if current != displayed:
                source = current
            translated = self._language.text(source)
            if current != translated:
                artist.set_text(translated)
            self._text_sources[artist] = (source, translated)

    def refresh_language(self) -> None:
        self._translate_figure_text()
        self.request_draw()

    def request_draw(self) -> None:
        widget = self.get_tk_widget()
        if not self._active or not widget.winfo_ismapped():
            self._draw_pending = True
            return
        self._draw_pending = False
        super().draw_idle()

    def set_active(self, active: bool) -> None:
        self._active = active
        if not active:
            self._cancel_resize_job()
            return
        if self._pending_size is not None:
            width, height = self._pending_size
            self.resize(SimpleNamespace(width=width, height=height))
        elif self._draw_pending:
            self.get_tk_widget().after_idle(self.request_draw)

    def _apply_pending_resize(self) -> None:
        self._resize_job = None
        if self._pending_size is None:
            return
        widget = self.get_tk_widget()
        if not self._active or not widget.winfo_ismapped():
            self._draw_pending = True
            return
        width, height = self._pending_size
        self._pending_size = None
        super().resize(SimpleNamespace(width=width, height=height))

    def _handle_map(self, _event=None) -> None:
        if self._active:
            self.get_tk_widget().after_idle(self._resume_visible_canvas)

    def _resume_visible_canvas(self) -> None:
        if self._pending_size is not None:
            width, height = self._pending_size
            self.resize(SimpleNamespace(width=width, height=height))
        elif self._draw_pending:
            self.request_draw()

    def _cancel_resize_job(self) -> None:
        if self._resize_job is None:
            return
        try:
            self.get_tk_widget().after_cancel(self._resize_job)
        except tk.TclError:
            pass
        self._resize_job = None


def set_canvas_tree_active(widget: tk.Misc, active: bool) -> None:
    controller = getattr(widget, "_plif_mpl_canvas", None)
    if controller is not None:
        controller.set_active(active)
    for child in widget.winfo_children():
        set_canvas_tree_active(child, active)
