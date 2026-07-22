from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from PLIFv2.algorithms.mixing import shift_image
from PLIFv2.algorithms.visualization import blacked_jet_colormap
from PLIFv2.core.image_io import read_image
from PLIFv2.ui.components.display_pane import HorizontalDisplayPane
from PLIFv2.ui.mpl_canvas import DebouncedFigureCanvasTkAgg

class ViewerPanel(ttk.Frame):
    def __init__(
        self,
        master,
        path: Path,
        show_histogram: bool = True,
        raw_single: np.ndarray | None = None,
        point_selected_callback=None,
    ):
        super().__init__(master)
        self.show_histogram = show_histogram
        self.path = Path()
        self.original = np.empty((0, 0), dtype=np.float64)
        self.raw_single = np.empty((0, 0), dtype=np.float64)
        self.value_image = np.empty((0, 0), dtype=np.float64)
        self.display = None
        self.zoom = 1.0
        self.render_scale = 1.0
        self.image_offset = (0, 0)
        self.cursor_position: tuple[int, int] | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.canvas_image_id: int | None = None
        self._canvas_image_size: tuple[int, int] | None = None
        self.point_selected_callback = point_selected_callback
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        self.display_pane = None
        self.histogram_panel = None
        image_host = body
        if self.show_histogram:
            self.display_pane = HorizontalDisplayPane(body)
            self.display_pane.pack(fill="both", expand=True)
            image_host = self.display_pane.primary_host
            self.histogram_panel = self.display_pane.secondary_host
        self.canvas = tk.Canvas(
            image_host,
            width=640,
            height=480,
            highlightthickness=1,
            highlightbackground="#aaa",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Motion>", self.mouse_info)
        self.canvas.bind("<Button-1>", self._select_point)
        self.figure = None
        self.axis = None
        self.hist_canvas = None
        self.hist_toolbar = None
        if self.show_histogram:
            self.figure = Figure(figsize=(3, 4), dpi=100)
            self.axis = self.figure.add_subplot(111)
            self.hist_canvas = DebouncedFigureCanvasTkAgg(self.figure, master=self.histogram_panel)
            self.hist_canvas.get_tk_widget().pack(fill="both", expand=True)
            toolbar_frame = ttk.Frame(self.histogram_panel)
            toolbar_frame.pack(fill="x")
            self.hist_toolbar = NavigationToolbar2Tk(self.hist_canvas, toolbar_frame)
            message_label = getattr(self.hist_toolbar, "_message_label", None)
            if message_label is not None:
                message_label.configure(width=28, anchor="e")
        self.info = ttk.Label(self, text="坐标(-, -)  值=-  最小=-  最大=-")
        self.info.pack(anchor="w")
        self.set_path(path, raw_single)

    def set_path(self, path: Path, raw_single: np.ndarray | None = None) -> None:
        self.path = path
        if raw_single is None:
            self.original = read_image(path)
            self.raw_single = _to_16bit_single_channel(self.original)
        else:
            self.original = raw_single
            self.raw_single = _to_16bit_single_channel(raw_single)
        self.value_image = self.raw_single
        self._refresh_info()

    def update_display(
        self,
        gain,
        auto_gain,
        zoom,
        rotate,
        shift,
        denoise,
        pseudo,
        center,
        ellipse,
        crosshair,
        line_width,
        quad,
        zero_threshold_8bit: float = 0,
    ) -> None:
        image = self.raw_single
        if zero_threshold_8bit < 0:
            raise ValueError("去噪阈值不能小于0")
        if zero_threshold_8bit > 0:
            image = image.copy()
            image[image <= zero_threshold_8bit * 256] = 0
        if denoise > 0:
            if image is self.raw_single:
                image = image.copy()
            image[image < np.max(image) * denoise] = 0
        if rotate:
            h, w = image.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), -rotate, 1)
            image = cv2.warpAffine(image, matrix, (w, h))
        if shift:
            image = shift_image(image, shift)
        self.value_image = image
        self._refresh_info()
        computed_gain = None
        if auto_gain:
            maximum = float(np.max(image)) if image.size else 0.0
            computed_gain = 65535.0 / maximum if maximum > 0 else None
            gain = computed_gain if computed_gain is not None else 0.0
        scaled = _scale_gray_for_display(image, gain)
        display = cv2.applyColorMap(scaled, blacked_jet_colormap()) if pseudo else cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
        if crosshair:
            cv2.line(display, (center[0], 0), (center[0], display.shape[0]), (0, 0, 255), line_width)
            cv2.line(display, (0, center[1]), (display.shape[1], center[1]), (0, 0, 255), line_width)
        if ellipse:
            cv2.ellipse(display, center, ellipse, 0, 0, 360, (0, 0, 255), line_width)
        if quad is not None:
            cv2.polylines(display, [quad], True, (0, 0, 255), line_width)
        self.display = display
        if zoom <= 0:
            raise ValueError("缩小倍数必须大于0")
        self.zoom = zoom
        self._render_canvas()
        if self.show_histogram and self.axis is not None and self.hist_canvas is not None:
            values = _quad_values(image, quad)
            self.axis.clear()
            self.axis.hist(values.ravel(), bins=100)
            self.axis.set_title("灰度直方图")
            if self.hist_toolbar is not None:
                self.hist_toolbar.update()
            self.hist_canvas.request_draw()
        return computed_gain

    def _render_canvas(self) -> None:
        if self.display is None or not self.winfo_exists():
            return
        image_height, image_width = self.display.shape[:2]
        self.render_scale = 1 / self.zoom
        shown_width = max(1, int(round(image_width / self.zoom)))
        shown_height = max(1, int(round(image_height / self.zoom)))
        shown = cv2.resize(self.display, (shown_width, shown_height), interpolation=cv2.INTER_NEAREST)
        frame_image = Image.fromarray(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB))
        self.image_offset = (0, 0)
        frame_size = (shown_width, shown_height)
        if self.tk_image is None or self._canvas_image_size != frame_size:
            self.tk_image = ImageTk.PhotoImage(frame_image)
            self._canvas_image_size = frame_size
            self.canvas.config(width=shown_width, height=shown_height)
            if self.canvas_image_id is None:
                self.canvas_image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
            else:
                self.canvas.itemconfigure(self.canvas_image_id, image=self.tk_image)
        else:
            self.tk_image.paste(frame_image)

    def mouse_info(self, event) -> None:
        x = int((event.x - self.image_offset[0]) / self.render_scale)
        y = int((event.y - self.image_offset[1]) / self.render_scale)
        self.cursor_position = (x, y)
        self._refresh_info()

    def _select_point(self, event) -> None:
        self.mouse_info(event)
        if self.point_selected_callback is None or self.cursor_position is None:
            return
        x, y = self.cursor_position
        if 0 <= x < self.value_image.shape[1] and 0 <= y < self.value_image.shape[0]:
            self.point_selected_callback(x, y)

    def _refresh_info(self) -> None:
        image = self.value_image
        if image.size == 0:
            return
        if self.cursor_position is not None:
            x, y = self.cursor_position
        else:
            x, y = -1, -1
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            self.info.config(
                text=f"坐标({x}, {y})  值={image[y, x]:g}  最小={np.min(image):g}  最大={np.max(image):g}"
            )
        else:
            self.info.config(text=f"坐标(-, -)  值=-  最小={np.min(image):g}  最大={np.max(image):g}")

def _to_16bit_single_channel(image: np.ndarray) -> np.ndarray:
    source_dtype = image.dtype
    if image.ndim == 3:
        image = image[:, :, 1]
    if source_dtype == np.uint8:
        return image.astype(np.uint16) * 256
    if source_dtype == np.uint16:
        return image
    return image.astype(np.float64)

def _scale_gray_for_display(image: np.ndarray, gain: float) -> np.ndarray:
    scaled = image.astype(np.float32, copy=False) * np.float32(gain / 256)
    return np.clip(scaled, 0, 255).astype(np.uint8)

def _quad_values(image: np.ndarray, quad):
    if quad is None:
        return image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [quad], 255)
    return image[mask > 0]
