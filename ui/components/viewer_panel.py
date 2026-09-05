from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from PLIFv2.core.image_io import read_image, to_16bit_plif_signal, write_tif
from PLIFv2.core.image_rendering import (
    ComputedRenderOptions,
    OverlayOptions,
    RawRenderOptions,
    copy_display_for_save as _copy_display_for_save,
    draw_overlays_on as _draw_overlays_impl,
    finite_range as _finite_range,
    quad_values as _quad_values,
    render_computed_image,
    render_raw_image,
    rotate_quarter_turns as _rotate_quarter_turns,
    rotate_special_points_quarter_turns as _rotate_special_points_quarter_turns,
    scale_gray_for_display as _scale_gray_for_display,
)
from PLIFv2.ui.coordinate_mapping import ImageCoordinateMapping
from PLIFv2.ui.components.display_pane import HorizontalDisplayPane
from PLIFv2.ui.mpl_canvas import DebouncedFigureCanvasTkAgg


class ViewerPanel(ttk.Frame):
    """Shared image canvas for file-backed and computed image displays."""

    def __init__(
        self,
        master,
        path: Path | None = None,
        show_histogram: bool = True,
        raw_single: np.ndarray | None = None,
        point_selected_callback=None,
        name: str | None = None,
        show_save_button: bool = False,
        retain_base_display: bool = True,
    ):
        super().__init__(master)
        self.show_histogram = show_histogram
        self.name = name or ""
        self.path = Path()
        self.original = np.empty((0, 0), dtype=np.float64)
        self.raw_single = np.empty((0, 0), dtype=np.float64)
        self.value_image = np.empty((0, 0), dtype=np.float64)
        self.value_scale = 1.0
        self._value_min = 0.0
        self._value_max = 0.0
        self._raw_value_range = (0.0, 0.0)
        self.image: np.ndarray | None = None
        self.display: np.ndarray | None = None
        self.base_display: np.ndarray | None = None
        self.zoom = 1.0
        self.render_scale = 1.0
        self.image_offset = (0, 0)
        self.cursor_position: tuple[int, int] | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.canvas_image_id: int | None = None
        self._canvas_image_size: tuple[int, int] | None = None
        self.point_selected_callback = point_selected_callback
        self.retain_base_display = retain_base_display
        self.save_name_suffix = ""
        self.save_folder = Path("manual_outputs")
        self._last_crosshair: tuple[int, int, int] | None = None
        self._last_ellipse: tuple[int, int, int, int, int] | None = None
        self._last_special_points = None
        self.coordinate_mapping: ImageCoordinateMapping | None = None

        if name is not None:
            header = ttk.Frame(self)
            header.pack(anchor="w")
            ttk.Label(header, text=name, width=12).pack(side="left")
            self.info = ttk.Label(header, text="坐标(-, -)  值=-  最小=-  最大=-")
            self.info.pack(side="left")
            if show_save_button:
                ttk.Button(header, text="保存图像", command=self.save_image).pack(side="left", padx=(4, 0))

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
        self.canvas = tk.Canvas(image_host, width=640, height=480, highlightthickness=1, highlightbackground="#aaa")
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
        if name is None:
            self.info = ttk.Label(self, text="坐标(-, -)  值=-  最小=-  最大=-")
            self.info.pack(anchor="w")
        if path is not None:
            self.set_path(path, raw_single)

    def set_path(
        self,
        path: Path,
        raw_single: np.ndarray | None = None,
        raw_range: tuple[float, float] | None = None,
    ) -> None:
        self.path = path
        self.coordinate_mapping = None
        source = read_image(path) if raw_single is None else raw_single
        self.original = source
        self.image = source
        if raw_single is not None and source.ndim == 2:
            self.raw_single = source
            self.value_scale = 256.0 if source.dtype == np.uint8 else 1.0
        else:
            self.raw_single = _to_16bit_single_channel(source)
            self.value_scale = 1.0
        self._raw_value_range = raw_range or _finite_range(self.raw_single)
        self._set_value_image(self.raw_single, self.value_scale, self._raw_value_range)
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
    ) -> float | None:
        image, display, computed_gain, value_range = self._build_display(
            self.raw_single,
            gain=gain,
            auto_gain=auto_gain,
            rotate=rotate,
            shift=shift,
            denoise=denoise,
            pseudo=pseudo,
            center=center,
            ellipse=ellipse,
            crosshair=crosshair,
            line_width=line_width,
            quad=quad,
            zero_threshold_8bit=zero_threshold_8bit,
            value_scale=self.value_scale,
            value_range=self._raw_value_range,
        )
        self.coordinate_mapping = ImageCoordinateMapping.from_parameters(
            self.raw_single.shape[:2], rotation_degrees=rotate, shift=shift
        )
        self._set_value_image(image, self.value_scale, value_range)
        self._refresh_info()
        self.display = display
        self.base_display = self.display.copy() if self.retain_base_display else None
        self._set_zoom_and_render(zoom)
        if self.show_histogram and self.axis is not None and self.hist_canvas is not None:
            values = _quad_values(image, quad).astype(np.float32, copy=False) * self.value_scale
            self.axis.clear()
            self.axis.hist(values.ravel(), bins=100)
            self.axis.set_title("灰度直方图")
            if self.hist_toolbar is not None:
                self.hist_toolbar.update()
            self.hist_canvas.request_draw()
        return computed_gain

    def render_array_for_export(
        self,
        source: np.ndarray,
        gain,
        auto_gain,
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
    ) -> np.ndarray:
        """Render one source image with the same parameters as the viewer."""
        _, display, _, _ = self._build_display(
            _to_16bit_single_channel(source),
            gain=gain,
            auto_gain=auto_gain,
            rotate=rotate,
            shift=shift,
            denoise=denoise,
            pseudo=pseudo,
            center=center,
            ellipse=ellipse,
            crosshair=crosshair,
            line_width=line_width,
            quad=quad,
            zero_threshold_8bit=zero_threshold_8bit,
            value_scale=1.0,
            value_range=None,
        )
        return display

    def _build_display(
        self,
        source: np.ndarray,
        *,
        gain,
        auto_gain,
        rotate,
        shift,
        denoise,
        pseudo,
        center,
        ellipse,
        crosshair,
        line_width,
        quad,
        zero_threshold_8bit: float,
        value_scale: float,
        value_range: tuple[float, float] | None,
    ) -> tuple[np.ndarray, np.ndarray, float | None, tuple[float, float] | None]:
        result = render_raw_image(
            source,
            RawRenderOptions(
                gain=gain,
                auto_gain=auto_gain,
                rotate=rotate,
                shift=shift,
                denoise=denoise,
                pseudo=pseudo,
                value_scale=value_scale,
                value_range=value_range,
                zero_threshold_8bit=zero_threshold_8bit,
                overlays=OverlayOptions(
                    center=center,
                    ellipse=ellipse,
                    crosshair=crosshair,
                    line_width=line_width,
                    quad=quad,
                ),
            ),
        )
        return result.image, result.display, result.computed_gain, result.value_range

    def update_image(
        self,
        image: np.ndarray,
        gain: float = 1,
        colormap: str = "jet",
        zoom: float = 1,
        crosshair: tuple[int, int, int] | None = None,
        ellipse: tuple[int, int, int, int, int] | None = None,
        denoise: float = 0,
        rotate: float = 0,
        rotation_center: tuple[int, int] | None = None,
        save_name_suffix: str = "",
        special_points=None,
        save_folder: str | Path | None = None,
    ) -> None:
        """Render a computed scalar or BGR image with all legacy ImageDisplay features."""
        result = render_computed_image(
            image,
            ComputedRenderOptions(
                gain=gain,
                colormap=colormap,
                rotate=rotate,
                denoise=denoise,
                rotation_center=rotation_center,
                overlays=OverlayOptions(
                    center=(crosshair[0], crosshair[1]) if crosshair else (0, 0),
                    ellipse=ellipse,
                    crosshair=crosshair is not None,
                    line_width=crosshair[2] if crosshair else 1,
                    special_points=special_points,
                ),
                retain_base_display=self.retain_base_display,
            ),
        )
        self.image = result.image
        self.original = result.image
        self.save_name_suffix = save_name_suffix
        if save_folder is not None:
            self.save_folder = Path(save_folder)
        self._set_value_image(result.image, 1.0)
        self.display = result.display
        self.base_display = result.base_display
        self._last_crosshair = crosshair
        self._last_ellipse = ellipse
        self._last_special_points = result.special_points
        self._refresh_info()
        self._set_zoom_and_render(zoom)

    def display_to_source(self, x: int, y: int) -> tuple[int, int] | None:
        if self.coordinate_mapping is None:
            return None
        return self.coordinate_mapping.display_to_source(x, y)

    def _draw_overlays(self, center, ellipse, crosshair, line_width, quad, special_points) -> None:
        if self.display is None:
            return
        self._draw_overlays_on(self.display, center, ellipse, crosshair, line_width, quad, special_points)

    @staticmethod
    def _draw_overlays_on(display, center, ellipse, crosshair, line_width, quad, special_points) -> None:
        _draw_overlays_on(display, center, ellipse, crosshair, line_width, quad, special_points)

    def copy_for_save(
        self,
        include_crosshair: bool,
        include_ellipse: bool,
        crosshair: tuple[int, int, int] | None = None,
        ellipse: tuple[int, int, int, int, int] | None = None,
    ) -> np.ndarray | None:
        return _copy_display_for_save(
            self.base_display,
            include_crosshair,
            include_ellipse,
            crosshair,
            ellipse,
            last_crosshair=self._last_crosshair,
            last_ellipse=self._last_ellipse,
            last_special_points=self._last_special_points,
        )

    def _set_zoom_and_render(self, zoom: float) -> None:
        if zoom <= 0:
            raise ValueError("缩小倍数必须大于0")
        self.zoom = zoom
        self._render_canvas()

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
        x, y = self.cursor_position if self.cursor_position is not None else (-1, -1)
        minimum = self._value_min
        maximum = self._value_max
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            value = image[y, x]
            if np.ndim(value):
                value_text = ", ".join(f"{float(item) * self.value_scale:g}" for item in value)
            else:
                value_text = f"{float(value) * self.value_scale:g}"
            self.info.config(text=f"坐标({x}, {y})  值={value_text}  最小={minimum:g}  最大={maximum:g}")
        else:
            self.info.config(text=f"坐标(-, -)  值=-  最小={minimum:g}  最大={maximum:g}")

    def _set_value_image(
        self,
        image: np.ndarray,
        value_scale: float,
        value_range: tuple[float, float] | None = None,
    ) -> None:
        self.value_image = image
        self.value_scale = value_scale
        native_minimum, native_maximum = value_range or _finite_range(image)
        self._value_min = native_minimum * value_scale
        self._value_max = native_maximum * value_scale

    def save_image(self) -> None:
        if self.display is None:
            return
        folder = self.save_folder / "images" / self.name
        name = self.save_name_suffix or self.name
        write_tif(folder / f"{name}-{self.name}.tif", self.display)


def _to_16bit_single_channel(image: np.ndarray) -> np.ndarray:
    return to_16bit_plif_signal(image)


def _draw_overlays_on(display, center, ellipse, crosshair, line_width, quad, special_points) -> None:
    _draw_overlays_impl(
        display,
        OverlayOptions(
            center=center,
            ellipse=ellipse,
            crosshair=crosshair,
            line_width=line_width,
            quad=quad,
            special_points=special_points,
        ),
    )
