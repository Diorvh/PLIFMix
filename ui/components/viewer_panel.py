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
from PLIFv2.algorithms.visualization import blacked_jet_colormap, blue_white_red_colormap
from PLIFv2.core.image_io import read_image, to_16bit_plif_signal, write_tif
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
    ):
        super().__init__(master)
        self.show_histogram = show_histogram
        self.name = name or ""
        self.path = Path()
        self.original = np.empty((0, 0), dtype=np.float64)
        self.raw_single = np.empty((0, 0), dtype=np.float64)
        self.value_image = np.empty((0, 0), dtype=np.float64)
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
        self.save_name_suffix = ""
        self.save_folder = Path("manual_outputs")
        self._last_crosshair: tuple[int, int, int] | None = None
        self._last_ellipse: tuple[int, int, int, int, int] | None = None
        self._last_special_points = None

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

    def set_path(self, path: Path, raw_single: np.ndarray | None = None) -> None:
        self.path = path
        source = read_image(path) if raw_single is None else raw_single
        self.original = source
        self.image = source
        self.raw_single = _to_16bit_single_channel(source)
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
    ) -> float | None:
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
            height, width = image.shape
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -rotate, 1)
            image = cv2.warpAffine(image, matrix, (width, height))
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
        self.display = cv2.applyColorMap(scaled, blacked_jet_colormap()) if pseudo else cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
        self.base_display = self.display.copy()
        self._draw_overlays(center, ellipse, crosshair, line_width, quad, None)
        self._set_zoom_and_render(zoom)
        if self.show_histogram and self.axis is not None and self.hist_canvas is not None:
            values = _quad_values(image, quad)
            self.axis.clear()
            self.axis.hist(values.ravel(), bins=100)
            self.axis.set_title("灰度直方图")
            if self.hist_toolbar is not None:
                self.hist_toolbar.update()
            self.hist_canvas.request_draw()
        return computed_gain

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
        if rotate:
            height, width = image.shape[:2]
            center = rotation_center or (width / 2, height / 2)
            matrix = cv2.getRotationMatrix2D(center, -rotate, 1)
            image = cv2.warpAffine(image, matrix, (width, height))
            special_points = _rotate_special_points(special_points, matrix, image.shape[:2])
        self.image = image
        self.original = image
        self.save_name_suffix = save_name_suffix
        if save_folder is not None:
            self.save_folder = Path(save_folder)
        if image.ndim == 2:
            self.value_image = image
            shown = image.copy()
            if denoise > 0:
                shown[shown < np.max(shown) * denoise] = 0
            if gain == 0:
                maximum = float(np.nanmax(shown)) if shown.size else 0
                gain = 65535 / maximum if maximum else 1
            shown = shown * gain
            if shown.dtype != np.uint8:
                shown = shown / 256
            shown = np.clip(shown, 0, 255).astype(np.uint8)
            if colormap == "bwr":
                self.display = cv2.applyColorMap(shown, blue_white_red_colormap())
            elif colormap == "none":
                self.display = cv2.cvtColor(shown, cv2.COLOR_GRAY2BGR)
            else:
                self.display = cv2.applyColorMap(shown, blacked_jet_colormap())
        else:
            self.value_image = image
            self.display = np.clip(image, 0, 255).astype(np.uint8)
        self.base_display = self.display.copy()
        self._last_crosshair = crosshair
        self._last_ellipse = ellipse
        self._last_special_points = special_points
        center = (crosshair[0], crosshair[1]) if crosshair else (0, 0)
        line_width = crosshair[2] if crosshair else 1
        self._draw_overlays(center, ellipse, crosshair is not None, line_width, None, special_points)
        self._refresh_info()
        self._set_zoom_and_render(zoom)

    def _draw_overlays(self, center, ellipse, crosshair, line_width, quad, special_points) -> None:
        if self.display is None:
            return
        self._draw_overlays_on(self.display, center, ellipse, crosshair, line_width, quad, special_points)

    @staticmethod
    def _draw_overlays_on(display, center, ellipse, crosshair, line_width, quad, special_points) -> None:
        if special_points is not None and len(special_points):
            try:
                y, x = special_points
                display[y, x] = [255, 255, 255]
            except (IndexError, TypeError, ValueError):
                pass
        if crosshair:
            cv2.line(display, (center[0], 0), (center[0], display.shape[0]), (0, 0, 255), line_width)
            cv2.line(display, (0, center[1]), (display.shape[1], center[1]), (0, 0, 255), line_width)
        if ellipse:
            if len(ellipse) == 2:
                cv2.ellipse(display, center, ellipse, 0, 0, 360, (0, 0, 255), line_width)
            else:
                x, y, major, minor, ellipse_width = ellipse
                cv2.ellipse(display, (x, y), (major, minor), 0, 0, 360, (0, 0, 255), ellipse_width)
        if quad is not None:
            cv2.polylines(display, [quad], True, (0, 0, 255), line_width)

    def copy_for_save(
        self,
        include_crosshair: bool,
        include_ellipse: bool,
        crosshair: tuple[int, int, int] | None = None,
        ellipse: tuple[int, int, int, int, int] | None = None,
    ) -> np.ndarray | None:
        if self.base_display is None:
            return None
        image = self.base_display.copy()
        selected_crosshair = crosshair if include_crosshair else None
        selected_ellipse = ellipse if include_ellipse else None
        if selected_crosshair is None and include_crosshair:
            selected_crosshair = self._last_crosshair
        if selected_ellipse is None and include_ellipse:
            selected_ellipse = self._last_ellipse
        center = (selected_crosshair[0], selected_crosshair[1]) if selected_crosshair else (0, 0)
        line_width = selected_crosshair[2] if selected_crosshair else 1
        self._draw_overlays_on(
            image,
            center,
            selected_ellipse,
            selected_crosshair is not None,
            line_width,
            None,
            self._last_special_points,
        )
        return image

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
        minimum = float(np.nanmin(image))
        maximum = float(np.nanmax(image))
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            value = image[y, x]
            value_text = ", ".join(f"{item:g}" for item in value) if np.ndim(value) else f"{value:g}"
            self.info.config(text=f"坐标({x}, {y})  值={value_text}  最小={minimum:g}  最大={maximum:g}")
        else:
            self.info.config(text=f"坐标(-, -)  值=-  最小={minimum:g}  最大={maximum:g}")

    def save_image(self) -> None:
        if self.display is None:
            return
        folder = self.save_folder / "images" / self.name
        name = self.save_name_suffix or self.name
        write_tif(folder / f"{name}-{self.name}.tif", self.display)


def _to_16bit_single_channel(image: np.ndarray) -> np.ndarray:
    return to_16bit_plif_signal(image)


def _rotate_special_points(special_points, matrix: np.ndarray, shape: tuple[int, int]):
    if special_points is None or len(special_points) != 2:
        return special_points
    try:
        y_points = np.asarray(special_points[0])
        x_points = np.asarray(special_points[1])
        if y_points.size != x_points.size:
            return special_points
        coordinates = np.column_stack((x_points, y_points, np.ones(x_points.size)))
        rotated = coordinates @ matrix.T
        x_rotated = np.rint(rotated[:, 0]).astype(np.int32)
        y_rotated = np.rint(rotated[:, 1]).astype(np.int32)
        height, width = shape
        valid = (0 <= x_rotated) & (x_rotated < width) & (0 <= y_rotated) & (y_rotated < height)
        return y_rotated[valid], x_rotated[valid]
    except (TypeError, ValueError):
        return special_points


def _scale_gray_for_display(image: np.ndarray, gain: float) -> np.ndarray:
    scaled = image.astype(np.float32, copy=False) * np.float32(gain / 256)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _quad_values(image: np.ndarray, quad):
    if quad is None:
        return image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [quad], 255)
    return image[mask > 0]
