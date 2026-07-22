from __future__ import annotations

import re
import time
import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from math import ceil
from pathlib import Path
from tkinter import filedialog, ttk

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from PLIFv2.algorithms.mixing import shift_image
from PLIFv2.algorithms.visualization import blacked_jet_colormap
from PLIFv2.core.image_io import read_image, write_tif
from PLIFv2.tools.image_tools import create_colorbar, export_gif, save_colorbar, top_k_values, transform_npy
from PLIFv2.ui.mpl_canvas import DebouncedFigureCanvasTkAgg


class ToolboxPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)
        for title, page_type in [
            ("图像查看器", ImageViewerTool),
            ("图片播放器", ImagePlayerTool),
            ("NPY矩阵", NpyTool),
            ("GIF导出", GifTool),
            ("Colorbar", ColorbarTool),
            ("分布对比", DistributionComparisonTool),
        ]:
            notebook.add(page_type(notebook), text=title)


class ImageViewerTool(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.vars = {name: tk.StringVar(value=value) for name, value in {
            "image_path": "", "folder_path": "", "experiment_number": "",
            "gain": "1", "zoom": "2", "rotate": "0", "shift": "0", "denoise": "0",
            "center_x": "0", "center_y": "0", "major": "100", "minor": "100",
            "line_width": "2",
            "quad_x1": "", "quad_y1": "", "quad_x2": "", "quad_y2": "",
            "quad_x3": "", "quad_y3": "", "quad_x4": "", "quad_y4": "",
        }.items()}
        self.input_mode = tk.StringVar(value="image")
        self.pseudo = tk.BooleanVar(value=False)
        self.auto_gain = tk.BooleanVar(value=False)
        self.auto_gain_text = tk.StringVar(value="（自动增益：-）")
        self.crosshair = tk.BooleanVar(value=False)
        self.ellipse = tk.BooleanVar(value=False)
        self.quad_visible = tk.BooleanVar(value=False)
        self.panel: ViewerPanel | None = None
        self._setup_controls()
        self.display_host = ttk.Frame(self)
        self.display_host.pack(fill="both", expand=True, padx=6, pady=4)

    def _setup_controls(self) -> None:
        row1 = ttk.Frame(self)

        row2 = ttk.Frame(self)
        row2.pack(anchor="w", padx=6, pady=2)
        ttk.Checkbutton(row2, text="伪彩图", variable=self.pseudo, command=self.update_current).pack(side="left", padx=3)
        self._entry(row2, "gain", "增益", 6)
        ttk.Checkbutton(row2, variable=self.auto_gain, command=self.update_current).pack(side="left", padx=(5, 1))
        ttk.Label(row2, textvariable=self.auto_gain_text).pack(side="left", padx=(0, 3))
        self._entry(row2, "zoom", "缩小", 6)
        self._entry(row2, "denoise", "相对去噪阈值", 7)
        for name, label, width in [("rotate", "旋转", 6), ("shift", "左移", 6)]:
            self._entry(row2, name, label, width)

        row3 = ttk.Frame(self)
        row3.pack(anchor="w", padx=6, pady=2)
        self._entry(row3, "line_width", "线宽", 6)
        ttk.Checkbutton(row3, text="十字线", variable=self.crosshair, command=self.update_current).pack(side="left", padx=3)
        self._entry(row3, "center_x", "X", 6)
        self._entry(row3, "center_y", "Y", 6)
        ttk.Checkbutton(row3, text="椭圆", variable=self.ellipse, command=self.update_current).pack(side="left", padx=3)
        self._entry(row3, "major", "长轴", 6)
        self._entry(row3, "minor", "短轴", 6)

        row4 = ttk.Frame(self)
        row4.pack(anchor="w", padx=6, pady=2)
        ttk.Checkbutton(row4, text="显示四边形", variable=self.quad_visible, command=self.update_current).pack(side="left", padx=3)
        for index in range(1, 5):
            self._entry(row4, f"quad_x{index}", f"X{index}", 6)
            self._entry(row4, f"quad_y{index}", f"Y{index}", 6)

        row5 = ttk.Frame(self)
        row5.pack(anchor="w", padx=6, pady=(2, 3))
        ttk.Button(row5, text="保存当前显示", command=self.save_current).pack(side="left")

        path_area = ttk.Frame(self)
        path_area.pack(fill="x", padx=6, pady=(5, 2), before=row2)
        ttk.Radiobutton(
            path_area,
            text="图片",
            variable=self.input_mode,
            value="image",
            command=lambda: self._load_path("image"),
        ).grid(
            row=0, column=0, sticky="w", pady=3
        )
        image_entry = ttk.Entry(path_area, textvariable=self.vars["image_path"], width=62)
        image_entry.grid(row=0, column=1, sticky="w", padx=(4, 0), pady=3)
        ttk.Button(path_area, text="...", width=3, command=self._choose_image_path).grid(
            row=0, column=2, sticky="w", padx=(3, 0), pady=3
        )
        image_entry.bind("<Return>", lambda _event: self._load_path("image"))

        ttk.Radiobutton(
            path_area,
            text="文件夹",
            variable=self.input_mode,
            value="folder",
            command=lambda: self._load_path("folder"),
        ).grid(
            row=1, column=0, sticky="w", pady=3
        )
        folder_entry = ttk.Entry(path_area, textvariable=self.vars["folder_path"], width=62)
        folder_entry.grid(row=1, column=1, sticky="w", padx=(4, 0), pady=3)
        ttk.Button(path_area, text="...", width=3, command=self._choose_folder_path).grid(
            row=1, column=2, sticky="w", padx=(3, 0), pady=3
        )
        ttk.Label(path_area, text="实验编号").grid(row=1, column=3, sticky="w", padx=(8, 2), pady=3)
        experiment_entry = ttk.Entry(path_area, textvariable=self.vars["experiment_number"], width=8)
        experiment_entry.grid(row=1, column=4, sticky="w", pady=3)
        ttk.Label(path_area, text=".npy").grid(row=1, column=5, sticky="w", padx=(2, 0), pady=3)
        ttk.Button(path_area, text="上一张", command=lambda: self.move_image(-1)).grid(
            row=1, column=6, sticky="w", padx=(8, 0), pady=3
        )
        ttk.Button(path_area, text="下一张", command=lambda: self.move_image(1)).grid(
            row=1, column=7, sticky="w", padx=(3, 0), pady=3
        )
        folder_entry.bind("<Return>", lambda _event: self._load_path("folder"))
        experiment_entry.bind("<Return>", lambda _event: self._load_path("folder"))

        self.load_status = ttk.Label(self, text="")
        self.load_status.pack(anchor="w", padx=9, before=row2)

    def _entry(self, row, name: str, label: str, width: int) -> None:
        ttk.Label(row, text=label).pack(side="left", padx=(4, 1))
        entry = ttk.Entry(row, textvariable=self.vars[name], width=width)
        entry.pack(side="left")
        entry.bind("<Return>", lambda event: self.update_current())

    def _choose_image_path(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("图像", "*.png *.tif *.tiff *.bmp *.npy")])
        if path:
            self.input_mode.set("image")
            self.vars["image_path"].set(path)
            self._load_path("image")

    def _choose_folder_path(self) -> None:
        folder = filedialog.askdirectory()
        if folder:
            self.input_mode.set("folder")
            self.vars["folder_path"].set(folder)
            self._load_path("folder")

    def _load_path(self, mode: str | None = None) -> None:
        mode = mode or self.input_mode.get()
        self.input_mode.set(mode)
        if mode == "image":
            path = Path(self.vars["image_path"].get().strip())
        else:
            path = _experiment_npy_path(
                self.vars["folder_path"].get(),
                self.vars["experiment_number"].get(),
            )
        if not str(path) or str(path) == "." or not path.is_file():
            self.load_status.config(text=f"文件不存在: {path}")
            return
        self.load_status.config(text="")
        self._show_path(path)

    def _show_path(self, path: Path) -> None:
        if self.panel is None:
            self.panel = ViewerPanel(self.display_host, path)
            self.panel.pack(fill="both", expand=True)
        else:
            self.panel.set_path(path)
        self.update_current()

    def current_panel(self):
        return self.panel

    def update_current(self) -> None:
        panel = self.current_panel()
        if panel is None:
            return
        try:
            auto_gain = self.auto_gain.get()
            computed_gain = panel.update_display(
                gain=1.0 if auto_gain else float(self.vars["gain"].get()),
                auto_gain=auto_gain,
                zoom=float(self.vars["zoom"].get()),
                rotate=float(self.vars["rotate"].get()), shift=int(self.vars["shift"].get()),
                denoise=float(self.vars["denoise"].get()), pseudo=self.pseudo.get(),
                center=(int(self.vars["center_x"].get()), int(self.vars["center_y"].get())),
                ellipse=(int(self.vars["major"].get()), int(self.vars["minor"].get())) if self.ellipse.get() else None,
                crosshair=self.crosshair.get(), line_width=int(self.vars["line_width"].get()),
                quad=_parse_quad_entries(self.vars) if self.quad_visible.get() else None,
            )
            self.auto_gain_text.set(
                f"（自动增益：{computed_gain:.2f}）" if computed_gain is not None else "（自动增益：-）"
            )
        except Exception as exc:
            panel.info.config(text=f"错误: {exc}")

    def move_image(self, delta: int) -> None:
        panel = self.current_panel()
        if panel is None:
            return
        target = _adjacent_image_path(panel.path, delta)
        if target is None:
            panel.info.config(text=f"未找到编号相邻的{'上一张' if delta < 0 else '下一张'}图片")
            return
        if self.input_mode.get() == "folder":
            match = re.fullmatch(r"实验编号\s*(\d+)", target.stem)
            if match:
                self.vars["folder_path"].set(str(target.parent))
                self.vars["experiment_number"].set(match.group(1))
        else:
            self.vars["image_path"].set(str(target))
        self.load_status.config(text="")
        self._show_path(target)

    def save_current(self) -> None:
        panel = self.current_panel()
        if panel is None or panel.display is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIF", "*.tif")])
        if path:
            write_tif(path, panel.display)


class ViewerPanel(ttk.Frame):
    def __init__(
        self,
        master,
        path: Path,
        show_histogram: bool = True,
        raw_single: np.ndarray | None = None,
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
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(
            body,
            width=640,
            height=480,
            highlightthickness=1,
            highlightbackground="#aaa",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Motion>", self.mouse_info)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        self.figure = None
        self.axis = None
        self.hist_canvas = None
        self.hist_toolbar = None
        self.histogram_panel = None
        if self.show_histogram:
            self.histogram_panel = ttk.Frame(body, width=460)
            self.histogram_panel.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
            self.histogram_panel.grid_propagate(False)
            body.columnconfigure(1, weight=0, minsize=460)
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
            self.original = np.empty((0, 0), dtype=np.float64)
            self.raw_single = raw_single
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
    image = image.astype(np.float64)
    if source_dtype == np.uint8:
        image *= 256
    return image


def _read_player_raw_frame(path: Path) -> np.ndarray:
    return _to_16bit_single_channel(read_image(path))


def _scale_gray_for_display(image: np.ndarray, gain: float) -> np.ndarray:
    scaled = image * gain / 256
    return np.clip(scaled, 0, 255).astype(np.uint8)


class ImagePlayerTool(ttk.Frame):
    PREFETCH_FRAME_COUNT = 10

    def __init__(self, master):
        super().__init__(master)
        self.files: list[Path] = []
        self.index = 0
        self.play_delta = 0
        self._prefetch_direction = 1
        self._prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="plif-player")
        self._prefetch_futures: dict[int, Future[np.ndarray]] = {}
        self.folder = tk.StringVar()
        self.frame_rate = tk.StringVar(value="20")
        self.current_frame = tk.StringVar(value="0")
        self.gain = tk.StringVar(value="1")
        self.auto_gain = tk.BooleanVar(value=False)
        self.auto_gain_text = tk.StringVar(value="（自动增益：-）")
        self.zoom = tk.StringVar(value="2")
        self.relative_denoise = tk.StringVar(value="0")
        self.rotate = tk.StringVar(value="0")
        self.shift = tk.StringVar(value="0")
        self.denoise_threshold = tk.StringVar(value="0")
        self.pseudo = tk.BooleanVar(value=False)

        row1 = ttk.Frame(self)
        row1.pack(anchor="w", padx=8, pady=(6, 2))
        ttk.Button(row1, text="选择文件夹", command=self.choose).pack(side="left")
        folder_entry = ttk.Entry(row1, textvariable=self.folder, width=62)
        folder_entry.pack(side="left", padx=3)
        folder_entry.bind("<Return>", self.load_folder)

        row2 = ttk.Frame(self)
        row2.pack(anchor="w", padx=8, pady=2)
        ttk.Label(row2, text="帧率/Hz").pack(side="left")
        ttk.Entry(row2, textvariable=self.frame_rate, width=7).pack(side="left", padx=(3, 10))
        ttk.Button(row2, text="反向播放", command=lambda: self.start(-1)).pack(side="left", padx=2)
        ttk.Button(row2, text="暂停", command=lambda: self.start(0)).pack(side="left", padx=2)
        ttk.Button(row2, text="正向播放", command=lambda: self.start(1)).pack(side="left", padx=2)

        row3 = ttk.Frame(self)
        row3.pack(anchor="w", padx=8, pady=(2, 4))
        ttk.Label(row3, text="当前帧").pack(side="left")
        frame_entry = ttk.Entry(row3, textvariable=self.current_frame, width=8)
        frame_entry.pack(side="left", padx=3)
        frame_entry.bind("<Return>", self.jump_to)
        ttk.Button(row3, text="上一张", command=lambda: self.step(-1)).pack(side="left", padx=2)
        ttk.Button(row3, text="下一张", command=lambda: self.step(1)).pack(side="left", padx=2)

        row4 = ttk.Frame(self)
        row4.pack(anchor="w", padx=8, pady=(2, 4))
        ttk.Checkbutton(row4, text="伪彩图", variable=self.pseudo, command=self.show).pack(side="left", padx=(0, 5))
        self._display_entry(row4, "增益", self.gain)
        ttk.Checkbutton(row4, variable=self.auto_gain, command=self.show).pack(side="left", padx=(5, 1))
        ttk.Label(row4, textvariable=self.auto_gain_text).pack(side="left", padx=(0, 3))
        self._display_entry(row4, "缩小", self.zoom)
        self._display_entry(row4, "相对去噪阈值", self.relative_denoise)
        self._display_entry(row4, "旋转", self.rotate)
        self._display_entry(row4, "左移", self.shift)
        self._display_entry(row4, "去噪阈值(8位)", self.denoise_threshold)

        self.info = ttk.Label(self, text="")
        self.info.pack(anchor="w", padx=8)
        self.display_host = ttk.Frame(self)
        self.display_host.pack(fill="both", expand=True, padx=6, pady=4)
        self.panel: ViewerPanel | None = None

    def _display_entry(self, row: ttk.Frame, label: str, variable: tk.StringVar) -> None:
        ttk.Label(row, text=label).pack(side="left", padx=(4, 2))
        entry = ttk.Entry(row, textvariable=variable, width=7)
        entry.pack(side="left")
        entry.bind("<Return>", lambda _event: self.show())

    def choose(self) -> None:
        folder = filedialog.askdirectory()
        if not folder:
            return
        self.folder.set(folder)
        self.load_folder()

    def load_folder(self, event=None) -> None:
        folder = Path(self.folder.get().strip())
        if not folder.is_dir():
            self.info.config(text=f"文件夹不存在: {folder}")
            return
        self.play_delta = 0
        self._reset_prefetch()
        self.files = sorted(
            path for path in folder.iterdir()
            if path.suffix.lower() in {".png", ".tif", ".tiff", ".bmp", ".npy"}
        )
        self.index = 0
        self.current_frame.set("0")
        if not self.files:
            self.info.config(text="文件夹中没有可读取的图像")
            return
        self.show()

    def show(self) -> None:
        if not self.files:
            return
        path = self.files[self.index]
        raw_single = self._take_prefetched_frame(self.index)
        try:
            auto_gain = self.auto_gain.get()
            gain = 1.0 if auto_gain else float(self.gain.get())
            zoom = float(self.zoom.get())
            relative_denoise = float(self.relative_denoise.get())
            rotate = float(self.rotate.get())
            shift = int(self.shift.get())
            denoise_threshold = float(self.denoise_threshold.get())
            if zoom <= 0:
                raise ValueError("缩小倍数必须大于0")
            if relative_denoise < 0 or denoise_threshold < 0:
                raise ValueError("去噪阈值不能小于0")
        except ValueError as exc:
            self.info.config(text=f"显示参数错误: {exc}")
            return
        if self.panel is None:
            self.panel = ViewerPanel(self.display_host, path, show_histogram=False, raw_single=raw_single)
            self.panel.pack(fill="both", expand=True)
        else:
            self.panel.set_path(path, raw_single)
        computed_gain = self.panel.update_display(
            gain=gain,
            auto_gain=auto_gain,
            zoom=zoom,
            rotate=rotate,
            shift=shift,
            denoise=relative_denoise,
            pseudo=self.pseudo.get(),
            center=(0, 0),
            ellipse=None,
            crosshair=False,
            line_width=1,
            quad=None,
            zero_threshold_8bit=denoise_threshold,
        )
        self.auto_gain_text.set(
            f"（自动增益：{computed_gain:.2f}）" if computed_gain is not None else "（自动增益：-）"
        )
        self.current_frame.set(str(self.index))
        self.info.config(text="")
        self._schedule_prefetch()

    def start(self, delta: int) -> None:
        self.play_delta = delta
        if delta:
            self._set_prefetch_direction(delta)
            self._tick()

    def _tick(self) -> None:
        if not self.play_delta or not self.files:
            return
        try:
            frame_rate = float(self.frame_rate.get())
            if frame_rate <= 0:
                raise ValueError
        except ValueError:
            self.play_delta = 0
            self.info.config(text="帧率必须是大于0的数值")
            return
        frame_started = time.perf_counter()
        self.index = (self.index + self.play_delta) % len(self.files)
        self.show()
        remaining_seconds = (1 / frame_rate) - (time.perf_counter() - frame_started)
        if remaining_seconds > 0:
            self.after(ceil(remaining_seconds * 1000), self._tick)
        else:
            self.after_idle(self._tick)

    def step(self, delta: int) -> None:
        if not self.files:
            return
        self.play_delta = 0
        self._set_prefetch_direction(delta)
        self.index = (self.index + delta) % len(self.files)
        self.show()

    def jump_to(self, event=None) -> None:
        if self.files:
            try:
                self.index = min(max(0, int(self.current_frame.get())), len(self.files) - 1)
            except ValueError:
                self.info.config(text="当前帧必须是整数")
                return
            self._reset_prefetch()
            self.show()

    def _set_prefetch_direction(self, direction: int) -> None:
        if direction != self._prefetch_direction:
            self._prefetch_direction = direction
            self._reset_prefetch()

    def _schedule_prefetch(self) -> None:
        if not self.files:
            return
        direction = self.play_delta or self._prefetch_direction
        desired = [
            (self.index + direction * offset) % len(self.files)
            for offset in range(1, self.PREFETCH_FRAME_COUNT + 1)
        ]
        desired_set = set(desired)
        for frame_index, future in tuple(self._prefetch_futures.items()):
            if frame_index not in desired_set:
                future.cancel()
                del self._prefetch_futures[frame_index]
        for frame_index in desired:
            if frame_index not in self._prefetch_futures:
                self._prefetch_futures[frame_index] = self._prefetch_executor.submit(
                    _read_player_raw_frame,
                    self.files[frame_index],
                )

    def _take_prefetched_frame(self, frame_index: int) -> np.ndarray | None:
        future = self._prefetch_futures.pop(frame_index, None)
        if future is None or not future.done():
            if future is not None:
                future.cancel()
            return None
        try:
            return future.result()
        except Exception:
            return None

    def _reset_prefetch(self) -> None:
        for future in self._prefetch_futures.values():
            future.cancel()
        self._prefetch_futures.clear()

    def destroy(self) -> None:
        self._reset_prefetch()
        self._prefetch_executor.shutdown(wait=False, cancel_futures=True)
        super().destroy()


class NpyTool(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.path = tk.StringVar()
        self.k = tk.StringVar(value="10")
        self.operation = tk.StringVar(value="multiply")
        self.value = tk.StringVar(value="1")
        self.array = None
        row = ttk.Frame(self)
        row.pack(anchor="w", padx=8, pady=6)
        ttk.Button(row, text="打开NPY", command=self.open).pack(side="left")
        ttk.Entry(row, textvariable=self.path, width=50).pack(side="left", padx=3)
        ttk.Label(row, text="Top K").pack(side="left")
        ttk.Entry(row, textvariable=self.k, width=6).pack(side="left")
        ttk.Button(row, text="查找", command=self.show_topk).pack(side="left", padx=3)
        ttk.Label(row, text="运算").pack(side="left")
        ttk.Combobox(row, textvariable=self.operation, values=["multiply", "add", "normalize", "reciprocal"], width=12, state="readonly").pack(side="left")
        ttk.Entry(row, textvariable=self.value, width=7).pack(side="left")
        ttk.Button(row, text="另存运算结果", command=self.transform).pack(side="left", padx=3)
        self.info = tk.Text(self, height=8, wrap="word")
        self.info.pack(fill="x", padx=8)
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self.colorbar = None
        self.canvas = DebouncedFigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=6)

    def open(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("NPY", "*.npy")])
        if not path:
            return
        self.path.set(path)
        self.array = np.load(path)
        self.info.delete("1.0", tk.END)
        self.info.insert(tk.END, f"shape={self.array.shape}\ndtype={self.array.dtype}\nmin={np.min(self.array)}\nmax={np.max(self.array)}\nmean={np.mean(self.array)}\nstd={np.std(self.array)}")
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None
        self.axis.clear()
        if self.array.ndim == 2:
            image = self.axis.imshow(self.array, cmap="viridis")
            self.colorbar = self.figure.colorbar(image, ax=self.axis, fraction=0.046)
        else:
            self.axis.plot(self.array.ravel())
        self.canvas.request_draw()

    def show_topk(self) -> None:
        if self.array is None:
            return
        result = top_k_values(self.array, int(self.k.get()))
        self.info.insert(tk.END, "\n\nTop K:\n" + "\n".join(f"{value}: {index}" for value, index in result))

    def transform(self) -> None:
        if not self.path.get():
            return
        output = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("NPY", "*.npy")])
        if output:
            transform_npy(Path(self.path.get()), Path(output), self.operation.get(), float(self.value.get()))
            self.info.insert(tk.END, f"\n已保存 {output}")


class GifTool(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.vars = {name: tk.StringVar(value=value) for name, value in {"folder": "", "output": "", "start": "0", "end": "200", "gain": "1", "duration": "50", "width": "400"}.items()}
        row = ttk.Frame(self)
        row.pack(anchor="w", padx=8, pady=8)
        ttk.Button(row, text="图像目录", command=self.choose_folder).pack(side="left")
        ttk.Entry(row, textvariable=self.vars["folder"], width=40).pack(side="left", padx=3)
        ttk.Button(row, text="输出GIF", command=self.choose_output).pack(side="left")
        ttk.Entry(row, textvariable=self.vars["output"], width=36).pack(side="left", padx=3)
        for name, label, width in [("start", "开始", 7), ("end", "结束", 7), ("gain", "增益", 7), ("duration", "帧间隔ms", 7), ("width", "最大宽度", 7)]:
            ttk.Label(row, text=label).pack(side="left")
            ttk.Entry(row, textvariable=self.vars[name], width=width).pack(side="left")
        ttk.Button(row, text="生成GIF", command=self.run).pack(side="left", padx=4)
        self.info = ttk.Label(self, text="")
        self.info.pack(anchor="w", padx=8)

    def choose_folder(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.vars["folder"].set(path)
            self.vars["output"].set(str(Path(path).parent / "spray.gif"))

    def choose_output(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".gif", filetypes=[("GIF", "*.gif")])
        if path:
            self.vars["output"].set(path)

    def run(self) -> None:
        try:
            result = export_gif(Path(self.vars["folder"].get()), Path(self.vars["output"].get()), int(self.vars["start"].get()), int(self.vars["end"].get()), float(self.vars["gain"].get()), int(self.vars["duration"].get()), int(self.vars["width"].get()))
            self.info.config(text=f"已生成 {result}")
        except Exception as exc:
            self.info.config(text=f"错误: {exc}")


class ColorbarTool(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.vars = {name: tk.StringVar(value=value) for name, value in {"height": "800", "width": "50", "minimum": "0", "maximum": "1", "top": "20", "bottom": "20"}.items()}
        row = ttk.Frame(self)
        row.pack(anchor="w", padx=8, pady=8)
        for name, label in [("height", "高度"), ("width", "宽度"), ("minimum", "最小"), ("maximum", "最大"), ("top", "上留白"), ("bottom", "下留白")]:
            ttk.Label(row, text=label).pack(side="left")
            ttk.Entry(row, textvariable=self.vars[name], width=8).pack(side="left", padx=2)
        ttk.Button(row, text="预览", command=self.preview).pack(side="left", padx=4)
        ttk.Button(row, text="保存", command=self.save).pack(side="left", padx=4)
        self.canvas = tk.Canvas(self, highlightthickness=1, highlightbackground="#aaa")
        self.canvas.pack(anchor="nw", padx=8, pady=6)
        self.image = None

    def preview(self) -> None:
        self.image = create_colorbar(int(self.vars["height"].get()), int(self.vars["width"].get()), float(self.vars["minimum"].get()), float(self.vars["maximum"].get()), int(self.vars["top"].get()), int(self.vars["bottom"].get()))
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.tk_image = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.config(width=rgb.shape[1], height=rgb.shape[0])
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def save(self) -> None:
        if self.image is None:
            self.preview()
        path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIF", "*.tif")])
        if path:
            save_colorbar(Path(path), self.image)


class DistributionComparisonTool(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        ttk.Button(self, text="绘制常见中心对称分布", command=self.draw).pack(anchor="w", padx=8, pady=6)
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self.canvas = DebouncedFigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8)
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=8)
        NavigationToolbar2Tk(self.canvas, toolbar)

    def draw(self) -> None:
        from scipy import stats
        x = np.linspace(-10, 10, 1000)
        distributions = {"Normal": stats.norm(), "Laplace": stats.laplace(), "Logistic": stats.logistic(), "t (df=5)": stats.t(5), "Cauchy": stats.cauchy()}
        self.axis.clear()
        for name, distribution in distributions.items():
            self.axis.plot(x, distribution.pdf(x), label=name)
        self.axis.legend()
        self.axis.set_title("中心对称分布比较")
        self.axis.grid(True, alpha=0.3)
        self.canvas.request_draw()


def _parse_quad_entries(variables: dict[str, tk.StringVar]) -> np.ndarray:
    values = []
    for index in range(1, 5):
        x = variables[f"quad_x{index}"].get().strip()
        y = variables[f"quad_y{index}"].get().strip()
        if not x or not y:
            raise ValueError("请完整填写四边形的8个坐标")
        values.append((int(x), int(y)))
    return np.array(values, dtype=np.int32)


def _experiment_npy_path(folder_text: str, number_text: str) -> Path:
    number = number_text.strip()
    if number.startswith("实验编号"):
        number = number.removeprefix("实验编号").strip()
    return Path(folder_text.strip()) / f"实验编号{number}.npy"


def _adjacent_image_path(path: Path, delta: int) -> Path | None:
    if path.stem.isdigit():
        number = int(path.stem) + delta
        if number < 0:
            return None
        target_stem = f"{number:04d}"
    else:
        match = re.fullmatch(r"实验编号\s*(\d+)", path.stem)
        if match is None:
            return None
        number = int(match.group(1)) + delta
        if number < 0:
            return None
        target_stem = f"实验编号{number}"

    for candidate in path.parent.iterdir():
        if candidate.is_file() and candidate.stem == target_stem and candidate.suffix.lower() == path.suffix.lower():
            return candidate
    return None


def _quad_values(image: np.ndarray, quad):
    if quad is None:
        return image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [quad], 255)
    return image[mask > 0]
