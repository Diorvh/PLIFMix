from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from PLIFv2.application.actions import PLIFActions
from PLIFv2.core.image_io import read_image
from PLIFv2.ui.mpl_canvas import DebouncedFigureCanvasTkAgg
from PLIFv2.ui.widgets import BoundForm, FormField, LogBox, run_in_background


class MatrixFitWorkspace:
    """Matrix-fit controls and the three linked diagnostic plots from v1 3.3."""

    def __init__(self, actions: PLIFActions, log: LogBox):
        self.actions = actions
        self.log = log
        self.settings = actions.state.settings.calibration
        self.analysis_result = None
        self.analysis_colorbar = None
        self.analysis_circle = None
        self.owner: tk.Misc | None = None
        self.source_var = tk.StringVar(value=_path_text(self.settings.source_folder))
        self.output_var = tk.StringVar(value=_path_text(self.settings.output_folder))
        self.vmax_var = tk.StringVar(value=str(self.settings.display_vmax))
        self.radius_var = tk.StringVar(value=str(self.settings.analysis_radius))

    def build_controls(self, master: tk.Misc) -> None:
        self.owner = master
        section = ttk.LabelFrame(master, text="修正矩阵拟合")
        section.pack(fill="x", padx=8, pady=4)
        paths = ttk.Frame(section)
        paths.pack(anchor="w", padx=6, pady=(4, 2))
        _path_row(paths, 0, "标定平均图目录", self.source_var, self._choose_source)
        _path_row(paths, 1, "修正矩阵输出目录", self.output_var, self._choose_output)

        controls = ttk.Frame(section)
        controls.pack(anchor="w", padx=6, pady=(2, 5))
        _labeled_entry(controls, "误差颜色上限(0自动)", self.vmax_var, 10)
        _labeled_entry(controls, "区域分析半径", self.radius_var, 8)
        ttk.Button(controls, text="拟合并分析", command=self.analyze).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="保存修正矩阵", command=self.save_matrix).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="计算区域平均误差", command=self.region_error).pack(side="left")
        self.info = ttk.Label(section, text="选择标定平均图目录后进行矩阵拟合")
        self.info.pack(anchor="w", padx=6, pady=(0, 4))

    def build_view(self, master: tk.Misc) -> None:
        section = ttk.LabelFrame(master, text="矩阵拟合结果")
        section.pack(fill="both", expand=True, padx=8, pady=4)
        self.figure = Figure(figsize=(11, 4.6), dpi=100)
        self.image_ax = self.figure.add_subplot(131)
        self.error_ax = self.figure.add_subplot(132)
        self.curve_ax = self.figure.add_subplot(133)
        self.canvas = DebouncedFigureCanvasTkAgg(self.figure, master=section)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._analysis_click)
        toolbar = ttk.Frame(section)
        toolbar.pack(fill="x")
        NavigationToolbar2Tk(self.canvas, toolbar)

    def analyze(self) -> None:
        try:
            self._save_settings()
        except Exception as exc:
            self.log.write(f"错误: {exc}")
            return
        if self.owner is None:
            return
        run_in_background(self.owner, self.log, self.actions.analyze_calibration, self._show_analysis, "开始拟合全部像素...")

    def save_matrix(self) -> None:
        if self.analysis_result is None:
            self.log.write("请先拟合标定图")
            return
        try:
            self._save_settings()
        except Exception as exc:
            self.log.write(f"错误: {exc}")
            return
        outputs = self.actions.save_calibration_matrix(self.analysis_result)
        self.log.write(f"修正矩阵已保存: {outputs}")

    def region_error(self) -> None:
        if self.analysis_result is None:
            self.log.write("请先拟合标定图")
            return
        try:
            self._save_settings()
        except Exception as exc:
            self.log.write(f"错误: {exc}")
            return
        value = self.analysis_result.region_average_error(self.settings.analysis_radius)
        self._draw_analysis_circle()
        self.info.config(text=f"圆形区域平均误差: {value:.4%}")

    def _show_analysis(self, result) -> None:
        self.analysis_result = result
        if self.analysis_colorbar is not None:
            self.analysis_colorbar.remove()
            self.analysis_colorbar = None
        self.image_ax.clear()
        self.error_ax.clear()
        self.curve_ax.clear()
        max_x_index = int(np.argmax(result.x_values))
        max_x_value = result.x_values[max_x_index]
        self.image_ax.imshow(result.image_stack[:, :, max_x_index], cmap="gray")
        self.image_ax.set_title(f"标定图（x={max_x_value:g}）")
        self.analysis_circle = None
        self._draw_analysis_circle()
        vmax = self.settings.display_vmax if self.settings.display_vmax > 0 else float(np.nanmax(result.error_map))
        image = self.error_ax.imshow(result.error_map, cmap="viridis", vmin=0, vmax=vmax)
        self.error_ax.set_title("最大相对误差")
        self.analysis_colorbar = self.figure.colorbar(image, ax=self.error_ax, fraction=0.046)
        self.canvas.request_draw()
        self.info.config(text=f"{len(result.files)} 张标定图；点击左图或误差图查看单点拟合")

    def _draw_analysis_circle(self) -> None:
        if self.analysis_result is None:
            return
        if self.analysis_circle is not None:
            self.analysis_circle.remove()
        height, width = self.analysis_result.error_map.shape
        self.analysis_circle = Circle(
            (width / 2, height / 2),
            radius=self.settings.analysis_radius,
            fill=False,
            edgecolor="red",
            linewidth=1.5,
        )
        self.image_ax.add_patch(self.analysis_circle)
        self.canvas.request_draw()

    def _analysis_click(self, event) -> None:
        if self.analysis_result is None or event.inaxes not in {self.image_ax, self.error_ax}:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        if not (0 <= x < self.analysis_result.error_map.shape[1] and 0 <= y < self.analysis_result.error_map.shape[0]):
            return
        stats = self.analysis_result.pixel_stats(x, y)
        order = np.argsort(self.analysis_result.x_values, kind="stable")
        x_values = self.analysis_result.x_values[order]
        values = np.asarray(stats["values"])[order]
        predicted = np.asarray(stats["predicted"])[order]
        self.curve_ax.clear()
        self.curve_ax.plot(x_values, values, "b.-", label="原始数据")
        self.curve_ax.plot(x_values, predicted, "r-", label="拟合")
        self.curve_ax.set_title(f"({x}, {y}) k={stats['k']:.4e}, R²={stats['r_squared']:.4f}")
        self.curve_ax.grid(True, alpha=0.3)
        self.curve_ax.legend()
        self.info.config(text=f"最大相对误差 {stats['max_error']:.2%}，位置 {stats['max_error_x']:g}")
        self.canvas.request_draw()

    def _save_settings(self) -> None:
        self.settings.source_folder = _to_path(self.source_var.get())
        self.settings.output_folder = _to_path(self.output_var.get())
        self.settings.display_vmax = float(self.vmax_var.get())
        self.settings.analysis_radius = int(self.radius_var.get())

    def _choose_source(self) -> None:
        path = filedialog.askdirectory(initialdir=self.source_var.get() or None)
        if path:
            self.source_var.set(path)

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(initialdir=self.output_var.get() or None)
        if path:
            self.output_var.set(path)


class CalibrationDiagnosticsNotebook(ttk.Notebook):
    """Low-frequency calibration tools kept outside the primary workflow."""

    def __init__(self, master: tk.Misc, actions: PLIFActions):
        super().__init__(master)
        self.actions = actions
        self.analysis_colorbar = None
        self.compare_colorbar = None
        self.stability_tab = ttk.Frame(self)
        self.compare_tab = ttk.Frame(self)
        self.add(self.stability_tab, text="平均图检验")
        self.add(self.compare_tab, text="矩阵偏差")
        self._setup_stability()
        self._setup_compare()

    def _setup_stability(self) -> None:
        controls = ttk.Frame(self.stability_tab)
        controls.pack(anchor="w", padx=8, pady=6)
        self.stability_folder = tk.StringVar()
        self.stability_average = tk.StringVar()
        self.stability_start = tk.StringVar(value="0")
        self.stability_end = tk.StringVar(value="5000")
        ttk.Button(controls, text="原始帧目录...", command=lambda: self._choose_dir(self.stability_folder)).pack(side="left")
        ttk.Entry(controls, textvariable=self.stability_folder, width=42).pack(side="left", padx=3)
        ttk.Button(controls, text="平均图...", command=lambda: self._choose_file(self.stability_average, self.load_stability_image)).pack(side="left")
        ttk.Entry(controls, textvariable=self.stability_average, width=36).pack(side="left", padx=3)
        ttk.Label(controls, text="帧").pack(side="left")
        ttk.Entry(controls, textvariable=self.stability_start, width=7).pack(side="left")
        ttk.Label(controls, text="~").pack(side="left")
        ttk.Entry(controls, textvariable=self.stability_end, width=7).pack(side="left")
        self.stability_info = ttk.Label(self.stability_tab, text="点击平均图中的像素查看全部帧波动")
        self.stability_info.pack(anchor="w", padx=8)
        self.stability_figure = Figure(figsize=(10, 5), dpi=100)
        self.stability_image_ax = self.stability_figure.add_subplot(121)
        self.stability_curve_ax = self.stability_figure.add_subplot(122)
        self.stability_canvas = DebouncedFigureCanvasTkAgg(self.stability_figure, master=self.stability_tab)
        self.stability_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=6)
        self.stability_canvas.mpl_connect("button_press_event", self._stability_click)

    def _setup_compare(self) -> None:
        settings = self.actions.state.settings.calibration
        self.compare_form = BoundForm(
            self.compare_tab,
            settings,
            [
                FormField("compare_folder", "矩阵目录", Path, 42, "dir"),
                FormField("compare_files", "文件名(逗号分隔，空为全部)", str, 42),
                FormField("compare_center_x", "圆心X", int, 8),
                FormField("compare_center_y", "圆心Y", int, 8),
                FormField("compare_radius", "半径", int, 8),
                FormField("single_image_path", "单张标定图", Path, 42, "file"),
                FormField("output_folder", "输出目录", Path, 42, "dir"),
            ],
            columns=2,
        )
        self.compare_form.pack(anchor="w", padx=8, pady=6)
        buttons = ttk.Frame(self.compare_tab)
        buttons.pack(anchor="w", padx=8)
        ttk.Button(buttons, text="比较多个矩阵", command=self.compare_matrices).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="单张图生成矩阵", command=self.single_matrix).pack(side="left")
        self.compare_info = ttk.Label(self.compare_tab, text="")
        self.compare_info.pack(anchor="w", padx=8, pady=4)
        self.compare_figure = Figure(figsize=(8, 5), dpi=100)
        self.compare_ax = self.compare_figure.add_subplot(111)
        self.compare_canvas = DebouncedFigureCanvasTkAgg(self.compare_figure, master=self.compare_tab)
        self.compare_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=6)
        self.compare_log = LogBox(self.compare_tab, height=4)
        self.compare_log.pack(fill="x", padx=8, pady=4)

    def load_stability_image(self) -> None:
        path = self.stability_average.get()
        if not path:
            return
        image = read_image(path)
        if image.ndim == 3:
            image = image[:, :, 1]
        self.stability_image_ax.clear()
        self.stability_image_ax.imshow(image, cmap="gray")
        self.stability_image_ax.set_title("点击像素")
        self.stability_canvas.request_draw()

    def _stability_click(self, event) -> None:
        if event.inaxes is not self.stability_image_ax or event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        folder = Path(self.stability_folder.get())
        start, end = int(self.stability_start.get()), int(self.stability_end.get())

        def work():
            return self.actions.calibration_pixel_stability(folder, x, y, start, end)

        run_in_background(self, LogProxy(self.stability_info), work, lambda result: self._show_stability(result, x, y), "读取像素序列...")

    def _show_stability(self, result, x: int, y: int) -> None:
        self.stability_curve_ax.clear()
        self.stability_curve_ax.plot(np.arange(result.frame_count), result.values, "b-")
        self.stability_curve_ax.set_title(f"像素 ({x}, {y})")
        self.stability_curve_ax.grid(True, alpha=0.3)
        self.stability_info.config(
            text=(
                f"平均值 {result.average:.2f}；极差 {result.value_range:.2f}；"
                f"相对极差 {result.relative_range:.2%}；相对标准差 {result.relative_std:.2%}"
            )
        )
        self.stability_canvas.request_draw()

    def compare_matrices(self) -> None:
        try:
            self.compare_form.save_to_model()
        except Exception as exc:
            self.compare_log.write(f"错误: {exc}")
            return
        run_in_background(self, self.compare_log, self.actions.compare_calibration_matrices, self._show_comparison, "开始比较矩阵...")

    def _show_comparison(self, result) -> None:
        self.compare_ax.clear()
        image = self.compare_ax.imshow(result.relative_ptp, cmap="viridis")
        if self.compare_colorbar is not None:
            self.compare_colorbar.remove()
        self.compare_colorbar = self.compare_figure.colorbar(image, ax=self.compare_ax, fraction=0.046)
        self.compare_ax.set_title("修正矩阵相对极差")
        self.compare_info.config(
            text=f"{result.file_count} 个矩阵；最大相对偏差 {result.max_relative_ptp:.4%}；位置 {result.max_index}"
        )
        self.compare_canvas.request_draw()

    def single_matrix(self) -> None:
        try:
            self.compare_form.save_to_model()
        except Exception as exc:
            self.compare_log.write(f"错误: {exc}")
            return
        run_in_background(
            self,
            self.compare_log,
            self.actions.create_single_image_matrix,
            lambda outputs: self.compare_log.write(f"已保存: {outputs}"),
            "生成单张修正矩阵...",
        )

    @staticmethod
    def _choose_dir(variable: tk.StringVar) -> None:
        path = filedialog.askdirectory()
        if path:
            variable.set(path)

    @staticmethod
    def _choose_file(variable: tk.StringVar, callback=None) -> None:
        path = filedialog.askopenfilename()
        if path:
            variable.set(path)
            if callback:
                callback()


class LogProxy:
    def __init__(self, label: ttk.Label):
        self.label = label

    def write(self, message: str) -> None:
        self.label.config(text=message)


def _path_row(parent, row: int, label: str, variable: tk.StringVar, command) -> None:
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="e", padx=(0, 4), pady=3)
    ttk.Entry(parent, textvariable=variable, width=48).grid(row=row, column=1, sticky="w", pady=3)
    ttk.Button(parent, text="...", width=3, command=command).grid(row=row, column=2, sticky="w", padx=(3, 0), pady=3)


def _labeled_entry(parent, label: str, variable: tk.StringVar, width: int) -> None:
    ttk.Label(parent, text=label).pack(side="left", padx=(0, 2))
    ttk.Entry(parent, textvariable=variable, width=width).pack(side="left", padx=(0, 8))


def _path_text(path: Path) -> str:
    return "" if str(path) == "." else str(path)


def _to_path(text: str) -> Path:
    return Path(text) if text.strip() else Path()
