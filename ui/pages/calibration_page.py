from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure

from PLIFv2.application.actions import PLIFActions
from PLIFv2.core.image_io import read_image
from PLIFv2.ui.mpl_canvas import DebouncedFigureCanvasTkAgg
from PLIFv2.ui.widgets import BoundForm, FormField, LogBox, run_in_background


class CalibrationPage(ttk.Frame):
    def __init__(self, master, actions: PLIFActions):
        super().__init__(master)
        self.actions = actions
        self.analysis_result = None
        self.analysis_colorbar = None
        self.compare_colorbar = None
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)
        self.analysis_tab = ttk.Frame(notebook)
        self.stability_tab = ttk.Frame(notebook)
        self.compare_tab = ttk.Frame(notebook)
        notebook.add(self.analysis_tab, text="拟合与误差诊断")
        notebook.add(self.stability_tab, text="平均程度检验")
        notebook.add(self.compare_tab, text="矩阵比较/单张矩阵")
        self._setup_analysis()
        self._setup_stability()
        self._setup_compare()

    def _setup_analysis(self) -> None:
        settings = self.actions.state.settings.calibration
        self.analysis_form = BoundForm(
            self.analysis_tab,
            settings,
            [
                FormField("source_folder", "标定平均图目录", Path, 42, "dir"),
                FormField("output_folder", "矩阵输出目录", Path, 42, "dir"),
                FormField("display_vmax", "误差颜色上限(0自动)", float, 12),
                FormField("analysis_radius", "区域分析半径", int, 12),
            ],
            columns=2,
        )
        self.analysis_form.pack(anchor="w", padx=8, pady=6)
        buttons = ttk.Frame(self.analysis_tab)
        buttons.pack(anchor="w", padx=8)
        ttk.Button(buttons, text="拟合并分析", command=self.analyze).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="保存修正矩阵", command=self.save_matrix).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="计算区域平均误差", command=self.region_error).pack(side="left")
        self.analysis_info = ttk.Label(self.analysis_tab, text="选择目录后开始分析")
        self.analysis_info.pack(anchor="w", padx=8, pady=4)

        plots = ttk.Frame(self.analysis_tab)
        plots.pack(fill="both", expand=True, padx=8, pady=4)
        self.analysis_figure = Figure(figsize=(11, 5), dpi=100)
        self.image_ax = self.analysis_figure.add_subplot(131)
        self.error_ax = self.analysis_figure.add_subplot(132)
        self.curve_ax = self.analysis_figure.add_subplot(133)
        self.analysis_canvas = DebouncedFigureCanvasTkAgg(self.analysis_figure, master=plots)
        self.analysis_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.analysis_canvas.mpl_connect("button_press_event", self._analysis_click)
        toolbar = ttk.Frame(plots)
        toolbar.pack(fill="x")
        NavigationToolbar2Tk(self.analysis_canvas, toolbar)
        self.analysis_log = LogBox(self.analysis_tab, height=4)
        self.analysis_log.pack(fill="x", padx=8, pady=4)

    def _setup_stability(self) -> None:
        controls = ttk.Frame(self.stability_tab)
        controls.pack(anchor="w", padx=8, pady=6)
        self.stability_folder = tk.StringVar()
        self.stability_average = tk.StringVar()
        self.stability_start = tk.StringVar(value="0")
        self.stability_end = tk.StringVar(value="5000")
        ttk.Button(controls, text="原始帧目录", command=lambda: self._choose_dir(self.stability_folder)).pack(side="left")
        ttk.Entry(controls, textvariable=self.stability_folder, width=42).pack(side="left", padx=3)
        ttk.Button(controls, text="平均图", command=lambda: self._choose_file(self.stability_average, self.load_stability_image)).pack(side="left")
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

    def analyze(self) -> None:
        try:
            self.analysis_form.save_to_model()
        except Exception as exc:
            self.analysis_log.write(f"错误: {exc}")
            return
        run_in_background(self, self.analysis_log, self.actions.analyze_calibration, self._show_analysis, "开始拟合全部像素...")

    def _show_analysis(self, result) -> None:
        self.analysis_result = result
        self.image_ax.clear()
        self.error_ax.clear()
        self.curve_ax.clear()
        self.image_ax.imshow(result.image_stack[:, :, -1], cmap="gray")
        self.image_ax.set_title("标定图")
        vmax_setting = self.actions.state.settings.calibration.display_vmax
        vmax = vmax_setting if vmax_setting > 0 else float(np.nanmax(result.error_map))
        image = self.error_ax.imshow(result.error_map, cmap="viridis", vmin=0, vmax=vmax)
        self.error_ax.set_title("最大相对误差")
        if self.analysis_colorbar is not None:
            self.analysis_colorbar.remove()
        self.analysis_colorbar = self.analysis_figure.colorbar(image, ax=self.error_ax, fraction=0.046)
        self.analysis_canvas.request_draw()
        self.analysis_info.config(text=f"{len(result.files)} 张标定图；点击左图或误差图查看单点拟合")

    def _analysis_click(self, event) -> None:
        if self.analysis_result is None or event.inaxes not in {self.image_ax, self.error_ax} or event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        if not (0 <= x < self.analysis_result.error_map.shape[1] and 0 <= y < self.analysis_result.error_map.shape[0]):
            return
        stats = self.analysis_result.pixel_stats(x, y)
        self.curve_ax.clear()
        self.curve_ax.plot(self.analysis_result.x_values, stats["values"], "b.-", label="原始数据")
        self.curve_ax.plot(self.analysis_result.x_values, stats["predicted"], "r-", label="拟合")
        self.curve_ax.set_title(f"({x},{y}) k={stats['k']:.4e}, R²={stats['r_squared']:.4f}")
        self.curve_ax.grid(True, alpha=0.3)
        self.curve_ax.legend()
        self.analysis_info.config(text=f"最大相对误差 {stats['max_error']:.2%}，位置 {stats['max_error_x']:g}")
        self.analysis_canvas.request_draw()

    def region_error(self) -> None:
        if self.analysis_result is None:
            self.analysis_log.write("请先拟合标定图")
            return
        self.analysis_form.save_to_model()
        value = self.analysis_result.region_average_error(self.actions.state.settings.calibration.analysis_radius)
        self.analysis_info.config(text=f"圆形区域平均误差: {value:.4%}")

    def save_matrix(self) -> None:
        if self.analysis_result is None:
            self.analysis_log.write("请先拟合标定图")
            return
        self.analysis_form.save_to_model()
        outputs = self.actions.save_calibration_matrix(self.analysis_result)
        self.analysis_log.write(f"修正矩阵已保存: {outputs}")

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
        self.stability_curve_ax.set_title(f"像素 ({x},{y})")
        self.stability_curve_ax.grid(True, alpha=0.3)
        self.stability_info.config(text=f"平均值 {result.average:.2f}；极差 {result.value_range:.2f}；相对极差 {result.relative_range:.2%}；相对标准差 {result.relative_std:.2%}")
        self.stability_canvas.request_draw()

    def compare_matrices(self) -> None:
        self.compare_form.save_to_model()
        run_in_background(self, self.compare_log, self.actions.compare_calibration_matrices, self._show_comparison, "开始比较矩阵...")

    def _show_comparison(self, result) -> None:
        self.compare_ax.clear()
        image = self.compare_ax.imshow(result.relative_ptp, cmap="viridis")
        if self.compare_colorbar is not None:
            self.compare_colorbar.remove()
        self.compare_colorbar = self.compare_figure.colorbar(image, ax=self.compare_ax, fraction=0.046)
        self.compare_ax.set_title("修正矩阵相对极差")
        self.compare_info.config(text=f"{result.file_count} 个矩阵；最大相对偏差 {result.max_relative_ptp:.4%}；位置 {result.max_index}")
        self.compare_canvas.request_draw()

    def single_matrix(self) -> None:
        self.compare_form.save_to_model()
        run_in_background(self, self.compare_log, self.actions.create_single_image_matrix, lambda outputs: self.compare_log.write(f"已保存: {outputs}"), "生成单张修正矩阵...")

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
