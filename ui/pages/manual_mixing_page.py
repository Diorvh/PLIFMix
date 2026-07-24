from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure

from PLIFv2.application.actions import PLIFActions
from PLIFv2.ui.components.shift_search_panel import ShiftSearchPanel
from PLIFv2.ui.image_display import ImageDisplay
from PLIFv2.ui.mpl_canvas import DebouncedFigureCanvasTkAgg
from PLIFv2.ui.widgets import LatestTaskRunner


class ManualMixingPage(ttk.Frame):
    """Manual F/O/FO alignment and mixing-characteristic calculation."""

    def __init__(self, master, actions: PLIFActions):
        super().__init__(master)
        self.actions = actions
        self.settings = actions.state.settings.manual
        self.is_pseudo = tk.BooleanVar(value=True)
        self.show_crosshair = tk.BooleanVar(value=True)
        self.show_ellipse = tk.BooleanVar(value=True)
        self.use_ellipse_region = tk.BooleanVar(value=self.settings.use_ellipse_region)
        self.vars: dict[str, tk.StringVar] = {}
        self.displays: dict[str, ImageDisplay] = {}
        self.latest_result = None
        self.request_id = 0
        self._setup_ui()
        self.live_task = LatestTaskRunner(self, LabelLog(self.summary_label))

    def _setup_ui(self) -> None:
        controls = ttk.Frame(self)
        controls.pack(fill=tk.X, padx=8, pady=6)
        self._row_image_source(controls).pack(anchor=tk.W, pady=1)
        self._row_output(controls).pack(anchor=tk.W, pady=1)
        self._row_display(controls).pack(anchor=tk.W, pady=1)
        self._row_geometry(controls).pack(anchor=tk.W, pady=1)

        self.result_tabs = ttk.Notebook(self)
        self.result_tabs.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        mixing_tab = ttk.Frame(self.result_tabs)
        search_tab = ttk.Frame(self.result_tabs)
        self.result_tabs.add(mixing_tab, text="混合特性")
        self.result_tabs.add(search_tab, text="平移搜索")

        self.summary_label = ttk.Label(mixing_tab, text="计算结果")
        self.summary_label.pack(anchor=tk.W)

        image_outer = ttk.Frame(mixing_tab)
        image_outer.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        canvas = tk.Canvas(image_outer, highlightthickness=0)
        scrollbar_y = ttk.Scrollbar(image_outer, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar_x = ttk.Scrollbar(image_outer, orient=tk.HORIZONTAL, command=canvas.xview)
        self.image_frame = ttk.Frame(canvas)
        self.image_frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.image_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")
        image_outer.rowconfigure(0, weight=1)
        image_outer.columnconfigure(0, weight=1)

        display_positions = [
            ("f", (0, 0)),
            ("o", (0, 1)),
            ("fo", (0, 2)),
            ("f+o-fo", (0, 3)),
            ("oDf+o", (1, 0)),
            ("fDfo", (1, 1)),
            ("oDfo", (1, 2)),
            ("fo-f-o", (1, 3)),
            ("ME_oDf+o", (2, 0)),
            ("ME_fDfo", (2, 1)),
            ("ME_oDfo", (2, 2)),
        ]
        for title, position in display_positions:
            display = ImageDisplay(self.image_frame, title)
            display.grid(row=position[0], column=position[1], padx=4, pady=4, sticky="nw")
            self.displays[title] = display

        self.curve_panel = MixCurvePanel(self.image_frame, self.export_curve_data)
        self.curve_panel.grid(row=2, column=3, padx=4, pady=4, sticky="nw")

        self.shift_search = ShiftSearchPanel(search_tab, self.actions, self._save_vars)
        self.shift_search.pack(fill=tk.BOTH, expand=True)

    def _row_image_source(self, parent) -> ttk.Frame:
        row = ttk.Frame(parent)
        self._path_entry(row, "image_folder", "图像文件夹", self._select_image_folder, width=68)
        self._entry(row, "f_exp_no", "F 编号", 5)
        self._entry(row, "o_exp_no", "O 编号", 5)
        self._entry(row, "fo_exp_no", "FO 编号", 5)
        return row

    def _row_output(self, parent) -> ttk.Frame:
        row = ttk.Frame(parent)
        self._path_entry(row, "output_folder", "输出目录", self._select_output_folder, width=78)
        self._entry(row, "save_name", "图像保存名称", 18)
        return row

    def _row_display(self, parent) -> ttk.Frame:
        row = ttk.Frame(parent)
        ttk.Checkbutton(row, text="伪彩图", variable=self.is_pseudo, command=self.process).pack(side=tk.LEFT, padx=(3, 5))
        self._entry(row, "gain", "增益", 6)
        self._entry(row, "zoom", "缩小倍数", 6)
        self._entry(row, "denoise_ratio", "相对去噪阈值", 7)
        self._entry(row, "f_shift", "F 平移", 6)
        self._entry(row, "o_shift", "O 平移", 6)
        self._entry(row, "global_of", "全局 O/F", 6)
        return row

    def _row_geometry(self, parent) -> ttk.Frame:
        row = ttk.Frame(parent)
        self._entry(row, "line_width", "线宽", 6)
        ttk.Checkbutton(row, text="十字线", variable=self.show_crosshair, command=self.process).pack(side=tk.LEFT, padx=(5, 2))
        ttk.Label(row, text="F").pack(side=tk.LEFT, padx=(5, 0))
        self._entry(row, "center_xf", "X", 6)
        self._entry(row, "center_yf", "Y", 6)
        ttk.Label(row, text="O").pack(side=tk.LEFT, padx=(5, 0))
        self._entry(row, "center_xo", "X", 6)
        self._entry(row, "center_yo", "Y", 6)
        ttk.Label(row, text="FO").pack(side=tk.LEFT, padx=(5, 0))
        self._entry(row, "center_xfo", "X", 6)
        self._entry(row, "center_yfo", "Y", 6)
        ttk.Checkbutton(row, text="椭圆", variable=self.show_ellipse, command=self.process).pack(side=tk.LEFT, padx=(5, 2))
        self._entry(row, "major_axis", "长轴", 6)
        self._entry(row, "minor_axis", "短轴", 6)
        ttk.Checkbutton(
            row,
            text="仅椭圆区域计算混合效率",
            variable=self.use_ellipse_region,
            command=self.process,
        ).pack(side=tk.LEFT, padx=(7, 2))
        return row

    def _path_entry(self, row: ttk.Frame, attr: str, label: str, command, width: int) -> None:
        ttk.Label(row, text=label).pack(side=tk.LEFT, padx=(4, 2))
        var = tk.StringVar(value=self._value_to_str(getattr(self.settings, attr)))
        self.vars[attr] = var
        entry = ttk.Entry(row, textvariable=var, width=width)
        entry.pack(side=tk.LEFT)
        ttk.Button(row, text="...", width=3, command=command).pack(side=tk.LEFT, padx=(3, 4))
        entry.bind("<Return>", self.process)

    def _entry(self, row: ttk.Frame, attr: str, label: str, width: int) -> None:
        ttk.Label(row, text=label).pack(side=tk.LEFT, padx=(4, 2))
        var = tk.StringVar(value=self._value_to_str(getattr(self.settings, attr)))
        self.vars[attr] = var
        entry = ttk.Entry(row, textvariable=var, width=width)
        entry.pack(side=tk.LEFT)
        entry.bind("<Return>", self.process)

    def _select_image_folder(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.vars["image_folder"].set(path)

    def _select_output_folder(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.vars["output_folder"].set(path)

    def _save_vars(self) -> None:
        for attr, var in self.vars.items():
            current = getattr(self.settings, attr)
            raw = var.get()
            if isinstance(current, Path):
                value = Path(raw) if raw else Path()
            elif isinstance(current, int):
                value = int(raw)
            elif isinstance(current, float):
                value = float(raw)
            else:
                value = raw
            setattr(self.settings, attr, value)
        self.settings.use_ellipse_region = self.use_ellipse_region.get()

    def process(self, event=None) -> None:
        try:
            self._save_vars()
        except Exception as exc:
            self.summary_label.config(text=f"错误: {exc}")
            return
        self.request_id += 1
        request_id = self.request_id

        def done(result) -> None:
            if request_id == self.request_id:
                self._show_result(result)

        self.live_task.submit(self.actions.run_manual_display, done, "计算混合特性...")

    def _show_result(self, result) -> None:
        try:
            self.latest_result = result
            summary = result.summary
            self.summary_label.config(
                text=(
                    f"差绝对值和: {summary.abs_value:.2f}; O-F 混合效率: {summary.me_o_f:.4f}; "
                    f"F-FO 混合效率: {summary.me_f_fo:.4f}; O-FO 混合效率: {summary.me_o_fo:.4f}; "
                    f"平均混合效率: {summary.average_me:.4f}; 实拍 O/F: {summary.record_of:.4f}"
                )
            )
            gain = float(self.settings.gain)
            zoom = float(self.settings.zoom)
            ellipse = (
                self.settings.center_xfo,
                self.settings.center_yfo,
                self.settings.major_axis,
                self.settings.minor_axis,
                self.settings.line_width,
            ) if self.show_ellipse.get() else None
            for item in result.images:
                display = self.displays.get(item.name)
                if display is None:
                    continue
                colormap = item.colormap if self.is_pseudo.get() else "none"
                crosshair = self._crosshair_for(item.name) if self.show_crosshair.get() else None
                display.update(
                    item.image,
                    gain=gain if item.colormap == "jet" else 1,
                    colormap=colormap,
                    zoom=zoom,
                    crosshair=crosshair,
                    ellipse=ellipse,
                    save_name_suffix=self.settings.save_name,
                    special_points=item.special_points,
                    save_folder=self.settings.output_folder if str(self.settings.output_folder) not in {"", "."} else Path(self.settings.image_folder).parent / "manual_results",
                )
            self.curve_panel.update(result.mix_curve)
        except Exception as exc:
            self.summary_label.config(text=f"错误: {exc}")

    def export_curve_data(self) -> None:
        if self.latest_result is None:
            self.summary_label.config(text="请先完成当前参数计算")
            return
        try:
            self._save_vars()
            output = self.actions.export_mix_curve_data(self.latest_result)
            self.summary_label.config(text=f"混合曲线数据已导出到: {output}")
        except Exception as exc:
            self.summary_label.config(text=f"导出错误: {exc}")

    def _crosshair_for(self, name: str) -> tuple[int, int, int]:
        if name == "f":
            return self.settings.center_xf, self.settings.center_yf, self.settings.line_width
        if name == "o":
            return self.settings.center_xo, self.settings.center_yo, self.settings.line_width
        return self.settings.center_xfo, self.settings.center_yfo, self.settings.line_width

    def export_ui_state(self) -> dict[str, object]:
        return {
            "is_pseudo": self.is_pseudo.get(),
            "show_crosshair": self.show_crosshair.get(),
            "show_ellipse": self.show_ellipse.get(),
        }

    def restore_ui_state(self, state: dict[str, object]) -> None:
        self.is_pseudo.set(bool(state.get("is_pseudo", self.is_pseudo.get())))
        self.show_crosshair.set(bool(state.get("show_crosshair", self.show_crosshair.get())))
        self.show_ellipse.set(bool(state.get("show_ellipse", self.show_ellipse.get())))

    @staticmethod
    def _value_to_str(value) -> str:
        if isinstance(value, Path):
            return "" if str(value) == "." else str(value)
        return str(value)


class MixCurvePanel(ttk.Frame):
    def __init__(self, master, export_command):
        super().__init__(master)
        header = ttk.Frame(self)
        header.pack(fill=tk.X)
        ttk.Label(header, text="mix_curve", width=12).pack(side=tk.LEFT)
        ttk.Button(header, text="导出数据", command=export_command).pack(side=tk.LEFT, padx=(4, 0))
        self.figure = Figure(figsize=(6.4, 4.8), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self.canvas = DebouncedFigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

    def update(self, curve: tuple) -> None:
        x_values, y_values = curve
        self.axis.clear()
        self.axis.plot(x_values, y_values, color="tab:blue", linewidth=1)
        self.axis.set_title("混合分数曲线")
        self.axis.set_xlabel("x")
        self.axis.set_ylabel("混合分数")
        self.axis.grid(True, alpha=0.3)
        self.canvas.request_draw()


class LabelLog:
    def __init__(self, label):
        self.label = label

    def write(self, message: str) -> None:
        self.label.config(text=message)
