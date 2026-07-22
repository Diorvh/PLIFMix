from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from PLIFv2.application.actions import PLIFActions
from PLIFv2.ui.image_display import ImageDisplay
from PLIFv2.ui.widgets import LatestTaskRunner


class ManualMixingPage(ttk.Frame):
    def __init__(self, master, actions: PLIFActions):
        super().__init__(master)
        self.actions = actions
        self.settings = actions.state.settings.manual
        self.is_pseudo = True
        self.show_crosshair = True
        self.show_ellipse = True
        self.show_q_center = "nonshow"
        self.show_img = "ori_img"
        self.vars: dict[str, tk.StringVar] = {}
        self.displays: dict[str, ImageDisplay] = {}
        self.latest_result = None
        self.request_id = 0
        self._setup_ui()
        self.live_task = LatestTaskRunner(self, LabelLog(self.lbl_res1))

    def _setup_ui(self) -> None:
        controls = ttk.Frame(self)
        controls.pack(fill=tk.X, padx=8, pady=6)

        self._row_files(controls).pack(anchor=tk.W, pady=1)
        self._row_display(controls).pack(anchor=tk.W, pady=1)
        self._row_centers(controls).pack(anchor=tk.W, pady=1)
        self._row_shift(controls).pack(anchor=tk.W, pady=1)
        self._row_fit(controls).pack(anchor=tk.W, pady=1)

        result_frame = ttk.Frame(self)
        result_frame.pack(fill=tk.X, padx=8)
        self.lbl_res1 = ttk.Label(result_frame, text="计算结果")
        self.lbl_res1.pack(anchor=tk.W)
        self.lbl_res2 = ttk.Label(result_frame, text="流量中心线/分布拟合结果", wraplength=1500, justify="left")
        self.lbl_res2.pack(anchor=tk.W)
        self.lbl_res3 = ttk.Label(result_frame, text="横纵向联合分布拟合结果", wraplength=1500, justify="left")
        self.lbl_res3.pack(anchor=tk.W)

        image_outer = ttk.Frame(self)
        image_outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
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

        display_title = ["f", "o", "fo", "f+o-fo", "fo-f-o", "oDf+o", "fDfo", "oDfo", "ME_oDf+o", "ME_fDfo", "ME_oDfo", "mix_curve"]
        display_position = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3)]
        for title, position in zip(display_title, display_position):
            display = ImageDisplay(self.image_frame, title)
            display.grid(row=position[0], column=position[1], padx=4, pady=4, sticky="nw")
            self.displays[title] = display

    def _row_files(self, parent) -> ttk.Frame:
        row = ttk.Frame(parent)
        ttk.Button(row, text="选择图像文件夹", command=self._select_image_folder).pack(side=tk.LEFT)
        self._entry(row, "image_folder", "目录", 42)
        self._entry(row, "fo_exp_no", "fo图像编号", 5)
        self._entry(row, "f_exp_no", "f图像编号", 5)
        self._entry(row, "o_exp_no", "o图像编号", 5)
        self._entry(row, "save_name", "保存名称", 18)
        ttk.Button(row, text="刷新", command=self.process).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="导出当前结果", command=self.export_current).pack(side=tk.LEFT, padx=4)
        return row

    def _row_display(self, parent) -> ttk.Frame:
        row = ttk.Frame(parent)
        self._entry(row, "global_of", "全局混合比", 6)
        self._entry(row, "zoom", "缩小倍数", 6)
        self.btn_pseudo = ttk.Button(row, text="关闭伪彩图", command=self.toggle_pseudo)
        self.btn_pseudo.pack(side=tk.LEFT, padx=4)
        self._entry(row, "gain", "增益", 6)
        ttk.Label(row, text="增益为0则按每张图最大值归一化显示").pack(side=tk.LEFT, padx=4)
        return row

    def _row_centers(self, parent) -> ttk.Frame:
        row = ttk.Frame(parent)
        ttk.Label(row, text="中心点 f").pack(side=tk.LEFT, padx=(0, 2))
        self._entry(row, "center_xf", "x", 5)
        self._entry(row, "center_yf", "y", 5)
        ttk.Label(row, text="o").pack(side=tk.LEFT, padx=(8, 2))
        self._entry(row, "center_xo", "x", 5)
        self._entry(row, "center_yo", "y", 5)
        ttk.Label(row, text="fo").pack(side=tk.LEFT, padx=(8, 2))
        self._entry(row, "center_xfo", "x", 5)
        self._entry(row, "center_yfo", "y", 5)
        self.btn_cross = ttk.Button(row, text="关闭十字线", command=self.toggle_crosshair)
        self.btn_cross.pack(side=tk.LEFT, padx=4)
        self._entry(row, "major_axis", "椭圆长轴", 5)
        self._entry(row, "minor_axis", "短轴", 5)
        self.btn_ellipse = ttk.Button(row, text="关闭椭圆", command=self.toggle_ellipse)
        self.btn_ellipse.pack(side=tk.LEFT, padx=4)
        self._entry(row, "line_width", "线宽", 5)
        return row

    def _row_shift(self, parent) -> ttk.Frame:
        row = ttk.Frame(parent)
        self._entry(row, "f_shift", "f图像左平移量", 5)
        self._entry(row, "o_shift", "o图像左平移量", 5)
        self._entry(row, "denoise_ratio", "去噪阈值", 7)
        self._entry(row, "output_folder", "输出目录", 28)
        return row

    def _row_fit(self, parent) -> ttk.Frame:
        row = ttk.Frame(parent)
        self._entry(row, "thres_contour", "轮廓流量比例", 6)
        self._entry(row, "q_center_up", "流量中心线上", 5)
        self._entry(row, "q_center_down", "下", 5)
        self._entry(row, "max_k", "最大值个数", 5)
        self._entry(row, "rotate_angle_f", "旋转角 f", 5)
        self._entry(row, "rotate_angle_o", "o", 5)
        self._entry(row, "rotate_angle_fo", "fo", 5)
        ttk.Button(row, text="显示流量轮廓", command=lambda: self.set_q_center("contour")).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="显示流量中心线", command=lambda: self.set_q_center("Q_center")).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="显示拟合流量中心线", command=lambda: self.set_q_center("Q_center_fitted")).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="不显示", command=lambda: self.set_q_center("nonshow")).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="掰直后图像", command=lambda: self.set_show_img("bent_img")).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="拟合图像", command=lambda: self.set_show_img("fit_img")).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="拟合-掰直", command=lambda: self.set_show_img("delta_img")).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="原始图像", command=lambda: self.set_show_img("ori_img")).pack(side=tk.LEFT, padx=2)
        return row

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
            self.process()

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

    def process(self, event=None) -> None:
        try:
            self._save_vars()
        except Exception as exc:
            self.lbl_res1.config(text=f"错误: {exc}")
            return
        self.request_id += 1
        request_id = self.request_id
        q_mode, image_mode = self.show_q_center, self.show_img
        def done(result) -> None:
            if request_id == self.request_id:
                self._show_result(result)
        self.live_task.submit(lambda: self.actions.run_manual_display(q_mode, image_mode), done, "计算中...")

    def _show_result(self, result) -> None:
        try:
            self.latest_result = result
            summary = result.summary
            self.lbl_res1.config(
                text=(
                    f"差绝对值和{summary.abs_value:.2f}; O-F混合效率: {summary.me_o_f:.4f}; "
                    f"F-FO混合效率: {summary.me_f_fo:.4f}; O-FO混合效率: {summary.me_o_fo:.4f}; "
                    f"平均混合效率:{summary.average_me:.4f}; 实拍OF:{summary.record_of:.4f}"
                )
            )
            self.lbl_res2.config(text=result.fit_text)
            self.lbl_res3.config(text=result.surface_text)
            gain = float(self.settings.gain)
            zoom = float(self.settings.zoom)
            ellipse = (
                self.settings.center_xfo,
                self.settings.center_yfo,
                self.settings.major_axis,
                self.settings.minor_axis,
                self.settings.line_width,
            ) if self.show_ellipse else None
            for item in result.images:
                display = self.displays.get(item.name)
                if display is None:
                    continue
                crosshair = self._crosshair_for(item.name) if self.show_crosshair else None
                colormap = item.colormap if self.is_pseudo else "none"
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
        except Exception as exc:
            self.lbl_res1.config(text=f"错误: {exc}")

    def export_current(self) -> None:
        if self.latest_result is None:
            self.lbl_res1.config(text="请先刷新并完成当前参数计算")
            return
        try:
            self._save_vars()
            output = self.actions.export_manual_result(self.latest_result)
            self.lbl_res1.config(text=f"当前结果已导出到: {output}")
        except Exception as exc:
            self.lbl_res1.config(text=f"导出错误: {exc}")

    def _crosshair_for(self, name: str) -> tuple[int, int, int]:
        if name == "f":
            return self.settings.center_xf, self.settings.center_yf, self.settings.line_width
        if name == "o":
            return self.settings.center_xo, self.settings.center_yo, self.settings.line_width
        return self.settings.center_xfo, self.settings.center_yfo, self.settings.line_width

    def toggle_pseudo(self) -> None:
        self.is_pseudo = not self.is_pseudo
        self.btn_pseudo.config(text="关闭伪彩图" if self.is_pseudo else "显示伪彩图")
        self.process()

    def toggle_crosshair(self) -> None:
        self.show_crosshair = not self.show_crosshair
        self.btn_cross.config(text="关闭十字线" if self.show_crosshair else "显示十字线")
        self.process()

    def toggle_ellipse(self) -> None:
        self.show_ellipse = not self.show_ellipse
        self.btn_ellipse.config(text="关闭椭圆" if self.show_ellipse else "显示椭圆")
        self.process()

    def set_q_center(self, mode: str) -> None:
        self.show_q_center = mode
        self.process()

    def set_show_img(self, mode: str) -> None:
        self.show_img = mode
        self.process()

    def export_ui_state(self) -> dict[str, object]:
        return {
            "is_pseudo": self.is_pseudo,
            "show_crosshair": self.show_crosshair,
            "show_ellipse": self.show_ellipse,
            "show_q_center": self.show_q_center,
            "show_img": self.show_img,
        }

    def restore_ui_state(self, state: dict[str, object]) -> None:
        self.is_pseudo = bool(state.get("is_pseudo", self.is_pseudo))
        self.show_crosshair = bool(state.get("show_crosshair", self.show_crosshair))
        self.show_ellipse = bool(state.get("show_ellipse", self.show_ellipse))
        self.show_q_center = str(state.get("show_q_center", self.show_q_center))
        self.show_img = str(state.get("show_img", self.show_img))
        self.btn_pseudo.config(text="关闭伪彩图" if self.is_pseudo else "显示伪彩图")
        self.btn_cross.config(text="关闭十字线" if self.show_crosshair else "显示十字线")
        self.btn_ellipse.config(text="关闭椭圆" if self.show_ellipse else "显示椭圆")

    @staticmethod
    def _value_to_str(value) -> str:
        if isinstance(value, Path):
            return "" if str(value) == "." else str(value)
        return str(value)


class LabelLog:
    def __init__(self, label):
        self.label = label

    def write(self, message: str) -> None:
        self.label.config(text=message)
