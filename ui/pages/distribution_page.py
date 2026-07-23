from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from PLIFv2.application.actions import PLIFActions
from PLIFv2.ui.image_display import ImageDisplay
from PLIFv2.ui.widgets import LatestTaskRunner


class DistributionPage(ttk.Frame):
    def __init__(self, master, actions: PLIFActions):
        super().__init__(master)
        self.actions = actions
        self.settings = actions.state.settings.distribution
        self.vars: dict[str, tk.StringVar] = {}
        self.show_q_center = "nonshow"
        self.show_img = "ori_img"
        self.show_crosshair = True
        self.is_pseudo = True
        self.latest_result = None
        self.request_id = 0
        self._setup_ui()
        self.live_task = LatestTaskRunner(self, LabelLog(self.status))

    def _setup_ui(self) -> None:
        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=8, pady=6)
        rows = [ttk.Frame(controls) for _ in range(4)]
        for row in rows:
            row.pack(anchor="w", pady=1)
        ttk.Button(rows[0], text="选择图像文件夹", command=self._choose_folder).pack(side="left")
        self._entry(rows[0], "image_folder", "目录", 78)
        self._entry(rows[0], "f_exp_no", "f编号", 5)
        self._entry(rows[0], "o_exp_no", "o编号", 5)
        self._entry(rows[0], "save_name", "保存名称", 16)
        self._entry(rows[0], "output_folder", "输出目录", 78)
        ttk.Button(rows[0], text="刷新", command=self.process).pack(side="left", padx=4)
        ttk.Button(rows[0], text="导出当前结果", command=self.export).pack(side="left", padx=4)

        self._entry(rows[1], "zoom", "缩小倍数", 5)
        self._entry(rows[1], "gain", "增益", 5)
        self._entry(rows[1], "denoise_ratio", "去噪阈值", 7)
        self._entry(rows[1], "hori_fit_width", "横向拟合半宽", 7)
        self.btn_pseudo = ttk.Button(rows[1], text="关闭伪彩图", command=self._toggle_pseudo)
        self.btn_pseudo.pack(side="left", padx=4)

        ttk.Label(rows[2], text="中心点 f").pack(side="left")
        self._entry(rows[2], "center_xf", "x", 5)
        self._entry(rows[2], "center_yf", "y", 5)
        ttk.Label(rows[2], text="o").pack(side="left", padx=(8, 0))
        self._entry(rows[2], "center_xo", "x", 5)
        self._entry(rows[2], "center_yo", "y", 5)
        self._entry(rows[2], "line_width", "线宽", 5)
        self.btn_cross = ttk.Button(rows[2], text="关闭十字线", command=self._toggle_crosshair)
        self.btn_cross.pack(side="left", padx=4)

        self._entry(rows[3], "thres_contour", "轮廓比例", 6)
        self._entry(rows[3], "q_center_up", "中心线上", 5)
        self._entry(rows[3], "q_center_down", "下", 5)
        self._entry(rows[3], "max_k", "最大值个数", 5)
        self._entry(rows[3], "rotate_angle_f", "旋转f", 5)
        self._entry(rows[3], "rotate_angle_o", "o", 5)
        for label, mode in [("轮廓", "contour"), ("中心线", "Q_center"), ("拟合中心线", "Q_center_fitted"), ("不显示", "nonshow")]:
            ttk.Button(rows[3], text=label, command=lambda value=mode: self._set_q(value)).pack(side="left", padx=2)
        for label, mode in [("掰直", "bent_img"), ("拟合", "fit_img"), ("拟合-掰直", "delta_img"), ("原图", "ori_img")]:
            ttk.Button(rows[3], text=label, command=lambda value=mode: self._set_img(value)).pack(side="left", padx=2)

        self.summary = ttk.Label(self, text="质量分布拟合结果", wraplength=1500, justify="left")
        self.summary.pack(anchor="w", padx=8)
        self.status = ttk.Label(self, text="")
        self.status.pack(anchor="w", padx=8)
        displays = ttk.Frame(self)
        displays.pack(fill="both", expand=True, padx=8, pady=6)
        self.display_f = ImageDisplay(displays, "f")
        self.display_o = ImageDisplay(displays, "o")
        self.display_f.grid(row=0, column=0, sticky="nw", padx=4)
        self.display_o.grid(row=0, column=1, sticky="nw", padx=4)

    def _entry(self, row, name: str, label: str, width: int) -> None:
        ttk.Label(row, text=label).pack(side="left", padx=(4, 2))
        value = getattr(self.settings, name)
        variable = tk.StringVar(value="" if isinstance(value, Path) and str(value) == "." else str(value))
        self.vars[name] = variable
        entry = ttk.Entry(row, textvariable=variable, width=width)
        entry.pack(side="left")
        entry.bind("<Return>", self.process)

    def _save(self) -> None:
        for name, variable in self.vars.items():
            current = getattr(self.settings, name)
            raw = variable.get()
            if isinstance(current, Path):
                value = Path(raw) if raw else Path()
            elif isinstance(current, int):
                value = int(raw)
            elif isinstance(current, float):
                value = float(raw)
            else:
                value = raw
            setattr(self.settings, name, value)

    def process(self, event=None) -> None:
        try:
            self._save()
        except Exception as exc:
            self.status.config(text=f"错误: {exc}")
            return
        self.request_id += 1
        request_id = self.request_id
        q_mode, image_mode = self.show_q_center, self.show_img
        def done(result) -> None:
            if request_id == self.request_id:
                self._show(result)
        self.live_task.submit(lambda: self.actions.run_distribution_display(q_mode, image_mode), done, "计算质量分布...")

    def _show(self, result) -> None:
        self.latest_result = result
        self.summary.config(text=result.fit_text + "\n" + result.surface_text)
        output = self.settings.output_folder if str(self.settings.output_folder) not in {"", "."} else Path(self.settings.image_folder).parent / "distribution_results"
        for item, display, cx, cy in [
            (result.images[0], self.display_f, self.settings.center_xf, self.settings.center_yf),
            (result.images[1], self.display_o, self.settings.center_xo, self.settings.center_yo),
        ]:
            display.update(
                item.image,
                gain=self.settings.gain,
                colormap=item.colormap if self.is_pseudo else "none",
                zoom=self.settings.zoom,
                crosshair=(cx, cy, self.settings.line_width) if self.show_crosshair else None,
                save_name_suffix=self.settings.save_name,
                special_points=item.special_points,
                save_folder=output,
            )
        self.status.config(text="计算完成")

    def export(self) -> None:
        if self.latest_result is None:
            self.status.config(text="请先完成一次计算")
            return
        self._save()
        output = self.actions.export_distribution_result(self.latest_result)
        self.status.config(text=f"结果已导出到 {output}")

    def _choose_folder(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.vars["image_folder"].set(path)
            self.process()

    def _toggle_pseudo(self) -> None:
        self.is_pseudo = not self.is_pseudo
        self.btn_pseudo.config(text="关闭伪彩图" if self.is_pseudo else "显示伪彩图")
        self.process()

    def _toggle_crosshair(self) -> None:
        self.show_crosshair = not self.show_crosshair
        self.btn_cross.config(text="关闭十字线" if self.show_crosshair else "显示十字线")
        self.process()

    def _set_q(self, mode: str) -> None:
        self.show_q_center = mode
        self.process()

    def _set_img(self, mode: str) -> None:
        self.show_img = mode
        self.process()

    def export_ui_state(self) -> dict[str, object]:
        return {
            "is_pseudo": self.is_pseudo,
            "show_crosshair": self.show_crosshair,
            "show_q_center": self.show_q_center,
            "show_img": self.show_img,
        }

    def restore_ui_state(self, state: dict[str, object]) -> None:
        self.is_pseudo = bool(state.get("is_pseudo", self.is_pseudo))
        self.show_crosshair = bool(state.get("show_crosshair", self.show_crosshair))
        self.show_q_center = str(state.get("show_q_center", self.show_q_center))
        self.show_img = str(state.get("show_img", self.show_img))
        self.btn_pseudo.config(text="关闭伪彩图" if self.is_pseudo else "显示伪彩图")
        self.btn_cross.config(text="关闭十字线" if self.show_crosshair else "显示十字线")


class LabelLog:
    def __init__(self, label):
        self.label = label

    def write(self, message: str) -> None:
        self.label.config(text=message)
