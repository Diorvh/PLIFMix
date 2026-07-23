from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from PLIFv2.application.actions import PLIFActions
from PLIFv2.ui.widgets import BoundForm, FormField, LogBox, run_in_background


class IntensifierPage(ttk.Frame):
    def __init__(self, master, actions: PLIFActions):
        super().__init__(master)
        self.actions = actions
        self.average_settings = actions.state.settings.intensifier_average
        self.average_vars = {
            "source_folder": tk.StringVar(value=_path_text(self.average_settings.source_folder)),
            "output_folder": tk.StringVar(value=_path_text(self.average_settings.output_folder)),
            "best_process_count": tk.StringVar(value=str(self.average_settings.best_process_count)),
            "denoise_threshold": tk.StringVar(value=str(self.average_settings.denoise_threshold)),
        }
        self._setup_average_controls()
        settings = actions.state.settings.gain
        self.form = BoundForm(
            self,
            settings,
            [
                FormField("image_folder", "图像目录", Path, 42, "dir"),
                FormField("matrix_path", "修正矩阵 .npy", Path, 42, "file"),
                FormField("matrix_folder", "修正矩阵目录", Path, 42, "dir"),
                FormField("excel_path", "实验增益 Excel", Path, 42, "file"),
                FormField("output_folder", "输出目录", Path, 42, "dir"),
                FormField("operation", "运算 multiply/subtract", str, 18),
                FormField("experiment_column", "实验编号列", int, 8),
                FormField("gain_column", "Gain列", int, 8),
                FormField("gain_scale", "Gain倍率", float, 8),
                FormField("matrix_name_template", "矩阵命名模板", str, 22),
                FormField("include_tif", "包含tif/png", bool, 10),
            ],
            columns=2,
        )
        self.form.pack(fill="x", padx=8, pady=8)
        buttons = ttk.Frame(self)
        buttons.pack(anchor="w", padx=8)
        ttk.Button(buttons, text="应用修正矩阵", command=self.apply_matrix).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="选择若干文件修正", command=self.apply_selected).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="按Excel和Gain批量修正", command=self.apply_excel).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="从标定图拟合矩阵", command=self.fit_matrix).pack(side="left")
        self.log = LogBox(self)
        self.log.pack(fill="both", expand=True, padx=8, pady=8)

    def _setup_average_controls(self) -> None:
        section = ttk.LabelFrame(self, text="标定平均图")
        section.pack(fill="x", padx=8, pady=(8, 4))
        paths = ttk.Frame(section)
        paths.pack(anchor="w", padx=6, pady=(4, 2))
        for row, (name, label) in enumerate(
            [("source_folder", "标定原始图像根目录"), ("output_folder", "平均图输出目录")]
        ):
            ttk.Label(paths, text=label).grid(row=row, column=0, sticky="e", padx=(0, 4), pady=3)
            ttk.Entry(paths, textvariable=self.average_vars[name], width=92).grid(row=row, column=1, sticky="w", pady=3)
            ttk.Button(paths, text="...", width=3, command=lambda key=name: self._choose_average_directory(key)).grid(
                row=row,
                column=2,
                sticky="w",
                padx=(3, 0),
                pady=3,
            )

        controls = ttk.Frame(section)
        controls.pack(anchor="w", padx=6, pady=(2, 5))
        _labeled_entry(controls, "最佳进程数", self.average_vars["best_process_count"], 6)
        _labeled_entry(controls, "去噪阈值", self.average_vars["denoise_threshold"], 8)
        ttk.Button(controls, text="计算标定平均图", command=self.run_intensifier_average).pack(side="left", padx=(4, 0))

    def run_intensifier_average(self) -> None:
        try:
            self._save_average_settings()
        except Exception as exc:
            self.log.write(f"错误: {exc}")
            return

        def done(result) -> None:
            self.log.write(
                f"完成标定平均图: 共 {result.selected_count} 张，{len(result.items)} 个标定目录，"
                f"图像计算耗时 {result.calculation_seconds:.2f} s"
            )
            for item in result.items:
                self.log.write(f"{item.source_folder.name}: {item.selected_count} 张 -> {item.outputs.get('npy')}")

        run_in_background(
            self,
            self.log,
            self.actions.run_intensifier_average,
            done,
            "开始计算像增强器标定平均图...",
        )

    def apply_matrix(self) -> None:
        try:
            self.form.save_to_model()
        except Exception as exc:
            self.log.write(f"错误: {exc}")
            return
        self._run(self.actions.apply_intensifier_correction, "开始应用单一修正矩阵...")

    def apply_selected(self) -> None:
        paths = [Path(path) for path in filedialog.askopenfilenames(filetypes=[("图像", "*.npy *.tif *.tiff *.png")])]
        if not paths:
            return
        try:
            self.form.save_to_model()
        except Exception as exc:
            self.log.write(f"错误: {exc}")
            return
        self._run(lambda: self.actions.apply_selected_intensifier_correction(paths), "开始修正所选文件...")

    def apply_excel(self) -> None:
        try:
            self.form.save_to_model()
        except Exception as exc:
            self.log.write(f"错误: {exc}")
            return
        self._run(self.actions.apply_excel_intensifier_correction, "开始读取Excel并匹配Gain矩阵...")

    def fit_matrix(self) -> None:
        try:
            self.form.save_to_model()
        except Exception as exc:
            self.log.write(f"错误: {exc}")
            return

        def done(result) -> None:
            self.log.write(f"完成修正矩阵拟合: {result.image_count} 张; 输出 {result.outputs}")

        run_in_background(self, self.log, self.actions.fit_intensifier_matrix, done, "开始拟合修正矩阵...")

    def _save_average_settings(self) -> None:
        self.average_settings.source_folder = _to_path(self.average_vars["source_folder"].get())
        self.average_settings.output_folder = _to_path(self.average_vars["output_folder"].get())
        self.average_settings.best_process_count = int(self.average_vars["best_process_count"].get())
        self.average_settings.denoise_threshold = float(self.average_vars["denoise_threshold"].get())

    def _choose_average_directory(self, name: str) -> None:
        path = filedialog.askdirectory(initialdir=self.average_vars[name].get() or None)
        if path:
            self.average_vars[name].set(path)

    def _run(self, work, message: str) -> None:
        def done(result) -> None:
            self.log.write(f"完成像增强器修正: {result.processed_count} 个文件")
            for record in result.records:
                gain_text = f", Gain={record.gain:g}" if record.gain is not None else ""
                self.log.write(f"{record.image_path.name}{gain_text} -> {record.matrix_path.name}")

        run_in_background(self, self.log, work, done, message)


def _labeled_entry(parent, label: str, variable: tk.StringVar, width: int) -> None:
    ttk.Label(parent, text=label).pack(side="left", padx=(0, 2))
    ttk.Entry(parent, textvariable=variable, width=width).pack(side="left", padx=(0, 8))


def _path_text(path: Path) -> str:
    return "" if str(path) == "." else str(path)


def _to_path(text: str) -> Path:
    return Path(text) if text.strip() else Path()
