from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from PLIFv2.application.actions import PLIFActions
from PLIFv2.ui.components.calibration_workspace import CalibrationDiagnosticsNotebook, MatrixFitWorkspace
from PLIFv2.ui.widgets import LogBox, run_in_background


class IntensifierPage(ttk.Frame):
    def __init__(self, master, actions: PLIFActions):
        super().__init__(master)
        self.actions = actions
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        self.workflow_tab = ttk.Frame(self.notebook)
        self.diagnostics_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.workflow_tab, text="处理流程")
        self.notebook.add(self.diagnostics_tab, text="高级诊断")

        status = ttk.LabelFrame(self.workflow_tab, text="状态")
        self.log = LogBox(status, height=3)
        self._setup_average_controls()
        self.fit_workspace = MatrixFitWorkspace(actions, self.log)
        self.fit_workspace.build_controls(self.workflow_tab)
        self._setup_correction_controls()
        self.fit_workspace.build_view(self.workflow_tab)
        status.pack(fill="x", padx=8, pady=(4, 8))
        self.log.pack(fill="x", padx=4, pady=3)

        self.diagnostics = CalibrationDiagnosticsNotebook(self.diagnostics_tab, actions)
        self.diagnostics.pack(fill="both", expand=True, padx=8, pady=8)

    def _setup_average_controls(self) -> None:
        self.average_settings = self.actions.state.settings.intensifier_average
        self.average_vars = {
            "source_folder": tk.StringVar(value=_path_text(self.average_settings.source_folder)),
            "output_folder": tk.StringVar(value=_path_text(self.average_settings.output_folder)),
            "best_process_count": tk.StringVar(value=str(self.average_settings.best_process_count)),
            "denoise_threshold": tk.StringVar(value=str(self.average_settings.denoise_threshold)),
        }
        section = ttk.LabelFrame(self.workflow_tab, text="标定平均图")
        section.pack(fill="x", padx=8, pady=(8, 4))
        paths = ttk.Frame(section)
        paths.pack(anchor="w", padx=6, pady=(4, 2))
        _path_row(paths, 0, "像增强器标定根目录", self.average_vars["source_folder"], self._choose_average_source)
        _path_row(paths, 1, "标定平均图根目录", self.average_vars["output_folder"], self._choose_average_output)
        controls = ttk.Frame(section)
        controls.pack(anchor="w", padx=6, pady=(2, 5))
        _labeled_entry(controls, "最佳进程数", self.average_vars["best_process_count"], 6)
        _labeled_entry(controls, "去噪阈值", self.average_vars["denoise_threshold"], 8)
        ttk.Button(controls, text="计算标定平均图", command=self.run_intensifier_average).pack(side="left", padx=(4, 0))

    def _setup_correction_controls(self) -> None:
        self.gain_settings = self.actions.state.settings.gain
        self.correction_vars = {
            "image_folder": tk.StringVar(value=_path_text(self.gain_settings.image_folder)),
            "matrix_path": tk.StringVar(value=_path_text(self.gain_settings.matrix_path)),
            "matrix_folder": tk.StringVar(value=_path_text(self.gain_settings.matrix_folder)),
            "excel_path": tk.StringVar(value=_path_text(self.gain_settings.excel_path)),
            "output_folder": tk.StringVar(value=_path_text(self.gain_settings.output_folder)),
            "operation": tk.StringVar(value=self.gain_settings.operation),
            "experiment_column": tk.StringVar(value=str(self.gain_settings.experiment_column)),
            "gain_column": tk.StringVar(value=str(self.gain_settings.gain_column)),
            "gain_scale": tk.StringVar(value=str(self.gain_settings.gain_scale)),
            "matrix_name_template": tk.StringVar(value=self.gain_settings.matrix_name_template),
        }
        self.include_tif = tk.BooleanVar(value=self.gain_settings.include_tif)
        section = ttk.LabelFrame(self.workflow_tab, text="实验图像修正")
        section.pack(fill="x", padx=8, pady=4)
        paths = ttk.Frame(section)
        paths.pack(anchor="w", padx=6, pady=(4, 2))
        _path_row(paths, 0, "待修正平均图目录", self.correction_vars["image_folder"], self._choose_image_folder)
        _path_row(paths, 1, "单个修正矩阵 .npy", self.correction_vars["matrix_path"], self._choose_matrix_path)
        _path_row(paths, 2, "修正矩阵目录", self.correction_vars["matrix_folder"], self._choose_matrix_folder)
        _path_row(paths, 3, "实验增益 Excel", self.correction_vars["excel_path"], self._choose_excel_path)
        _path_row(paths, 4, "修正结果输出目录", self.correction_vars["output_folder"], self._choose_correction_output)

        options = ttk.Frame(section)
        options.pack(anchor="w", padx=6, pady=(2, 2))
        ttk.Label(options, text="运算").pack(side="left", padx=(0, 2))
        ttk.Combobox(
            options,
            textvariable=self.correction_vars["operation"],
            values=("multiply", "subtract"),
            state="readonly",
            width=12,
        ).pack(side="left", padx=(0, 8))
        _labeled_entry(options, "实验编号列", self.correction_vars["experiment_column"], 5)
        _labeled_entry(options, "Gain列", self.correction_vars["gain_column"], 5)
        _labeled_entry(options, "Gain倍率", self.correction_vars["gain_scale"], 7)
        _labeled_entry(options, "矩阵命名模板", self.correction_vars["matrix_name_template"], 22)
        ttk.Checkbutton(options, text="包含 tif/png", variable=self.include_tif).pack(side="left", padx=(0, 8))

        buttons = ttk.Frame(section)
        buttons.pack(anchor="w", padx=6, pady=(2, 5))
        ttk.Button(buttons, text="应用单个修正矩阵", command=self.apply_matrix).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="选择若干文件修正", command=self.apply_selected).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="按 Excel 和 Gain 批量修正", command=self.apply_excel).pack(side="left")

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
                source_name = f"{item.source_folder.parent.name}/{item.source_folder.name}"
                self.log.write(f"{source_name}: {item.selected_count} 张 -> {item.outputs.get('npy')}")

        run_in_background(
            self,
            self.log,
            self.actions.run_intensifier_average,
            done,
            "开始计算像增强器标定平均图...",
        )

    def apply_matrix(self) -> None:
        try:
            self._save_correction_settings()
        except Exception as exc:
            self.log.write(f"错误: {exc}")
            return
        self._run(self.actions.apply_intensifier_correction, "开始应用单个修正矩阵...")

    def apply_selected(self) -> None:
        paths = [Path(path) for path in filedialog.askopenfilenames(filetypes=[("图像", "*.npy *.tif *.tiff *.png")])]
        if not paths:
            return
        try:
            self._save_correction_settings()
        except Exception as exc:
            self.log.write(f"错误: {exc}")
            return
        self._run(lambda: self.actions.apply_selected_intensifier_correction(paths), "开始修正所选文件...")

    def apply_excel(self) -> None:
        try:
            self._save_correction_settings()
        except Exception as exc:
            self.log.write(f"错误: {exc}")
            return
        self._run(self.actions.apply_excel_intensifier_correction, "开始读取 Excel 并匹配 Gain 矩阵...")

    def _save_average_settings(self) -> None:
        self.average_settings.source_folder = _to_path(self.average_vars["source_folder"].get())
        self.average_settings.output_folder = _to_path(self.average_vars["output_folder"].get())
        self.average_settings.best_process_count = int(self.average_vars["best_process_count"].get())
        self.average_settings.denoise_threshold = float(self.average_vars["denoise_threshold"].get())

    def _save_correction_settings(self) -> None:
        self.gain_settings.image_folder = _to_path(self.correction_vars["image_folder"].get())
        self.gain_settings.matrix_path = _to_path(self.correction_vars["matrix_path"].get())
        self.gain_settings.matrix_folder = _to_path(self.correction_vars["matrix_folder"].get())
        self.gain_settings.excel_path = _to_path(self.correction_vars["excel_path"].get())
        self.gain_settings.output_folder = _to_path(self.correction_vars["output_folder"].get())
        self.gain_settings.operation = self.correction_vars["operation"].get()
        self.gain_settings.experiment_column = int(self.correction_vars["experiment_column"].get())
        self.gain_settings.gain_column = int(self.correction_vars["gain_column"].get())
        self.gain_settings.gain_scale = float(self.correction_vars["gain_scale"].get())
        self.gain_settings.matrix_name_template = self.correction_vars["matrix_name_template"].get().strip()
        self.gain_settings.include_tif = self.include_tif.get()

    def _choose_average_source(self) -> None:
        self._choose_directory(self.average_vars["source_folder"])

    def _choose_average_output(self) -> None:
        self._choose_directory(self.average_vars["output_folder"])

    def _choose_image_folder(self) -> None:
        self._choose_directory(self.correction_vars["image_folder"])

    def _choose_matrix_folder(self) -> None:
        self._choose_directory(self.correction_vars["matrix_folder"])

    def _choose_correction_output(self) -> None:
        self._choose_directory(self.correction_vars["output_folder"])

    def _choose_matrix_path(self) -> None:
        self._choose_file(self.correction_vars["matrix_path"], [("NumPy 矩阵", "*.npy")])

    def _choose_excel_path(self) -> None:
        self._choose_file(self.correction_vars["excel_path"], [("Excel", "*.xlsx *.xls")])

    @staticmethod
    def _choose_directory(variable: tk.StringVar) -> None:
        path = filedialog.askdirectory(initialdir=variable.get() or None)
        if path:
            variable.set(path)

    @staticmethod
    def _choose_file(variable: tk.StringVar, filetypes) -> None:
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            variable.set(path)

    def _run(self, work, message: str) -> None:
        def done(result) -> None:
            self.log.write(f"完成像增强器修正: {result.processed_count} 个文件")
            for record in result.records:
                gain_text = f", Gain={record.gain:g}" if record.gain is not None else ""
                self.log.write(f"{record.image_path.name}{gain_text} -> {record.matrix_path.name}")

        run_in_background(self, self.log, work, done, message)


def _path_row(parent, row: int, label: str, variable: tk.StringVar, command) -> None:
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="e", padx=(0, 4), pady=3)
    ttk.Entry(parent, textvariable=variable, width=92).grid(row=row, column=1, sticky="w", pady=3)
    ttk.Button(parent, text="...", width=3, command=command).grid(row=row, column=2, sticky="w", padx=(3, 0), pady=3)


def _labeled_entry(parent, label: str, variable: tk.StringVar, width: int) -> None:
    ttk.Label(parent, text=label).pack(side="left", padx=(0, 2))
    ttk.Entry(parent, textvariable=variable, width=width).pack(side="left", padx=(0, 8))


def _path_text(path: Path) -> str:
    return "" if str(path) == "." else str(path)


def _to_path(text: str) -> Path:
    return Path(text) if text.strip() else Path()
