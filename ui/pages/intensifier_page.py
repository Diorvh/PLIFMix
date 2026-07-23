from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from PLIFv2.application.actions import PLIFActions
from PLIFv2.core.paths import path_is_set
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
        controls = ttk.Frame(self.workflow_tab)
        controls.pack(fill="x", padx=8, pady=(8, 0))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        left = ttk.Frame(controls)
        left.grid(row=0, column=0, sticky="new", padx=(0, 4))
        right = ttk.Frame(controls)
        right.grid(row=0, column=1, sticky="new", padx=(4, 0))

        self._setup_average_controls(left)
        self.fit_workspace = MatrixFitWorkspace(actions, self.log)
        self.fit_workspace.build_controls(left)
        self._setup_correction_controls(right)
        self.fit_workspace.build_view(self.workflow_tab)
        status.pack(fill="x", padx=8, pady=(4, 8))
        self.log.pack(fill="x", padx=4, pady=3)

        self.diagnostics = CalibrationDiagnosticsNotebook(self.diagnostics_tab, actions)
        self.diagnostics.pack(fill="both", expand=True, padx=8, pady=8)

    def _setup_average_controls(self, parent: tk.Misc) -> None:
        self.average_settings = self.actions.state.settings.intensifier_average
        self.average_vars = {
            "source_folder": tk.StringVar(value=_path_text(self.average_settings.source_folder)),
            "output_folder": tk.StringVar(value=_path_text(self.average_settings.output_folder)),
            "best_process_count": tk.StringVar(value=str(self.average_settings.best_process_count)),
            "denoise_threshold": tk.StringVar(value=str(self.average_settings.denoise_threshold)),
        }
        section = ttk.LabelFrame(parent, text="标定平均图")
        section.pack(fill="x", pady=(0, 4))
        paths = ttk.Frame(section)
        paths.pack(anchor="w", padx=6, pady=(4, 2))
        _path_row(paths, 0, "像增强器标定根目录", self.average_vars["source_folder"], self._choose_average_source)
        _path_row(paths, 1, "标定平均图根目录", self.average_vars["output_folder"], self._choose_average_output)
        controls = ttk.Frame(section)
        controls.pack(anchor="w", padx=6, pady=(2, 5))
        _labeled_entry(controls, "最佳进程数", self.average_vars["best_process_count"], 6)
        _labeled_entry(controls, "去噪阈值", self.average_vars["denoise_threshold"], 8)
        ttk.Button(controls, text="计算标定平均图", command=self.run_intensifier_average).pack(side="left", padx=(4, 0))

    def _setup_correction_controls(self, parent: tk.Misc) -> None:
        self.gain_settings = self.actions.state.settings.gain
        self.batch_vars = {
            "image_folder": tk.StringVar(value=_path_text(self.gain_settings.image_folder)),
            "matrix_folder": tk.StringVar(value=_path_text(self.gain_settings.matrix_folder)),
            "excel_path": tk.StringVar(value=_path_text(self.gain_settings.excel_path)),
            "output_folder": tk.StringVar(value=_path_text(self.gain_settings.output_folder)),
            "experiment_column": tk.StringVar(value=str(self.gain_settings.experiment_column)),
            "gain_column": tk.StringVar(value=str(self.gain_settings.gain_column)),
        }
        self.single_image_var = tk.StringVar()
        self.single_matrix_var = tk.StringVar(value=_path_text(self.gain_settings.matrix_path))
        self.single_output_var = tk.StringVar(value=_path_text(self.gain_settings.output_folder))
        section = ttk.LabelFrame(parent, text="实验图像修正")
        section.pack(fill="x")
        notebook = ttk.Notebook(section)
        notebook.pack(fill="both", expand=True, padx=6, pady=6)
        batch_tab = ttk.Frame(notebook)
        single_tab = ttk.Frame(notebook)
        notebook.add(batch_tab, text="按 Excel 批量修正")
        notebook.add(single_tab, text="单张修正")
        self._setup_batch_correction(batch_tab)
        self._setup_single_correction(single_tab)

    def _setup_batch_correction(self, parent: tk.Misc) -> None:
        paths = ttk.Frame(parent)
        paths.pack(anchor="w", padx=2, pady=(2, 4))
        _path_row(paths, 0, "待修正平均图目录", self.batch_vars["image_folder"], self._choose_batch_images)
        _path_row(paths, 1, "修正矩阵目录", self.batch_vars["matrix_folder"], self._choose_batch_matrices)
        _path_row(paths, 2, "实验增益 Excel", self.batch_vars["excel_path"], self._choose_batch_excel)
        _path_row(paths, 3, "修正结果输出目录", self.batch_vars["output_folder"], self._choose_batch_output)
        columns = ttk.Frame(parent)
        columns.pack(anchor="w", padx=2, pady=(0, 4))
        _labeled_entry(columns, "实验编号列", self.batch_vars["experiment_column"], 5)
        _labeled_entry(columns, "Gain列", self.batch_vars["gain_column"], 5)
        ttk.Button(parent, text="按 Excel 批量修正", command=self.apply_excel).pack(anchor="w", padx=2, pady=(0, 2))

    def _setup_single_correction(self, parent: tk.Misc) -> None:
        paths = ttk.Frame(parent)
        paths.pack(anchor="w", padx=2, pady=(2, 4))
        _path_row(paths, 0, "待修正图像", self.single_image_var, self._choose_single_image)
        _path_row(paths, 1, "修正矩阵 .npy", self.single_matrix_var, self._choose_single_matrix)
        _path_row(paths, 2, "修正结果输出目录", self.single_output_var, self._choose_single_output)
        ttk.Button(parent, text="修正所选图像", command=self.apply_single).pack(anchor="w", padx=2, pady=(0, 2))

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

    def apply_excel(self) -> None:
        try:
            self._save_batch_settings()
        except Exception as exc:
            self.log.write(f"错误: {exc}")
            return
        self._run(self.actions.apply_excel_intensifier_correction, "开始读取 Excel 并匹配 Gain 矩阵...")

    def apply_single(self) -> None:
        try:
            image_path = self._save_single_settings()
        except Exception as exc:
            self.log.write(f"错误: {exc}")
            return
        self._run(lambda: self.actions.apply_selected_intensifier_correction([image_path]), "开始修正所选图像...")

    def _save_average_settings(self) -> None:
        self.average_settings.source_folder = _to_path(self.average_vars["source_folder"].get())
        self.average_settings.output_folder = _to_path(self.average_vars["output_folder"].get())
        self.average_settings.best_process_count = int(self.average_vars["best_process_count"].get())
        self.average_settings.denoise_threshold = float(self.average_vars["denoise_threshold"].get())

    def _save_batch_settings(self) -> None:
        self.gain_settings.image_folder = _required_path(self.batch_vars["image_folder"].get(), "待修正平均图目录")
        self.gain_settings.matrix_folder = _required_path(self.batch_vars["matrix_folder"].get(), "修正矩阵目录")
        self.gain_settings.excel_path = _required_path(self.batch_vars["excel_path"].get(), "实验增益 Excel")
        self.gain_settings.output_folder = _required_path(self.batch_vars["output_folder"].get(), "修正结果输出目录")
        self.gain_settings.experiment_column = _positive_int(
            self.batch_vars["experiment_column"].get(), "实验编号列"
        )
        self.gain_settings.gain_column = _positive_int(self.batch_vars["gain_column"].get(), "Gain列")

    def _save_single_settings(self) -> Path:
        image_path = _required_path(self.single_image_var.get(), "待修正图像")
        self.gain_settings.matrix_path = _required_path(self.single_matrix_var.get(), "修正矩阵")
        self.gain_settings.output_folder = _required_path(self.single_output_var.get(), "修正结果输出目录")
        return image_path

    def _choose_average_source(self) -> None:
        self._choose_directory(self.average_vars["source_folder"])

    def _choose_average_output(self) -> None:
        self._choose_directory(self.average_vars["output_folder"])

    def _choose_batch_images(self) -> None:
        self._choose_directory(self.batch_vars["image_folder"])

    def _choose_batch_matrices(self) -> None:
        self._choose_directory(self.batch_vars["matrix_folder"])

    def _choose_batch_excel(self) -> None:
        self._choose_file(self.batch_vars["excel_path"], [("Excel", "*.xlsx *.xls")])

    def _choose_batch_output(self) -> None:
        self._choose_directory(self.batch_vars["output_folder"])

    def _choose_single_image(self) -> None:
        self._choose_file(self.single_image_var, [("NumPy 图像", "*.npy")])

    def _choose_single_matrix(self) -> None:
        self._choose_file(self.single_matrix_var, [("NumPy 矩阵", "*.npy")])

    def _choose_single_output(self) -> None:
        self._choose_directory(self.single_output_var)

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
    ttk.Entry(parent, textvariable=variable, width=48).grid(row=row, column=1, sticky="w", pady=3)
    ttk.Button(parent, text="...", width=3, command=command).grid(row=row, column=2, sticky="w", padx=(3, 0), pady=3)


def _labeled_entry(parent, label: str, variable: tk.StringVar, width: int) -> None:
    ttk.Label(parent, text=label).pack(side="left", padx=(0, 2))
    ttk.Entry(parent, textvariable=variable, width=width).pack(side="left", padx=(0, 8))


def _path_text(path: Path) -> str:
    return "" if str(path) == "." else str(path)


def _to_path(text: str) -> Path:
    return Path(text) if text.strip() else Path()


def _required_path(text: str, label: str) -> Path:
    path = _to_path(text)
    if not path_is_set(path):
        raise ValueError(f"请选择{label}")
    return path


def _positive_int(text: str, label: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"{label}必须是正整数") from exc
    if value < 1:
        raise ValueError(f"{label}必须是正整数")
    return value
