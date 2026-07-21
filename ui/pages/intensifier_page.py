from __future__ import annotations

from pathlib import Path
from tkinter import filedialog, ttk

from PLIFv2.application.actions import PLIFActions
from PLIFv2.ui.widgets import BoundForm, FormField, LogBox, run_in_background


class IntensifierPage(ttk.Frame):
    def __init__(self, master, actions: PLIFActions):
        super().__init__(master)
        self.actions = actions
        settings = actions.state.settings.gain
        self.form = BoundForm(
            self,
            settings,
            [
                FormField("image_folder", "图像/平均图目录", Path, 42, "dir"),
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

    def _run(self, work, message: str) -> None:
        def done(result) -> None:
            self.log.write(f"完成像增强器修正: {result.processed_count} 个文件")
            for record in result.records:
                gain_text = f", Gain={record.gain:g}" if record.gain is not None else ""
                self.log.write(f"{record.image_path.name}{gain_text} -> {record.matrix_path.name}")

        run_in_background(self, self.log, work, done, message)
