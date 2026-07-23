from __future__ import annotations

from pathlib import Path
from tkinter import ttk

from PLIFv2.application.actions import PLIFActions
from PLIFv2.ui.widgets import BoundForm, FormField, LogBox, run_in_background


class DavisPage(ttk.Frame):
    def __init__(self, master, actions: PLIFActions):
        super().__init__(master)
        self.actions = actions
        input_section = ttk.LabelFrame(self, text="准备 Davis 输入")
        input_section.pack(fill="x", padx=8, pady=(8, 4))
        self.input_form = BoundForm(
            input_section,
            actions.state.settings.davis,
            [
                FormField("corrected_folder", "像增强器修正后目录", Path, 78, "dir"),
                FormField("davis_input_folder", "Davis 输入目录", Path, 78, "dir"),
            ],
            columns=1,
        )
        self.input_form.pack(anchor="w", padx=6, pady=(4, 2))
        ttk.Button(input_section, text="准备 Davis 输入", command=self.prepare).pack(anchor="w", padx=6, pady=(0, 5))

        output_section = ttk.LabelFrame(self, text="整理 Davis 输出")
        output_section.pack(fill="x", padx=8, pady=4)
        self.output_form = BoundForm(
            output_section,
            actions.state.settings.davis,
            [
                FormField("davis_export_folder", "Davis 导出目录", Path, 78, "dir"),
                FormField("output_folder", "整理输出目录", Path, 78, "dir"),
            ],
            columns=1,
        )
        self.output_form.pack(anchor="w", padx=6, pady=(4, 2))
        ttk.Button(output_section, text="整理 Davis 输出", command=self.run).pack(anchor="w", padx=6, pady=(0, 5))
        self.log = LogBox(self)
        self.log.pack(fill="both", expand=True, padx=8, pady=8)

    def run(self) -> None:
        try:
            self.output_form.save_to_model()
        except Exception as exc:
            self.log.write(f"错误: {exc}")
            return
        run_in_background(
            self,
            self.log,
            self.actions.import_davis,
            lambda result: self.log.write(f"完成 Davis 整理: tif={result.tif_count}, npy={result.npy_count}, 输出 {result.output_folder}"),
            "开始整理 Davis 输出...",
        )

    def prepare(self) -> None:
        try:
            self.input_form.save_to_model()
        except Exception as exc:
            self.log.write(f"错误: {exc}")
            return
        run_in_background(
            self,
            self.log,
            self.actions.prepare_davis_input,
            lambda result: self.log.write(f"已复制 {result.copied_count} 个 GAIN tif 到 {result.output_folder}"),
            "开始准备 Davis 输入目录...",
        )
