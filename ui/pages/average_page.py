from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from PLIFv2.application.actions import PLIFActions
from PLIFv2.ui.widgets import LogBox, run_in_background


class AveragePage(ttk.Frame):
    def __init__(self, master, actions: PLIFActions):
        super().__init__(master)
        self.actions = actions
        self.settings = actions.state.settings.average
        self.vars = {
            "source_folder": tk.StringVar(value=_path_text(self.settings.source_folder)),
            "output_folder": tk.StringVar(value=_path_text(self.settings.output_folder)),
            "experiment_numbers": tk.StringVar(value=self.settings.experiment_numbers),
            "process_count": tk.StringVar(value=str(self.settings.process_count)),
            "denoise_threshold": tk.StringVar(value=str(self.settings.denoise_threshold)),
            "time_start": tk.StringVar(value=str(self.settings.time_start)),
            "time_end": tk.StringVar(value=str(self.settings.time_end)),
            "sample_frequency": tk.StringVar(value=str(self.settings.sample_frequency)),
            "frame_segments": tk.StringVar(value=self.settings.frame_segments),
        }
        self.use_experiment_numbers = tk.BooleanVar(
            value=bool(self.settings.experiment_numbers.strip()) and self.settings.use_experiment_numbers
        )
        self.use_frame_segments = tk.BooleanVar(value=self.settings.use_frame_segments)
        self._setup_ui()

    def _setup_ui(self) -> None:
        paths = ttk.Frame(self)
        paths.pack(anchor="w", padx=8, pady=(8, 4))
        self.path_entries = {}
        for row, (name, label) in enumerate(
            [("source_folder", "原始图像根目录"), ("output_folder", "平均图输出目录")]
        ):
            ttk.Label(paths, text=label).grid(row=row, column=0, sticky="e", padx=(0, 4), pady=3)
            entry = ttk.Entry(paths, textvariable=self.vars[name], width=124)
            entry.grid(row=row, column=1, sticky="w", pady=3)
            ttk.Button(paths, text="...", width=3, command=lambda key=name: self._choose_directory(key)).grid(
                row=row, column=2, sticky="w", padx=(3, 0), pady=3
            )
            self.path_entries[name] = entry

        ttk.Checkbutton(
            paths,
            text="实验编号",
            variable=self.use_experiment_numbers,
            command=self._sync_experiment_state,
        ).grid(row=0, column=3, sticky="w", padx=(12, 3), pady=3)
        self.experiment_entry = ttk.Entry(paths, textvariable=self.vars["experiment_numbers"], width=66)
        self.experiment_entry.grid(row=0, column=4, sticky="w", pady=3)

        processing_row = ttk.Frame(self)
        processing_row.pack(anchor="w", padx=8, pady=3)
        _labeled_entry(processing_row, "进程数", self.vars["process_count"], 6)
        _labeled_entry(processing_row, "去噪阈值", self.vars["denoise_threshold"], 8)

        self.range_row = ttk.Frame(self)
        self.range_row.pack(anchor="w", padx=8, pady=3)
        _labeled_entry(self.range_row, "开始时间/s", self.vars["time_start"], 8)
        _labeled_entry(self.range_row, "结束时间/s", self.vars["time_end"], 8)
        _labeled_entry(self.range_row, "采样频率/Hz", self.vars["sample_frequency"], 8)
        ttk.Checkbutton(
            self.range_row,
            text="非连续区间",
            variable=self.use_frame_segments,
            command=self._sync_segment_state,
        ).pack(side="left", padx=(8, 2))
        self.segment_entry = ttk.Entry(self.range_row, textvariable=self.vars["frame_segments"], width=34)
        self.segment_entry.pack(side="left")
        ttk.Label(self.range_row, text="格式：200:600,3200:3700（帧序号，左闭右开）").pack(side="left", padx=(4, 0))
        self._sync_segment_state()
        self._sync_experiment_state()

        ttk.Button(self, text="计算平均图", command=self.run).pack(anchor="w", padx=8, pady=(5, 0))
        self.log = LogBox(self)
        self.log.pack(fill="both", expand=True, padx=8, pady=8)

    def run(self) -> None:
        try:
            self._save_to_settings()
        except Exception as exc:
            self.log.write(f"错误: {exc}")
            return

        def done(result) -> None:
            self.log.write(
                f"完成平均图: 共 {result.selected_count} 张，{len(result.items)} 个实验，"
                f"图像计算耗时 {result.calculation_seconds:.2f} s"
            )
            for item in result.items:
                self.log.write(f"{item.source_folder.name}: {item.selected_count} 张 -> {item.outputs.get('npy')}")

        run_in_background(self, self.log, self.actions.run_average, done, "开始计算平均图...")

    def _save_to_settings(self) -> None:
        self.settings.source_folder = _to_path(self.vars["source_folder"].get())
        self.settings.output_folder = _to_path(self.vars["output_folder"].get())
        self.settings.experiment_numbers = self.vars["experiment_numbers"].get().strip()
        self.settings.use_experiment_numbers = self.use_experiment_numbers.get()
        self.settings.process_count = int(self.vars["process_count"].get())
        self.settings.denoise_threshold = float(self.vars["denoise_threshold"].get())
        self.settings.time_start = float(self.vars["time_start"].get())
        self.settings.time_end = float(self.vars["time_end"].get())
        self.settings.sample_frequency = float(self.vars["sample_frequency"].get())
        self.settings.use_frame_segments = self.use_frame_segments.get()
        self.settings.frame_segments = self.vars["frame_segments"].get().strip()

    def _choose_directory(self, name: str) -> None:
        path = filedialog.askdirectory(initialdir=self.vars[name].get() or None)
        if path:
            self.vars[name].set(path)

    def _sync_segment_state(self) -> None:
        self.segment_entry.configure(state="normal" if self.use_frame_segments.get() else "disabled")

    def _sync_experiment_state(self) -> None:
        self.experiment_entry.configure(state="normal" if self.use_experiment_numbers.get() else "disabled")


def _labeled_entry(parent, label: str, variable: tk.StringVar, width: int) -> None:
    ttk.Label(parent, text=label).pack(side="left", padx=(0, 2))
    ttk.Entry(parent, textvariable=variable, width=width).pack(side="left", padx=(0, 8))


def _path_text(path: Path) -> str:
    return "" if str(path) == "." else str(path)


def _to_path(text: str) -> Path:
    return Path(text) if text.strip() else Path()
