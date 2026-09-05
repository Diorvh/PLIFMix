from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from tkinter import ttk

from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure

from PLIFv2.application.actions import PLIFActions
from PLIFv2.ui.mpl_canvas import DebouncedFigureCanvasTkAgg
from PLIFv2.ui.widgets import BoundForm, FormField, LogBox, run_in_background


class NoisePage(ttk.Frame):
    def __init__(self, master, actions: PLIFActions):
        super().__init__(master)
        self.actions = actions
        self.form = BoundForm(
            self,
            actions.state.settings.noise,
            [
                FormField("source_folder", "原始帧目录", Path, 42, "dir"),
                FormField("frame_start", "起始帧", int, 10),
                FormField("frame_end", "结束帧", int, 10),
                FormField("process_count", "进程数", int, 8),
            ],
            columns=2,
        )
        self.form.pack(anchor="w", padx=8, pady=6)
        ttk.Button(self, text="计算底噪最大值趋势", command=self.run).pack(anchor="w", padx=8)
        self.info = ttk.Label(self, text="")
        self.info.pack(anchor="w", padx=8, pady=4)
        self.figure = Figure(figsize=(10, 5), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self.canvas = DebouncedFigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8)
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=8)
        NavigationToolbar2Tk(self.canvas, toolbar)
        self.log = LogBox(self, height=3)
        self.log.pack(fill="x", padx=8, pady=4)

    def run(self) -> None:
        try:
            self.form.save_to_model()
        except Exception as exc:
            self.log.write(f"错误: {exc}")
            return
        settings = deepcopy(self.actions.state.settings.noise)
        run_in_background(self, self.log, lambda: self.actions.run_noise_analysis(settings), self._show, "开始分析底噪...")

    def _show(self, result) -> None:
        self.axis.clear()
        self.axis.plot(result.frame_indices, result.max_values, color="#2E86C1", linewidth=1.5)
        self.axis.set_xlabel("帧编号")
        self.axis.set_ylabel("最大灰度值")
        self.axis.grid(True, linestyle="--", alpha=0.5)
        self.info.config(text=f"帧数 {len(result.max_values)}；平均最大值 {result.average:.2f}；全局最大值 {result.maximum:.2f}")
        self.canvas.request_draw()
