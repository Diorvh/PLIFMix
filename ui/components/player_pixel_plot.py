from __future__ import annotations

import tkinter as tk
from PLIFMix.ui import localized_widgets as ttk

import numpy as np
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure

from PLIFMix.ui.mpl_canvas import DebouncedFigureCanvasTkAgg

class PlayerPixelPlot(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.series: np.ndarray | None = None
        self.point: tuple[int, int] | None = None
        self.frame_labels: list[str] = []
        self.fixed_y_axis = tk.BooleanVar(value=False)
        self.status = ttk.Label(self, text="单击左侧图像中的像素，绘制该点的去噪灰度值。")
        self.status.pack(anchor="w", padx=6, pady=(4, 0))
        axis_controls = ttk.Frame(self)
        axis_controls.pack(fill="x", padx=6, pady=(2, 0))
        ttk.Checkbutton(
            axis_controls,
            text="固定纵轴 0-65535",
            variable=self.fixed_y_axis,
            command=self._update_y_axis,
        ).pack(side="left")
        self.figure = Figure(figsize=(4, 5), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self.axis.set_xlabel("图片序号")
        self.axis.set_ylabel("灰度值(16位)")
        self.axis.grid(True, alpha=0.3)
        self.canvas = DebouncedFigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(fill="x", padx=4)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        table_frame = ttk.Frame(self)
        table_frame.pack(fill="x", padx=6, pady=(2, 6))
        self.top_values = ttk.Treeview(
            table_frame,
            columns=("rank", "frame", "value"),
            show="headings",
            height=5,
        )
        self.top_values.heading("rank", text="#")
        self.top_values.heading("frame", text="图片序号")
        self.top_values.heading("value", text="灰度值")
        self.top_values.column("rank", width=42, stretch=False, anchor="e")
        self.top_values.column("frame", width=76, stretch=False, anchor="e")
        self.top_values.column("value", width=110, stretch=True, anchor="e")
        self.top_values.pack(fill="x")

    def set_loading(self, x: int, y: int) -> None:
        self.status.config(text=f"正在读取坐标({x}, {y})的去噪灰度值...")

    def set_series(
        self,
        x: int,
        y: int,
        values: np.ndarray,
        frame_axis: np.ndarray,
        frame_labels: list[str],
        top_k: int,
    ) -> None:
        self.series = values
        self.point = (x, y)
        self.frame_labels = frame_labels
        self.axis.clear()
        self.axis.plot(frame_axis, values, linewidth=0.8)
        self.axis.set_xlabel("图片序号")
        self.axis.set_ylabel("灰度值(16位)")
        self.axis.set_title(f"去噪像素 ({x}, {y})")
        self.axis.grid(True, alpha=0.3)
        self._apply_y_axis_limit()
        self.status.config(text=f"像素 ({x}, {y})，共 {len(values)} 张图片")
        self.canvas.request_draw()
        self.update_top_k(top_k)

    def set_error(self, message: str) -> None:
        self.status.config(text=message)

    def reset(self) -> None:
        self.series = None
        self.point = None
        self.frame_labels = []
        self.axis.clear()
        self.axis.set_xlabel("图片序号")
        self.axis.set_ylabel("灰度值(16位)")
        self.axis.grid(True, alpha=0.3)
        self.status.config(text="单击左侧图像中的像素，绘制该点的去噪灰度值。")
        self.top_values.delete(*self.top_values.get_children())
        self.canvas.request_draw()

    def update_top_k(self, top_k: int) -> None:
        if self.series is None:
            return
        self.top_values.delete(*self.top_values.get_children())
        count = min(top_k, len(self.series))
        for rank, frame_index in enumerate(np.argsort(self.series)[::-1][:count], start=1):
            label = self.frame_labels[frame_index] if frame_index < len(self.frame_labels) else str(frame_index)
            self.top_values.insert("", "end", values=(rank, label, f"{self.series[frame_index]:g}"))

    def _update_y_axis(self) -> None:
        if self.series is None:
            return
        self._apply_y_axis_limit()
        self.canvas.request_draw()

    def _apply_y_axis_limit(self) -> None:
        if self.fixed_y_axis.get():
            self.axis.set_ylim(0, 65535)
        else:
            self.axis.relim()
            self.axis.autoscale_view(scalex=False, scaley=True)
