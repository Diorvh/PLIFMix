from __future__ import annotations

from pathlib import Path
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure

from PLIFv2.application.actions import PLIFActions
from PLIFv2.ui.mpl_canvas import DebouncedFigureCanvasTkAgg
from PLIFv2.ui.widgets import BoundForm, FormField, LogBox, run_in_background


class ShiftSearchPage(ttk.Frame):
    def __init__(self, master, actions: PLIFActions):
        super().__init__(master)
        self.actions = actions
        self.form = BoundForm(
            self,
            actions.state.settings.shift,
            [
                FormField("image_folder", "修正后图像目录", Path, 78, "dir"),
                FormField("f_exp_no", "F 实验编号", str, 12),
                FormField("o_exp_no", "O 实验编号", str, 12),
                FormField("fo_exp_no", "FO 实验编号", str, 12),
                FormField("f_min", "F 平移最小", int, 12),
                FormField("f_max", "F 平移最大", int, 12),
                FormField("o_min", "O 平移最小", int, 12),
                FormField("o_max", "O 平移最大", int, 12),
                FormField("ellipse_x", "椭圆中心 X", int, 12),
                FormField("ellipse_y", "椭圆中心 Y", int, 12),
                FormField("major_axis", "长轴", int, 12),
                FormField("minor_axis", "短轴", int, 12),
                FormField("global_of", "全局 O/F", float, 12),
                FormField("process_count", "进程数", int, 12),
            ],
        )
        self.form.pack(fill="x", padx=8, pady=8)
        ttk.Button(self, text="搜索平移量", command=self.run).pack(anchor="w", padx=8)
        self.log = LogBox(self, height=4)
        self.log.pack(fill="x", padx=8, pady=8)
        self._setup_plot_area()

    def _setup_plot_area(self) -> None:
        plot_frame = ttk.Frame(self)
        plot_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.figures: list[Figure] = []
        self.axes = []
        self.canvases: list[DebouncedFigureCanvasTkAgg] = []
        titles = ["F信号+O信号与FO信号的差绝对值和", "混合效率极差", "平均混合效率"]
        for index, title in enumerate(titles):
            panel = ttk.Frame(plot_frame)
            panel.grid(row=0, column=index, sticky="nsew", padx=4)
            fig = Figure(figsize=(5.2, 4.2), dpi=100)
            ax = fig.add_subplot(111, projection="3d")
            ax.set_title(title)
            canvas = DebouncedFigureCanvasTkAgg(fig, master=panel)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            toolbar_frame = ttk.Frame(panel)
            toolbar_frame.pack(fill="x")
            NavigationToolbar2Tk(canvas, toolbar_frame)
            self.figures.append(fig)
            self.axes.append(ax)
            self.canvases.append(canvas)
            plot_frame.columnconfigure(index, weight=1)
        plot_frame.rowconfigure(0, weight=1)

    def run(self) -> None:
        try:
            self.form.save_to_model()
        except Exception as exc:
            self.log.write(f"错误: {exc}")
            return
        run_in_background(self, self.log, self.actions.run_shift_search, self._show_result, "开始搜索平移量...")

    def _show_result(self, result) -> None:
        best_abs = result.best_abs
        best_me = result.best_me_ptp
        self.log.write(
            f"差绝对值最小: F={best_abs[0]}, O={best_abs[1]}, value={best_abs[2]:.4f}\n"
            f"混合效率极差最小: F={best_me[0]}, O={best_me[1]}, value={best_me[2]:.4f}"
        )
        self._update_plots(result)

    def _update_plots(self, result) -> None:
        x_grid, y_grid = np.meshgrid(result.f_shifts, result.o_shifts)
        surfaces = [
            (result.abs_matrix.T, "F信号+O信号与FO信号的差绝对值和", "差绝对值和", "viridis"),
            (result.me_ptp_matrix.T, "混合效率极差", "混合效率极差", "plasma"),
            (((result.me1_matrix + result.me2_matrix + result.me3_matrix) / 3).T, "平均混合效率", "混合效率", "plasma"),
        ]
        for ax, canvas, (z_data, title, z_label, cmap) in zip(self.axes, self.canvases, surfaces):
            ax.clear()
            ax.plot_surface(x_grid, y_grid, z_data, cmap=cmap)
            ax.set_title(title)
            ax.set_xlabel("F平移量", fontsize=10, labelpad=10)
            ax.set_ylabel("O平移量", fontsize=10, labelpad=10)
            ax.set_zlabel(z_label, fontsize=10, labelpad=10)
            canvas.request_draw()
