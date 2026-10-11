from __future__ import annotations

import tkinter as tk
from copy import deepcopy
from PLIFMix.ui import localized_widgets as ttk
from typing import Callable

import numpy as np
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure

from PLIFMix.application.actions import PLIFActions
from PLIFMix.ui.mpl_canvas import DebouncedFigureCanvasTkAgg
from PLIFMix.ui.widgets import LogBox, run_in_background


class ShiftSearchPanel(ttk.Frame):
    """Brute-force shift search using the manual-mixing inputs supplied by its parent."""

    def __init__(self, master, actions: PLIFActions, save_manual_settings: Callable[[], None]):
        super().__init__(master)
        self.actions = actions
        self.settings = actions.state.settings.shift
        self.save_manual_settings = save_manual_settings
        self.vars: dict[str, tk.StringVar] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        controls = ttk.Frame(self)
        controls.pack(anchor=tk.W, padx=8, pady=(8, 4))
        self._entry(controls, "f_min", "F 平移最小", 6)
        self._entry(controls, "f_max", "最大", 6)
        self._entry(controls, "o_min", "O 平移最小", 6)
        self._entry(controls, "o_max", "最大", 6)
        self._entry(controls, "process_count", "进程数", 6)
        ttk.Button(controls, text="搜索平移量", command=self.run).pack(side=tk.LEFT, padx=6)

        self.log = LogBox(self, height=3)
        self.log.pack(fill=tk.X, padx=8, pady=(0, 6))
        self._setup_plot_area()

    def _entry(self, parent: ttk.Frame, attr: str, label: str, width: int) -> None:
        ttk.Label(parent, text=label).pack(side=tk.LEFT, padx=(4, 2))
        var = tk.StringVar(value=str(getattr(self.settings, attr)))
        self.vars[attr] = var
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.pack(side=tk.LEFT)
        entry.bind("<Return>", self.run)

    def _setup_plot_area(self) -> None:
        plot_frame = ttk.Frame(self)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.axes = []
        self.canvases: list[DebouncedFigureCanvasTkAgg] = []
        titles = ["F+O 与 FO 的差绝对值和", "混合效率极差", "平均混合效率"]
        for index, title in enumerate(titles):
            panel = ttk.Frame(plot_frame)
            panel.grid(row=0, column=index, sticky="nsew", padx=4)
            figure = Figure(figsize=(5.2, 4.2), dpi=100)
            axis = figure.add_subplot(111, projection="3d")
            axis.set_title(title)
            canvas = DebouncedFigureCanvasTkAgg(figure, master=panel)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            toolbar_frame = ttk.Frame(panel)
            toolbar_frame.pack(fill=tk.X)
            NavigationToolbar2Tk(canvas, toolbar_frame)
            self.axes.append(axis)
            self.canvases.append(canvas)
            plot_frame.columnconfigure(index, weight=1)
        plot_frame.rowconfigure(0, weight=1)

    def _save(self) -> None:
        values = {attr: int(var.get()) for attr, var in self.vars.items()}
        self.settings.__dict__.update(values)

    def run(self, event=None) -> None:
        try:
            self.save_manual_settings()
            self._save()
        except Exception as exc:
            self.log.write(f"错误: {exc}")
            return
        manual = deepcopy(self.actions.state.settings.manual)
        search = deepcopy(self.settings)
        run_in_background(self, self.log, lambda: self.actions.run_manual_shift_search(manual, search), self._show_result, "开始搜索平移量...")

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
            (result.abs_matrix.T, "F+O 与 FO 的差绝对值和", "差绝对值和", "viridis"),
            (result.me_ptp_matrix.T, "混合效率极差", "混合效率极差", "plasma"),
            (((result.me1_matrix + result.me2_matrix + result.me3_matrix) / 3).T, "平均混合效率", "混合效率", "plasma"),
        ]
        for axis, canvas, (z_data, title, z_label, cmap) in zip(self.axes, self.canvases, surfaces):
            axis.clear()
            axis.plot_surface(x_grid, y_grid, z_data, cmap=cmap)
            axis.set_title(title)
            axis.set_xlabel("F 平移量", fontsize=10, labelpad=10)
            axis.set_ylabel("O 平移量", fontsize=10, labelpad=10)
            axis.set_zlabel(z_label, fontsize=10, labelpad=10)
            canvas.request_draw()
