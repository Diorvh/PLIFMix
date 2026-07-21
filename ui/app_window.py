from __future__ import annotations

from collections.abc import Callable
import tkinter as tk
from tkinter import ttk

from PLIFv2.application.actions import PLIFActions
from PLIFv2.application.app_state import AppState
from PLIFv2.core.plot_style import configure_matplotlib
from PLIFv2.core.ui_state_io import UIStateStore
from PLIFv2.ui.mpl_canvas import set_canvas_tree_active
from PLIFv2.ui.pages.average_page import AveragePage
from PLIFv2.ui.pages.calibration_page import CalibrationPage
from PLIFv2.ui.pages.davis_page import DavisPage
from PLIFv2.ui.pages.distribution_page import DistributionPage
from PLIFv2.ui.pages.intensifier_page import IntensifierPage
from PLIFv2.ui.pages.manual_mixing_page import ManualMixingPage
from PLIFv2.ui.pages.noise_page import NoisePage
from PLIFv2.ui.pages.shift_search_page import ShiftSearchPage
from PLIFv2.ui.pages.toolbox_page import ToolboxPage


class LazyPage(ttk.Frame):
    def __init__(self, master, page_key: str, factory: Callable[[tk.Misc], ttk.Frame], ui_state: UIStateStore):
        super().__init__(master)
        self.page_key = page_key
        self._factory = factory
        self._ui_state = ui_state
        self.content: ttk.Frame | None = None

    def ensure_loaded(self) -> ttk.Frame:
        if self.content is None:
            self.content = self._factory(self)
            self.content.pack(fill="both", expand=True)
            self._ui_state.restore_page(self.page_key, self.content)
        return self.content

    def capture_state(self) -> None:
        if self.content is not None:
            self._ui_state.capture_page(self.page_key, self.content)


class PLIFAppWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        configure_matplotlib()
        self.title("PLIFv2 - 互击式喷嘴混合特性处理")
        self.geometry("1180x760")
        self.minsize(960, 620)
        self.state_model = AppState()
        self.ui_state = UIStateStore()
        self.actions = PLIFActions(self.state_model)
        self._setup_style()
        self._setup_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_style(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")

    def _setup_ui(self) -> None:
        header = ttk.Frame(self)
        header.pack(fill="x", padx=10, pady=(10, 4))
        ttk.Label(header, text="PLIFv2", font=("Microsoft YaHei UI", 16, "bold")).pack(side="left")
        ttk.Label(header, text="平均图 -> 像增强器修正 -> Davis -> 平移搜索 -> 手动调参").pack(side="left", padx=16)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        page_specs = [
            (lambda master: AveragePage(master, self.actions), "1 平均图"),
            (lambda master: IntensifierPage(master, self.actions), "2 像增强器"),
            (lambda master: CalibrationPage(master, self.actions), "2.1 标定诊断"),
            (lambda master: DavisPage(master, self.actions), "3 Davis 整理"),
            (lambda master: ShiftSearchPage(master, self.actions), "4 平移搜索"),
            (lambda master: ManualMixingPage(master, self.actions), "5 手动调参"),
            (lambda master: DistributionPage(master, self.actions), "6 质量分布拟合"),
            (lambda master: NoisePage(master, self.actions), "底噪分析"),
            (lambda master: ToolboxPage(master), "工具箱"),
        ]
        self.pages = [
            LazyPage(self.notebook, title, factory, self.ui_state)
            for factory, title in page_specs
        ]
        for page, (_, title) in zip(self.pages, page_specs):
            self.notebook.add(page, text=title)
        self.notebook.bind("<<NotebookTabChanged>>", self._activate_current_page)
        self.after_idle(self._activate_current_page)

    def _activate_current_page(self, _event=None) -> None:
        selected = self.notebook.nametowidget(self.notebook.select())
        if isinstance(selected, LazyPage):
            selected.ensure_loaded()
        for page in self.pages:
            set_canvas_tree_active(page, page is selected)

    def _on_close(self) -> None:
        for page in self.pages:
            page.capture_state()
        self.ui_state.save()
        self.state_model.save()
        self.destroy()
