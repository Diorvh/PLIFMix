from __future__ import annotations

from collections.abc import Callable
import tkinter as tk
from tkinter import ttk

from PLIFv2.application.actions import PLIFActions
from PLIFv2.application.app_state import AppState
from PLIFv2.core.plot_style import configure_matplotlib
from PLIFv2.core.ui_state_io import UIStateStore
from PLIFv2.ui.mpl_canvas import set_canvas_tree_active
from PLIFv2.ui.task_runner import close_task_runners
from PLIFv2.ui.pages.average_page import AveragePage
from PLIFv2.ui.pages.davis_page import DavisPage
from PLIFv2.ui.pages.distribution_page import DistributionPage
from PLIFv2.ui.pages.intensifier_page import IntensifierPage
from PLIFv2.ui.pages.manual_mixing_page import ManualMixingPage
from PLIFv2.ui.pages.noise_page import NoisePage
from PLIFv2.ui.pages.colorbar_page import ColorbarPage
from PLIFv2.ui.pages.distribution_comparison_page import DistributionComparisonPage
from PLIFv2.ui.pages.gif_export_page import GifExportPage
from PLIFv2.ui.pages.image_player_page import ImagePlayerPage
from PLIFv2.ui.pages.image_viewer_page import ImageViewerPage
from PLIFv2.ui.pages.npy_page import NpyPage


MANUAL_MIXING_PAGE_KEY = "4 手动调参"
DISTRIBUTION_PAGE_KEY = "5 质量分布拟合"


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

    def notify_hidden(self) -> None:
        if self.content is not None:
            callback = getattr(self.content, "on_page_hidden", None)
            if callable(callback):
                callback()


class TabRow(ttk.Frame):
    def __init__(self, master, command: Callable[[str], None]):
        super().__init__(master)
        self._command = command
        self._buttons: dict[str, ttk.Button] = {}

    def add_tab(self, key: str, label: str) -> None:
        button = ttk.Button(
            self,
            text=label,
            style="Navigation.TButton",
            command=lambda: self._command(key),
        )
        button.pack(side="left", padx=(0, 2))
        self._buttons[key] = button

    def select(self, key: str | None) -> None:
        for tab_key, button in self._buttons.items():
            button.state(["pressed"] if tab_key == key else ["!pressed"])


class PLIFAppWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        configure_matplotlib()
        self.title("PLIFv2 - 互击式喷嘴混合特性处理")
        self.geometry("1180x760")
        self.minsize(960, 620)
        self.after_idle(lambda: self.state("zoomed"))
        self.state_model = AppState()
        self.ui_state = UIStateStore(self.state_model.settings.ui)
        self.actions = PLIFActions(self.state_model)
        self._setup_style()
        self._setup_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_style(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Navigation.TButton", padding=(10, 4))
        style.map(
            "Navigation.TButton",
            relief=[("pressed", "sunken"), ("!pressed", "raised")],
        )

    def _setup_ui(self) -> None:
        header = ttk.Frame(self)
        header.pack(fill="x", padx=10, pady=(10, 4))
        ttk.Label(header, text="PLIFv2", font=("Microsoft YaHei UI", 16, "bold")).pack(side="left")
        ttk.Label(header, text="平均图 -> 像增强器修正 -> Davis -> 手动调参 -> 质量分布拟合").pack(side="left", padx=16)

        self.workflow_specs = [
            ("1 平均图", lambda master: AveragePage(master, self.actions)),
            ("2 像增强器", lambda master: IntensifierPage(master, self.actions)),
            ("3 Davis 整理", lambda master: DavisPage(master, self.actions)),
            (MANUAL_MIXING_PAGE_KEY, lambda master: ManualMixingPage(master, self.actions)),
            (DISTRIBUTION_PAGE_KEY, lambda master: DistributionPage(master, self.actions)),
            ("底噪分析", lambda master: NoisePage(master, self.actions)),
        ]
        self.secondary_specs = [
            ("图像查看器", ImageViewerPage),
            ("图片播放器", ImagePlayerPage),
            ("NPY矩阵", NpyPage),
            ("GIF导出", GifExportPage),
            ("Colorbar", ColorbarPage),
            ("分布对比", DistributionComparisonPage),
        ]
        self.content_host = ttk.Frame(self)
        navigation = ttk.Frame(self)
        navigation.pack(fill="x", padx=10, pady=(2, 0))
        self.primary_tabs = TabRow(navigation, self._select_primary_tab)
        self.primary_tabs.pack(fill="x")
        self.secondary_tabs = TabRow(navigation, self._select_tool_tab)
        self.secondary_tabs.pack(fill="x", pady=(2, 0))
        self.content_host.pack(fill="both", expand=True, padx=10, pady=(2, 10))

        self.pages: dict[str, LazyPage] = {}
        for key, factory in [*self.workflow_specs, *self.secondary_specs]:
            self.pages[key] = LazyPage(self.content_host, key, factory, self.ui_state)
        for key, _factory in self.workflow_specs:
            self.primary_tabs.add_tab(key, key)
        for key, _factory in self.secondary_specs:
            self.secondary_tabs.add_tab(key, key)

        self._migrate_saved_page_keys()
        self.active_page_key: str | None = None
        self.last_secondary_key = self.ui_state.get_app_value("last_secondary_page", self.secondary_specs[0][0])
        if self.last_secondary_key not in self.pages or self.last_secondary_key in {key for key, _factory in self.workflow_specs}:
            self.last_secondary_key = self.secondary_specs[0][0]
        saved_key = self.ui_state.get_app_value("active_page", None)
        if saved_key not in self.pages:
            saved_key = self._legacy_saved_page_key()
        self.after_idle(lambda: self._show_page(saved_key))

    def _migrate_saved_page_keys(self) -> None:
        page_key_mapping = [
            ("2.1 标定诊断", "2 像增强器"),
            ("4 平移搜索", MANUAL_MIXING_PAGE_KEY),
            ("5 手动调参", MANUAL_MIXING_PAGE_KEY),
            ("6 质量分布拟合", DISTRIBUTION_PAGE_KEY),
        ]
        pages = self.ui_state.data.pages
        for old_key, new_key in page_key_mapping:
            old_state = pages.pop(old_key, None)
            if old_state is not None:
                pages[new_key] = old_state

        active_page = self.ui_state.get_app_value("active_page")
        for old_key, new_key in page_key_mapping:
            if active_page == old_key:
                self.ui_state.set_app_value("active_page", new_key)
                break

    def _legacy_saved_page_key(self) -> str:
        try:
            saved_tab = int(self.ui_state.get_app_value("main_tab", 0))
        except (TypeError, ValueError):
            saved_tab = 0
        if 0 <= saved_tab < len(self.workflow_specs):
            return self.workflow_specs[saved_tab][0]
        return self.last_secondary_key

    def _select_primary_tab(self, key: str) -> None:
        self._show_page(key)

    def _select_tool_tab(self, key: str) -> None:
        self.last_secondary_key = key
        self._show_page(key)

    def _show_page(self, key: str) -> None:
        if key not in self.pages:
            key = self.workflow_specs[0][0]
        selected = self.pages[key]
        if self.active_page_key != key:
            if self.active_page_key is not None:
                self.pages[self.active_page_key].notify_hidden()
                self.pages[self.active_page_key].pack_forget()
            selected.pack(fill="both", expand=True)
            self.active_page_key = key
        selected.ensure_loaded()
        is_secondary = key in {page_key for page_key, _factory in self.secondary_specs}
        self.primary_tabs.select(None if is_secondary else key)
        self.secondary_tabs.select(key if is_secondary else None)
        for page_key, page in self.pages.items():
            set_canvas_tree_active(page, page_key == key)

    def _on_close(self) -> None:
        if getattr(self, "_closing", False):
            return
        self._closing = True
        self._closing_runners = close_task_runners(self)
        for page in self.pages.values():
            page.notify_hidden()
        self.title("PLIFv2 - 正在等待当前任务结束并关闭")
        self._finish_close()

    def _finish_close(self) -> None:
        if any(runner.worker_running for runner in self._closing_runners):
            self.after(50, self._finish_close)
            return
        self.ui_state.set_app_value("active_page", self.active_page_key)
        self.ui_state.set_app_value("last_secondary_page", self.last_secondary_key)
        if self.active_page_key in {key for key, _factory in self.workflow_specs}:
            self.ui_state.set_app_value("main_tab", [key for key, _factory in self.workflow_specs].index(self.active_page_key))
        else:
            self.ui_state.set_app_value("main_tab", len(self.workflow_specs))
        for page in self.pages.values():
            page.capture_state()
        self.state_model.save()
        self.actions.close()
        self.destroy()
