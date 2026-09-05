from __future__ import annotations

import tkinter as tk
from copy import deepcopy
from pathlib import Path
from tkinter import filedialog, ttk

from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure

from PLIFv2.application.actions import PLIFActions
from PLIFv2.core.image_io import write_png
from PLIFv2.core.result_snapshot import require_current_result
from PLIFv2.core.mixing_presets import (
    MixingPreset,
    create_mixing_presets_from_excel,
    load_mixing_presets,
    save_mixing_presets,
)
from PLIFv2.ui.components.experiment_group_controls import (
    ExperimentTripletControls,
    MixingPresetControls,
    split_save_name,
)
from PLIFv2.ui.components.shift_search_panel import ShiftSearchPanel
from PLIFv2.ui.image_display import ImageDisplay
from PLIFv2.ui.mpl_canvas import DebouncedFigureCanvasTkAgg
from PLIFv2.ui.settings_binding import set_dataclass_text_fields
from PLIFv2.ui.widgets import LatestTaskRunner


RESULT_LAYOUT = (
    ("f", (0, 0)),
    ("o", (0, 1)),
    ("fo", (0, 2)),
    ("f+o-fo", (0, 3)),
    ("oDf+o", (1, 0)),
    ("fDfo", (1, 1)),
    ("oDfo", (1, 2)),
    ("fo-f-o", (1, 3)),
    ("ME_oDf+o", (2, 0)),
    ("ME_fDfo", (2, 1)),
    ("ME_oDfo", (2, 2)),
    ("mix_curve", (2, 3)),
)


class ManualMixingPage(ttk.Frame):
    """Manual F/O/FO alignment and mixing-characteristic calculation."""

    persist_widget_values = False

    def __init__(self, master, actions: PLIFActions):
        super().__init__(master)
        self.actions = actions
        self.settings = actions.state.settings.manual
        self.is_pseudo = tk.BooleanVar(value=True)
        self.auto_gain = tk.BooleanVar(value=self.settings.auto_gain)
        self.show_crosshair = tk.BooleanVar(value=True)
        self.show_ellipse = tk.BooleanVar(value=True)
        self.use_ellipse_region = tk.BooleanVar(value=self.settings.use_ellipse_region)
        self.use_calibrated_display = tk.BooleanVar(value=True)
        self.save_crosshair = tk.BooleanVar(value=True)
        self.save_ellipse = tk.BooleanVar(value=True)
        self.save_targets = {name: tk.BooleanVar(value=True) for name, _position in RESULT_LAYOUT}
        self.vars: dict[str, tk.StringVar] = {}
        self.displays: dict[str, ImageDisplay] = {}
        self.preset_groups: tuple[MixingPreset, ...] = ()
        self.preset_index: int | None = None
        self.preset_group_var = tk.StringVar()
        self.latest_result = None
        self.request_id = 0
        self.completed_request_id = -1
        self._setup_ui()
        self.live_task = LatestTaskRunner(self, LabelLog(self.status_label))

    def _setup_ui(self) -> None:
        control_area = ttk.Frame(self)
        control_area.pack(fill=tk.X, padx=8, pady=6)
        controls = ttk.Frame(control_area)
        self._quick_save_panel(control_area).pack(side=tk.RIGHT, anchor=tk.NE)
        controls.pack(side=tk.LEFT, anchor=tk.NW)
        self._row_image_source(controls).pack(anchor=tk.W, pady=1)
        self._row_preset(controls).pack(anchor=tk.W, pady=1)
        self._row_display(controls).pack(anchor=tk.W, pady=1)
        self._row_geometry(controls).pack(anchor=tk.W, pady=1)

        self.result_tabs = ttk.Notebook(self)
        self.result_tabs.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        mixing_tab = ttk.Frame(self.result_tabs)
        search_tab = ttk.Frame(self.result_tabs)
        self.result_tabs.add(mixing_tab, text="混合特性")
        self.result_tabs.add(search_tab, text="平移搜索")

        summary_area = ttk.Frame(mixing_tab)
        summary_area.pack(fill=tk.X)
        self.summary_label = ttk.Label(summary_area, text="计算结果")
        self.summary_label.pack(anchor=tk.W)
        self.efficiency_label = ttk.Label(summary_area, text="")
        self.efficiency_label.pack(anchor=tk.W)
        self.raw_efficiency_label = ttk.Label(summary_area, text="")
        self.raw_efficiency_label.pack(anchor=tk.W)
        self.status_label = ttk.Label(summary_area, text="")
        self.status_label.pack(anchor=tk.W)

        image_outer = ttk.Frame(mixing_tab)
        image_outer.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        canvas = tk.Canvas(image_outer, highlightthickness=0)
        scrollbar_y = ttk.Scrollbar(image_outer, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar_x = ttk.Scrollbar(image_outer, orient=tk.HORIZONTAL, command=canvas.xview)
        self.image_frame = ttk.Frame(canvas)
        self.image_frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.image_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")
        image_outer.rowconfigure(0, weight=1)
        image_outer.columnconfigure(0, weight=1)

        for title, position in RESULT_LAYOUT:
            if title == "mix_curve":
                continue
            display = ImageDisplay(self.image_frame, title)
            display.grid(row=position[0], column=position[1], padx=4, pady=4, sticky="nw")
            self.displays[title] = display

        self.curve_panel = MixCurvePanel(self.image_frame)
        self.curve_panel.grid(row=2, column=3, padx=4, pady=4, sticky="nw")
        self._bind_result_wheel(image_outer, canvas)

        self.shift_search = ShiftSearchPanel(search_tab, self.actions, self._save_vars)
        self.shift_search.pack(fill=tk.BOTH, expand=True)

    @staticmethod
    def _bind_result_wheel(container: tk.Misc, scroll_canvas: tk.Canvas) -> None:
        def scroll(event, direction: int | None = None):
            if direction is None:
                if not event.delta:
                    return None
                direction = -1 if event.delta > 0 else 1
                steps = max(1, abs(event.delta) // 120)
            else:
                steps = 1
            if event.state & 0x0001:
                scroll_canvas.xview_scroll(direction * steps, "units")
            else:
                scroll_canvas.yview_scroll(direction * steps, "units")
            return "break"

        def bind_descendants(widget: tk.Misc) -> None:
            widget.bind("<MouseWheel>", scroll, add="+")
            widget.bind("<Button-4>", lambda event: scroll(event, -1), add="+")
            widget.bind("<Button-5>", lambda event: scroll(event, 1), add="+")
            for child in widget.winfo_children():
                bind_descendants(child)

        bind_descendants(container)

    def _quick_save_panel(self, parent) -> ttk.LabelFrame:
        panel = ttk.LabelFrame(parent, text="一键保存")
        top_row = ttk.Frame(panel)
        top_row.pack(anchor=tk.W, padx=4, pady=(3, 1))
        ttk.Button(top_row, text="一键保存", command=self.save_selected_outputs).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Checkbutton(top_row, text="绘制十字线", variable=self.save_crosshair).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Checkbutton(top_row, text="绘制椭圆", variable=self.save_ellipse).pack(side=tk.LEFT)

        selections = ttk.Frame(panel)
        selections.pack(anchor=tk.W, padx=4, pady=(1, 3))
        for name, (row, column) in RESULT_LAYOUT:
            ttk.Checkbutton(selections, text=name, variable=self.save_targets[name]).grid(
                row=row,
                column=column,
                sticky=tk.W,
                padx=4,
            )
        return panel

    def _row_image_source(self, parent) -> ttk.Frame:
        return ExperimentTripletControls(
            parent,
            self.settings,
            self.vars,
            self._select_image_folder,
            self.process,
        )

    def _row_preset(self, parent) -> ttk.Frame:
        row = MixingPresetControls(
            parent,
            self.settings,
            self.vars,
            self.preset_group_var,
            self._select_preset_path,
            self._create_preset_from_excel,
            self._load_preset_file,
            self._move_preset,
            self._go_to_preset,
            self.process,
        )
        self.preset_group_entry = row.group_entry
        self.preset_total_label = row.total_label
        return row

    def _row_display(self, parent) -> ttk.Frame:
        row = ttk.Frame(parent)
        ttk.Checkbutton(row, text="伪彩图", variable=self.is_pseudo, command=self._redraw_cached_result).pack(side=tk.LEFT, padx=(3, 5))
        self._display_entry(row, "gain", "增益", 6)
        ttk.Checkbutton(row, text="自动", variable=self.auto_gain, command=self._redraw_cached_result).pack(side=tk.LEFT, padx=(4, 5))
        self._display_entry(row, "zoom", "缩小倍数", 6)
        self._entry(row, "denoise_ratio", "相对去噪阈值", 7)
        self._entry(row, "f_shift", "F 平移", 6)
        self._entry(row, "o_shift", "O 平移", 6)
        self._entry(row, "global_of", "全局 O/F", 6)
        ttk.Checkbutton(
            row,
            text="绘图使用标定总量",
            variable=self.use_calibrated_display,
            command=self._redraw_cached_result,
        ).pack(side=tk.LEFT, padx=(6, 0))
        return row

    def _row_geometry(self, parent) -> ttk.Frame:
        row = ttk.Frame(parent)
        self._display_entry(row, "line_width", "线宽", 6)
        ttk.Checkbutton(row, text="十字线", variable=self.show_crosshair, command=self._redraw_cached_result).pack(side=tk.LEFT, padx=(5, 2))
        ttk.Label(row, text="F").pack(side=tk.LEFT, padx=(5, 0))
        self._entry(row, "center_xf", "X", 6)
        self._entry(row, "center_yf", "Y", 6)
        ttk.Button(row, text="一键填充", command=self._fill_centers_from_f).pack(side=tk.LEFT, padx=(4, 2))
        ttk.Label(row, text="O").pack(side=tk.LEFT, padx=(5, 0))
        self._entry(row, "center_xo", "X", 6)
        self._entry(row, "center_yo", "Y", 6)
        ttk.Label(row, text="FO").pack(side=tk.LEFT, padx=(5, 0))
        self._entry(row, "center_xfo", "X", 6)
        self._entry(row, "center_yfo", "Y", 6)
        ttk.Checkbutton(row, text="椭圆", variable=self.show_ellipse, command=self._redraw_cached_result).pack(side=tk.LEFT, padx=(5, 2))
        self._entry(row, "major_axis", "长轴", 6)
        self._entry(row, "minor_axis", "短轴", 6)
        ttk.Checkbutton(
            row,
            text="仅椭圆区域计算混合效率",
            variable=self.use_ellipse_region,
            command=self.process,
        ).pack(side=tk.LEFT, padx=(7, 2))
        return row

    def _entry(self, row: ttk.Frame, attr: str, label: str, width: int) -> None:
        ttk.Label(row, text=label).pack(side=tk.LEFT, padx=(4, 2))
        self._text_entry(row, attr, self._value_to_str(getattr(self.settings, attr)), width)

    def _display_entry(self, row: ttk.Frame, attr: str, label: str, width: int) -> None:
        ttk.Label(row, text=label).pack(side=tk.LEFT, padx=(4, 2))
        self._text_entry(
            row,
            attr,
            self._value_to_str(getattr(self.settings, attr)),
            width,
            self._redraw_cached_result,
        )

    def _text_entry(self, row: ttk.Frame, attr: str, value: str, width: int, command=None) -> None:
        var = tk.StringVar(value=value)
        self.vars[attr] = var
        entry = ttk.Entry(row, textvariable=var, width=width)
        entry.pack(side=tk.LEFT)
        entry.bind("<Return>", command or self.process)

    def _select_image_folder(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.vars["image_folder"].set(path)

    def _select_preset_path(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("混合预设", "*.json"), ("所有文件", "*.*")],
        )
        if path:
            self.vars["preset_path"].set(path)
            self._load_preset_file()

    def _create_preset_from_excel(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Excel 文件", "*.xlsx"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            excel_path = Path(path)
            output_path = excel_path.with_suffix(".mixing-presets.json")
            catalog = create_mixing_presets_from_excel(excel_path)
            save_mixing_presets(catalog, output_path)
            self.vars["preset_path"].set(str(output_path))
            self._load_preset_file()
        except Exception as exc:
            self.status_label.config(text=f"创建预设错误: {exc}")

    def _load_preset_file(self, event=None) -> None:
        try:
            path = Path(self.vars["preset_path"].get().strip())
            catalog = load_mixing_presets(path)
            self.preset_groups = catalog.groups
            self.preset_index = self._matching_preset_index()
            if self.preset_index is None:
                self.preset_index = 0
            self._apply_preset(self.preset_index)
        except Exception as exc:
            self.preset_groups = ()
            self.preset_index = None
            self.preset_group_var.set("")
            self.preset_total_label.config(text="/-")
            self.status_label.config(text=f"加载预设错误: {exc}")

    def _move_preset(self, offset: int) -> None:
        if not self.preset_groups:
            self._load_preset_file()
            return
        current = self.preset_index if self.preset_index is not None else 0
        self._apply_preset(min(max(current + offset, 0), len(self.preset_groups) - 1))

    def _go_to_preset(self, event=None) -> str:
        if not self.preset_groups:
            self._load_preset_file()
            if not self.preset_groups:
                return "break"

        try:
            group_number = int(self.preset_group_var.get().strip())
        except ValueError:
            self.status_label.config(text=f"组号必须是 1 到 {len(self.preset_groups)} 的整数")
            if self.preset_index is not None:
                self.preset_group_var.set(str(self.preset_index + 1))
            return "break"

        if not 1 <= group_number <= len(self.preset_groups):
            self.status_label.config(text=f"组号必须在 1 到 {len(self.preset_groups)} 之间")
            if self.preset_index is not None:
                self.preset_group_var.set(str(self.preset_index + 1))
            return "break"

        self._apply_preset(group_number - 1)
        return "break"

    def _matching_preset_index(self) -> int | None:
        current = (
            self.vars["f_exp_no"].get().strip(),
            self.vars["fo_exp_no"].get().strip(),
            self.vars["o_exp_no"].get().strip(),
        )
        for index, preset in enumerate(self.preset_groups):
            if current == (preset.f_exp_no, preset.fo_exp_no, preset.o_exp_no):
                return index
        return None

    def _apply_preset(self, index: int) -> None:
        preset = self.preset_groups[index]
        self.preset_index = index
        self.vars["f_exp_no"].set(preset.f_exp_no)
        self.vars["fo_exp_no"].set(preset.fo_exp_no)
        self.vars["o_exp_no"].set(preset.o_exp_no)
        self.vars["save_dr"].set(preset.dr)
        self.vars["save_u"].set(preset.u)
        self.vars["global_of"].set(str(preset.global_of))
        self.preset_group_var.set(str(index + 1))
        self.preset_total_label.config(text=f"/{len(self.preset_groups)}")
        self.process()

    def _save_vars(self) -> None:
        candidate = deepcopy(self.settings)
        set_dataclass_text_fields(candidate, {
            name: variable.get() for name, variable in self.vars.items()
            if name not in {"save_dr", "save_u"}
        })
        dr_value = self.vars["save_dr"].get().strip()
        u_value = self.vars["save_u"].get().strip()
        candidate.save_name = f"DR{dr_value}_U{u_value}" if dr_value or u_value else ""
        candidate.use_ellipse_region = self.use_ellipse_region.get()
        candidate.auto_gain = self.auto_gain.get()
        self.settings.__dict__.update(vars(candidate))

    def _fill_centers_from_f(self) -> None:
        x_value = self.vars["center_xf"].get()
        y_value = self.vars["center_yf"].get()
        self.vars["center_xo"].set(x_value)
        self.vars["center_yo"].set(y_value)
        self.vars["center_xfo"].set(x_value)
        self.vars["center_yfo"].set(y_value)
        self.process()

    def _redraw_cached_result(self) -> None:
        try:
            self.settings.gain = float(self.vars["gain"].get())
            self.settings.zoom = float(self.vars["zoom"].get())
            self.settings.line_width = int(self.vars["line_width"].get())
            self.settings.auto_gain = self.auto_gain.get()
            if not self.settings.auto_gain and self.settings.gain <= 0:
                raise ValueError("关闭自动增益后，手动增益必须大于 0")
            if self.settings.zoom <= 0 or self.settings.line_width < 1:
                raise ValueError("缩小倍数必须大于 0，线宽必须至少为 1")
            if self.latest_result is not None:
                self._show_result(self.latest_result)
        except Exception as exc:
            self.status_label.config(text=f"错误: {exc}")

    def process(self, event=None) -> None:
        try:
            self._save_vars()
            if not self.settings.auto_gain and self.settings.gain <= 0:
                raise ValueError("关闭自动增益后，手动增益必须大于 0")
        except Exception as exc:
            self.status_label.config(text=f"错误: {exc}")
            return
        self.request_id += 1
        request_id = self.request_id
        settings = deepcopy(self.settings)

        def done(result) -> None:
            if request_id == self.request_id:
                self.completed_request_id = request_id
                self._show_result(result)

        self.live_task.submit(lambda: self.actions.run_manual_display(settings), done, "计算混合特性...")

    def _show_result(self, result) -> None:
        try:
            self.latest_result = result
            summary = result.summary
            self.status_label.config(text="")
            self.summary_label.config(
                text=f"实拍 O/F: {summary.record_of:.4f}; 差绝对值和: {summary.abs_value:.2f}"
            )
            self.efficiency_label.config(
                text=(
                    f"标定总量后：O-F 混合效率: {summary.me_o_f:.4f}; F-FO 混合效率: {summary.me_f_fo:.4f}; "
                    f"O-FO 混合效率: {summary.me_o_fo:.4f}; 平均混合效率: {summary.average_me:.4f}"
                )
            )
            self.raw_efficiency_label.config(
                text=(
                    f"原始混合效率：O-F 混合效率: {summary.raw_me_o_f:.4f}; F-FO 混合效率: {summary.raw_me_f_fo:.4f}; "
                    f"O-FO 混合效率: {summary.raw_me_o_fo:.4f}; 平均混合效率: {summary.raw_average_me:.4f}"
                )
            )
            display_images = result.images if self.use_calibrated_display.get() else result.raw_images
            mix_curve = result.mix_curve if self.use_calibrated_display.get() else result.raw_mix_curve
            gain = float(self.settings.gain)
            zoom = float(self.settings.zoom)
            ellipse = (
                self.settings.center_xfo,
                self.settings.center_yfo,
                self.settings.major_axis,
                self.settings.minor_axis,
                self.settings.line_width,
            ) if self.show_ellipse.get() else None
            for item in display_images:
                display = self.displays.get(item.name)
                if display is None:
                    continue
                colormap = item.colormap if self.is_pseudo.get() else "none"
                crosshair = self._crosshair_for(item.name) if self.show_crosshair.get() else None
                display.update(
                    item.image,
                    gain=0 if self.settings.auto_gain and item.colormap == "jet" else gain if item.colormap == "jet" else 1,
                    colormap=colormap,
                    zoom=zoom,
                    crosshair=crosshair,
                    ellipse=ellipse,
                    save_name_suffix=self.settings.save_name,
                    special_points=item.special_points,
                )
            self.curve_panel.update(mix_curve)
        except Exception as exc:
            self.status_label.config(text=f"错误: {exc}")

    def save_selected_outputs(self) -> None:
        try:
            self._save_vars()
            if self.completed_request_id != self.request_id:
                raise ValueError("请等待当前参数计算完成后再保存。")
            require_current_result(self.latest_result, self.settings)
            self._show_result(self.latest_result)
            if not self.vars["save_dr"].get().strip() or not self.vars["save_u"].get().strip():
                raise ValueError("请填写 DR 和 U 保存名称")

            output_root = Path(self.settings.image_folder).parent
            output_name = f"{self.settings.save_name}-降噪{self.settings.denoise_ratio:g}"
            ellipse = (
                self.settings.center_xfo,
                self.settings.center_yfo,
                self.settings.major_axis,
                self.settings.minor_axis,
                self.settings.line_width,
            )
            image_outputs: list[Path] = []
            for name, _position in RESULT_LAYOUT:
                if name == "mix_curve" or not self.save_targets[name].get():
                    continue
                image = self.displays[name].copy_for_save(
                    include_crosshair=self.save_crosshair.get(),
                    include_ellipse=self.save_ellipse.get(),
                    crosshair=self._crosshair_for(name),
                    ellipse=ellipse,
                )
                if image is None:
                    continue
                image_outputs.append(write_png(output_root / name / f"{output_name}.png", image))

            curve_output = None
            if self.save_targets["mix_curve"].get():
                curve_output = self.actions.export_mix_curve_data(
                    self.latest_result,
                    use_calibrated=self.use_calibrated_display.get(),
                    output_name=output_name,
                )

            saved_count = len(image_outputs) + int(curve_output is not None)
            self.status_label.config(text=f"已保存 {saved_count} 项结果到: {output_root}")
        except Exception as exc:
            self.status_label.config(text=f"保存错误: {exc}")

    def _crosshair_for(self, name: str) -> tuple[int, int, int]:
        if name == "f":
            return self.settings.center_xf, self.settings.center_yf, self.settings.line_width
        if name == "o":
            return self.settings.center_xo, self.settings.center_yo, self.settings.line_width
        return self.settings.center_xfo, self.settings.center_yfo, self.settings.line_width

    def _sync_input_vars_from_settings(self) -> None:
        """Use structured settings as the stable source after UI layout changes."""
        for attr, var in self.vars.items():
            if attr not in {"save_dr", "save_u"}:
                var.set(self._value_to_str(getattr(self.settings, attr)))
        dr_value, u_value = split_save_name(self.settings.save_name)
        self.vars["save_dr"].set(dr_value)
        self.vars["save_u"].set(u_value)

    def export_ui_state(self) -> dict[str, object]:
        try:
            self._save_vars()
        except (TypeError, ValueError):
            pass
        return {
            "is_pseudo": self.is_pseudo.get(),
            "auto_gain": self.auto_gain.get(),
            "show_crosshair": self.show_crosshair.get(),
            "show_ellipse": self.show_ellipse.get(),
            "use_calibrated_display": self.use_calibrated_display.get(),
            "save_crosshair": self.save_crosshair.get(),
            "save_ellipse": self.save_ellipse.get(),
            "save_targets": {name: target.get() for name, target in self.save_targets.items()},
        }

    def restore_ui_state(self, state: dict[str, object]) -> None:
        self._sync_input_vars_from_settings()
        self.is_pseudo.set(bool(state.get("is_pseudo", self.is_pseudo.get())))
        self.auto_gain.set(bool(state.get("auto_gain", self.auto_gain.get())))
        self.show_crosshair.set(bool(state.get("show_crosshair", self.show_crosshair.get())))
        self.show_ellipse.set(bool(state.get("show_ellipse", self.show_ellipse.get())))
        self.use_calibrated_display.set(bool(state.get("use_calibrated_display", self.use_calibrated_display.get())))
        self.save_crosshair.set(bool(state.get("save_crosshair", self.save_crosshair.get())))
        self.save_ellipse.set(bool(state.get("save_ellipse", self.save_ellipse.get())))
        targets = state.get("save_targets", {})
        if isinstance(targets, dict):
            for name, target in self.save_targets.items():
                target.set(bool(targets.get(name, target.get())))

    @staticmethod
    def _value_to_str(value) -> str:
        if isinstance(value, Path):
            return "" if str(value) == "." else str(value)
        return str(value)

class MixCurvePanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        header = ttk.Frame(self)
        header.pack(fill=tk.X)
        ttk.Label(header, text="mix_curve", width=12).pack(side=tk.LEFT)
        self.figure = Figure(figsize=(6.4, 4.8), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self.canvas = DebouncedFigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

    def update(self, curve: tuple) -> None:
        x_values, y_values = curve
        self.axis.clear()
        self.axis.plot(x_values, y_values, color="tab:blue", linewidth=1)
        self.axis.set_title("混合分数曲线")
        self.axis.set_xlabel("x")
        self.axis.set_ylabel("混合分数")
        self.axis.grid(True, alpha=0.3)
        self.canvas.request_draw()


class LabelLog:
    def __init__(self, label):
        self.label = label

    def write(self, message: str) -> None:
        self.label.config(text=message)
