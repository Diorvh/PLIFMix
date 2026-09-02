from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, simpledialog, ttk

from PLIFv2.application.actions import PLIFActions
from PLIFv2.core.image_io import write_png
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
from PLIFv2.ui.image_display import ImageDisplay
from PLIFv2.ui.settings_binding import set_dataclass_text_field
from PLIFv2.ui.widgets import LatestTaskRunner


DISTRIBUTION_SAVE_LAYOUT = (
    ("f", (0, 0)),
    ("o", (0, 1)),
    ("fo", (0, 2)),
    ("fit_data", (1, 0)),
)


class DistributionPage(ttk.Frame):
    persist_widget_values = False

    def __init__(self, master, actions: PLIFActions):
        super().__init__(master)
        self.actions = actions
        self.settings = actions.state.settings.distribution
        self.vars: dict[str, tk.StringVar] = {}
        self.is_pseudo = tk.BooleanVar(value=True)
        self.auto_gain = tk.BooleanVar(value=self.settings.gain <= 0)
        self.show_crosshair = tk.BooleanVar(value=True)
        self.show_contour = tk.BooleanVar(value=False)
        self.show_center_line = tk.BooleanVar(value=False)
        self.show_fitted_center_line = tk.BooleanVar(value=False)
        self.image_mode = tk.StringVar(value="ori_img")
        self.show_bent = tk.BooleanVar(value=False)
        self.save_crosshair = tk.BooleanVar(value=True)
        self.save_targets = {name: tk.BooleanVar(value=True) for name, _position in DISTRIBUTION_SAVE_LAYOUT}
        self.preset_groups: tuple[MixingPreset, ...] = ()
        self.preset_index: int | None = None
        self.preset_group_var = tk.StringVar()
        self.displays: dict[str, ImageDisplay] = {}
        self.latest_result = None
        self.request_id = 0
        self.pending_status_notice = ""
        self._setup_ui()
        self.live_task = LatestTaskRunner(self, LabelLog(self.status))
        self.registration_task = LatestTaskRunner(self, LabelLog(self.status), delay_ms=0)

    def _setup_ui(self) -> None:
        control_area = ttk.Frame(self)
        control_area.pack(fill=tk.X, padx=8, pady=6)
        controls = ttk.Frame(control_area)
        self._quick_save_panel(control_area).pack(side=tk.RIGHT, anchor=tk.NE)
        controls.pack(side=tk.LEFT, anchor=tk.NW)
        self._row_image_source(controls).pack(anchor=tk.W, pady=1)
        self._row_preset(controls).pack(anchor=tk.W, pady=1)
        self._row_display(controls).pack(anchor=tk.W, pady=1)
        self._row_display_options(controls).pack(anchor=tk.W, pady=1)
        self._row_fit_parameters(controls).pack(anchor=tk.W, pady=1)
        self._row_fit_result_file(controls).pack(anchor=tk.W, pady=1)

        self.summary = DistributionSummaryPanel(self)
        self.summary.pack(fill=tk.X, padx=8)
        self.status = ttk.Label(self, text="")
        self.status.pack(anchor=tk.W, padx=8)

        displays = ttk.Frame(self)
        displays.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        for column, (name, label) in enumerate((("f", "F"), ("o", "O"), ("fo", "FO"))):
            self._image_parameter_panel(displays, name, label).grid(row=0, column=column, sticky="nw", padx=4)

    def _quick_save_panel(self, parent) -> ttk.LabelFrame:
        panel = ttk.LabelFrame(parent, text="一键保存")
        top_row = ttk.Frame(panel)
        top_row.pack(anchor=tk.W, padx=4, pady=(3, 1))
        ttk.Button(top_row, text="一键保存", command=self.save_selected_outputs).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Checkbutton(top_row, text="绘制十字线", variable=self.save_crosshair).pack(side=tk.LEFT)

        selections = ttk.Frame(panel)
        selections.pack(anchor=tk.W, padx=4, pady=(1, 3))
        for name, (row, column) in DISTRIBUTION_SAVE_LAYOUT:
            label = "拟合数据" if name == "fit_data" else name
            ttk.Checkbutton(selections, text=label, variable=self.save_targets[name]).grid(
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
            self._choose_folder,
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
        return row

    def _row_display_options(self, parent) -> ttk.Frame:
        panel = ttk.Frame(parent)

        overlay_row = ttk.Frame(panel)
        overlay_row.pack(anchor=tk.W)
        ttk.Label(overlay_row, text="覆盖件").pack(side=tk.LEFT, padx=(3, 6))
        ttk.Checkbutton(
            overlay_row,
            text="十字线",
            variable=self.show_crosshair,
            command=self._redraw_cached_result,
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Checkbutton(
            overlay_row,
            text="轮廓",
            variable=self.show_contour,
            command=self.process,
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Checkbutton(
            overlay_row,
            text="中心线",
            variable=self.show_center_line,
            command=lambda: self._on_center_line_changed("center"),
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Checkbutton(
            overlay_row,
            text="拟合中心线",
            variable=self.show_fitted_center_line,
            command=lambda: self._on_center_line_changed("fitted"),
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(overlay_row, text="还原", command=self._reset_display_options).pack(side=tk.LEFT, padx=(4, 0))

        data_row = ttk.Frame(panel)
        data_row.pack(anchor=tk.W)
        ttk.Label(data_row, text="图像数据").pack(side=tk.LEFT, padx=(3, 6))
        for label, mode in (("原图", "ori_img"), ("拟合", "fit_img"), ("拟合-原图绝对值", "delta_img")):
            ttk.Radiobutton(data_row, text=label, variable=self.image_mode, value=mode, command=self.process).pack(
                side=tk.LEFT,
                padx=(0, 5),
            )
        ttk.Checkbutton(data_row, text="掰直", variable=self.show_bent, command=self.process).pack(
            side=tk.LEFT,
            padx=(2, 0),
        )
        return panel

    def _row_fit_parameters(self, parent) -> ttk.Frame:
        row = ttk.Frame(parent)
        self._display_entry(row, "line_width", "线宽", 6)
        self._entry(row, "thres_contour", "轮廓比例", 6)
        self._entry(row, "max_k", "最大值个数", 5)
        ttk.Button(row, text="自动原点和偏转角", command=self.auto_registration).pack(side=tk.LEFT, padx=(8, 0))
        return row

    def _row_fit_result_file(self, parent) -> ttk.Frame:
        row = ttk.Frame(parent)
        ttk.Label(row, text="拟合结果写入文件").pack(side=tk.LEFT, padx=(4, 2))
        self._text_entry(
            row,
            "fit_result_path",
            self._value_to_str(self.settings.fit_result_path),
            52,
        )
        ttk.Button(row, text="...", width=3, command=self._choose_fit_result_file).pack(side=tk.LEFT, padx=(3, 6))
        ttk.Button(row, text="写入拟合结果", command=self.write_fit_results).pack(side=tk.LEFT)
        return row

    def _image_parameter_panel(self, parent, name: str, label: str) -> ttk.Frame:
        panel = ttk.Frame(parent)
        title_row = ttk.Frame(panel)
        title_row.pack(anchor=tk.W)
        ttk.Label(title_row, text=label, font=("Microsoft YaHei UI", 10, "bold")).pack(side=tk.LEFT, padx=(4, 8))
        self._entry(title_row, f"center_x{name}", "十字线 X", 5)
        self._entry(title_row, f"center_y{name}", "Y", 5)
        ttk.Button(title_row, text="保存", command=lambda: self.save_display(name)).pack(side=tk.LEFT, padx=(8, 0))

        fit_row = ttk.Frame(panel)
        fit_row.pack(anchor=tk.W)
        self._entry(fit_row, f"q_center_up_{name}", "中心线上界", 5)
        self._entry(fit_row, f"q_center_down_{name}", "下界", 5)
        self._entry(fit_row, f"rotate_angle_{name}", "旋转角度", 5)

        display = ImageDisplay(panel, name)
        display.pack(anchor=tk.W)
        self.displays[name] = display
        return panel

    def save_display(self, name: str) -> None:
        display = self.displays.get(name)
        if display is None or display.display is None:
            self.status.config(text=f"{name.upper()} 当前没有可保存的渲染图像")
            return
        folder_text = filedialog.askdirectory(title=f"选择 {name.upper()} 图像保存目录")
        if not folder_text:
            return
        filename = simpledialog.askstring(
            "输入保存名称",
            f"请输入 {name.upper()} 图像名称（不含扩展名）",
            initialvalue=name.upper(),
            parent=self,
        )
        if not filename or not filename.strip():
            return
        filename = Path(filename.strip()).name
        if not filename.lower().endswith(".png"):
            filename += ".png"
        output = Path(folder_text) / filename
        write_png(output, display.display.copy())
        self.status.config(text=f"已保存 {name.upper()} 图像：{output}")

    def _entry(self, row, name: str, label: str, width: int) -> None:
        ttk.Label(row, text=label).pack(side=tk.LEFT, padx=(4, 2))
        self._text_entry(row, name, self._value_to_str(getattr(self.settings, name)), width)

    def _display_entry(self, row, name: str, label: str, width: int) -> None:
        ttk.Label(row, text=label).pack(side=tk.LEFT, padx=(4, 2))
        self._text_entry(
            row,
            name,
            self._value_to_str(getattr(self.settings, name)),
            width,
            self._redraw_cached_result,
        )

    def _text_entry(self, row, name: str, value: str, width: int, command=None) -> None:
        var = tk.StringVar(value=value)
        self.vars[name] = var
        entry = ttk.Entry(row, textvariable=var, width=width)
        entry.pack(side=tk.LEFT)
        entry.bind("<Return>", command or self.process)

    def _save_vars(self) -> None:
        for name, variable in self.vars.items():
            if name in {"save_dr", "save_u"}:
                continue
            set_dataclass_text_field(self.settings, name, variable.get())
        dr_value = self.vars["save_dr"].get().strip()
        u_value = self.vars["save_u"].get().strip()
        self.settings.save_name = f"DR{dr_value}_U{u_value}" if dr_value or u_value else ""

    def _choose_folder(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.vars["image_folder"].set(path)

    def _choose_fit_result_file(self) -> None:
        path = filedialog.askopenfilename(
            title="选择拟合结果 Excel",
            filetypes=[("Excel 文件", "*.xlsx"), ("所有文件", "*.*")],
        )
        if path:
            self.vars["fit_result_path"].set(path)
            self._save_vars()

    def write_fit_results(self) -> None:
        try:
            self._save_vars()
            if self.latest_result is None:
                raise ValueError("请先完成当前 F、O、FO 的拟合计算。")
            output = self.actions.write_distribution_fit_results(self.latest_result)
            self.status.config(text=f"拟合结果已写入: {output}")
        except Exception as exc:
            self.status.config(text=f"写入拟合结果失败: {exc}")

    def _select_preset_path(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("混合预设", "*.json"), ("所有文件", "*.*")])
        if path:
            self.vars["preset_path"].set(path)
            self._load_preset_file()

    def _create_preset_from_excel(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Excel 文件", "*.xlsx"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            excel_path = Path(path)
            output_path = excel_path.with_suffix(".mixing-presets.json")
            save_mixing_presets(create_mixing_presets_from_excel(excel_path), output_path)
            self.vars["preset_path"].set(str(output_path))
            self._load_preset_file()
        except Exception as exc:
            self.status.config(text=f"创建预设错误: {exc}")

    def _load_preset_file(self, event=None) -> None:
        try:
            self._save_vars()
            catalog = load_mixing_presets(self.settings.preset_path)
            self.preset_groups = catalog.groups
            self.preset_index = self._matching_preset_index()
            self._apply_preset(self.preset_index if self.preset_index is not None else 0)
        except Exception as exc:
            self.preset_groups = ()
            self.preset_index = None
            self.preset_group_var.set("")
            self.preset_total_label.config(text="/-")
            self.status.config(text=f"加载预设错误: {exc}")

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
            self.status.config(text=f"组号必须是 1 到 {len(self.preset_groups)} 的整数")
            return "break"
        if not 1 <= group_number <= len(self.preset_groups):
            self.status.config(text=f"组号必须在 1 到 {len(self.preset_groups)} 之间")
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
        self.preset_group_var.set(str(index + 1))
        self.preset_total_label.config(text=f"/{len(self.preset_groups)}")
        self.process()

    def _redraw_cached_result(self) -> None:
        try:
            self.settings.gain = float(self.vars["gain"].get())
            self.settings.zoom = float(self.vars["zoom"].get())
            self.settings.line_width = int(self.vars["line_width"].get())
            if not self.auto_gain.get() and self.settings.gain <= 0:
                raise ValueError("关闭自动增益后，手动增益必须大于 0")
            if self.settings.zoom <= 0 or self.settings.line_width < 1:
                raise ValueError("缩小倍数必须大于 0，线宽必须至少为 1")
            if self.latest_result is not None:
                self._show(self.latest_result)
        except Exception as exc:
            self.status.config(text=f"错误: {exc}")

    def _on_center_line_changed(self, selected_line: str) -> None:
        if selected_line == "center" and self.show_center_line.get():
            self.show_fitted_center_line.set(False)
        elif selected_line == "fitted" and self.show_fitted_center_line.get():
            self.show_center_line.set(False)
        self.process()

    def _reset_display_options(self) -> None:
        for variable in (
            self.show_crosshair,
            self.show_contour,
            self.show_center_line,
            self.show_fitted_center_line,
        ):
            variable.set(False)
        self.image_mode.set("ori_img")
        self.show_bent.set(False)
        self.process()

    def _annotations(self) -> frozenset[str]:
        annotations = set()
        if self.show_contour.get():
            annotations.add("contour")
        if self.show_center_line.get():
            annotations.add("center_line")
        if self.show_fitted_center_line.get():
            annotations.add("fitted_center_line")
        return frozenset(annotations)

    def _image_mode(self) -> str:
        return self.image_mode.get()

    def process(self, event=None) -> None:
        try:
            self._save_vars()
        except Exception as exc:
            self.status.config(text=f"错误: {exc}")
            return
        self.request_id += 1
        request_id = self.request_id
        annotations, image_mode, show_bent = self._annotations(), self._image_mode(), self.show_bent.get()

        def done(result) -> None:
            if request_id == self.request_id:
                self._show(result)

        self.live_task.submit(
            lambda: self.actions.run_distribution_display(annotations, image_mode, show_bent),
            done,
            "计算质量分布...",
        )

    def auto_registration(self) -> None:
        try:
            self._save_vars()
        except Exception as exc:
            self.status.config(text=f"错误: {exc}")
            return
        self.request_id += 1
        request_id = self.request_id

        def done(result) -> None:
            if request_id != self.request_id:
                return
            estimates = {"f": result.f, "o": result.o, "fo": result.fo}
            labels = {"f": "F", "o": "O", "fo": "FO"}
            notices = []
            for name, estimate in estimates.items():
                self.vars[f"center_x{name}"].set(str(estimate.center_x))
                self.vars[f"center_y{name}"].set(str(estimate.center_y))
                self.vars[f"rotate_angle_{name}"].set(str(estimate.angle_degrees))
                quality = (
                    "回退初值，需人工检查"
                    if estimate.used_fallback
                    else f"两侧亮脊对称误差={estimate.symmetry_error_pixels:.1f} px"
                )
                notices.append(f"{labels[name]}: {quality}")
            self.pending_status_notice = "；".join(notices)
            self.process()

        self.registration_task.submit(
            self.actions.estimate_distribution_registration,
            done,
            "自动计算 F、O、FO 原点和偏转角...",
        )

    def _show(self, result) -> None:
        self.latest_result = result
        self.summary.set_rows(result.summary_rows)
        output = self._output_folder()
        for item in result.images:
            display = self.displays.get(item.name)
            if display is None:
                continue
            display.update(
                item.image,
                gain=0 if self.auto_gain.get() else self.settings.gain,
                colormap=item.colormap if self.is_pseudo.get() else "none",
                zoom=self.settings.zoom,
                crosshair=self._crosshair_for(item.name) if self.show_crosshair.get() else None,
                rotate=0,
                rotation_center=None,
                save_name_suffix=self.settings.save_name,
                special_points=item.special_points,
                save_folder=output,
            )
        status = "计算完成"
        if result.warnings:
            status += f"；{len(result.warnings)} 项拟合警告，详细原因见上方结果"
        if self.pending_status_notice:
            status += f"；{self.pending_status_notice}"
            self.pending_status_notice = ""
        self.status.config(text=status)

    def save_selected_outputs(self) -> None:
        try:
            self._save_vars()
            if self.latest_result is None:
                raise ValueError("请先完成当前参数计算")
            output = self._output_folder()
            name = self.settings.save_name or f"F{self.settings.f_exp_no}_FO{self.settings.fo_exp_no}_O{self.settings.o_exp_no}"
            output_name = f"{name}-降噪{self.settings.denoise_ratio:g}"
            image_outputs: list[Path] = []
            for image_name in ("f", "o", "fo"):
                if not self.save_targets[image_name].get():
                    continue
                image = self.displays[image_name].copy_for_save(
                    include_crosshair=self.save_crosshair.get(),
                    include_ellipse=False,
                    crosshair=self._crosshair_for(image_name),
                )
                if image is not None:
                    image_outputs.append(write_png(output / image_name / f"{output_name}.png", image))
            fit_output = self.actions.export_distribution_result(self.latest_result) if self.save_targets["fit_data"].get() else None
            saved_count = len(image_outputs) + int(fit_output is not None)
            self.status.config(text=f"已保存 {saved_count} 项结果到: {output}")
        except Exception as exc:
            self.status.config(text=f"保存错误: {exc}")

    def _crosshair_for(self, name: str) -> tuple[int, int, int]:
        center_x, center_y = self._center_for(name)
        return center_x, center_y, self.settings.line_width

    def _center_for(self, name: str) -> tuple[int, int]:
        if name == "f":
            return self.settings.center_xf, self.settings.center_yf
        if name == "o":
            return self.settings.center_xo, self.settings.center_yo
        return self.settings.center_xfo, self.settings.center_yfo

    def _output_folder(self) -> Path:
        if str(self.settings.output_folder) not in {"", "."}:
            return Path(self.settings.output_folder)
        return Path(self.settings.image_folder).parent / "distribution_results"

    def export_ui_state(self) -> dict[str, object]:
        try:
            self._save_vars()
        except (TypeError, ValueError):
            pass
        return {
            "is_pseudo": self.is_pseudo.get(),
            "auto_gain": self.auto_gain.get(),
            "show_crosshair": self.show_crosshair.get(),
            "show_contour": self.show_contour.get(),
            "show_center_line": self.show_center_line.get(),
            "show_fitted_center_line": self.show_fitted_center_line.get(),
            "image_mode": self.image_mode.get(),
            "show_bent": self.show_bent.get(),
            "save_crosshair": self.save_crosshair.get(),
            "save_targets": {name: target.get() for name, target in self.save_targets.items()},
        }

    def restore_ui_state(self, state: dict[str, object]) -> None:
        self._sync_input_vars_from_settings()
        self.is_pseudo.set(bool(state.get("is_pseudo", self.is_pseudo.get())))
        self.auto_gain.set(bool(state.get("auto_gain", self.auto_gain.get())))
        self.show_crosshair.set(bool(state.get("show_crosshair", self.show_crosshair.get())))
        self.show_contour.set(bool(state.get("show_contour", False)))
        self.show_center_line.set(bool(state.get("show_center_line", False)))
        self.show_fitted_center_line.set(bool(state.get("show_fitted_center_line", False)))
        if self.show_fitted_center_line.get():
            self.show_center_line.set(False)
        legacy_image_mode = "delta_img" if state.get("show_delta_image") else "fit_img" if state.get("show_fit_image") else "bent_img" if state.get("show_bent_image") else "ori_img"
        image_mode = str(state.get("image_mode", legacy_image_mode))
        legacy_bent = bool(state.get("show_bent", state.get("show_bent_image", False)))
        if "show_bent" not in state and image_mode in {"fit_img", "delta_img"}:
            legacy_bent = True
        if image_mode == "bent_img":
            image_mode = "ori_img"
            legacy_bent = True
        self.image_mode.set(image_mode if image_mode in {"ori_img", "fit_img", "delta_img"} else "ori_img")
        self.show_bent.set(legacy_bent)
        self.save_crosshair.set(bool(state.get("save_crosshair", self.save_crosshair.get())))
        targets = state.get("save_targets", {})
        if isinstance(targets, dict):
            for name, target in self.save_targets.items():
                target.set(bool(targets.get(name, target.get())))

    def _sync_input_vars_from_settings(self) -> None:
        for name, variable in self.vars.items():
            if name not in {"save_dr", "save_u"}:
                variable.set(self._value_to_str(getattr(self.settings, name)))
        dr_value, u_value = split_save_name(self.settings.save_name)
        self.vars["save_dr"].set(dr_value)
        self.vars["save_u"].set(u_value)

    @staticmethod
    def _value_to_str(value) -> str:
        if isinstance(value, Path):
            return "" if str(value) == "." else str(value)
        return str(value)

class DistributionSummaryPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.cells: dict[tuple[str, str, str], ttk.Label] = {}
        self._build_group(
            "geometry",
            "轮廓与中心线",
            (
                ("contour", "轮廓值"),
                ("center_line", "中心线方程"),
                ("center_a", "a"),
                ("center_b", "b"),
                ("center_r_squared", "R²"),
            ),
        ).pack(fill=tk.X, pady=(0, 3))
        distribution_group = ttk.LabelFrame(self, text="强度分布拟合")
        distribution_group.pack(fill=tk.X)
        fit_sections = ttk.Frame(distribution_group)
        fit_sections.pack(anchor=tk.W)
        self._build_fit_section(
            fit_sections,
            "horizontal",
            "横向拟合",
            (
                ("horizontal_equation", "方程"),
                ("horizontal_a", "a"),
                ("horizontal_b", "b"),
                ("horizontal_c", "c"),
                ("horizontal_r_squared", "R²"),
            ),
        ).pack(side=tk.LEFT, anchor=tk.NW)
        self._build_fit_section(
            fit_sections,
            "vertical",
            "纵向拟合",
            (
                ("vertical_equation", "方程"),
                ("vertical_a", "a"),
                ("vertical_b", "b"),
                ("vertical_c", "c"),
                ("vertical_r_squared", "R²"),
            ),
        ).pack(side=tk.LEFT, anchor=tk.NW, padx=(10, 0))
        self._build_fit_section(
            fit_sections,
            "student",
            "纯椭圆",
            (
                ("student_equation", "方程"),
                ("student_scale_x", "sₓ"),
                ("student_scale_y", "sᵧ"),
                ("student_exponent", "m"),
                ("student_r_squared", "R²"),
            ),
        ).pack(side=tk.LEFT, anchor=tk.NW, padx=(10, 0))
        self._build_fit_section(
            fit_sections,
            "compact",
            "紧支撑",
            (
                ("compact_equation", "方程"),
                ("compact_scale_x", "sₓ"),
                ("compact_scale_y", "sᵧ"),
                ("compact_exponent", "m"),
                ("compact_cutoff_radius", "R"),
                ("compact_r_squared", "R²"),
            ),
        ).pack(side=tk.LEFT, anchor=tk.NW, padx=(10, 0))

    def _build_group(self, group_name: str, title: str, columns: tuple[tuple[str, str], ...]) -> ttk.LabelFrame:
        group = ttk.LabelFrame(self, text=title)
        ttk.Label(group, text="图像", width=5, anchor=tk.W).grid(row=0, column=0, padx=4, sticky="w")
        for column_index, (_field, heading) in enumerate(columns, start=1):
            ttk.Label(group, text=heading, anchor=tk.W).grid(
                row=0,
                column=column_index,
                padx=4,
                sticky="w",
            )

        for row_index, image_name in enumerate(("F", "O", "FO"), start=1):
            ttk.Label(group, text=image_name, width=5, anchor=tk.W).grid(
                row=row_index,
                column=0,
                padx=4,
                pady=1,
                sticky="w",
            )
            for column_index, (field, _heading) in enumerate(columns, start=1):
                numeric = field in {"contour", "center_a", "center_b", "center_r_squared"}
                cell = ttk.Label(group, text="-", anchor=tk.E if numeric else tk.W)
                cell.grid(
                    row=row_index,
                    column=column_index,
                    padx=4,
                    pady=1,
                    sticky="e" if numeric else "w",
                )
                self.cells[(group_name, image_name, field)] = cell
        return group

    def _build_fit_section(
        self,
        parent,
        group_name: str,
        title: str,
        columns: tuple[tuple[str, str], ...],
    ) -> ttk.Frame:
        section = ttk.Frame(parent)
        ttk.Label(section, text=title, width=9, anchor=tk.W).grid(
            row=0,
            column=0,
            rowspan=4,
            padx=(4, 8),
            sticky="nw",
        )
        ttk.Label(section, text="图像", width=5, anchor=tk.W).grid(row=0, column=1, padx=4, sticky="w")
        for column_index, (_field, heading) in enumerate(columns, start=2):
            ttk.Label(section, text=heading, anchor=tk.W).grid(
                row=0,
                column=column_index,
                padx=4,
                sticky="w",
            )

        for row_index, image_name in enumerate(("F", "O", "FO"), start=1):
            ttk.Label(section, text=image_name, width=5, anchor=tk.W).grid(
                row=row_index,
                column=1,
                padx=4,
                pady=1,
                sticky="w",
            )
            for column_index, (field, _heading) in enumerate(columns, start=2):
                numeric = not field.endswith("_equation")
                cell = ttk.Label(section, text="-", anchor=tk.E if numeric else tk.W)
                cell.grid(
                    row=row_index,
                    column=column_index,
                    padx=4,
                    pady=1,
                    sticky="e" if numeric else "w",
                )
                self.cells[(group_name, image_name, field)] = cell
        return section

    def set_rows(self, rows) -> None:
        for row in rows:
            for group_name, fields in (
                ("geometry", ("contour", "center_line", "center_a", "center_b", "center_r_squared")),
                (
                    "horizontal",
                    (
                        "horizontal_equation",
                        "horizontal_a",
                        "horizontal_b",
                        "horizontal_c",
                        "horizontal_r_squared",
                    ),
                ),
                (
                    "vertical",
                    (
                        "vertical_equation",
                        "vertical_a",
                        "vertical_b",
                        "vertical_c",
                        "vertical_r_squared",
                    ),
                ),
                (
                    "student",
                    (
                        "student_equation",
                        "student_scale_x",
                        "student_scale_y",
                        "student_exponent",
                        "student_r_squared",
                    ),
                ),
                (
                    "compact",
                    (
                        "compact_equation",
                        "compact_scale_x",
                        "compact_scale_y",
                        "compact_exponent",
                        "compact_cutoff_radius",
                        "compact_r_squared",
                    ),
                ),
            ):
                for field in fields:
                    cell = self.cells.get((group_name, row.image_name, field))
                    if cell is not None:
                        cell.config(text=getattr(row, field))


class LabelLog:
    def __init__(self, label):
        self.label = label

    def write(self, message: str) -> None:
        self.label.config(text=message)
