from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

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
from PLIFv2.ui.widgets import LatestTaskRunner


DISTRIBUTION_SAVE_LAYOUT = (
    ("f", (0, 0)),
    ("o", (0, 1)),
    ("fo", (0, 2)),
    ("fit_data", (1, 0)),
)


class DistributionPage(ttk.Frame):
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

        self.summary = ttk.Label(self, text="质量分布拟合结果", wraplength=1500, justify=tk.LEFT)
        self.summary.pack(anchor=tk.W, padx=8)
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
        row = ttk.Frame(parent)
        ttk.Checkbutton(row, text="十字线", variable=self.show_crosshair, command=self._redraw_cached_result).pack(side=tk.LEFT, padx=(3, 5))
        ttk.Checkbutton(row, text="轮廓", variable=self.show_contour, command=self.process).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Checkbutton(
            row,
            text="中心线",
            variable=self.show_center_line,
            command=lambda: self._on_center_line_changed("center"),
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Checkbutton(
            row,
            text="拟合中心线",
            variable=self.show_fitted_center_line,
            command=lambda: self._on_center_line_changed("fitted"),
        ).pack(side=tk.LEFT, padx=(0, 5))
        for label, mode in (("原图", "ori_img"), ("掰直", "bent_img"), ("拟合", "fit_img"), ("拟合-掰直绝对值", "delta_img")):
            ttk.Radiobutton(row, text=label, variable=self.image_mode, value=mode, command=self.process).pack(
                side=tk.LEFT,
                padx=(0, 5),
            )
        ttk.Button(row, text="还原", command=self._reset_display_options).pack(side=tk.LEFT, padx=(4, 0))
        return row

    def _row_fit_parameters(self, parent) -> ttk.Frame:
        row = ttk.Frame(parent)
        self._display_entry(row, "line_width", "线宽", 6)
        self._entry(row, "thres_contour", "轮廓比例", 6)
        self._entry(row, "max_k", "最大值个数", 5)
        self._entry(row, "hori_fit_width", "横向拟合半宽", 7)
        ttk.Button(row, text="自动原点和偏转角", command=self.auto_registration).pack(side=tk.LEFT, padx=(8, 0))
        return row

    def _image_parameter_panel(self, parent, name: str, label: str) -> ttk.Frame:
        panel = ttk.Frame(parent)
        title_row = ttk.Frame(panel)
        title_row.pack(anchor=tk.W)
        ttk.Label(title_row, text=label, font=("Microsoft YaHei UI", 10, "bold")).pack(side=tk.LEFT, padx=(4, 8))
        self._entry(title_row, f"center_x{name}", "十字线 X", 5)
        self._entry(title_row, f"center_y{name}", "Y", 5)

        fit_row = ttk.Frame(panel)
        fit_row.pack(anchor=tk.W)
        self._entry(fit_row, f"q_center_up_{name}", "中心线上界", 5)
        self._entry(fit_row, f"q_center_down_{name}", "下界", 5)
        self._entry(fit_row, f"rotate_angle_{name}", "旋转角度", 5)

        display = ImageDisplay(panel, name)
        display.pack(anchor=tk.W)
        self.displays[name] = display
        return panel

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
            current = getattr(self.settings, name)
            raw = variable.get()
            if isinstance(current, Path):
                value = Path(raw) if raw else Path()
            elif isinstance(current, int):
                value = int(raw)
            elif isinstance(current, float):
                value = float(raw)
            else:
                value = raw
            setattr(self.settings, name, value)
        dr_value = self.vars["save_dr"].get().strip()
        u_value = self.vars["save_u"].get().strip()
        self.settings.save_name = f"DR{dr_value}_U{u_value}" if dr_value or u_value else ""

    def _choose_folder(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.vars["image_folder"].set(path)

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
        annotations, image_mode = self._annotations(), self._image_mode()

        def done(result) -> None:
            if request_id == self.request_id:
                self._show(result)

        self.live_task.submit(
            lambda: self.actions.run_distribution_display(annotations, image_mode),
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
                self.vars[f"rotate_angle_{name}"].set(f"{estimate.angle_degrees:.2f}")
                quality = "回退初值，需人工检查" if estimate.used_fallback else f"拟合 R²={estimate.fit_r_squared:.3f}"
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
        self.summary.config(text=result.fit_text + "\n" + result.surface_text)
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
                rotate=self._rotation_for(item.name),
                rotation_center=self._center_for(item.name),
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

    def _rotation_for(self, name: str) -> float:
        if name == "f":
            return self.settings.rotate_angle_f
        if name == "o":
            return self.settings.rotate_angle_o
        return self.settings.rotate_angle_fo

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
        self.image_mode.set(image_mode if image_mode in {"ori_img", "bent_img", "fit_img", "delta_img"} else "ori_img")
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

class LabelLog:
    def __init__(self, label):
        self.label = label

    def write(self, message: str) -> None:
        self.label.config(text=message)
