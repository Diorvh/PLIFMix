from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from PLIFv2.application.actions import PLIFActions
from PLIFv2.core.image_io import write_png
from PLIFv2.core.mixing_presets import (
    MixingPreset,
    create_mixing_presets_from_excel,
    default_mixing_preset_path,
    load_mixing_presets,
    save_mixing_presets,
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
        self.use_calibrated_display = tk.BooleanVar(value=True)
        self.save_crosshair = tk.BooleanVar(value=True)
        self.save_targets = {name: tk.BooleanVar(value=True) for name, _position in DISTRIBUTION_SAVE_LAYOUT}
        self.show_q_center = "nonshow"
        self.show_img = "ori_img"
        self.preset_groups: tuple[MixingPreset, ...] = ()
        self.preset_index: int | None = None
        self.preset_group_var = tk.StringVar()
        self.displays: dict[str, ImageDisplay] = {}
        self.latest_result = None
        self.request_id = 0
        self._setup_ui()
        self.live_task = LatestTaskRunner(self, LabelLog(self.status))

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
        self._row_fit_options(controls).pack(anchor=tk.W, pady=1)

        self.summary = ttk.Label(self, text="质量分布拟合结果", wraplength=1500, justify=tk.LEFT)
        self.summary.pack(anchor=tk.W, padx=8)
        self.status = ttk.Label(self, text="")
        self.status.pack(anchor=tk.W, padx=8)

        displays = ttk.Frame(self)
        displays.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        for column, name in enumerate(("f", "o", "fo")):
            display = ImageDisplay(displays, name)
            display.grid(row=0, column=column, sticky="nw", padx=4)
            self.displays[name] = display

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
        row = ttk.Frame(parent)
        self._path_entry(row, "image_folder", "图像文件夹", self._choose_folder, width=68)
        self._entry(row, "f_exp_no", "F 编号", 5)
        self._entry(row, "fo_exp_no", "FO 编号", 5)
        self._entry(row, "o_exp_no", "O 编号", 5)
        return row

    def _row_preset(self, parent) -> ttk.Frame:
        row = ttk.Frame(parent)
        ttk.Label(row, text="预设数据组").pack(side=tk.LEFT, padx=(4, 2))
        configured_path = self._value_to_str(self.settings.preset_path)
        default_path = default_mixing_preset_path()
        initial_path = configured_path or (str(default_path) if default_path.exists() else "")
        var = tk.StringVar(value=initial_path)
        self.vars["preset_path"] = var
        entry = ttk.Entry(row, textvariable=var, width=68)
        entry.pack(side=tk.LEFT)
        entry.bind("<Return>", self._load_preset_file)
        ttk.Button(row, text="...", width=3, command=self._select_preset_path).pack(side=tk.LEFT, padx=(3, 2))
        ttk.Button(row, text="Excel 转预设", command=self._create_preset_from_excel).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(row, text="加载", command=self._load_preset_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row, text="上一组", command=lambda: self._move_preset(-1)).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(row, text="下一组", command=lambda: self._move_preset(1)).pack(side=tk.LEFT)
        group_selector = ttk.Frame(row)
        group_selector.pack(side=tk.LEFT, padx=(8, 0))
        self.preset_group_entry = ttk.Entry(group_selector, textvariable=self.preset_group_var, width=4, justify=tk.RIGHT)
        self.preset_group_entry.pack(side=tk.LEFT)
        self.preset_group_entry.bind("<Return>", self._go_to_preset)
        self.preset_total_label = ttk.Label(group_selector, text="/-")
        self.preset_total_label.pack(side=tk.LEFT)

        dr_value, u_value = self._split_save_name(self.settings.save_name)
        ttk.Label(row, text="图像保存名称").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Label(row, text="DR:").pack(side=tk.LEFT, padx=(4, 2))
        self._text_entry(row, "save_dr", dr_value, width=8)
        ttk.Label(row, text="U").pack(side=tk.LEFT, padx=(4, 2))
        self._text_entry(row, "save_u", u_value, width=8)
        return row

    def _row_display(self, parent) -> ttk.Frame:
        row = ttk.Frame(parent)
        ttk.Checkbutton(row, text="伪彩图", variable=self.is_pseudo, command=self._redraw_cached_result).pack(side=tk.LEFT, padx=(3, 5))
        self._entry(row, "gain", "增益", 6)
        ttk.Checkbutton(row, text="自动", variable=self.auto_gain, command=self._redraw_cached_result).pack(side=tk.LEFT, padx=(4, 5))
        self._entry(row, "zoom", "缩小倍数", 6)
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
        self._entry(row, "line_width", "线宽", 6)
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
        return row

    def _row_fit_options(self, parent) -> ttk.Frame:
        row = ttk.Frame(parent)
        self._entry(row, "thres_contour", "轮廓比例", 6)
        self._entry(row, "q_center_up", "中心线上", 5)
        self._entry(row, "q_center_down", "中心线下", 5)
        self._entry(row, "max_k", "最大值个数", 5)
        self._entry(row, "hori_fit_width", "横向拟合半宽", 7)
        self._entry(row, "rotate_angle_f", "旋转 F", 5)
        self._entry(row, "rotate_angle_o", "O", 5)
        self._entry(row, "rotate_angle_fo", "FO", 5)
        for label, mode in [("轮廓", "contour"), ("中心线", "Q_center"), ("拟合中心线", "Q_center_fitted"), ("不显示", "nonshow")]:
            ttk.Button(row, text=label, command=lambda value=mode: self._set_q(value)).pack(side=tk.LEFT, padx=2)
        for label, mode in [("掰直", "bent_img"), ("拟合", "fit_img"), ("拟合-掰直", "delta_img"), ("原图", "ori_img")]:
            ttk.Button(row, text=label, command=lambda value=mode: self._set_img(value)).pack(side=tk.LEFT, padx=2)
        return row

    def _path_entry(self, row, name: str, label: str, command, width: int) -> None:
        ttk.Label(row, text=label).pack(side=tk.LEFT, padx=(4, 2))
        var = tk.StringVar(value=self._value_to_str(getattr(self.settings, name)))
        self.vars[name] = var
        entry = ttk.Entry(row, textvariable=var, width=width)
        entry.pack(side=tk.LEFT)
        ttk.Button(row, text="...", width=3, command=command).pack(side=tk.LEFT, padx=(3, 4))
        entry.bind("<Return>", self.process)

    def _entry(self, row, name: str, label: str, width: int) -> None:
        ttk.Label(row, text=label).pack(side=tk.LEFT, padx=(4, 2))
        self._text_entry(row, name, self._value_to_str(getattr(self.settings, name)), width)

    def _text_entry(self, row, name: str, value: str, width: int) -> None:
        var = tk.StringVar(value=value)
        self.vars[name] = var
        entry = ttk.Entry(row, textvariable=var, width=width)
        entry.pack(side=tk.LEFT)
        entry.bind("<Return>", self.process)

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
        self.vars["global_of"].set(str(preset.global_of))
        self.preset_group_var.set(str(index + 1))
        self.preset_total_label.config(text=f"/{len(self.preset_groups)}")
        self.process()

    def _fill_centers_from_f(self) -> None:
        x_value = self.vars["center_xf"].get()
        y_value = self.vars["center_yf"].get()
        self.vars["center_xo"].set(x_value)
        self.vars["center_yo"].set(y_value)
        self.vars["center_xfo"].set(x_value)
        self.vars["center_yfo"].set(y_value)
        self.process()

    def _redraw_cached_result(self) -> None:
        if self.latest_result is not None:
            self._show(self.latest_result)

    def process(self, event=None) -> None:
        try:
            self._save_vars()
        except Exception as exc:
            self.status.config(text=f"错误: {exc}")
            return
        self.request_id += 1
        request_id = self.request_id
        q_mode, image_mode = self.show_q_center, self.show_img

        def done(result) -> None:
            if request_id == self.request_id:
                self._show(result)

        self.live_task.submit(lambda: self.actions.run_distribution_display(q_mode, image_mode), done, "计算质量分布...")

    def _show(self, result) -> None:
        self.latest_result = result
        self.summary.config(text=result.fit_text + "\n" + result.surface_text)
        images = result.images if self.use_calibrated_display.get() else result.raw_images
        output = self._output_folder()
        for item in images:
            display = self.displays.get(item.name)
            if display is None:
                continue
            display.update(
                item.image,
                gain=0 if self.auto_gain.get() else self.settings.gain,
                colormap=item.colormap if self.is_pseudo.get() else "none",
                zoom=self.settings.zoom,
                crosshair=self._crosshair_for(item.name) if self.show_crosshair.get() else None,
                save_name_suffix=self.settings.save_name,
                special_points=item.special_points,
                save_folder=output,
            )
        self.status.config(text="计算完成")

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
        if name == "f":
            return self.settings.center_xf, self.settings.center_yf, self.settings.line_width
        if name == "o":
            return self.settings.center_xo, self.settings.center_yo, self.settings.line_width
        return self.settings.center_xfo, self.settings.center_yfo, self.settings.line_width

    def _output_folder(self) -> Path:
        if str(self.settings.output_folder) not in {"", "."}:
            return Path(self.settings.output_folder)
        return Path(self.settings.image_folder).parent / "distribution_results"

    def _set_q(self, mode: str) -> None:
        self.show_q_center = mode
        self.process()

    def _set_img(self, mode: str) -> None:
        self.show_img = mode
        self.process()

    def export_ui_state(self) -> dict[str, object]:
        try:
            self._save_vars()
        except (TypeError, ValueError):
            pass
        return {
            "is_pseudo": self.is_pseudo.get(),
            "auto_gain": self.auto_gain.get(),
            "show_crosshair": self.show_crosshair.get(),
            "use_calibrated_display": self.use_calibrated_display.get(),
            "show_q_center": self.show_q_center,
            "show_img": self.show_img,
            "save_crosshair": self.save_crosshair.get(),
            "save_targets": {name: target.get() for name, target in self.save_targets.items()},
        }

    def restore_ui_state(self, state: dict[str, object]) -> None:
        self._sync_input_vars_from_settings()
        self.is_pseudo.set(bool(state.get("is_pseudo", self.is_pseudo.get())))
        self.auto_gain.set(bool(state.get("auto_gain", self.auto_gain.get())))
        self.show_crosshair.set(bool(state.get("show_crosshair", self.show_crosshair.get())))
        self.use_calibrated_display.set(bool(state.get("use_calibrated_display", self.use_calibrated_display.get())))
        self.show_q_center = str(state.get("show_q_center", self.show_q_center))
        self.show_img = str(state.get("show_img", self.show_img))
        self.save_crosshair.set(bool(state.get("save_crosshair", self.save_crosshair.get())))
        targets = state.get("save_targets", {})
        if isinstance(targets, dict):
            for name, target in self.save_targets.items():
                target.set(bool(targets.get(name, target.get())))

    def _sync_input_vars_from_settings(self) -> None:
        for name, variable in self.vars.items():
            if name not in {"save_dr", "save_u"}:
                variable.set(self._value_to_str(getattr(self.settings, name)))
        dr_value, u_value = self._split_save_name(self.settings.save_name)
        self.vars["save_dr"].set(dr_value)
        self.vars["save_u"].set(u_value)

    @staticmethod
    def _value_to_str(value) -> str:
        if isinstance(value, Path):
            return "" if str(value) == "." else str(value)
        return str(value)

    @staticmethod
    def _split_save_name(save_name: str) -> tuple[str, str]:
        name = save_name.strip()
        if name.startswith("DR") and "_U" in name:
            dr_value, u_value = name[2:].split("_U", maxsplit=1)
            return dr_value, u_value
        return name, ""


class LabelLog:
    def __init__(self, label):
        self.label = label

    def write(self, message: str) -> None:
        self.label.config(text=message)
