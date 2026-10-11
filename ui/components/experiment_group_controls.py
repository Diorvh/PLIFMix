from __future__ import annotations

import tkinter as tk
from pathlib import Path
from PLIFMix.ui import localized_widgets as ttk

from PLIFMix.core.mixing_presets import default_mixing_preset_path


class ExperimentTripletControls(ttk.Frame):
    """Common source row for workflows that load one F/O/FO experiment group."""

    def __init__(
        self,
        master,
        settings,
        variables: dict[str, tk.StringVar],
        choose_folder,
        on_change,
    ):
        super().__init__(master)
        _path_entry(
            self,
            variables,
            "image_folder",
            "图像文件夹",
            _value_to_str(settings.image_folder),
            68,
            choose_folder,
            on_change,
        )
        for name, label in (("f_exp_no", "F 编号"), ("fo_exp_no", "FO 编号"), ("o_exp_no", "O 编号")):
            _entry(self, variables, name, label, _value_to_str(getattr(settings, name)), 5, on_change)


class MixingPresetControls(ttk.Frame):
    """Common preset navigation row used by mixing and distribution pages."""

    def __init__(
        self,
        master,
        settings,
        variables: dict[str, tk.StringVar],
        group_variable: tk.StringVar,
        select_preset,
        create_preset,
        load_preset,
        move_preset,
        go_to_preset,
        on_change,
    ):
        super().__init__(master)
        ttk.Label(self, text="预设数据组").pack(side=tk.LEFT, padx=(4, 2))
        configured_path = _value_to_str(settings.preset_path)
        default_path = default_mixing_preset_path()
        initial_path = configured_path or (str(default_path) if default_path.exists() else "")
        preset_path = tk.StringVar(value=initial_path)
        variables["preset_path"] = preset_path
        path_entry = ttk.Entry(self, textvariable=preset_path, width=68)
        path_entry.pack(side=tk.LEFT)
        path_entry.bind("<Return>", load_preset)
        ttk.Button(self, text="...", width=3, command=select_preset).pack(side=tk.LEFT, padx=(3, 2))
        ttk.Button(self, text="Excel 转预设", command=create_preset).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(self, text="加载", command=load_preset).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self, text="上一组", command=lambda: move_preset(-1)).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(self, text="下一组", command=lambda: move_preset(1)).pack(side=tk.LEFT)

        group_selector = ttk.Frame(self)
        group_selector.pack(side=tk.LEFT, padx=(8, 0))
        self.group_entry = ttk.Entry(group_selector, textvariable=group_variable, width=4, justify=tk.RIGHT)
        self.group_entry.pack(side=tk.LEFT)
        self.group_entry.bind("<Return>", go_to_preset)
        self.total_label = ttk.Label(group_selector, text="/-")
        self.total_label.pack(side=tk.LEFT)

        dr_value, u_value = split_save_name(settings.save_name)
        ttk.Label(self, text="图像保存名称").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Label(self, text="DR:").pack(side=tk.LEFT, padx=(4, 2))
        _plain_entry(self, variables, "save_dr", dr_value, 8, on_change)
        ttk.Label(self, text="U").pack(side=tk.LEFT, padx=(4, 2))
        _plain_entry(self, variables, "save_u", u_value, 8, on_change)


def split_save_name(save_name: str) -> tuple[str, str]:
    name = save_name.strip()
    if name.startswith("DR") and "_U" in name:
        dr_value, u_value = name[2:].split("_U", maxsplit=1)
        return dr_value, u_value
    return name, ""


def _path_entry(
    row,
    variables,
    name,
    label,
    value,
    width,
    choose_path,
    on_change,
) -> None:
    ttk.Label(row, text=label).pack(side=tk.LEFT, padx=(4, 2))
    variable = tk.StringVar(value=value)
    variables[name] = variable
    entry = ttk.Entry(row, textvariable=variable, width=width)
    entry.pack(side=tk.LEFT)
    ttk.Button(row, text="...", width=3, command=choose_path).pack(side=tk.LEFT, padx=(3, 4))
    entry.bind("<Return>", on_change)


def _entry(row, variables, name, label, value, width, on_change) -> None:
    ttk.Label(row, text=label).pack(side=tk.LEFT, padx=(4, 2))
    _plain_entry(row, variables, name, value, width, on_change)


def _plain_entry(row, variables, name, value, width, on_change) -> None:
    variable = tk.StringVar(value=value)
    variables[name] = variable
    entry = ttk.Entry(row, textvariable=variable, width=width)
    entry.pack(side=tk.LEFT)
    entry.bind("<Return>", on_change)


def _value_to_str(value) -> str:
    if isinstance(value, Path):
        return "" if str(value) == "." else str(value)
    return str(value)
