from __future__ import annotations

import re
import tkinter as tk
from pathlib import Path
from PLIFMix.ui import localized_dialogs as filedialog, localized_widgets as ttk

import numpy as np

from PLIFMix.core.image_io import read_image, write_tif
from PLIFMix.ui.components.viewer_panel import ViewerPanel

class ImageViewerPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.vars = {name: tk.StringVar(value=value) for name, value in {
            "image_path": "", "folder_path": "", "experiment_number": "",
            "gain": "1", "zoom": "2", "rotate": "0", "shift": "0", "denoise": "0",
            "center_x": "0", "center_y": "0", "major": "100", "minor": "100",
            "line_width": "2",
            "quad_x1": "", "quad_y1": "", "quad_x2": "", "quad_y2": "",
            "quad_x3": "", "quad_y3": "", "quad_x4": "", "quad_y4": "",
        }.items()}
        self.input_mode = tk.StringVar(value="image")
        self.batch_range = tk.StringVar(value="0800-0850")
        self.pseudo = tk.BooleanVar(value=False)
        self.auto_gain = tk.BooleanVar(value=False)
        self.auto_gain_text = tk.StringVar(value="（自动增益：-）")
        self.crosshair = tk.BooleanVar(value=False)
        self.ellipse = tk.BooleanVar(value=False)
        self.quad_visible = tk.BooleanVar(value=False)
        self.panel: ViewerPanel | None = None
        self._setup_controls()
        self.display_host = ttk.Frame(self)
        self.display_host.pack(fill="both", expand=True, padx=6, pady=4)

    def _setup_controls(self) -> None:
        row1 = ttk.Frame(self)

        row2 = ttk.Frame(self)
        row2.pack(anchor="w", padx=6, pady=2)
        ttk.Checkbutton(row2, text="伪彩图", variable=self.pseudo, command=self.update_current).pack(side="left", padx=3)
        self._entry(row2, "gain", "增益", 6)
        ttk.Checkbutton(row2, variable=self.auto_gain, command=self.update_current).pack(side="left", padx=(5, 1))
        ttk.Label(row2, textvariable=self.auto_gain_text).pack(side="left", padx=(0, 3))
        self._entry(row2, "zoom", "缩小", 6)
        self._entry(row2, "denoise", "相对去噪阈值", 7)
        for name, label, width in [("rotate", "旋转", 6), ("shift", "左移", 6)]:
            self._entry(row2, name, label, width)

        row3 = ttk.Frame(self)
        row3.pack(anchor="w", padx=6, pady=2)
        self._entry(row3, "line_width", "线宽", 6)
        ttk.Checkbutton(row3, text="十字线", variable=self.crosshair, command=self.update_current).pack(side="left", padx=3)
        self._entry(row3, "center_x", "X", 6)
        self._entry(row3, "center_y", "Y", 6)
        ttk.Checkbutton(row3, text="椭圆", variable=self.ellipse, command=self.update_current).pack(side="left", padx=3)
        self._entry(row3, "major", "长轴", 6)
        self._entry(row3, "minor", "短轴", 6)

        row4 = ttk.Frame(self)
        row4.pack(anchor="w", padx=6, pady=2)
        ttk.Checkbutton(row4, text="显示四边形", variable=self.quad_visible, command=self.update_current).pack(side="left", padx=3)
        for index in range(1, 5):
            self._entry(row4, f"quad_x{index}", f"X{index}", 6)
            self._entry(row4, f"quad_y{index}", f"Y{index}", 6)

        row5 = ttk.Frame(self)
        row5.pack(anchor="w", padx=6, pady=(2, 3))
        ttk.Label(row5, text="编号范围").pack(side="left", padx=(4, 1))
        ttk.Entry(row5, textvariable=self.batch_range, width=12).pack(side="left")
        ttk.Button(row5, text="批量导出TIF", command=self.batch_export).pack(side="left", padx=(6, 6))
        ttk.Button(row5, text="保存当前显示", command=self.save_current).pack(side="left")

        path_area = ttk.Frame(self)
        path_area.pack(fill="x", padx=6, pady=(5, 2), before=row2)
        ttk.Radiobutton(
            path_area,
            text="图片",
            variable=self.input_mode,
            value="image",
            command=lambda: self._load_path("image"),
        ).grid(
            row=0, column=0, sticky="w", pady=3
        )
        image_entry = ttk.Entry(path_area, textvariable=self.vars["image_path"], width=62)
        image_entry.grid(row=0, column=1, sticky="w", padx=(4, 0), pady=3)
        ttk.Button(path_area, text="...", width=3, command=self._choose_image_path).grid(
            row=0, column=2, sticky="w", padx=(3, 0), pady=3
        )
        image_entry.bind("<Return>", lambda _event: self._load_path("image"))

        ttk.Radiobutton(
            path_area,
            text="文件夹",
            variable=self.input_mode,
            value="folder",
            command=lambda: self._load_path("folder"),
        ).grid(
            row=1, column=0, sticky="w", pady=3
        )
        folder_entry = ttk.Entry(path_area, textvariable=self.vars["folder_path"], width=62)
        folder_entry.grid(row=1, column=1, sticky="w", padx=(4, 0), pady=3)
        ttk.Button(path_area, text="...", width=3, command=self._choose_folder_path).grid(
            row=1, column=2, sticky="w", padx=(3, 0), pady=3
        )
        ttk.Label(path_area, text="实验编号").grid(row=1, column=3, sticky="w", padx=(8, 2), pady=3)
        experiment_entry = ttk.Entry(path_area, textvariable=self.vars["experiment_number"], width=8)
        experiment_entry.grid(row=1, column=4, sticky="w", pady=3)
        ttk.Label(path_area, text=".npy").grid(row=1, column=5, sticky="w", padx=(2, 0), pady=3)
        ttk.Button(path_area, text="上一张", command=lambda: self.move_image(-1)).grid(
            row=1, column=6, sticky="w", padx=(8, 0), pady=3
        )
        ttk.Button(path_area, text="下一张", command=lambda: self.move_image(1)).grid(
            row=1, column=7, sticky="w", padx=(3, 0), pady=3
        )
        folder_entry.bind("<Return>", lambda _event: self._load_path("folder"))
        experiment_entry.bind("<Return>", lambda _event: self._load_path("folder"))

        self.load_status = ttk.Label(self, text="")
        self.load_status.pack(anchor="w", padx=9, before=row2)

    def _entry(self, row, name: str, label: str, width: int) -> None:
        ttk.Label(row, text=label).pack(side="left", padx=(4, 1))
        entry = ttk.Entry(row, textvariable=self.vars[name], width=width)
        entry.pack(side="left")
        entry.bind("<Return>", lambda event: self.update_current())

    def _choose_image_path(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("图像", "*.png *.tif *.tiff *.bmp *.npy")])
        if path:
            self.input_mode.set("image")
            self.vars["image_path"].set(path)
            self._load_path("image")

    def _choose_folder_path(self) -> None:
        folder = filedialog.askdirectory()
        if folder:
            self.input_mode.set("folder")
            self.vars["folder_path"].set(folder)
            self._load_path("folder")

    def _load_path(self, mode: str | None = None) -> None:
        mode = mode or self.input_mode.get()
        self.input_mode.set(mode)
        if mode == "image":
            path = Path(self.vars["image_path"].get().strip())
        else:
            path = _experiment_npy_path(
                self.vars["folder_path"].get(),
                self.vars["experiment_number"].get(),
            )
        if not str(path) or str(path) == "." or not path.is_file():
            self.load_status.config(text=f"文件不存在: {path}")
            return
        self.load_status.config(text="")
        self._show_path(path)

    def _show_path(self, path: Path) -> None:
        if self.panel is None:
            self.panel = ViewerPanel(self.display_host, path)
            self.panel.pack(fill="both", expand=True)
        else:
            self.panel.set_path(path)
        self.update_current()

    def current_panel(self):
        return self.panel

    def update_current(self) -> None:
        panel = self.current_panel()
        if panel is None:
            return
        try:
            computed_gain = panel.update_display(
                zoom=float(self.vars["zoom"].get()),
                **self._display_parameters(),
            )
            self.auto_gain_text.set(
                f"（自动增益：{computed_gain:.2f}）" if computed_gain is not None else "（自动增益：-）"
            )
        except Exception as exc:
            panel.info.config(text=f"错误: {exc}")

    def _display_parameters(self) -> dict:
        auto_gain = self.auto_gain.get()
        return {
            "gain": 1.0 if auto_gain else float(self.vars["gain"].get()),
            "auto_gain": auto_gain,
            "rotate": float(self.vars["rotate"].get()),
            "shift": int(self.vars["shift"].get()),
            "denoise": float(self.vars["denoise"].get()),
            "pseudo": self.pseudo.get(),
            "center": (int(self.vars["center_x"].get()), int(self.vars["center_y"].get())),
            "ellipse": (int(self.vars["major"].get()), int(self.vars["minor"].get())) if self.ellipse.get() else None,
            "crosshair": self.crosshair.get(),
            "line_width": int(self.vars["line_width"].get()),
            "quad": _parse_quad_entries(self.vars) if self.quad_visible.get() else None,
        }

    def batch_export(self) -> None:
        panel = self.current_panel()
        if panel is None or panel.path == Path() or not panel.path.parent.is_dir():
            self.load_status.config(text="请先打开一张图像")
            return
        try:
            start, end = _parse_image_range(self.batch_range.get())
            output_text = filedialog.askdirectory(
                initialdir=str(panel.path.parent),
                title="选择批量导出 TIF 目录",
            )
            if not output_text:
                return
            source_paths = _numbered_image_paths(panel.path.parent, panel.path.suffix)
            parameters = self._display_parameters()
            output_folder = Path(output_text)
            total = end - start + 1
            exported = 0
            missing: list[int] = []
            failed: list[tuple[int, str]] = []
            for index, number in enumerate(range(start, end + 1), start=1):
                self.load_status.config(text=f"正在导出 {index}/{total}: {number:04d}")
                self.update_idletasks()
                source_path = source_paths.get(number)
                if source_path is None:
                    missing.append(number)
                    continue
                try:
                    display = panel.render_array_for_export(read_image(source_path), **parameters)
                    write_tif(output_folder / f"{number:04d}.tif", display)
                    exported += 1
                except Exception as exc:
                    failed.append((number, str(exc)))

            status = f"批量导出完成：{exported}/{total} 张；输出目录：{output_folder}"
            if missing:
                status += f"；缺少 {len(missing)} 张（例如 {', '.join(f'{number:04d}' for number in missing[:5])}）"
            if failed:
                status += f"；失败 {len(failed)} 张（例如 {failed[0][0]:04d}: {failed[0][1]}）"
            self.load_status.config(text=status)
        except Exception as exc:
            self.load_status.config(text=f"批量导出失败: {exc}")

    def move_image(self, delta: int) -> None:
        panel = self.current_panel()
        if panel is None:
            return
        target = _adjacent_image_path(panel.path, delta)
        if target is None:
            panel.info.config(text=f"未找到编号相邻的{'上一张' if delta < 0 else '下一张'}图片")
            return
        if self.input_mode.get() == "folder":
            match = re.fullmatch(r"实验编号\s*(\d+)", target.stem)
            if match:
                self.vars["folder_path"].set(str(target.parent))
                self.vars["experiment_number"].set(match.group(1))
        else:
            self.vars["image_path"].set(str(target))
        self.load_status.config(text="")
        self._show_path(target)

    def save_current(self) -> None:
        panel = self.current_panel()
        if panel is None or panel.display is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIF", "*.tif")])
        if path:
            write_tif(path, panel.display)

def _parse_quad_entries(variables: dict[str, tk.StringVar]) -> np.ndarray:
    values = []
    for index in range(1, 5):
        x = variables[f"quad_x{index}"].get().strip()
        y = variables[f"quad_y{index}"].get().strip()
        if not x or not y:
            raise ValueError("请完整填写四边形的8个坐标")
        values.append((int(x), int(y)))
    return np.array(values, dtype=np.int32)

def _experiment_npy_path(folder_text: str, number_text: str) -> Path:
    number = number_text.strip()
    if number.startswith("实验编号"):
        number = number.removeprefix("实验编号").strip()
    return Path(folder_text.strip()) / f"实验编号{number}.npy"

def _adjacent_image_path(path: Path, delta: int) -> Path | None:
    if path.stem.isdigit():
        number = int(path.stem) + delta
        if number < 0:
            return None
        target_stem = f"{number:04d}"
    else:
        match = re.fullmatch(r"实验编号\s*(\d+)", path.stem)
        if match is None:
            return None
        number = int(match.group(1)) + delta
        if number < 0:
            return None
        target_stem = f"实验编号{number}"

    for candidate in path.parent.iterdir():
        if candidate.is_file() and candidate.stem == target_stem and candidate.suffix.lower() == path.suffix.lower():
            return candidate
    return None


def _parse_image_range(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*(?:-|:|到)\s*(\d+)\s*", value)
    if match is None:
        raise ValueError("编号范围格式应为 0800-0850 或 0800到0850")
    start, end = int(match.group(1)), int(match.group(2))
    if start > end:
        raise ValueError("编号范围的起始值不能大于结束值")
    return start, end


def _numbered_image_paths(folder: Path, preferred_suffix: str) -> dict[int, Path]:
    suffixes = {".png", ".tif", ".tiff", ".bmp", ".npy"}
    candidates = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in suffixes and path.stem.isdigit()
        ),
        key=lambda path: (path.suffix.lower() != preferred_suffix.lower(), path.name.lower()),
    )
    return {int(path.stem): path for path in reversed(candidates)}
