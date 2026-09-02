from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from PLIFv2.tools.image_tools import export_gif


class GifExportPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.vars = {
            name: tk.StringVar(value=value)
            for name, value in {
                "folder": "",
                "output": "",
                "start": "0",
                "end": "200",
                "gain": "1",
                "duration": "50",
                "width": "400",
            }.items()
        }

        folder_row = ttk.Frame(self)
        folder_row.pack(anchor="w", padx=8, pady=(8, 2))
        ttk.Button(folder_row, text="图像目录", command=self.choose_folder).pack(side="left")
        ttk.Entry(folder_row, textvariable=self.vars["folder"], width=80).pack(side="left", padx=3)

        output_row = ttk.Frame(self)
        output_row.pack(anchor="w", padx=8, pady=2)
        ttk.Button(output_row, text="输出GIF", command=self.choose_output).pack(side="left")
        ttk.Entry(output_row, textvariable=self.vars["output"], width=80).pack(side="left", padx=3)

        range_row = ttk.Frame(self)
        range_row.pack(anchor="w", padx=8, pady=2)
        self._entry(range_row, "start", "开始编号", 8)
        self._entry(range_row, "end", "结束编号", 8)

        option_row = ttk.Frame(self)
        option_row.pack(anchor="w", padx=8, pady=2)
        self._entry(option_row, "gain", "增益", 8)
        self._entry(option_row, "duration", "帧间隔/ms", 8)
        self._entry(option_row, "width", "最大宽度", 8)
        ttk.Button(option_row, text="生成GIF", command=self.run).pack(side="left", padx=(8, 0))

        self.info = ttk.Label(self, text="")
        self.info.pack(anchor="w", padx=8, pady=(4, 0))

    def _entry(self, row, name: str, label: str, width: int) -> None:
        ttk.Label(row, text=label).pack(side="left", padx=(4, 1))
        ttk.Entry(row, textvariable=self.vars[name], width=width).pack(side="left", padx=(0, 8))

    def choose_folder(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.vars["folder"].set(path)
            self.vars["output"].set(str(Path(path).parent / "spray.gif"))

    def choose_output(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".gif", filetypes=[("GIF", "*.gif")])
        if path:
            self.vars["output"].set(path)

    def run(self) -> None:
        try:
            result = export_gif(
                Path(self.vars["folder"].get().strip()),
                Path(self.vars["output"].get().strip()),
                int(self.vars["start"].get()),
                int(self.vars["end"].get()),
                float(self.vars["gain"].get()),
                int(self.vars["duration"].get()),
                int(self.vars["width"].get()),
            )
            self.info.config(text=f"已生成 {result}")
        except Exception as exc:
            self.info.config(text=f"错误: {exc}")
