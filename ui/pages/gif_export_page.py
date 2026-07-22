from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from PLIFv2.tools.image_tools import export_gif

class GifExportPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.vars = {name: tk.StringVar(value=value) for name, value in {"folder": "", "output": "", "start": "0", "end": "200", "gain": "1", "duration": "50", "width": "400"}.items()}
        row = ttk.Frame(self)
        row.pack(anchor="w", padx=8, pady=8)
        ttk.Button(row, text="图像目录", command=self.choose_folder).pack(side="left")
        ttk.Entry(row, textvariable=self.vars["folder"], width=40).pack(side="left", padx=3)
        ttk.Button(row, text="输出GIF", command=self.choose_output).pack(side="left")
        ttk.Entry(row, textvariable=self.vars["output"], width=36).pack(side="left", padx=3)
        for name, label, width in [("start", "开始", 7), ("end", "结束", 7), ("gain", "增益", 7), ("duration", "帧间隔ms", 7), ("width", "最大宽度", 7)]:
            ttk.Label(row, text=label).pack(side="left")
            ttk.Entry(row, textvariable=self.vars[name], width=width).pack(side="left")
        ttk.Button(row, text="生成GIF", command=self.run).pack(side="left", padx=4)
        self.info = ttk.Label(self, text="")
        self.info.pack(anchor="w", padx=8)

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
            result = export_gif(Path(self.vars["folder"].get()), Path(self.vars["output"].get()), int(self.vars["start"].get()), int(self.vars["end"].get()), float(self.vars["gain"].get()), int(self.vars["duration"].get()), int(self.vars["width"].get()))
            self.info.config(text=f"已生成 {result}")
        except Exception as exc:
            self.info.config(text=f"错误: {exc}")
