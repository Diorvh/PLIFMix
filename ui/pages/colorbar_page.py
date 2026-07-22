from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import cv2
from PIL import Image, ImageTk

from PLIFv2.tools.image_tools import create_colorbar, save_colorbar

class ColorbarPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.vars = {name: tk.StringVar(value=value) for name, value in {"height": "800", "width": "50", "minimum": "0", "maximum": "1", "top": "20", "bottom": "20"}.items()}
        row = ttk.Frame(self)
        row.pack(anchor="w", padx=8, pady=8)
        for name, label in [("height", "高度"), ("width", "宽度"), ("minimum", "最小"), ("maximum", "最大"), ("top", "上留白"), ("bottom", "下留白")]:
            ttk.Label(row, text=label).pack(side="left")
            ttk.Entry(row, textvariable=self.vars[name], width=8).pack(side="left", padx=2)
        ttk.Button(row, text="预览", command=self.preview).pack(side="left", padx=4)
        ttk.Button(row, text="保存", command=self.save).pack(side="left", padx=4)
        self.canvas = tk.Canvas(self, highlightthickness=1, highlightbackground="#aaa")
        self.canvas.pack(anchor="nw", padx=8, pady=6)
        self.image = None

    def preview(self) -> None:
        self.image = create_colorbar(int(self.vars["height"].get()), int(self.vars["width"].get()), float(self.vars["minimum"].get()), float(self.vars["maximum"].get()), int(self.vars["top"].get()), int(self.vars["bottom"].get()))
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.tk_image = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.config(width=rgb.shape[1], height=rgb.shape[0])
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def save(self) -> None:
        if self.image is None:
            self.preview()
        path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIF", "*.tif")])
        if path:
            save_colorbar(Path(path), self.image)
