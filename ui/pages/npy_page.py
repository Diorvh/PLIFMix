from __future__ import annotations

import tkinter as tk
from pathlib import Path
from PLIFMix.ui import localized_dialogs as filedialog, localized_widgets as ttk

import numpy as np
from matplotlib.figure import Figure

from PLIFMix.tools.image_tools import top_k_values, transform_npy
from PLIFMix.ui.mpl_canvas import DebouncedFigureCanvasTkAgg

class NpyPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.path = tk.StringVar()
        self.k = tk.StringVar(value="10")
        self.operation = tk.StringVar(value="multiply")
        self.value = tk.StringVar(value="1")
        self.array = None
        row = ttk.Frame(self)
        row.pack(anchor="w", padx=8, pady=6)
        ttk.Button(row, text="打开NPY", command=self.open).pack(side="left")
        ttk.Entry(row, textvariable=self.path, width=50).pack(side="left", padx=3)
        ttk.Label(row, text="Top K").pack(side="left")
        ttk.Entry(row, textvariable=self.k, width=6).pack(side="left")
        ttk.Button(row, text="查找", command=self.show_topk).pack(side="left", padx=3)
        ttk.Label(row, text="运算").pack(side="left")
        ttk.Combobox(row, textvariable=self.operation, values=["multiply", "add", "normalize", "reciprocal"], width=12, state="readonly").pack(side="left")
        ttk.Entry(row, textvariable=self.value, width=7).pack(side="left")
        ttk.Button(row, text="另存运算结果", command=self.transform).pack(side="left", padx=3)
        self.info = ttk.MessageText(self, height=8, wrap="word")
        self.info.pack(fill="x", padx=8)
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self.colorbar = None
        self.canvas = DebouncedFigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=6)

    def open(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("NPY", "*.npy")])
        if not path:
            return
        self.path.set(path)
        self.array = np.load(path)
        self.info.clear_messages()
        self.info.append_message(f"shape={self.array.shape}\ndtype={self.array.dtype}\nmin={np.min(self.array)}\nmax={np.max(self.array)}\nmean={np.mean(self.array)}\nstd={np.std(self.array)}")
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None
        self.axis.clear()
        if self.array.ndim == 2:
            image = self.axis.imshow(self.array, cmap="viridis")
            self.colorbar = self.figure.colorbar(image, ax=self.axis, fraction=0.046)
        else:
            self.axis.plot(self.array.ravel())
        self.canvas.request_draw()

    def show_topk(self) -> None:
        if self.array is None:
            return
        result = top_k_values(self.array, int(self.k.get()))
        self.info.append_message("\n\nTop K:\n" + "\n".join(f"{value}: {index}" for value, index in result))

    def transform(self) -> None:
        if not self.path.get():
            return
        output = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("NPY", "*.npy")])
        if output:
            transform_npy(Path(self.path.get()), Path(output), self.operation.get(), float(self.value.get()))
            self.info.append_message(f"\n已保存 {output}")
