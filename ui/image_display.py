from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from PLIFv2.algorithms.visualization import blacked_jet_colormap, blue_white_red_colormap
from PLIFv2.core.image_io import write_tif


class ImageDisplay(ttk.Frame):
    def __init__(self, master, name: str):
        super().__init__(master)
        self.name = name
        self.image: np.ndarray | None = None
        self.display: np.ndarray | None = None
        self.zoom = 1.0
        self.save_name_suffix = ""
        self.save_folder = Path("manual_outputs")
        self._setup_ui()

    def _setup_ui(self) -> None:
        row = ttk.Frame(self)
        row.pack(anchor=tk.W)
        ttk.Label(row, text=self.name, width=12).pack(side=tk.LEFT)
        self.info = ttk.Label(row, text="坐标: (0, 0) 值: 0")
        self.info.pack(side=tk.LEFT)
        ttk.Button(row, text="保存图像", command=self.save_image).pack(side=tk.LEFT, padx=(4, 0))
        self.canvas = tk.Canvas(self, highlightthickness=1, highlightbackground="#cccccc")
        self.canvas.pack(anchor=tk.W)
        self.canvas.bind("<Motion>", self.update_mouse_info)

    def update(
        self,
        image: np.ndarray,
        gain: float = 1,
        colormap: str = "jet",
        zoom: float = 1,
        crosshair: tuple[int, int, int] | None = None,
        ellipse: tuple[int, int, int, int, int] | None = None,
        save_name_suffix: str = "",
        special_points=None,
        save_folder: str | Path | None = None,
    ) -> None:
        self.image = image
        self.zoom = zoom if zoom else 1
        self.save_name_suffix = save_name_suffix
        if save_folder is not None:
            self.save_folder = Path(save_folder)
        img = image.copy()

        if img.ndim == 2:
            if gain == 0:
                max_value = float(np.nanmax(img)) if img.size else 0
                gain = 65535 / max_value if max_value else 1
            img = img * gain
            if img.dtype != np.uint8:
                img = img / 256
            img = np.clip(img, 0, 255).astype(np.uint8)
            if colormap == "bwr":
                img = cv2.applyColorMap(img, blue_white_red_colormap())
            elif colormap == "none":
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = cv2.applyColorMap(img, blacked_jet_colormap())
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

        if special_points is not None and len(special_points):
            try:
                y, x = special_points
                img[y, x] = [255, 255, 255]
            except Exception:
                pass
        if crosshair:
            x, y, line_width = crosshair
            cv2.line(img, (x, 0), (x, img.shape[0]), (0, 0, 255), line_width)
            cv2.line(img, (0, y), (img.shape[1], y), (0, 0, 255), line_width)
        if ellipse:
            x, y, major, minor, line_width = ellipse
            cv2.ellipse(img, (x, y), (major, minor), 0, 0, 360, (0, 0, 255), line_width)

        self.display = img
        height, width = img.shape[:2]
        show_width = max(1, int(width / self.zoom))
        show_height = max(1, int(height / self.zoom))
        if self.zoom != 1:
            img = cv2.resize(img, (show_width, show_height))
        self.canvas.config(width=show_width, height=show_height)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(show_width // 2, show_height // 2, image=self.tk_img)

    def update_mouse_info(self, event) -> None:
        if self.image is None:
            return
        x = int(event.x * self.zoom)
        y = int(event.y * self.zoom)
        value = 0
        if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
            value = self.image[y, x]
        max_value = float(np.nanmax(self.image)) if self.image.size else 0
        self.info.config(text=f"坐标: ({x}, {y}) 值: {value:.2f} 最大值: {max_value:.2f}")

    def save_image(self) -> None:
        if self.display is None:
            return
        folder = self.save_folder / "images" / self.name
        name = self.save_name_suffix or self.name
        write_tif(folder / f"{name}-{self.name}.tif", self.display)
