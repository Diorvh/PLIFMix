from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from PLIFv2.core.plot_style import configure_matplotlib

ZERO_COLOR = [128, 128, 128]


@lru_cache(maxsize=1)
def blacked_jet_colormap() -> np.ndarray:
    colormap = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_JET)
    colormap[0] = ZERO_COLOR
    return colormap


def blue_white_red_colormap() -> np.ndarray:
    colormap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(0, 129):
        ratio = i / 128.0
        colormap[i] = [255, int(255 * ratio), int(255 * ratio)]
    for i in range(129, 256):
        ratio = (i - 128) / 127.0
        colormap[i] = [int(255 * (1 - ratio)), int(255 * (1 - ratio)), 255]
    colormap[0] = ZERO_COLOR
    return colormap.reshape(256, 1, 3)


def plot_data_image(
    x_data: np.ndarray,
    y_data: np.ndarray,
    width: int = 1308,
    height: int = 877,
    dpi: int = 100,
    title: str = "数据图表",
    xlabel: str = "X轴",
    ylabel: str = "Y轴",
    axislimit: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    configure_matplotlib()
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi, layout="constrained")
    ax = fig.add_subplot(111)
    ax.plot(x_data, y_data, "b-", linewidth=2)
    ax.scatter(x_data, y_data, color="red", s=8, zorder=5)
    ax.tick_params(axis="both", labelsize=20)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    if axislimit:
        ax.set_xlim(axislimit[0])
        ax.set_ylim(axislimit[1])
    ax.set_title(title, fontsize=20, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.7)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_rgba = np.asarray(canvas.buffer_rgba())
    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
    return img_bgr
