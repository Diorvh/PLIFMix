from __future__ import annotations

import os
import re
import time
import tkinter as tk
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import ceil
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np

from PLIFv2.core.image_io import extract_plif_signal, list_image_files, read_image
from PLIFv2.ui.components.display_pane import HorizontalDisplayPane
from PLIFv2.ui.components.player_pixel_plot import PlayerPixelPlot
from PLIFv2.ui.components.viewer_panel import ViewerPanel

def _read_player_raw_frame(path: Path) -> np.ndarray:
    image = extract_plif_signal(read_image(path))
    # Make the cached green channel own its compact 8-bit buffer instead of
    # retaining the three-channel source image through a NumPy view.
    return np.ascontiguousarray(image)

def _read_player_raw_batch(indexed_paths: tuple[tuple[int, Path], ...]) -> tuple[list[tuple[int, np.ndarray | None]], int]:
    frames: list[tuple[int, np.ndarray | None]] = []
    failed = 0
    for frame_index, path in indexed_paths:
        try:
            frames.append((frame_index, _read_player_raw_frame(path)))
        except Exception:
            frames.append((frame_index, None))
            failed += 1
    return frames, failed

class ImagePlayerPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.files: list[Path] = []
        self.index = 0
        self.play_delta = 0
        self._frame_cache: list[np.ndarray | None] = []
        self._cache_loaded = 0
        self._cache_failed = 0
        self._cache_bytes = 0
        self.folder = tk.StringVar()
        self.frame_rate = tk.StringVar(value="20")
        self.current_frame = tk.StringVar(value="0")
        self.gain = tk.StringVar(value="1")
        self.auto_gain = tk.BooleanVar(value=False)
        self.auto_gain_text = tk.StringVar(value="（自动增益：-）")
        self.zoom = tk.StringVar(value="2")
        self.relative_denoise = tk.StringVar(value="0")
        self.rotate = tk.StringVar(value="0")
        self.shift = tk.StringVar(value="0")
        self.denoise_threshold = tk.StringVar(value="0")
        self.pseudo = tk.BooleanVar(value=False)
        default_cache_processes = max(1, min(8, (os.cpu_count() or 1) - 1))
        self.cache_processes = tk.StringVar(value=str(default_cache_processes))
        self.use_frame_segments = tk.BooleanVar(value=False)
        self.frame_segments = tk.StringVar()
        self._plot_index_set: set[int] = set()
        self.pixel_x = tk.StringVar(value="0")
        self.pixel_y = tk.StringVar(value="0")
        self.pixel_top_k = tk.StringVar(value="10")
        self.cache_status = tk.StringVar(value="缓存：空闲")

        row1 = ttk.Frame(self)
        row1.pack(anchor="w", padx=8, pady=(6, 2))
        ttk.Button(row1, text="选择文件夹", command=self.choose).pack(side="left")
        folder_entry = ttk.Entry(row1, textvariable=self.folder, width=62)
        folder_entry.pack(side="left", padx=3)
        folder_entry.bind("<Return>", self.load_folder)
        ttk.Label(row1, text="缓存进程数").pack(side="left", padx=(8, 2))
        process_entry = ttk.Entry(row1, textvariable=self.cache_processes, width=5)
        process_entry.pack(side="left")
        process_entry.bind("<Return>", self.load_folder)
        ttk.Checkbutton(
            row1,
            text="非连续区间",
            variable=self.use_frame_segments,
            command=self._update_plot_range,
        ).pack(side="left", padx=(8, 2))
        self.segment_entry = ttk.Entry(row1, textvariable=self.frame_segments, width=28)
        self.segment_entry.pack(side="left")
        self.segment_entry.bind("<Return>", self._update_plot_range)

        row2 = ttk.Frame(self)
        row2.pack(anchor="w", padx=8, pady=2)
        ttk.Label(row2, text="帧率/Hz").pack(side="left")
        ttk.Entry(row2, textvariable=self.frame_rate, width=7).pack(side="left", padx=(3, 10))
        ttk.Button(row2, text="反向播放", command=lambda: self.start(-1)).pack(side="left", padx=2)
        ttk.Button(row2, text="暂停", command=lambda: self.start(0)).pack(side="left", padx=2)
        ttk.Button(row2, text="正向播放", command=lambda: self.start(1)).pack(side="left", padx=2)

        row3 = ttk.Frame(self)
        row3.pack(anchor="w", padx=8, pady=(2, 4))
        ttk.Label(row3, text="当前帧").pack(side="left")
        frame_entry = ttk.Entry(row3, textvariable=self.current_frame, width=8)
        frame_entry.pack(side="left", padx=3)
        frame_entry.bind("<Return>", self.jump_to)
        ttk.Button(row3, text="上一张", command=lambda: self.step(-1)).pack(side="left", padx=2)
        ttk.Button(row3, text="下一张", command=lambda: self.step(1)).pack(side="left", padx=2)

        row4 = ttk.Frame(self)
        row4.pack(anchor="w", padx=8, pady=(2, 4))
        ttk.Checkbutton(row4, text="伪彩图", variable=self.pseudo, command=self.show).pack(side="left", padx=(0, 5))
        self._display_entry(row4, "增益", self.gain)
        ttk.Checkbutton(row4, variable=self.auto_gain, command=self.show).pack(side="left", padx=(5, 1))
        ttk.Label(row4, textvariable=self.auto_gain_text).pack(side="left", padx=(0, 3))
        self._display_entry(row4, "缩小", self.zoom)
        self._display_entry(row4, "相对去噪阈值", self.relative_denoise)
        self._display_entry(row4, "旋转", self.rotate)
        self._display_entry(row4, "左移", self.shift)
        self._display_entry(row4, "去噪阈值(8位)", self.denoise_threshold)

        row5 = ttk.Frame(self)
        row5.pack(anchor="w", padx=8, pady=(0, 3))
        self._pixel_entry(row5, "X", self.pixel_x, self._request_manual_pixel_series)
        self._pixel_entry(row5, "Y", self.pixel_y, self._request_manual_pixel_series)
        ttk.Button(row5, text="更新像素数据", command=self._request_manual_pixel_series).pack(side="left", padx=(4, 10))
        self._pixel_entry(row5, "Top K", self.pixel_top_k, self._update_top_k, width=5)
        ttk.Button(row5, text="更新 Top K", command=self._update_top_k).pack(side="left", padx=4)
        ttk.Label(row5, textvariable=self.cache_status).pack(side="left", padx=(12, 0))

        self.info = ttk.Label(self, text="")
        self.info.pack(anchor="w", padx=8)
        self.display_host = ttk.Frame(self)
        self.display_host.pack(fill="both", expand=True, padx=6, pady=4)
        self.display_pane = HorizontalDisplayPane(self.display_host)
        self.display_pane.pack(fill="both", expand=True)
        self.image_host = self.display_pane.primary_host
        self.pixel_plot = PlayerPixelPlot(self.display_pane.secondary_host)
        self.pixel_plot.pack(fill="both", expand=True)
        self.panel: ViewerPanel | None = None

    def _display_entry(self, row: ttk.Frame, label: str, variable: tk.StringVar) -> None:
        ttk.Label(row, text=label).pack(side="left", padx=(4, 2))
        entry = ttk.Entry(row, textvariable=variable, width=7)
        entry.pack(side="left")
        entry.bind("<Return>", self._refresh_display)

    def _pixel_entry(self, row: ttk.Frame, label: str, variable: tk.StringVar, command, width: int = 7) -> None:
        ttk.Label(row, text=label).pack(side="left", padx=(4, 2))
        entry = ttk.Entry(row, textvariable=variable, width=width)
        entry.pack(side="left")
        entry.bind("<Return>", command)

    def _refresh_display(self, event=None) -> None:
        self.show()

    def choose(self) -> None:
        folder = filedialog.askdirectory()
        if not folder:
            return
        self.folder.set(folder)
        self.load_folder()

    def load_folder(self, event=None) -> None:
        folder = Path(self.folder.get().strip())
        if not folder.is_dir():
            self.info.config(text=f"文件夹不存在: {folder}")
            return
        self.play_delta = 0
        self.pixel_plot.reset()
        self._frame_cache = []
        self._plot_index_set = set()
        self.files = list_image_files(folder, (".png", ".tif", ".tiff", ".bmp", ".npy"))
        if not self.files:
            self.info.config(text="文件夹中没有可读取的图像")
            return
        self.index = 0
        self.current_frame.set("0")
        if not self._cache_frames():
            return
        self._update_plot_range()
        self.show()

    def _update_plot_range(self, event=None) -> None:
        if not self.files:
            return
        if not self.use_frame_segments.get():
            self._plot_index_set = set(range(len(self.files)))
        elif not self.frame_segments.get().strip():
            self.pixel_plot.set_error("非连续区间已启用，但尚未输入区间")
            return
        else:
            indices: list[int] = []
            for segment in re.split(r"[,，;；]+", self.frame_segments.get()):
                try:
                    start_text, end_text = segment.strip().split(":", 1)
                    start, end = int(start_text), int(end_text)
                except ValueError:
                    self.pixel_plot.set_error(f"非连续区间格式错误: {segment}")
                    return
                if start < 0 or end <= start or end > len(self.files):
                    self.pixel_plot.set_error(f"非连续区间超出范围: {start}:{end}，可用范围为 0:{len(self.files)}")
                    return
                indices.extend(range(start, end))
            self._plot_index_set = set(indices)
        if self.pixel_plot.point is not None:
            self._request_pixel_series(*self.pixel_plot.point)

    def _cache_frames(self) -> bool:
        cache_started = time.perf_counter()
        try:
            requested_processes = int(self.cache_processes.get())
            if requested_processes < 1:
                raise ValueError
        except ValueError:
            self.info.config(text="缓存进程数必须是大于等于 1 的整数")
            return False

        total = len(self.files)
        worker_count = min(requested_processes, total)
        batch_size, remainder = divmod(total, worker_count)
        indexed_files = tuple(enumerate(self.files))
        batches: list[tuple[tuple[int, Path], ...]] = []
        start_index = 0
        for batch_index in range(worker_count):
            stop_index = start_index + batch_size + (1 if batch_index < remainder else 0)
            batches.append(indexed_files[start_index:stop_index])
            start_index = stop_index

        self._frame_cache = [None] * total
        self._cache_loaded = 0
        self._cache_failed = 0
        self._cache_bytes = 0
        self.cache_status.set(f"正在用 {worker_count} 个进程缓存：0/{total} 张")
        self.update_idletasks()

        if worker_count == 1:
            results = [_read_player_raw_batch(batches[0])]
        else:
            results = []
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_batches = {
                    executor.submit(_read_player_raw_batch, indexed_paths): indexed_paths
                    for indexed_paths in batches
                }
                for future in as_completed(future_batches):
                    indexed_paths = future_batches[future]
                    try:
                        results.append(future.result())
                    except Exception:
                        results.append(([(index, None) for index, _path in indexed_paths], len(indexed_paths)))

        for frames, failed in results:
            self._cache_failed += failed
            for frame_index, frame in frames:
                self._frame_cache[frame_index] = frame
                if frame is not None:
                    self._cache_loaded += 1
                    self._cache_bytes += frame.nbytes
        memory_gib = self._cache_bytes / (1024 ** 3)
        elapsed_seconds = time.perf_counter() - cache_started
        text = f"已缓存：{self._cache_loaded}/{total} 张，{memory_gib:.2f} GiB，耗时：{elapsed_seconds:.2f} s"
        if self._cache_failed:
            text += f"，失败：{self._cache_failed}"
        self.cache_status.set(text)
        return True

    def _cached_frame(self, frame_index: int) -> np.ndarray | None:
        if 0 <= frame_index < len(self._frame_cache):
            return self._frame_cache[frame_index]
        return None

    def show(self) -> None:
        if not self.files:
            return
        path = self.files[self.index]
        raw_single = self._cached_frame(self.index)
        if raw_single is None:
            self.info.config(text="Selected image could not be read into cache.")
            return
        try:
            auto_gain = self.auto_gain.get()
            gain = 1.0 if auto_gain else float(self.gain.get())
            zoom = float(self.zoom.get())
            relative_denoise = float(self.relative_denoise.get())
            rotate = float(self.rotate.get())
            shift = int(self.shift.get())
            denoise_threshold = float(self.denoise_threshold.get())
            if zoom <= 0:
                raise ValueError("缩小倍数必须大于0")
            if relative_denoise < 0 or denoise_threshold < 0:
                raise ValueError("去噪阈值不能小于0")
        except ValueError as exc:
            self.info.config(text=f"显示参数错误: {exc}")
            return
        if self.panel is None:
            self.panel = ViewerPanel(
                self.image_host,
                path,
                show_histogram=False,
                raw_single=raw_single,
                point_selected_callback=self._request_pixel_series,
            )
            self.panel.pack(fill="both", expand=True)
        else:
            self.panel.set_path(path, raw_single)
        computed_gain = self.panel.update_display(
            gain=gain,
            auto_gain=auto_gain,
            zoom=zoom,
            rotate=rotate,
            shift=shift,
            denoise=relative_denoise,
            pseudo=self.pseudo.get(),
            center=(0, 0),
            ellipse=None,
            crosshair=False,
            line_width=1,
            quad=None,
            zero_threshold_8bit=denoise_threshold,
        )
        self.auto_gain_text.set(
            f"（自动增益：{computed_gain:.2f}）" if computed_gain is not None else "（自动增益：-）"
        )
        self.current_frame.set(str(self.index))
        self.info.config(text="")

    def start(self, delta: int) -> None:
        self.play_delta = delta
        if delta:
            self._tick()

    def _tick(self) -> None:
        if not self.play_delta or not self.files:
            return
        try:
            frame_rate = float(self.frame_rate.get())
            if frame_rate <= 0:
                raise ValueError
        except ValueError:
            self.play_delta = 0
            self.info.config(text="帧率必须是大于0的数值")
            return
        next_index = (self.index + self.play_delta) % len(self.files)
        if self._cached_frame(next_index) is None:
            self.play_delta = 0
            self.info.config(text=f"Image {next_index} was not cached; playback stopped.")
            return
        frame_started = time.perf_counter()
        self.index = next_index
        self.show()
        remaining_seconds = (1 / frame_rate) - (time.perf_counter() - frame_started)
        if remaining_seconds > 0:
            self.after(ceil(remaining_seconds * 1000), self._tick)
        else:
            self.after_idle(self._tick)

    def step(self, delta: int) -> None:
        if not self.files:
            return
        self.play_delta = 0
        self.index = (self.index + delta) % len(self.files)
        self.show()

    def jump_to(self, event=None) -> None:
        if self.files:
            try:
                self.index = min(max(0, int(self.current_frame.get())), len(self.files) - 1)
            except ValueError:
                self.info.config(text="当前帧必须是整数")
                return
            self.show()

    def _request_pixel_series(self, x: int, y: int) -> None:
        self.pixel_x.set(str(x))
        self.pixel_y.set(str(y))
        self._request_manual_pixel_series()

    def _request_manual_pixel_series(self, event=None) -> None:
        if not self.files:
            return
        try:
            x = int(self.pixel_x.get())
            y = int(self.pixel_y.get())
            zero_threshold_8bit = float(self.denoise_threshold.get())
            relative_denoise = float(self.relative_denoise.get())
            top_k = int(self.pixel_top_k.get())
            if x < 0 or y < 0 or zero_threshold_8bit < 0 or relative_denoise < 0 or top_k < 1:
                raise ValueError
        except ValueError:
            self.pixel_plot.set_error("X/Y must be non-negative integers; Top K must be at least 1.")
            return
        self.pixel_plot.set_loading(x, y)

        values = np.zeros(len(self.files), dtype=np.float64)
        for frame_index, raw_single in enumerate(self._frame_cache):
            if frame_index not in self._plot_index_set:
                continue
            if raw_single is None:
                self.pixel_plot.set_error(f"Image {frame_index} was not cached.")
                return
            if y >= raw_single.shape[0] or x >= raw_single.shape[1]:
                self.pixel_plot.set_error(f"Point ({x}, {y}) is outside image {frame_index}.")
                return
            value_scale = 256.0 if raw_single.dtype == np.uint8 else 1.0
            pixel_value = float(raw_single[y, x]) * value_scale
            if pixel_value <= zero_threshold_8bit * 256:
                continue
            if relative_denoise > 0 and pixel_value < float(np.max(raw_single)) * value_scale * relative_denoise:
                continue
            values[frame_index] = pixel_value
        frame_axis, frame_labels = self._frame_axis(tuple(self.files))
        self.pixel_plot.set_series(x, y, values, frame_axis, frame_labels, top_k)

    @staticmethod
    def _frame_axis(paths: tuple[Path, ...]) -> tuple[np.ndarray, list[str]]:
        labels = [path.stem for path in paths]
        if all(label.isdigit() for label in labels):
            return np.asarray([int(label) for label in labels]), labels
        return np.arange(len(paths)), labels

    def _update_top_k(self, event=None) -> None:
        try:
            top_k = int(self.pixel_top_k.get())
            if top_k < 1:
                raise ValueError
        except ValueError:
            self.pixel_plot.set_error("Top K must be at least 1.")
            return
        self.pixel_plot.update_top_k(top_k)

    def destroy(self) -> None:
        super().destroy()

    def on_page_hidden(self) -> None:
        self.play_delta = 0
