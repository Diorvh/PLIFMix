from __future__ import annotations

import os
from pathlib import Path
from statistics import fmean
import time
import tkinter as tk
import unittest

import cv2
import numpy as np
from PIL import Image, ImageTk

from PLIFMix.algorithms.visualization import blacked_jet_colormap
from PLIFMix.core.image_io import extract_plif_signal, read_image
from PLIFMix.ui.components.viewer_panel import _scale_gray_for_display


class PlayerPerformanceTests(unittest.TestCase):
    def test_player_pipeline_timing(self) -> None:
        folder_text = os.environ.get("PLIF_PLAYER_BENCHMARK_FOLDER", "").strip()
        if not folder_text:
            self.skipTest("Set PLIF_PLAYER_BENCHMARK_FOLDER to run the player benchmark.")

        files = sorted(
            path
            for path in Path(folder_text).iterdir()
            if path.suffix.lower() in {".png", ".tif", ".tiff", ".bmp"}
        )
        if not files:
            self.skipTest(f"No supported image files in {folder_text}")

        sample_count = min(int(os.environ.get("PLIF_PLAYER_BENCHMARK_SAMPLES", "120")), len(files))
        sample_paths = files[:sample_count]
        root, canvas, frame_var, info = _create_tk_context()

        timings: dict[str, list[float]] = {}
        canvas_state: dict[str, object | None] = {"photo": None, "item": None, "size": None}

        def measure(name: str, callback):
            started = time.perf_counter_ns()
            value = callback()
            timings.setdefault(name, []).append((time.perf_counter_ns() - started) / 1_000_000)
            return value

        def render_frame(path: Path, frame_index: int) -> None:
            frame_started = time.perf_counter_ns()
            original = measure("read and decode", lambda: read_image(path))
            raw_single = measure(
                "compact 8-bit green channel",
                lambda: np.ascontiguousarray(extract_plif_signal(original)),
            )
            image = raw_single

            def apply_threshold() -> None:
                nonlocal image
                image = image.copy()
                image[image <= 5] = 0

            measure("8-bit denoise threshold", apply_threshold)
            minimum, maximum = measure(
                "minimum and maximum scan",
                lambda: (np.min(image) * 256, np.max(image) * 256),
            )
            scaled = measure(
                "gain scaling and uint8 conversion",
                lambda: _scale_gray_for_display(image, 10.0, value_scale=256.0),
            )
            display = measure("colormap", lambda: cv2.applyColorMap(scaled, blacked_jet_colormap()))
            shown = measure("zoom resize", lambda: cv2.resize(display, (display.shape[1], display.shape[0]), interpolation=cv2.INTER_NEAREST))
            frame_image = measure("BGR to RGB and PIL image", lambda: Image.fromarray(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)))

            if root is not None and canvas is not None and frame_var is not None and info is not None:
                def update_photo() -> None:
                    frame_size = frame_image.size
                    photo = canvas_state["photo"]
                    if photo is None or canvas_state["size"] != frame_size:
                        photo = ImageTk.PhotoImage(frame_image)
                        canvas_state["photo"] = photo
                        canvas_state["size"] = frame_size
                        canvas.config(width=frame_size[0], height=frame_size[1])
                        if canvas_state["item"] is None:
                            canvas_state["item"] = canvas.create_image(0, 0, anchor="nw", image=photo)
                        else:
                            canvas.itemconfigure(canvas_state["item"], image=photo)
                    else:
                        photo.paste(frame_image)

                measure("PhotoImage update", update_photo)
                measure("Tk canvas flush", root.update_idletasks)
                measure(
                    "Tk control updates",
                    lambda: (frame_var.set(str(frame_index)), info.config(text=f"min={minimum:g}, max={maximum:g}")),
                )
            timings.setdefault("full frame", []).append((time.perf_counter_ns() - frame_started) / 1_000_000)

        try:
            for warmup_index, path in enumerate(sample_paths[:8]):
                render_frame(path, warmup_index)
            timings.clear()
            for frame_index, path in enumerate(sample_paths):
                render_frame(path, frame_index)
        finally:
            if root is not None:
                root.destroy()

        report = [(name, fmean(values), _percentile(values, 0.95)) for name, values in timings.items()]
        print(f"\nPlayer benchmark: {len(sample_paths)} frames from {Path(folder_text).name}")
        if root is None:
            print("Tk canvas timing unavailable: this Python interpreter has no usable Tcl/Tk runtime.")
        for name, average_ms, p95_ms in report:
            print(f"{name:34} mean={average_ms:6.2f} ms  p95={p95_ms:6.2f} ms")
        full_frame_ms = fmean(timings["full frame"])
        print(f"60 Hz budget: 16.67 ms; measured mean: {full_frame_ms:.2f} ms; upper bound: {1000 / full_frame_ms:.1f} Hz")
        self.assertEqual(len(timings["full frame"]), len(sample_paths))


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * quantile)))
    return ordered[index]


def _create_tk_context():
    try:
        root = tk.Tk()
    except tk.TclError:
        return None, None, None, None
    root.title("PLIF player performance benchmark")
    canvas = tk.Canvas(root, width=1, height=1)
    canvas.pack()
    frame_var = tk.StringVar(root)
    info = tk.Label(root, text="")
    info.pack()
    root.update()
    return root, canvas, frame_var, info
