from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from tkinter import ttk

from PLIFv2.ui.pages.image_player_page import ImagePlayerPage


class _Value:
    def __init__(self, value) -> None:
        self.value = value

    def get(self):
        return self.value


class _Info:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def config(self, *, text: str) -> None:
        self.messages.append(text)


class PlayerTimerTests(unittest.TestCase):
    def test_start_restart_keeps_one_tick_chain_and_moves_one_frame_each_time(self) -> None:
        page = _make_page(frame_count=4)
        with patch("PLIFv2.ui.pages.image_player_page.time.perf_counter", return_value=0.0):
            page.start(1)
            first_job = page._tick_after_id
            page.start(-1)

        self.assertEqual(page.index, 0)
        self.assertEqual(page.play_delta, -1)
        self.assertEqual(page.cancelled_jobs, [first_job])
        self.assertEqual(len(page.scheduled_jobs), 2)
        self.assertEqual(page._tick_after_id, page.scheduled_jobs[-1][0])
        self.assertEqual(page.scheduled_callbacks[-1][1], page._tick)

    def test_tick_uses_remaining_frame_duration_and_stores_after_job(self) -> None:
        page = _make_page(frame_count=3, frame_rate="1")
        page.play_delta = 1
        clock_values = iter((10.0, 10.25))
        with patch(
            "PLIFv2.ui.pages.image_player_page.time.perf_counter",
            side_effect=lambda: next(clock_values),
        ):
            page._tick()

        self.assertEqual(page.index, 1)
        self.assertEqual(page.scheduled_jobs, [("after-1", 750)])
        self.assertEqual(page._tick_after_id, "after-1")

    def test_overrun_uses_one_idle_callback_without_skipping_a_frame(self) -> None:
        page = _make_page(frame_count=3, frame_rate="20")
        page.play_delta = 1
        clock_values = iter((10.0, 10.1))
        with patch(
            "PLIFv2.ui.pages.image_player_page.time.perf_counter",
            side_effect=lambda: next(clock_values),
        ):
            page._tick()

        self.assertEqual(page.index, 1)
        self.assertEqual(page.scheduled_callbacks[-1][1], page._tick)
        self.assertEqual(page._tick_after_id, "idle-1")

    def test_stop_step_hide_and_destroy_cancel_pending_tick(self) -> None:
        page = _make_page(frame_count=3)
        with patch("PLIFv2.ui.pages.image_player_page.time.perf_counter", return_value=0.0):
            page.start(1)
        first_job = page._tick_after_id

        page.start(0)
        self.assertEqual(page.cancelled_jobs, [first_job])
        self.assertEqual(page._tick_after_id, None)
        self.assertEqual(page.play_delta, 0)

        with patch("PLIFv2.ui.pages.image_player_page.time.perf_counter", return_value=0.0):
            page.start(1)
        second_job = page._tick_after_id
        page.step(1)
        self.assertEqual(page.cancelled_jobs, [first_job, second_job])
        self.assertEqual(page.play_delta, 0)

        with patch("PLIFv2.ui.pages.image_player_page.time.perf_counter", return_value=0.0):
            page.start(1)
        third_job = page._tick_after_id
        page.on_page_hidden()
        self.assertEqual(page.cancelled_jobs, [first_job, second_job, third_job])
        self.assertEqual(page._tick_after_id, None)
        self.assertEqual(page.play_delta, 0)

        with patch.object(ttk.Frame, "destroy") as destroy:
            page.destroy()
        self.assertEqual(page._tick_after_id, None)
        destroy.assert_called_once_with()


def _make_page(frame_count: int, frame_rate: str = "20") -> ImagePlayerPage:
    page = ImagePlayerPage.__new__(ImagePlayerPage)
    page.files = [Path(f"{index}.npy") for index in range(frame_count)]
    page.index = 0
    page.play_delta = 0
    page._tick_after_id = None
    page._frame_cache = [np.ones((1, 1), dtype=np.uint8) for _ in range(frame_count)]
    page.frame_rate = _Value(frame_rate)
    page.info = _Info()
    page.scheduled_jobs: list[tuple[str, object] | tuple[str, int]] = []
    page.scheduled_callbacks: list[tuple[str, object]] = []
    page.cancelled_jobs: list[str] = []
    job_counter = iter(range(1, 20))

    def after(delay: int, callback):
        job_id = f"after-{next(job_counter)}"
        page.scheduled_jobs.append((job_id, delay))
        page.scheduled_callbacks.append((job_id, callback))
        return job_id

    def after_idle(callback):
        job_id = f"idle-{next(job_counter)}"
        page.scheduled_jobs.append((job_id, callback))
        page.scheduled_callbacks.append((job_id, callback))
        return job_id

    page.after = after
    page.after_idle = after_idle
    page.after_cancel = page.cancelled_jobs.append
    page.show = lambda: None
    page.update_idletasks = lambda: None
    return page


if __name__ == "__main__":
    unittest.main()
