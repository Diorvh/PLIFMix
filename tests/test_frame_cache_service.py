from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from PLIFv2.services.frame_cache_service import load_frame_cache, split_frame_cache_tasks


class FrameCacheServiceTests(unittest.TestCase):
    def test_tasks_are_continuous_and_balanced(self) -> None:
        paths = [Path(f"{index}.png") for index in range(10)]

        batches = split_frame_cache_tasks(paths, worker_count=3)

        self.assertEqual(
            [[index for index, _path in batch] for batch in batches],
            [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]],
        )

    def test_serial_and_spawn_cache_results_match_with_corrupt_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            paths = _write_sample_frames(folder)

            serial = load_frame_cache(paths, worker_count=1)
            spawned = load_frame_cache(paths, worker_count=3)

        self.assertEqual(serial.loaded_count, 4)
        self.assertEqual(serial.failed_count, 1)
        self.assertEqual(serial.ranges, spawned.ranges)
        self.assertEqual(serial.loaded_count, spawned.loaded_count)
        self.assertEqual(serial.failed_count, spawned.failed_count)
        self.assertEqual(serial.cache_bytes, spawned.cache_bytes)
        self.assertGreaterEqual(serial.elapsed_seconds, 0)
        self.assertGreaterEqual(spawned.elapsed_seconds, 0)
        for serial_frame, spawned_frame in zip(serial.cache, spawned.cache):
            if serial_frame is None or spawned_frame is None:
                self.assertIsNone(serial_frame)
                self.assertIsNone(spawned_frame)
            else:
                self.assertEqual(serial_frame.dtype, spawned_frame.dtype)
                np.testing.assert_array_equal(serial_frame, spawned_frame)

    def test_direct_main_import_does_not_import_gui(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        probe = (
            "import sys\n"
            "import main\n"
            "assert 'PLIFv2.ui.app_window' not in sys.modules\n"
            "assert 'PLIFv2.ui.pages.image_player_page' not in sys.modules\n"
            "assert 'tkinter' not in sys.modules\n"
        )

        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout, "")

    def test_spawn_child_imports_main_without_gui_modules(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            probe_path = Path(temp_dir) / "spawn_import_probe.py"
            probe_path.write_text(
                "import json\n"
                "import multiprocessing as mp\n"
                "import sys\n"
                f"sys.path.insert(0, {str(repo)!r})\n"
                "import main\n"
                "from PLIFv2.services.frame_cache_service import read_frame_batch\n"
                "from concurrent.futures import ProcessPoolExecutor\n"
                "\n"
                "def report_modules():\n"
                "    names = ('tkinter', 'PLIFv2.ui.app_window', 'PLIFv2.ui.pages.image_player_page')\n"
                "    return {name: name in sys.modules for name in names}\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn')) as executor:\n"
                "        print(json.dumps(executor.submit(report_modules).result(), sort_keys=True))\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [sys.executable, str(probe_path)],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertEqual(
            json.loads(completed.stdout),
            {
                "PLIFv2.ui.app_window": False,
                "PLIFv2.ui.pages.image_player_page": False,
                "tkinter": False,
            },
        )


def _write_sample_frames(folder: Path) -> list[Path]:
    paths: list[Path] = []

    first = np.zeros((2, 3, 3), dtype=np.uint8)
    first[:, :, 1] = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    first_path = folder / "0000.png"
    _write_png(first_path, first)
    paths.append(first_path)

    second = np.array([[256, 512, 768], [1024, 1280, 1536]], dtype=np.uint16)
    second_path = folder / "0001.npy"
    np.save(second_path, second)
    paths.append(second_path)

    third = np.zeros((2, 3, 3), dtype=np.uint8)
    third[:, :, 1] = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.uint8)
    third_path = folder / "0002.png"
    _write_png(third_path, third)
    paths.append(third_path)

    fourth = np.array([[2048, 2304, 2560], [2816, 3072, 3328]], dtype=np.uint16)
    fourth_path = folder / "0003.npy"
    np.save(fourth_path, fourth)
    paths.append(fourth_path)

    corrupt_path = folder / "0004.png"
    corrupt_path.write_bytes(b"not a valid image")
    paths.append(corrupt_path)
    return paths


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise AssertionError(f"could not encode {path}")
    path.write_bytes(encoded.tobytes())


if __name__ == "__main__":
    unittest.main()
