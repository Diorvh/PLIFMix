from __future__ import annotations

import json
import tempfile
import tkinter as tk
import unittest
from pathlib import Path

import cv2
import numpy as np
import openpyxl

from PLIFv2.algorithms.average import average_image_sets, average_images, split_average_tasks
from PLIFv2.core.config_io import load_settings, save_settings
from PLIFv2.core.experiment_images import load_experiment_triplet
from PLIFv2.core.image_io import write_tif
from PLIFv2.domain.settings import (
    AverageSettings,
    CalibrationSettings,
    DavisSettings,
    DistributionSettings,
    GainCorrectionSettings,
    ManualMixingSettings,
    NoiseSettings,
    ShiftSearchSettings,
)
from PLIFv2.services.average_service import AverageImageService
from PLIFv2.services.calibration_service import CalibrationService
from PLIFv2.services.davis_service import DavisService
from PLIFv2.services.distribution_service import DistributionFitService
from PLIFv2.services.intensifier_service import IntensifierCorrectionService
from PLIFv2.services.mixing_service import MixingService
from PLIFv2.services.noise_service import NoiseAnalysisService
from PLIFv2.tools.image_tools import create_colorbar, export_gif, top_k_values, transform_npy
from PLIFv2.ui.image_display import ImageDisplay
from PLIFv2.ui.app_window import DISTRIBUTION_PAGE_KEY, MANUAL_MIXING_PAGE_KEY, PLIFAppWindow


class WorkflowSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_average_multiple_experiments_and_segments(self) -> None:
        source = self.root / "raw"
        for experiment in ["实验编号1", "实验编号2"]:
            folder = source / experiment
            folder.mkdir(parents=True)
            for index in range(4):
                image = np.zeros((8, 10, 3), dtype=np.uint16)
                image[:, :, 0] = 5000
                image[:, :, 1] = (index + 1) * 100
                image[:, :, 2] = 6000
                write_tif(folder / f"capture_{index}.tif", image)
        result = AverageImageService().run(
            AverageSettings(
                source_folder=source,
                output_folder=self.root / "avg",
                experiment_numbers="1,2",
                use_frame_segments=True,
                frame_segments="1:3",
                denoise_threshold=250,
                best_process_count=3,
            )
        )
        self.assertEqual(len(result.items), 2)
        self.assertEqual(result.selected_count, 4)
        self.assertGreaterEqual(result.calculation_seconds, 0)
        self.assertTrue((self.root / "avg" / "实验编号1.npy").exists())
        self.assertAlmostEqual(float(np.load(self.root / "avg" / "实验编号1.npy")[0, 0]), 150)
        self.assertTrue((source / "实验编号1" / "0000.TIF").exists())

    def test_average_time_range_when_segments_disabled(self) -> None:
        folder = self.root / "实验编号17"
        folder.mkdir()
        for index in range(4):
            image = np.zeros((5, 6, 3), dtype=np.uint16)
            image[:, :, 1] = (index + 1) * 100
            write_tif(folder / f"{index:04d}.tif", image)
        result = AverageImageService().run(
            AverageSettings(
                source_folder=folder,
                output_folder=self.root / "avg",
                experiment_numbers="17",
                time_start=0.25,
                time_end=0.75,
                sample_frequency=4,
                use_frame_segments=False,
                frame_segments="invalid-but-ignored",
                denoise_threshold=0,
                best_process_count=1,
            )
        )
        self.assertEqual(result.selected_count, 2)
        self.assertAlmostEqual(float(np.load(self.root / "avg" / "实验编号17.npy")[0, 0]), 250)

    def test_average_uses_source_root_when_experiment_numbers_are_disabled(self) -> None:
        folder = self.root / "custom-name"
        folder.mkdir()
        for index, value in enumerate([100, 300]):
            image = np.zeros((4, 5, 3), dtype=np.uint16)
            image[:, :, 1] = value
            write_tif(folder / f"{index:04d}.tif", image)

        result = AverageImageService().run(
            AverageSettings(
                source_folder=folder,
                output_folder=self.root / "avg",
                experiment_numbers="17,18",
                use_experiment_numbers=False,
                time_start=0,
                time_end=1,
                sample_frequency=2,
                best_process_count=1,
            )
        )
        self.assertEqual(len(result.items), 1)
        self.assertTrue((self.root / "avg" / "custom-name.npy").exists())
        self.assertAlmostEqual(float(np.load(self.root / "avg" / "custom-name.npy")[0, 0]), 200)

    def test_average_uint8_threshold_is_applied_before_scaling(self) -> None:
        folder = self.root / "uint8"
        folder.mkdir()
        first = np.zeros((2, 3, 3), dtype=np.uint8)
        second = np.zeros((2, 3, 3), dtype=np.uint8)
        first[:, :, 1] = 2
        second[:, :, 1] = 3
        cv2.imwrite(str(folder / "0000.png"), first)
        cv2.imwrite(str(folder / "0001.png"), second)

        average = average_images(
            [folder / "0000.png", folder / "0001.png"],
            best_process_count=1,
            denoise_threshold=2,
        )

        self.assertTrue(np.all(average == 384))

    def test_average_uint8_multiprocessing_matches_reference(self) -> None:
        folder = self.root / "uint8-multiprocessing"
        folder.mkdir()
        rng = np.random.default_rng(17)
        green_frames = rng.integers(0, 16, size=(7, 6, 8), dtype=np.uint8)
        paths = []
        for index, green in enumerate(green_frames):
            image = np.empty((6, 8, 3), dtype=np.uint8)
            image[:, :, 0] = 201
            image[:, :, 1] = green
            image[:, :, 2] = 99
            path = folder / f"{index:04d}.png"
            cv2.imwrite(str(path), image)
            paths.append(path)

        filtered = green_frames.astype(np.float64)
        filtered[filtered <= 2] = 0
        expected = filtered.mean(axis=0) * 256
        single = average_images(paths, best_process_count=1, denoise_threshold=2)
        multiple = average_images(paths, best_process_count=3, denoise_threshold=2)

        np.testing.assert_array_equal(single, expected)
        np.testing.assert_array_equal(multiple, expected)

    def test_average_batch_uses_balanced_shared_tasks(self) -> None:
        image_sets = [
            [Path(f"experiment_{experiment}_{frame}.png") for frame in range(12)]
            for experiment in range(3)
        ]
        tasks = split_average_tasks(image_sets, best_process_count=10)
        self.assertGreater(len(tasks), 10)
        for experiment in range(3):
            assigned = [path for task_experiment, paths in tasks if task_experiment == experiment for path in paths]
            self.assertEqual(assigned, image_sets[experiment])

        equal_tasks = split_average_tasks(image_sets, best_process_count=1)
        self.assertEqual(len(equal_tasks), 3)

    def test_average_batch_matches_individual_averages(self) -> None:
        folder = self.root / "batch-average"
        folder.mkdir()
        image_sets = []
        for experiment, values in enumerate([(4, 8, 12), (20, 24, 28)]):
            paths = []
            for frame, value in enumerate(values):
                image = np.zeros((4, 5, 3), dtype=np.uint8)
                image[:, :, 1] = value
                path = folder / f"experiment_{experiment}_{frame}.png"
                cv2.imwrite(str(path), image)
                paths.append(path)
            image_sets.append(paths)

        averages = average_image_sets(image_sets, best_process_count=3, denoise_threshold=0)
        self.assertAlmostEqual(float(averages[0][0, 0]), 8 * 256)
        self.assertAlmostEqual(float(averages[1][0, 0]), 24 * 256)

    def test_average_reports_experiment_when_worker_cannot_read_image(self) -> None:
        source = self.root / "raw"
        good_folder = source / "实验编号1"
        bad_folder = source / "实验编号2"
        good_folder.mkdir(parents=True)
        bad_folder.mkdir(parents=True)
        write_tif(good_folder / "0000.tif", np.ones((4, 5), dtype=np.uint16))
        (bad_folder / "0000.png").write_bytes(b"not an image")

        with self.assertRaisesRegex(RuntimeError, "实验编号2"):
            AverageImageService().run(
                AverageSettings(
                    source_folder=source,
                    experiment_numbers="1,2",
                    use_frame_segments=True,
                    frame_segments="0:1",
                    best_process_count=2,
                )
            )

    def test_excel_gain_correction(self) -> None:
        images = self.root / "images"
        matrices = self.root / "matrices"
        images.mkdir()
        matrices.mkdir()
        np.save(images / "实验编号1.npy", np.ones((6, 7)) * 10)
        np.save(matrices / "Gain60.npy", np.ones((6, 7)) * 2)
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append([1, None, None, 0.6])
        excel = self.root / "experiments.xlsx"
        workbook.save(excel)
        settings = GainCorrectionSettings(image_folder=images, matrix_folder=matrices, output_folder=self.root / "corrected", excel_path=excel)
        result = IntensifierCorrectionService().apply_by_excel(settings)
        self.assertEqual(result.processed_count, 1)
        self.assertAlmostEqual(float(np.load(self.root / "corrected" / "实验编号1.npy")[0, 0]), 20)

    def test_intensifier_fit_uses_v1_single_image_signal_path(self) -> None:
        folder = self.root / "intensifier-fit"
        folder.mkdir()
        x_values = [100, 200, 300]
        green_stack = []
        for exposure in x_values:
            green = np.array([[2, 4], [8, 16]], dtype=np.uint16) * exposure
            image = np.zeros((2, 2, 3), dtype=np.uint16)
            image[:, :, 0] = 1
            image[:, :, 1] = green
            image[:, :, 2] = 65535
            path = folder / f"{exposure}ns.tif"
            write_tif(path, image)
            green_stack.append(green)

        result = IntensifierCorrectionService().fit_matrix_from_folder(folder, self.root / "fit-output")
        expected_stack = np.dstack(green_stack)
        expected_k = np.tensordot(expected_stack.astype(np.float64), np.array(x_values), axes=([2], [0]))
        expected_k /= float(np.dot(x_values, x_values))
        expected = np.where(expected_k > 0, np.max(expected_k) / expected_k, 0)

        self.assertEqual(result.image_count, 3)
        np.testing.assert_array_equal(np.load(result.outputs["npy"]), expected)

    def test_davis_prepare_and_import(self) -> None:
        corrected = self.root / "corrected"
        corrected.mkdir()
        write_tif(corrected / "实验编号1-GAIN2.tif", np.ones((5, 6), dtype=np.uint16) * 20)
        settings = DavisSettings(corrected_folder=corrected, davis_input_folder=self.root / "davis_in")
        prepared = DavisService().prepare_input_folder(settings)
        self.assertEqual(prepared.copied_count, 1)
        export = self.root / "davis_export"
        export.mkdir()
        write_tif(export / "prefix_实验编号1-GAIN2_Image.tif", np.ones((5, 6), dtype=np.uint16) * 20)
        settings.davis_export_folder = export
        settings.output_folder = self.root / "davis_out"
        imported = DavisService().import_export_folder(settings)
        self.assertEqual(imported.npy_count, 1)
        self.assertAlmostEqual(float(np.load(self.root / "davis_out" / "实验编号1.npy")[0, 0]), 10)

    def test_davis_rejects_duplicate_experiment_output(self) -> None:
        export = self.root / "davis-duplicate"
        export.mkdir()
        write_tif(export / "a_实验编号1-GAIN2_Image.tif", np.ones((2, 2), dtype=np.uint16))
        write_tif(export / "b_实验编号1-GAIN2_Image.tif", np.ones((2, 2), dtype=np.uint16))
        with self.assertRaisesRegex(FileExistsError, "Duplicate Davis output"):
            DavisService().import_export_folder(DavisSettings(davis_export_folder=export))

    def test_davis_restores_decimal_gain(self) -> None:
        export = self.root / "davis-decimal-gain"
        export.mkdir()
        write_tif(
            export / "实验编号7-GAIN2.5_Image.tif",
            np.ones((2, 2), dtype=np.uint16) * 25,
        )
        output = self.root / "davis-decimal-output"
        result = DavisService().import_export_folder(
            DavisSettings(davis_export_folder=export, output_folder=output)
        )
        self.assertEqual(result.npy_count, 1)
        self.assertAlmostEqual(float(np.load(output / "实验编号7.npy")[0, 0]), 10)

    def test_calibration_noise_and_matrix_tools(self) -> None:
        folder = self.root / "calibration"
        folder.mkdir()
        base = np.arange(1, 49, dtype=np.float64).reshape(6, 8)
        for exposure in [100, 200, 300]:
            np.save(folder / f"{exposure}ns.npy", base * exposure)
        service = CalibrationService()
        settings = CalibrationSettings(source_folder=folder, output_folder=self.root / "cal_out", analysis_radius=2)
        result = service.analyze(settings)
        self.assertLess(float(np.max(result.error_map)), 1e-12)
        self.assertTrue(service.save_correction(settings, result)["npy"].exists())
        settings.compare_folder = self.root / "compare"
        settings.compare_folder.mkdir()
        np.save(settings.compare_folder / "a.npy", np.ones((6, 8)))
        np.save(settings.compare_folder / "b.npy", np.ones((6, 8)) * 1.1)
        settings.compare_center_x = 4
        settings.compare_center_y = 3
        settings.compare_radius = 3
        comparison = service.compare_matrices(settings)
        self.assertGreater(comparison.max_relative_ptp, 0)
        noise = NoiseAnalysisService().run(NoiseSettings(source_folder=folder, frame_start=0, frame_end=3, process_count=1))
        self.assertEqual(len(noise.max_values), 3)

    def test_noise_uses_green_channel_for_color_frames(self) -> None:
        folder = self.root / "noise-green"
        folder.mkdir()
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        image[:, :, 0] = 255
        image[:, :, 1] = 7
        image[:, :, 2] = 200
        write_tif(folder / "0000.tif", image)

        result = NoiseAnalysisService().run(NoiseSettings(source_folder=folder, frame_start=0, frame_end=1, process_count=1))
        self.assertEqual(float(result.maximum), 7.0)

    def test_manual_display_uses_shared_viewport_for_scalar_and_color_images(self) -> None:
        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest("Tk is unavailable")
        root.withdraw()
        try:
            display = ImageDisplay(root, "result")
            scalar = np.array([[0, 256], [512, 1024]], dtype=np.float64)
            display.update(scalar, gain=1, zoom=2, crosshair=(1, 1, 1), ellipse=(1, 1, 1, 1, 1), special_points=[])
            self.assertEqual(display.display.shape, (2, 2, 3))
            self.assertEqual(display.value_image.shape, (2, 2))

            color = np.full((3, 4, 3), 50, dtype=np.uint8)
            display.update(color, colormap="none", zoom=1)
            self.assertEqual(display.display.shape, (3, 4, 3))
            self.assertEqual(display.value_image.shape, (3, 4, 3))
        finally:
            root.destroy()

    def test_manual_and_distribution_pages_load(self) -> None:
        try:
            app = PLIFAppWindow()
        except tk.TclError:
            self.skipTest("Tk is unavailable")
        try:
            app.withdraw()
            app._show_page(MANUAL_MIXING_PAGE_KEY)
            app.update_idletasks()
            manual_page = app.pages[MANUAL_MIXING_PAGE_KEY].ensure_loaded()
            self.assertEqual(app.active_page_key, MANUAL_MIXING_PAGE_KEY)
            self.assertEqual(
                [manual_page.result_tabs.tab(tab_id, "text") for tab_id in manual_page.result_tabs.tabs()],
                ["混合特性", "平移搜索"],
            )

            app._show_page(DISTRIBUTION_PAGE_KEY)
            app.update_idletasks()
            self.assertEqual(app.active_page_key, DISTRIBUTION_PAGE_KEY)
            self.assertIsNotNone(app.pages[DISTRIBUTION_PAGE_KEY].content)
        finally:
            app.destroy()

    def test_shift_manual_and_distribution(self) -> None:
        folder = self.root / "mixing"
        folder.mkdir()
        y, x = np.indices((80, 120))
        f = np.exp(-((x - 45) ** 2 + (y - 40) ** 2) / 500) * 1000
        o = np.exp(-((x - 70) ** 2 + (y - 40) ** 2) / 500) * 900
        fo = f + o
        for number, image in [(1, f), (2, o), (3, fo)]:
            np.save(folder / f"实验编号{number}.npy", image)
        service = MixingService()
        shift = service.run_shift_search(ShiftSearchSettings(image_folder=folder, f_exp_no="1", o_exp_no="2", fo_exp_no="3", f_min=-1, f_max=1, o_min=-1, o_max=1, ellipse_x=58, ellipse_y=40, major_axis=50, minor_axis=30, process_count=1))
        self.assertEqual(shift.abs_matrix.shape, (3, 3))
        manual_settings = ManualMixingSettings(image_folder=folder, f_exp_no="1", o_exp_no="2", fo_exp_no="3", center_xf=45, center_yf=40, center_xo=70, center_yo=40, center_xfo=58, center_yfo=40, major_axis=50, minor_axis=30, global_of=1, zoom=4, output_folder=self.root / "manual")
        manual = service.run_manual_display(manual_settings)
        self.assertEqual(len(manual.images), 11)
        self.assertTrue(service.export_mix_curve(manual_settings, manual).exists())
        distribution_settings = DistributionSettings(
            image_folder=folder,
            f_exp_no="1",
            o_exp_no="2",
            fo_exp_no="3",
            center_xf=45,
            center_yf=40,
            center_xo=70,
            center_yo=40,
            center_xfo=58,
            center_yfo=40,
            q_center_up_f=10,
            q_center_down_f=70,
            q_center_up_o=10,
            q_center_down_o=70,
            q_center_up_fo=10,
            q_center_down_fo=70,
            max_k=5,
            output_folder=self.root / "distribution",
        )
        distribution_service = DistributionFitService()
        distribution = distribution_service.run(distribution_settings, frozenset(), "ori_img")
        self.assertEqual(len(distribution.images), 3)
        self.assertTrue(distribution_service.export(distribution_settings, distribution).exists())

    def test_experiment_triplet_rejects_shape_mismatch(self) -> None:
        folder = self.root / "shape-mismatch"
        folder.mkdir()
        np.save(folder / "实验编号1.npy", np.ones((3, 4)))
        np.save(folder / "实验编号2.npy", np.ones((3, 4)))
        np.save(folder / "实验编号3.npy", np.ones((4, 4)))

        with self.assertRaisesRegex(ValueError, "尺寸不一致"):
            load_experiment_triplet(folder, "1", "2", "3")

    def test_tool_algorithms(self) -> None:
        array = np.arange(12).reshape(3, 4)
        npy = self.root / "array.npy"
        np.save(npy, array)
        self.assertEqual(top_k_values(array, 1)[0][0], 11)
        transformed = transform_npy(npy, self.root / "normalized.npy", "normalize")
        self.assertAlmostEqual(float(np.max(np.load(transformed))), 1)
        colorbar = create_colorbar(120, 20, 0, 1, 10, 10)
        self.assertEqual(colorbar.shape[0], 120)
        frames = self.root / "frames"
        frames.mkdir()
        for index in range(3):
            write_tif(frames / f"{index:04d}.tif", np.ones((20, 30), dtype=np.uint16) * index * 100)
        gif = export_gif(frames, self.root / "test.gif", end=3)
        self.assertTrue(gif.exists())

    def test_single_settings_file_keeps_ui_state(self) -> None:
        path = self.root / "settings.json"
        app_settings = load_settings(path)
        app_settings.average = AverageSettings(source_folder=self.root / "raw")
        app_settings.ui.app["active_page"] = "图片播放器"
        app_settings.ui.pages["图片播放器"] = {"entry": {"kind": "text", "value": "24"}}
        save_settings(app_settings, path)

        restored = load_settings(path)
        self.assertEqual(restored.average.source_folder, self.root / "raw")
        self.assertEqual(restored.ui.app["active_page"], "图片播放器")
        self.assertEqual(restored.ui.pages["图片播放器"]["entry"]["value"], "24")

    def test_settings_migrates_legacy_ui_state_once(self) -> None:
        path = self.root / "settings.json"
        path.write_text("{}", encoding="utf-8")
        legacy = {"app": {"active_page": "图像查看器"}, "pages": {"图像查看器": {}}}
        path.with_name("ui_state.json").write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")

        restored = load_settings(path)
        self.assertEqual(restored.ui.app["active_page"], "图像查看器")


if __name__ == "__main__":
    unittest.main()
