from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from PLIFMix.application.actions import PLIFActions
from PLIFMix.core.result_snapshot import ResultSnapshot, require_current_result
from PLIFMix.domain.settings import AppSettings, DistributionSettings
from PLIFMix.services.distribution_service import DistributionFitResult
from PLIFMix.ui.pages.distribution_page import DistributionPage


class ResultSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.settings = DistributionSettings(f_exp_no="1", o_exp_no="2", fo_exp_no="3")

    def test_snapshot_is_independent_and_allows_output_changes(self):
        snapshot = ResultSnapshot.capture(self.settings, ())
        self.settings.save_name = "DR1_U40"
        self.settings.fit_result_path = Path("chosen.xlsx")
        self.settings.image_save_folder = Path("saved-images")
        self.settings.gain = 3
        exported = snapshot.export_settings(self.settings)
        self.assertEqual(exported.save_name, "DR1_U40")
        self.assertEqual(exported.fit_result_path, Path("chosen.xlsx"))
        self.assertEqual(exported.image_save_folder, Path("saved-images"))
        self.assertEqual(dict(snapshot.values)["save_name"], "")
        self.settings.f_exp_no = "4"
        with self.assertRaisesRegex(ValueError, "实验编号"):
            snapshot.export_settings(self.settings)

    def test_single_image_save_folder_does_not_invalidate_a_result(self):
        snapshot = ResultSnapshot.capture(self.settings, ())
        changed = deepcopy(self.settings)
        changed.image_save_folder = Path("another-folder")
        snapshot.require_current(changed)

    def test_every_compute_field_invalidates_old_result(self):
        snapshot = ResultSnapshot.capture(self.settings, ())
        for field in ("denoise_ratio", "f_shift", "center_xf", "q_center_up_f", "rotate_angle_f", "max_k"):
            changed = deepcopy(self.settings)
            setattr(changed, field, getattr(changed, field) + 1)
            with self.subTest(field=field), self.assertRaises(ValueError):
                snapshot.require_current(changed)

    def test_source_replacement_invalidates(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "source.npy"
            path.write_bytes(b"old")
            snapshot = ResultSnapshot.capture(self.settings, (path,))
            snapshot.require_current(self.settings)
            path.write_bytes(b"new image")
            with self.assertRaisesRegex(ValueError, "源图像"):
                snapshot.require_current(self.settings)

    def test_actions_use_submission_settings_not_later_global_state(self):
        state = SimpleNamespace(settings=AppSettings(), save=Mock())
        actions = PLIFActions(state)
        self.addCleanup(actions.close)
        cached_result = DistributionFitResult("", "", [], (), {}, (), ())
        actions.distribution_service.run = Mock(return_value=cached_result)
        submitted = deepcopy(self.settings)
        state.settings.distribution.f_exp_no = "999"
        result = actions.run_distribution_display(frozenset(), "ori_img", False, submitted)
        self.assertEqual(actions.distribution_service.run.call_args.args[0].f_exp_no, "1")
        self.assertEqual(dict(result.snapshot.values)["f_exp_no"], "1")
        self.assertIsNone(cached_result.snapshot)
        self.assertIsNot(result, cached_result)
        self.assertIs(result.images, cached_result.images)
        actions.distribution_service.write_fit_results_to_excel = Mock()
        with self.assertRaises(ValueError):
            actions.write_distribution_fit_results(result)
        actions.distribution_service.write_fit_results_to_excel.assert_not_called()

    def test_page_rejects_pending_result(self):
        page = SimpleNamespace(request_id=2, completed_request_id=1)
        with self.assertRaisesRegex(ValueError, "等待"):
            DistributionPage._require_current_result(page)

    def test_missing_provenance_cannot_be_exported(self):
        with self.assertRaises(ValueError):
            require_current_result(SimpleNamespace(), self.settings)


if __name__ == "__main__":
    unittest.main()
