from copy import deepcopy
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest
from unittest.mock import patch

from PLIFv2.application.app_state import AppState
from PLIFv2.core.config_io import load_settings, save_settings
from PLIFv2.domain.settings import AppSettings
from PLIFv2.ui.app_window import PLIFAppWindow


class UIStateIntegrationTests(unittest.TestCase):
    def test_actual_pages_roundtrip_semantic_state_and_invalid_input_is_atomic(self):
        state = AppState(settings=AppSettings())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            state.save = lambda: save_settings(state.settings, path)
            with patch("PLIFv2.ui.app_window.AppState", return_value=state):
                app = PLIFAppWindow()
            app.withdraw()
            try:
                for page in app.pages.values():
                    page.ensure_loaded()
                average = app.pages["1 平均图"].content
                average.vars["time_start"].set("1.25")
                average.vars["time_end"].set("8.75")
                average.use_experiment_numbers.set(True)
                average._save_to_settings()
                old = deepcopy(average.settings)
                average.vars["time_end"].set("invalid")
                average.vars["time_start"].set("3.5")
                with self.assertRaises(ValueError):
                    average._save_to_settings()
                self.assertEqual(average.settings, old)
                average.vars["time_end"].set("8.75")
                average.vars["time_start"].set("1.25")
                for page in app.pages.values():
                    page.capture_state()
                for name, values in state.settings.ui.pages.items():
                    with self.subTest(page=name):
                        self.assertFalse(any(key.startswith("!") for key in values))
                state.save()
                restored = load_settings(path)
                self.assertEqual(restored.average.time_start, 1.25)
                self.assertEqual(restored.average.time_end, 8.75)
                self.assertEqual(restored.ui.pages, state.settings.ui.pages)
                average.vars["time_start"].set("0")
                average.use_experiment_numbers.set(False)
                average._sync_experiment_state()
                app.ui_state.restore_page("1 平均图", average)
                self.assertEqual(average.vars["time_start"].get(), "1.25")
                self.assertTrue(average.use_experiment_numbers.get())
                self.assertFalse(average.experiment_entry.instate(["disabled"]))
            finally:
                app.actions.close()
                app.destroy()


if __name__ == "__main__":
    unittest.main()
