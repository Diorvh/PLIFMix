from __future__ import annotations

import json
import tkinter as tk
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from PLIFv2.core.config_io import load_settings, save_settings
from PLIFv2.core.ui_state_io import UIStateStore
from PLIFv2.domain.settings import (
    SETTINGS_SCHEMA_VERSION,
    AverageSettings,
    DistributionSettings,
    UiStateSettings,
)
from PLIFv2.ui.settings_binding import set_dataclass_text_field, set_dataclass_text_fields


class SettingsPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_modern_average_times_are_restored_without_frame_recalculation(self) -> None:
        for schema_version in (None, SETTINGS_SCHEMA_VERSION):
            path = self.root / f"settings-{schema_version}.json"
            payload = {
                "average": {
                    "time_start": 1.25,
                    "time_end": 2.5,
                    "frame_start": 100,
                    "frame_end": 300,
                    "sample_frequency": 100,
                },
                "ui": {"app": {}, "pages": {}},
            }
            if schema_version is not None:
                payload["schema_version"] = schema_version
            path.write_text(json.dumps(payload), encoding="utf-8")

            restored = load_settings(path)

            self.assertEqual(restored.average.time_start, 1.25)
            self.assertEqual(restored.average.time_end, 2.5)

    def test_legacy_average_frames_still_migrate_missing_times(self) -> None:
        path = self.root / "legacy-settings.json"
        path.write_text(
            json.dumps(
                {
                    "average": {
                        "frame_start": 100,
                        "frame_end": 300,
                        "sample_frequency": 100,
                    }
                }
            ),
            encoding="utf-8",
        )

        restored = load_settings(path)

        self.assertEqual(restored.average.time_start, 1.0)
        self.assertEqual(restored.average.time_end, 3.0)
        self.assertEqual(restored.schema_version, SETTINGS_SCHEMA_VERSION)

    def test_save_writes_explicit_schema_version(self) -> None:
        path = self.root / "settings.json"

        save_settings(load_settings(path), path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], SETTINGS_SCHEMA_VERSION)

    def test_legacy_ui_file_is_used_once_after_settings_is_saved(self) -> None:
        path = self.root / "settings.json"
        legacy_path = path.with_name("ui_state.json")
        path.write_text("{}", encoding="utf-8")
        legacy_path.write_text(
            json.dumps({"app": {"active_page": "图像查看器"}, "pages": {"图像查看器": {}}}),
            encoding="utf-8",
        )

        first = load_settings(path)
        save_settings(first, path)
        legacy_path.write_text(
            json.dumps({"app": {"active_page": "被忽略的旧值"}, "pages": {}}),
            encoding="utf-8",
        )

        second = load_settings(path)

        self.assertEqual(second.ui.app["active_page"], "图像查看器")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["schema_version"], SETTINGS_SCHEMA_VERSION)

    def test_semantic_state_survives_variable_and_nested_component_reordering(self) -> None:
        interpreter = tk.Tcl()
        first_root = _semantic_root(
            interpreter,
            first="first-captured",
            second="second-captured",
            enabled=True,
            source="source-captured",
            radius="42",
        )
        store = UIStateStore(UiStateSettings())

        store.capture_page("page", first_root)
        saved = store.data.pages["page"]

        self.assertIn("vars.first", saved)
        self.assertIn("vars.second", saved)
        self.assertIn("enabled", saved)
        self.assertIn("calibration_workspace.source_var", saved)
        self.assertIn("calibration_workspace.compare_form.vars.radius", saved)
        self.assertFalse(any(key.startswith("!") for key in saved))

        second_root = _semantic_root(
            interpreter,
            first="first-default",
            second="second-default",
            enabled=False,
            source="source-default",
            radius="0",
            reverse=True,
        )
        store.restore_page("page", second_root)

        self.assertEqual(second_root.vars["first"].get(), "first-captured")
        self.assertEqual(second_root.vars["second"].get(), "second-captured")
        self.assertTrue(second_root.enabled.get())
        self.assertEqual(second_root.calibration_workspace.source_var.get(), "source-captured")
        self.assertEqual(second_root.calibration_workspace.compare_form.vars["radius"].get(), "42")

    def test_legacy_widget_paths_are_recovered_and_promoted_to_semantic_state(self) -> None:
        interpreter = tk.Tcl()
        value = tk.StringVar(master=interpreter, value="default")
        entry = _LegacyEntry(value)
        root = SimpleNamespace(
            _w=".page",
            persist_widget_values=True,
            vars={"value": value},
            winfo_children=lambda: [entry],
        )
        store = UIStateStore(
            UiStateSettings(
                pages={"page": {"!entry": {"kind": "text", "value": "recovered"}}}
            )
        )

        store.restore_page("page", root)

        self.assertEqual(value.get(), "recovered")
        self.assertIn("vars.value", store.data.pages["page"])
        self.assertFalse(any(key.startswith("!") for key in store.data.pages["page"]))

    def test_disabled_widget_persistence_keeps_custom_hook_contract(self) -> None:
        interpreter = tk.Tcl()
        variable = tk.StringVar(master=interpreter, value="not-captured")
        captured: list[dict[str, object]] = []
        root = SimpleNamespace(
            persist_widget_values=False,
            variable=variable,
            export_ui_state=lambda: {"custom": True},
            restore_ui_state=lambda state: captured.append(state),
        )
        store = UIStateStore(UiStateSettings())

        store.capture_page("page", root)
        store.restore_page("page", root)

        self.assertEqual(store.data.pages["page"], {"__custom__": {"custom": True}})
        self.assertEqual(captured, [{"custom": True}])

    def test_bulk_binding_validates_before_mutating_and_preserves_rotation_rule(self) -> None:
        settings = DistributionSettings(line_width=2, rotate_angle_f=7)

        set_dataclass_text_fields(settings, {"line_width": "4", "rotate_angle_f": "9"})
        self.assertEqual(settings.line_width, 4)
        self.assertEqual(settings.rotate_angle_f, 9)

        with self.assertRaisesRegex(ValueError, "旋转角度必须是整数"):
            set_dataclass_text_fields(settings, {"rotate_angle_f": "4.5", "line_width": "invalid"})
        self.assertEqual(settings.line_width, 4)
        self.assertEqual(settings.rotate_angle_f, 9)

        with self.assertRaisesRegex(ValueError, "旋转角度必须是整数"):
            set_dataclass_text_field(settings, "rotate_angle_f", "4.5")
        self.assertEqual(settings.rotate_angle_f, 9)


def _semantic_root(
    interpreter: tk.Tcl,
    *,
    first: str,
    second: str,
    enabled: bool,
    source: str,
    radius: str,
    reverse: bool = False,
) -> SimpleNamespace:
    variables = {
        "first": tk.StringVar(master=interpreter, value=first),
        "second": tk.StringVar(master=interpreter, value=second),
    }
    if reverse:
        variables = {"second": variables["second"], "first": variables["first"]}
    compare_vars = {"radius": tk.StringVar(master=interpreter, value=radius)}
    return SimpleNamespace(
        persist_widget_values=True,
        vars=variables,
        enabled=tk.BooleanVar(master=interpreter, value=enabled),
        calibration_workspace=SimpleNamespace(
            source_var=tk.StringVar(master=interpreter, value=source),
            compare_form=SimpleNamespace(vars=compare_vars),
        ),
    )


class _LegacyEntry:
    _w = ".page.!entry"

    def __init__(self, variable: tk.StringVar):
        self.variable = variable

    def winfo_children(self) -> list[object]:
        return []

    def winfo_class(self) -> str:
        return "TEntry"

    def cget(self, name: str) -> str:
        return str(self.variable) if name == "textvariable" else ""

    def setvar(self, _name: str, value: object) -> None:
        self.variable.set(value)


if __name__ == "__main__":
    unittest.main()
