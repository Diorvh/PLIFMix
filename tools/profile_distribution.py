"""Small staged performance harness for the current distribution workflow.

The harness deliberately calls DistributionFitService.run directly. It does not
save settings, export files, or modify production modules on disk.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any

# Direct execution puts tools/ on sys.path, while the package lives one level
# below the repository parent.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT.parent))

import numpy as np
from concurrent.futures.process import BrokenProcessPool

from PLIFv2.algorithms import distribution as distribution_algorithms
from PLIFv2.core.config_io import default_config_path, load_settings
from PLIFv2.core.experiment_images import experiment_npy_path
from PLIFv2.domain.settings import DistributionSettings, dataclass_to_jsonable
from PLIFv2.services import distribution_service as distribution_service_module
from PLIFv2.services import distribution_workers as distribution_workers_module
from PLIFv2.services.distribution_service import DistributionFitResult, DistributionFitService
from PLIFv2.services.distribution_workers import (
    SurfaceFitTask,
    SurfaceFitWorkerResult,
)


_IMAGE_NAMES = ("f_img", "o_img", "fo_img")

# Keep references to the unmodified callables. Child processes install wrappers
# around these references once, on their first task.
_ORIGINAL_PREPARE = distribution_algorithms.prepare_surface_fit_data
_ORIGINAL_PURE = distribution_algorithms.fit_elliptical_student_prepared
_ORIGINAL_COMPACT = distribution_algorithms.fit_compact_elliptical_student_prepared
_ORIGINAL_OPTIMIZE = distribution_algorithms._optimize_surface
_ORIGINAL_WORKER = distribution_workers_module.fit_surface_models_worker

_ORIGINAL_SERVICE_CENTERLINE = distribution_service_module.fit_center_line
_ORIGINAL_SERVICE_FUNC_FIT_3 = distribution_service_module.func_fit_3
_ORIGINAL_SERVICE_STRAIGHTEN = distribution_service_module.straighten_center_line_image
_ORIGINAL_SERVICE_UNSTRAIGHTEN = distribution_service_module.unstraighten_center_line_image


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _error_record(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _array_fingerprint(array: np.ndarray | None) -> dict[str, Any] | None:
    if array is None:
        return None
    value = np.ascontiguousarray(np.asarray(array))
    return {
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "sum": _finite_float(np.sum(value)),
        "minimum": _finite_float(np.min(value)) if value.size else None,
        "maximum": _finite_float(np.max(value)) if value.size else None,
    }


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class _WorkerState:
    image_name: str
    prepare_s: float = 0.0
    pure_s: float = 0.0
    compact_s: float = 0.0
    phase: str = "worker"
    optimizers: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class WorkerTiming:
    pid: int
    image_name: str
    elapsed_s: float
    core_s: float
    fingerprint_s: float
    prepare_s: float
    pure_s: float
    compact_s: float
    optimizers: tuple[dict[str, Any], ...]
    selected_prediction: dict[str, Any] | None
    error: dict[str, str] | None


@dataclass(frozen=True)
class TimedWorkerResult:
    worker_result: SurfaceFitWorkerResult | None
    timing: WorkerTiming


_WORKER_STATE: _WorkerState | None = None
_WORKER_PATCHES: list[tuple[Any, str, Any]] | None = None


def _optimizer_record(
    phase: str,
    wall_s: float,
    evaluation_s: float,
    evaluation_calls: int,
    result: Any = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    return {
        "model": phase,
        "wall_s": wall_s,
        "evaluation_s": evaluation_s,
        "evaluation_calls": evaluation_calls,
        "nit": None if result is None else getattr(result, "nit", None),
        "nfev": None if result is None else getattr(result, "nfev", None),
        "status": None if result is None else getattr(result, "status", None),
        "success": None if result is None else bool(getattr(result, "success", False)),
        "objective": None if result is None else _finite_float(getattr(result, "fun", None)),
        "message": None if result is None else str(getattr(result, "message", "")),
        "error": None if error is None else _error_record(error),
    }


def _timed_prepare_surface_fit_data(
    target: np.ndarray,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    initial_scale_x: float,
    initial_scale_y: float,
    initial_exponent: float,
):
    state = _WORKER_STATE
    if state is None:
        return _ORIGINAL_PREPARE(
            target,
            pixel_x,
            pixel_y,
            initial_scale_x,
            initial_scale_y,
            initial_exponent,
        )
    started = time.perf_counter()
    try:
        return _ORIGINAL_PREPARE(
            target,
            pixel_x,
            pixel_y,
            initial_scale_x,
            initial_scale_y,
            initial_exponent,
        )
    finally:
        state.prepare_s += time.perf_counter() - started


def _timed_pure(fit_data):
    state = _WORKER_STATE
    if state is None:
        return _ORIGINAL_PURE(fit_data)
    state.phase = "pure"
    started = time.perf_counter()
    try:
        return _ORIGINAL_PURE(fit_data)
    finally:
        state.pure_s += time.perf_counter() - started


def _timed_compact(fit_data, initial_scale_x=None, initial_scale_y=None, initial_exponent=None):
    state = _WORKER_STATE
    if state is None:
        return _ORIGINAL_COMPACT(fit_data, initial_scale_x, initial_scale_y, initial_exponent)
    state.phase = "compact"
    started = time.perf_counter()
    try:
        return _ORIGINAL_COMPACT(fit_data, initial_scale_x, initial_scale_y, initial_exponent)
    finally:
        state.compact_s += time.perf_counter() - started


def _timed_optimize_surface(initial, bounds, predict, fit_data):
    state = _WORKER_STATE
    if state is None:
        return _ORIGINAL_OPTIMIZE(initial, bounds, predict, fit_data)

    evaluation_s = 0.0
    evaluation_calls = 0

    def timed_predict(parameters):
        nonlocal evaluation_s, evaluation_calls
        started = time.perf_counter()
        try:
            return predict(parameters)
        finally:
            evaluation_calls += 1
            evaluation_s += time.perf_counter() - started

    started = time.perf_counter()
    try:
        result = _ORIGINAL_OPTIMIZE(initial, bounds, timed_predict, fit_data)
    except Exception as exc:
        state.optimizers.append(
            _optimizer_record(
                state.phase,
                time.perf_counter() - started,
                evaluation_s,
                evaluation_calls,
                error=exc,
            )
        )
        raise
    state.optimizers.append(
        _optimizer_record(
            state.phase,
            time.perf_counter() - started,
            evaluation_s,
            evaluation_calls,
            result=result[0],
        )
    )
    return result


def _install_worker_patches() -> None:
    global _WORKER_PATCHES
    if _WORKER_PATCHES is not None:
        return
    patches = [
        (distribution_algorithms, "prepare_surface_fit_data", _timed_prepare_surface_fit_data),
        (distribution_algorithms, "fit_elliptical_student_prepared", _timed_pure),
        (distribution_algorithms, "fit_compact_elliptical_student_prepared", _timed_compact),
        (distribution_algorithms, "_optimize_surface", _timed_optimize_surface),
        (distribution_workers_module, "prepare_surface_fit_data", _timed_prepare_surface_fit_data),
        (distribution_workers_module, "fit_elliptical_student_prepared", _timed_pure),
        (distribution_workers_module, "fit_compact_elliptical_student_prepared", _timed_compact),
    ]
    _WORKER_PATCHES = []
    for module, name, replacement in patches:
        _WORKER_PATCHES.append((module, name, getattr(module, name)))
        setattr(module, name, replacement)


def _restore_worker_patches() -> None:
    global _WORKER_PATCHES
    if _WORKER_PATCHES is None:
        return
    for module, name, original in reversed(_WORKER_PATCHES):
        setattr(module, name, original)
    _WORKER_PATCHES = None


def timed_fit_surface_models_worker(task: SurfaceFitTask) -> TimedWorkerResult:
    """Picklable worker entry point; production worker behavior is unchanged."""
    _install_worker_patches()
    global _WORKER_STATE
    previous_state = _WORKER_STATE
    state = _WorkerState(task.image_name)
    _WORKER_STATE = state
    started = time.perf_counter()
    worker_result: SurfaceFitWorkerResult | None = None
    error: dict[str, str] | None = None
    try:
        try:
            worker_result = _ORIGINAL_WORKER(task)
        except Exception as exc:
            error = _error_record(exc)
        core_s = time.perf_counter() - started
        fingerprint_started = time.perf_counter()
        selected_prediction = (
            None if worker_result is None else _array_fingerprint(worker_result.selected_prediction)
        )
        fingerprint_s = time.perf_counter() - fingerprint_started
        timing = WorkerTiming(
            pid=os.getpid(),
            image_name=task.image_name,
            elapsed_s=time.perf_counter() - started,
            core_s=core_s,
            fingerprint_s=fingerprint_s,
            prepare_s=state.prepare_s,
            pure_s=state.pure_s,
            compact_s=state.compact_s,
            optimizers=tuple(state.optimizers),
            selected_prediction=selected_prediction,
            error=error,
        )
        return TimedWorkerResult(worker_result, timing)
    finally:
        _WORKER_STATE = previous_state


@dataclass
class _ImageTiming:
    name: str
    centerline_s: float = 0.0
    horizontal_s: float = 0.0
    vertical_s: float = 0.0
    straighten_s: float = 0.0
    restore_s: float = 0.0
    fit3_calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "centerline_s": self.centerline_s,
            "horizontal_fit_s": self.horizontal_s,
            "vertical_fit_s": self.vertical_s,
            "straighten_s": self.straighten_s,
            "restore_s": self.restore_s,
            "fit3_calls": self.fit3_calls,
        }


@dataclass
class _RunTiming:
    label: str
    image_mode: str
    show_bent: bool
    images: list[_ImageTiming] = field(default_factory=lambda: [_ImageTiming(name) for name in _IMAGE_NAMES])
    started: float = 0.0
    elapsed_s: float = 0.0
    pre_s: float = 0.0
    process_batch_s: float = 0.0
    post_s: float = 0.0
    batch_called: bool = False
    pool_reused: bool | None = None
    pool_map_s: float = 0.0
    worker_metrics: list[WorkerTiming] = field(default_factory=list)
    fallback_worker_metrics: list[WorkerTiming] = field(default_factory=list)
    process_errors: list[dict[str, str]] = field(default_factory=list)
    fallback: dict[str, Any] | None = None
    success: bool = False
    error: dict[str, str] | None = None
    result: dict[str, Any] | None = None
    next_centerline: int = 0
    active_image: int = 0
    restore_index: int = 0


_ACTIVE_RUN: _RunTiming | None = None


def _timed_centerline(*args, **kwargs):
    state = _ACTIVE_RUN
    if state is None:
        return _ORIGINAL_SERVICE_CENTERLINE(*args, **kwargs)
    index = min(state.next_centerline, len(state.images) - 1)
    state.next_centerline += 1
    state.active_image = index
    state.images[index].fit3_calls = 0
    started = time.perf_counter()
    try:
        return _ORIGINAL_SERVICE_CENTERLINE(*args, **kwargs)
    finally:
        state.images[index].centerline_s += time.perf_counter() - started


def _timed_func_fit_3(*args, **kwargs):
    state = _ACTIVE_RUN
    if state is None:
        return _ORIGINAL_SERVICE_FUNC_FIT_3(*args, **kwargs)
    image = state.images[state.active_image]
    call_number = image.fit3_calls
    image.fit3_calls += 1
    started = time.perf_counter()
    try:
        return _ORIGINAL_SERVICE_FUNC_FIT_3(*args, **kwargs)
    finally:
        elapsed = time.perf_counter() - started
        if call_number == 0:
            image.horizontal_s += elapsed
        else:
            image.vertical_s += elapsed


def _timed_straighten(*args, **kwargs):
    state = _ACTIVE_RUN
    if state is None:
        return _ORIGINAL_SERVICE_STRAIGHTEN(*args, **kwargs)
    started = time.perf_counter()
    try:
        return _ORIGINAL_SERVICE_STRAIGHTEN(*args, **kwargs)
    finally:
        state.images[state.active_image].straighten_s += time.perf_counter() - started


def _timed_unstraighten(*args, **kwargs):
    state = _ACTIVE_RUN
    if state is None:
        return _ORIGINAL_SERVICE_UNSTRAIGHTEN(*args, **kwargs)
    index = min(state.restore_index, len(state.images) - 1)
    state.restore_index += 1
    started = time.perf_counter()
    try:
        return _ORIGINAL_SERVICE_UNSTRAIGHTEN(*args, **kwargs)
    finally:
        state.images[index].restore_s += time.perf_counter() - started


@contextmanager
def _parent_timing_patches():
    patches = [
        (distribution_service_module, "fit_center_line", _timed_centerline),
        (distribution_service_module, "func_fit_3", _timed_func_fit_3),
        (distribution_service_module, "straighten_center_line_image", _timed_straighten),
        (distribution_service_module, "unstraighten_center_line_image", _timed_unstraighten),
    ]
    originals = [(module, name, getattr(module, name)) for module, name, _ in patches]
    try:
        for module, name, replacement in patches:
            setattr(module, name, replacement)
        yield
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)


def _worker_dict(metric: WorkerTiming) -> dict[str, Any]:
    return {
        "pid": metric.pid,
        "image": metric.image_name,
        "elapsed_s": metric.elapsed_s,
        "core_s": metric.core_s,
        "fingerprint_s": metric.fingerprint_s,
        "prepare_s": metric.prepare_s,
        "pure_s": metric.pure_s,
        "compact_s": metric.compact_s,
        "optimizers": list(metric.optimizers),
        "selected_prediction": metric.selected_prediction,
        "error": metric.error,
    }


def _result_fingerprints(result: DistributionFitResult) -> dict[str, Any]:
    return {
        "fit_rows": result.fit_rows,
        "fit_rows_sha256": _json_sha256(result.fit_rows),
        "warnings": list(result.warnings),
        "images": {
            item.name: _array_fingerprint(item.image)
            for item in result.images
        },
    }


def _run_dict(run: _RunTiming) -> dict[str, Any]:
    worker_elapsed_sum = sum(item.elapsed_s for item in run.worker_metrics)
    worker_core_sum = sum(item.core_s for item in run.worker_metrics)
    fallback_elapsed_sum = sum(item.elapsed_s for item in run.fallback_worker_metrics)
    return {
        "label": run.label,
        "image_mode": run.image_mode,
        "show_bent": run.show_bent,
        "success": run.success,
        "error": run.error,
        "elapsed_s": run.elapsed_s,
        "wall_s": {
            "pre_s": run.pre_s,
            "process_batch_s": run.process_batch_s,
            "post_s": run.post_s,
        },
        "parent_per_image": {
            item.name: item.as_dict() for item in run.images
        },
        "process_batch": {
            "called": run.batch_called,
            "pool_reused": run.pool_reused,
            "pool_map_s": run.pool_map_s,
            "worker_elapsed_sum_s": worker_elapsed_sum,
            "worker_core_sum_s": worker_core_sum,
            "worker_pids": sorted({item.pid for item in run.worker_metrics}),
            "workers": [_worker_dict(item) for item in run.worker_metrics],
            "errors": run.process_errors,
            "fallback": run.fallback,
            "fallback_worker_elapsed_sum_s": fallback_elapsed_sum,
            "fallback_workers": [_worker_dict(item) for item in run.fallback_worker_metrics],
        },
        "result": run.result,
    }


class TimedDistributionFitService(DistributionFitService):
    """Minimal measurement subclass that still uses the production pool path."""

    def __init__(self):
        super().__init__()
        self.run_records: list[_RunTiming] = []
        self._next_label = "run"

    def run_stage(
        self,
        label: str,
        settings: DistributionSettings,
        image_mode: str = "fit_img",
        show_bent: bool = False,
    ) -> DistributionFitResult:
        self._next_label = label
        try:
            return self.run(settings, frozenset(), image_mode, show_bent)
        finally:
            self._next_label = "run"

    def run(self, settings, annotations=frozenset(), image_mode="ori_img", show_bent=None):
        global _ACTIVE_RUN
        timing = _RunTiming(self._next_label, image_mode, bool(show_bent))
        timing.started = time.perf_counter()
        self.run_records.append(timing)
        previous = _ACTIVE_RUN
        _ACTIVE_RUN = timing
        try:
            result = super().run(settings, annotations, image_mode, show_bent)
        except Exception as exc:
            timing.error = _error_record(exc)
            raise
        else:
            timing.success = True
        finally:
            timing.elapsed_s = time.perf_counter() - timing.started
            if timing.batch_called:
                timing.post_s = max(
                    0.0,
                    timing.elapsed_s - timing.pre_s - timing.process_batch_s,
                )
            else:
                timing.pre_s = timing.elapsed_s
            _ACTIVE_RUN = previous
        timing.result = _result_fingerprints(result)
        return result

    def _fit_surface_models_batch(self, tasks: list[SurfaceFitTask]):
        timing = _ACTIVE_RUN
        started = time.perf_counter()
        if timing is not None:
            timing.batch_called = True
            timing.pre_s = started - timing.started
            timing.pool_reused = self._surface_process_pool is not None

        try:
            try:
                process_pool = self._get_surface_process_pool()
                map_started = time.perf_counter()
                envelopes = list(process_pool.map(timed_fit_surface_models_worker, tasks))
                if timing is not None:
                    timing.pool_map_s = time.perf_counter() - map_started
            except BrokenProcessPool as exc:
                if timing is not None:
                    timing.process_errors.append(_error_record(exc))
                self._reset_surface_process_pool()
                fallback_started = time.perf_counter()
                fallback_envelopes = []
                fallback_error = None
                try:
                    _install_worker_patches()
                    fallback_envelopes = [timed_fit_surface_models_worker(task) for task in tasks]
                except Exception as fallback_exc:
                    fallback_error = _error_record(fallback_exc)
                    if timing is not None:
                        timing.fallback = {
                            "used": True,
                            "reason": _error_record(exc),
                            "wall_s": time.perf_counter() - fallback_started,
                            "error": _error_record(fallback_exc),
                        }
                    raise
                if timing is not None:
                    timing.fallback = {
                        "used": True,
                        "reason": _error_record(exc),
                        "wall_s": time.perf_counter() - fallback_started,
                        "error": fallback_error,
                    }
                    timing.fallback_worker_metrics.extend(item.timing for item in fallback_envelopes)
                envelopes = fallback_envelopes
            except Exception as exc:
                if timing is not None:
                    timing.process_errors.append(_error_record(exc))
                raise

            used_fallback = timing is not None and timing.fallback is not None
            if timing is not None and not used_fallback:
                timing.worker_metrics.extend(item.timing for item in envelopes)
                worker_errors = [
                    {"image": item.timing.image_name, **item.timing.error}
                    for item in envelopes
                    if item.timing.error is not None
                ]
                timing.process_errors.extend(worker_errors)
                if worker_errors:
                    raise RuntimeError(f"instrumented worker error: {worker_errors}")
            elif timing is not None:
                worker_errors = [
                    {"image": item.timing.image_name, **item.timing.error}
                    for item in envelopes
                    if item.timing.error is not None
                ]
                timing.process_errors.extend(worker_errors)
                if worker_errors:
                    timing.fallback["errors"] = worker_errors
                    raise RuntimeError(f"instrumented fallback worker error: {worker_errors}")

            if any(item.worker_result is None for item in envelopes):
                raise RuntimeError("instrumented worker returned no production result")
            return [self._surface_model_fits_from_worker(item.worker_result) for item in envelopes]
        finally:
            if timing is not None:
                timing.process_batch_s = time.perf_counter() - started

    def close(self) -> None:
        with self._cache_lock:
            process_pool = self._surface_process_pool
            self._surface_process_pool = None
        if process_pool is not None:
            process_pool.shutdown(wait=True)


def _clear_surface_result_caches(service: TimedDistributionFitService) -> None:
    """Force a fresh surface fit while preserving processed images and the pool."""
    with service._cache_lock:
        service._result_cache_key = None
        service._result_cache = None
        service._surface_fit_cache.clear()


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "numpy": getattr(np, "__version__", None),
    }
    for module_name, package_name in (
        ("scipy", "scipy"),
        ("cv2", "opencv-python"),
        ("matplotlib", "matplotlib"),
        ("openpyxl", "openpyxl"),
    ):
        try:
            module = __import__(module_name)
            versions[package_name] = getattr(module, "__version__", None)
        except Exception:
            versions[package_name] = None
    return versions


def _input_snapshot(settings: DistributionSettings) -> list[dict[str, Any]]:
    values = (
        ("f_img", settings.f_exp_no),
        ("o_img", settings.o_exp_no),
        ("fo_img", settings.fo_exp_no),
    )
    snapshot = []
    for label, number in values:
        path = experiment_npy_path(settings.image_folder, number).resolve()
        entry: dict[str, Any] = {"image": label, "experiment": str(number), "path": str(path)}
        try:
            entry["size_bytes"] = path.stat().st_size
        except OSError as exc:
            entry["stat_error"] = _error_record(exc)
        snapshot.append(entry)
    return snapshot


def _progress(message: str) -> None:
    print(f"[profile_distribution] {message}", flush=True)


def _run_profile(settings: DistributionSettings, repeats: int) -> dict[str, Any]:
    service = TimedDistributionFitService()
    results: list[DistributionFitResult] = []

    def execute(label: str) -> None:
        _progress(f"{label}: start")
        try:
            result = service.run_stage(label, settings, "fit_img", False)
        except Exception as exc:
            _progress(f"{label}: error {type(exc).__name__}: {exc}")
        else:
            results.append(result)
            record = service.run_records[-1]
            pids = sorted({item.pid for item in record.worker_metrics})
            _progress(f"{label}: {record.elapsed_s:.3f}s; worker_pids={pids}")

    try:
        execute("cold")
        for index in range(1, repeats + 1):
            _clear_surface_result_caches(service)
            execute(f"warm_{index}")
    finally:
        try:
            service.close()
        finally:
            _restore_worker_patches()

    dimensions = None
    if results and results[0].images:
        dimensions = {
            "height": int(results[0].images[0].image.shape[0]),
            "width": int(results[0].images[0].image.shape[1]),
        }
    worker_pids = sorted(
        {
            item.pid
            for record in service.run_records
            for item in record.worker_metrics + record.fallback_worker_metrics
        }
    )
    return {
        "dimensions": dimensions,
        "worker_pids": worker_pids,
        "runs": [_run_dict(record) for record in service.run_records],
        "successful_runs": sum(record.success for record in service.run_records),
        "failed_runs": sum(not record.success for record in service.run_records),
    }


def _build_report(settings, settings_path: Path, repeats: int) -> dict[str, Any]:
    raw_settings = settings_path.read_bytes()
    distribution = settings.distribution
    profile = _run_profile(distribution, repeats)
    return {
        "harness": {
            "name": "profile_distribution",
            "version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "repo_root": str(_REPO_ROOT),
            "command": sys.argv,
        },
        "settings": {
            "path": str(settings_path),
            "sha256": hashlib.sha256(raw_settings).hexdigest(),
            "distribution": dataclass_to_jsonable(distribution),
        },
        "environment": {
            "parent_pid": os.getpid(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "versions": _package_versions(),
            "expected_surface_processes": getattr(distribution_service_module, "_SURFACE_PROCESS_COUNT", None),
        },
        "inputs": {
            "files": _input_snapshot(distribution),
            "dimensions": profile["dimensions"],
        },
        "stages": {
            "cold": "new service, new persistent spawn pool",
            "warm": "fresh surface/result caches, same service and pool",
            "repeats": repeats,
            "image_mode": "fit_img",
            "show_bent": False,
            "annotations": [],
            "ui_render": "not measured",
        },
        "summary": {
            "successful_runs": profile["successful_runs"],
            "failed_runs": profile["failed_runs"],
            "worker_pids": profile["worker_pids"],
            "timing_note": "worker_elapsed_sum_s is a sum across concurrent workers and must be compared with process_batch_s, not substituted for it",
        },
        "runs": profile["runs"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="number of warm recomputes after cold (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(tempfile.gettempdir()) / "plif-distribution-profile.json",
        help="JSON report path (default: system temp directory)",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=None,
        help="optional settings JSON; default uses runtime/settings.json via load_settings()",
    )
    args = parser.parse_args()
    if args.repeats < 0:
        parser.error("--repeats must be >= 0")
    return args


def main() -> int:
    mp.freeze_support()
    args = _parse_args()
    settings_path = (args.settings or default_config_path()).resolve()
    try:
        settings = load_settings(args.settings)
        with _parent_timing_patches():
            report = _build_report(settings, settings_path, args.repeats)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        _progress(f"fatal: {type(exc).__name__}: {exc}")
        return 1
    _progress(f"report: {args.output.resolve()}")
    return 0 if report["summary"]["failed_runs"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
