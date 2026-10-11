from __future__ import annotations

import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import numpy as np

from PLIFMix.algorithms.distribution import (
    DistributionRegistrationEstimate,
    estimate_distribution_registration,
)
from PLIFMix.algorithms.mixing import (
    FitConvergenceError,
    calc_flowrate_contour,
    extrapolate_fitted_center_line,
    fit_center_line,
    func_fit_3,
    map_center_line_points,
    sample_horizontal_profile,
    shift_image,
    straighten_center_line_image,
    unstraighten_center_line_image,
)
from PLIFMix.core.experiment_images import (
    ExperimentImageTriplet,
    ExperimentTripletCache,
    experiment_npy_path,
)
from PLIFMix.domain.settings import DistributionSettings
from PLIFMix.services.distribution_export import (
    export_distribution_result,
    write_fit_results_to_excel,
)
from PLIFMix.services.distribution_presentation import (
    CenterLinePresentation,
    ProfilePresentation,
    build_summary_row,
    format_center_line_failure,
    format_center_line_fit,
    format_profile_failure,
    format_profile_fit,
    format_profile_not_executed,
    format_surface_fits,
    synthesize_distribution_image,
)
from PLIFMix.services.distribution_results import (
    DistributionFitParameters,
    DistributionFitResult,
    DistributionImageData,
    DistributionRegistrationResult,
    DistributionSummaryRow,
)
from PLIFMix.services.distribution_workers import (
    SurfaceFitParameters,
    SurfaceFitTask,
    SurfaceFitWorkerResult,
    fit_surface_models_worker,
)


_SURFACE_PROCESS_COUNT = 3


@dataclass(frozen=True)
class _SurfaceModelFits:
    student_fit: SurfaceFitParameters | None
    compact_fit: SurfaceFitParameters | None
    selected_prediction: np.ndarray | None
    student_error: str | None
    compact_error: str | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _SurfaceFitCacheEntry:
    key: tuple
    fits: _SurfaceModelFits


@dataclass
class _PreparedDistributionImage:
    source: DistributionImageData
    image_name: str
    display_name: str
    image: np.ndarray
    normalized: np.ndarray
    center_x: int
    center_y: int
    rotate_angle: int
    contour_value: float
    boundary_points: set[tuple[int, int]]
    center_a: float
    center_b: float
    center_r_squared: float
    center_presentation: CenterLinePresentation
    origin_point: tuple[np.ndarray, np.ndarray]
    fitted_point: tuple[np.ndarray, np.ndarray]
    symmetric_fitted_point: tuple[np.ndarray, np.ndarray]
    horizontal_a: float
    horizontal_b: float
    horizontal_c: float
    horizontal_r_squared: float
    horizontal_presentation: ProfilePresentation
    vertical_a: float
    vertical_b: float
    vertical_c: float
    vertical_r_squared: float
    vertical_presentation: ProfilePresentation
    bent_image: np.ndarray | None
    surface_task: SurfaceFitTask | None


class DistributionFitService:
    """Fit independently normalized F, O, and FO mass distributions."""

    def __init__(self, image_cache: ExperimentTripletCache | None = None):
        self.image_cache = image_cache or ExperimentTripletCache()
        self._processed_cache_key: tuple | None = None
        self._processed_cache: tuple[DistributionImageData, ...] | None = None
        self._result_cache_key: tuple | None = None
        self._result_cache: DistributionFitResult | None = None
        self._surface_fit_cache: dict[str, _SurfaceFitCacheEntry] = {}
        self._surface_process_pool: ProcessPoolExecutor | None = None
        self._cache_lock = RLock()

    def close(self) -> None:
        """Release the persistent Windows process pool without blocking the GUI."""
        with self._cache_lock:
            process_pool = self._surface_process_pool
            self._surface_process_pool = None
        if process_pool is not None:
            process_pool.shutdown(wait=False, cancel_futures=True)

    def run(
        self,
        settings: DistributionSettings,
        annotations: frozenset[str] = frozenset(),
        image_mode: str = "ori_img",
        show_bent: bool | None = None,
    ) -> DistributionFitResult:
        if image_mode == "bent_img":
            image_mode = "ori_img"
            show_bent = True
        elif show_bent is None:
            # Preserve the old API behavior for callers that did not pass the new flag.
            show_bent = image_mode in {"fit_img", "delta_img"}
        if image_mode == "rot_img":
            # The rotation view already straightens the jet, so straightening on top
            # of it would tilt the ridge again. Ignore the flag instead of combining.
            show_bent = False
        if image_mode not in {"ori_img", "rot_img", "fit_img", "delta_img"}:
            raise ValueError(f"不支持的图像数据类型: {image_mode}")
        source_images = self._distribution_images(settings)
        result_cache_key = self._result_cache_signature(settings, annotations, image_mode, show_bent)
        with self._cache_lock:
            if result_cache_key == self._result_cache_key and self._result_cache is not None:
                return self._result_cache
        surface_fit_cache_keys = tuple(
            self._surface_fit_cache_signature(settings, source, index)
            for index, source in enumerate(source_images)
        )
        with self._cache_lock:
            cached_surface_fits = tuple(
                self._surface_fit_cache[image_name].fits
                if image_name in self._surface_fit_cache
                and self._surface_fit_cache[image_name].key == cache_key
                else None
                for image_name, cache_key in zip(
                    ("f_img", "o_img", "fo_img"),
                    surface_fit_cache_keys,
                )
            )

        fit_text = ""
        surface_text = ""
        display_images = []
        numeric_results: list[DistributionFitParameters] = []
        curve_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        warnings: list[str] = []
        summary_rows: list[DistributionSummaryRow] = []
        centers = [
            (settings.center_xf, settings.center_yf),
            (settings.center_xo, settings.center_yo),
            (settings.center_xfo, settings.center_yfo),
        ]
        rotate_angles = [settings.rotate_angle_f, settings.rotate_angle_o, settings.rotate_angle_fo]
        center_line_bounds = [
            (settings.q_center_up_f, settings.q_center_down_f),
            (settings.q_center_up_o, settings.q_center_down_o),
            (settings.q_center_up_fo, settings.q_center_down_fo),
        ]
        image_names = ["f_img", "o_img", "fo_img"]
        display_names = ["f", "o", "fo"]
        height, width = source_images[0].denoised.shape
        need_surface_fit = image_mode in {"fit_img", "delta_img"}
        need_bent_image = bool(show_bent or need_surface_fit)
        need_symmetric_restore = bool(need_surface_fit and not show_bent)
        pixel_x_axis = pixel_y_axis = None
        if need_surface_fit:
            pixel_x_axis = np.arange(width, dtype=np.float64)
            pixel_y_axis = np.arange(height, dtype=np.float64)

        prepared_images: list[_PreparedDistributionImage] = []
        for index, source in enumerate(source_images):
            image = source.denoised
            norm_img = source.normalized
            cx, cy = centers[index]
            q_center_up, q_center_down = center_line_bounds[index]
            image_name = image_names[index]
            display_name = display_names[index]
            contour_value = 0.0
            boundary_points: set[tuple[int, int]] = set()
            if "contour" in annotations:
                contour_value, boundary_points = calc_flowrate_contour(image, settings.thres_contour)

            empty_points = (np.array([], dtype=np.int32), np.array([], dtype=np.int32))
            try:
                a, b, r_squared, origin_point, fitted_point, verti_px_rotated = fit_center_line(
                    image,
                    q_center_up,
                    q_center_down,
                    settings.max_k,
                    cx,
                    cy,
                    rotate_angles[index],
                )
                center_presentation = format_center_line_fit(a, b, r_squared)
            except FitConvergenceError as exc:
                a = b = r_squared = np.nan
                origin_point = fitted_point = empty_points
                verti_px_rotated = np.array([], dtype=np.float64)
                center_presentation = format_center_line_failure(str(exc))
                warnings.append(f"{image_name} {center_presentation.text}")
            symmetric_fitted_point = empty_points
            if len(fitted_point[0]):
                symmetric_fitted_point = extrapolate_fitted_center_line(
                    image.shape,
                    a,
                    b,
                    cx,
                    cy,
                    0,
                )
            fit_text += f"{image_name}流量轮廓值:{contour_value};{center_presentation.text}     "

            hori_px, hori_intensity = sample_horizontal_profile(
                norm_img,
                cx,
                cy,
                rotate_angles[index],
            )

            def fit_func(x: np.ndarray, aa: float, bb: float, cc: float) -> np.ndarray:
                return aa / (x**2 * bb**2 + 1) ** cc

            try:
                ah, bh, ch, r_squared_h, hori_predicted = func_fit_3(hori_px, hori_intensity, fit_func)
                curve_data[f"横向分布_{image_name}"] = (hori_px, hori_intensity, hori_predicted)
                horizontal_presentation = format_profile_fit(
                    "horizontal",
                    ah,
                    bh,
                    ch,
                    r_squared_h,
                )
            except FitConvergenceError as exc:
                ah = bh = ch = r_squared_h = np.nan
                horizontal_presentation = format_profile_failure(str(exc))
                warnings.append(f"{image_name} {horizontal_presentation.text}")
            fit_text += f"{image_name}{horizontal_presentation.text}     "

            verti_py, verti_px = origin_point
            if len(verti_py):
                verti_intensity = norm_img[verti_py, verti_px]
                try:
                    av, bv, cv, r_squared_v, vertical_predicted = func_fit_3(
                        verti_px_rotated,
                        verti_intensity,
                        fit_func,
                    )
                    curve_data[f"纵向分布_{image_name}"] = (
                        verti_px_rotated,
                        verti_intensity,
                        vertical_predicted,
                    )
                    vertical_presentation = format_profile_fit(
                        "vertical",
                        av,
                        bv,
                        cv,
                        r_squared_v,
                    )
                except FitConvergenceError as exc:
                    av = bv = cv = r_squared_v = np.nan
                    vertical_presentation = format_profile_failure(str(exc))
                    warnings.append(f"{image_name} {vertical_presentation.text}")
            else:
                av = bv = cv = r_squared_v = np.nan
                vertical_presentation = format_profile_not_executed()
            fit_text += f"{image_name}{vertical_presentation.text}\n"

            bent_img: np.ndarray | None = None
            if need_bent_image:
                if len(fitted_point[0]) == 0:
                    raise FitConvergenceError(f"{image_name} 无有效拟合中心线，不能生成掰直图像。")
                bent_img = straighten_center_line_image(norm_img, fitted_point, cx)

            surface_task: SurfaceFitTask | None = None
            if need_surface_fit:
                assert pixel_x_axis is not None and pixel_y_axis is not None and bent_img is not None
                if not np.all(np.isfinite([ah, bh, ch, av, bv, cv])):
                    raise FitConvergenceError(f"{image_name} 横向或纵向拟合失败，不能执行二维 Student 拟合。")
                surface_task = SurfaceFitTask(
                    bent_img,
                    pixel_x_axis - cx,
                    pixel_y_axis - cy,
                    1.0 / max(abs(bh), 1e-12),
                    1.0 / max(abs(bv), 1e-12),
                    (ch + cv) / 2.0,
                    image_name,
                )

            prepared_images.append(
                _PreparedDistributionImage(
                    source=source,
                    image_name=image_name,
                    display_name=display_name,
                    image=image,
                    normalized=norm_img,
                    center_x=cx,
                    center_y=cy,
                    rotate_angle=rotate_angles[index],
                    contour_value=contour_value,
                    boundary_points=boundary_points,
                    center_a=a,
                    center_b=b,
                    center_r_squared=r_squared,
                    center_presentation=center_presentation,
                    origin_point=origin_point,
                    fitted_point=fitted_point,
                    symmetric_fitted_point=symmetric_fitted_point,
                    horizontal_a=ah,
                    horizontal_b=bh,
                    horizontal_c=ch,
                    horizontal_r_squared=r_squared_h,
                    horizontal_presentation=horizontal_presentation,
                    vertical_a=av,
                    vertical_b=bv,
                    vertical_c=cv,
                    vertical_r_squared=r_squared_v,
                    vertical_presentation=vertical_presentation,
                    bent_image=bent_img,
                    surface_task=surface_task,
                )
            )

        surface_fits = list(cached_surface_fits)
        if need_surface_fit:
            miss_indices = [
                index
                for index, surface_fit in enumerate(surface_fits)
                if surface_fit is None
            ]
            miss_tasks = [prepared_images[index].surface_task for index in miss_indices]
            if any(task is None for task in miss_tasks):
                raise FitConvergenceError("存在无法生成二维拟合任务的图像。")
            if miss_indices:
                fitted_misses = self._fit_surface_models_batch(
                    [task for task in miss_tasks if task is not None]
                )
                if len(fitted_misses) != len(miss_indices):
                    raise RuntimeError("二维拟合任务返回数量与缓存 miss 数量不一致。")
                with self._cache_lock:
                    for index, fits in zip(miss_indices, fitted_misses):
                        surface_fits[index] = fits
                        self._surface_fit_cache[("f_img", "o_img", "fo_img")[index]] = _SurfaceFitCacheEntry(
                            surface_fit_cache_keys[index],
                            fits,
                        )

        for prepared, surface_fit_result in zip(prepared_images, surface_fits):
            image_name = prepared.image_name
            display_name = prepared.display_name
            source = prepared.source
            image = prepared.image
            norm_img = prepared.normalized
            cx = prepared.center_x
            cy = prepared.center_y
            fitted_point = prepared.fitted_point
            symmetric_fitted_point = prepared.symmetric_fitted_point
            bent_img = prepared.bent_image

            if surface_fit_result is not None:
                student_fit = surface_fit_result.student_fit
                compact_fit = surface_fit_result.compact_fit
                surface_prediction = surface_fit_result.selected_prediction
                warnings.extend(surface_fit_result.warnings)
            else:
                student_fit = compact_fit = None
                surface_prediction = None

            surface_presentation = format_surface_fits(
                image_name,
                student_fit,
                compact_fit,
                surface_fit_result.student_error if surface_fit_result is not None else None,
                surface_fit_result.compact_error if surface_fit_result is not None else None,
                surface_fit_result is not None,
            )
            surface_text += surface_presentation.text
            if surface_fit_result is not None and surface_prediction is None:
                raise FitConvergenceError(f"{image_name} 两种二维 Student 模型均拟合失败。")

            numeric_results.append(
                DistributionFitParameters(
                    image_name=image_name,
                    center_x=_finite_or_none(cx),
                    center_y=_finite_or_none(cy),
                    rotate_angle=_finite_or_none(prepared.rotate_angle),
                    center_a=_finite_or_none(prepared.center_a),
                    center_b=_finite_or_none(prepared.center_b),
                    center_r_squared=_finite_or_none(prepared.center_r_squared),
                    horizontal_a=_finite_or_none(prepared.horizontal_a),
                    horizontal_b=_finite_or_none(prepared.horizontal_b),
                    horizontal_c=_finite_or_none(prepared.horizontal_c),
                    horizontal_r_squared=_finite_or_none(prepared.horizontal_r_squared),
                    vertical_a=_finite_or_none(prepared.vertical_a),
                    vertical_b=_finite_or_none(prepared.vertical_b),
                    vertical_c=_finite_or_none(prepared.vertical_c),
                    vertical_r_squared=_finite_or_none(prepared.vertical_r_squared),
                    student_scale_x=_finite_or_none(
                        student_fit.scale_x if student_fit is not None else np.nan
                    ),
                    student_scale_y=_finite_or_none(
                        student_fit.scale_y if student_fit is not None else np.nan
                    ),
                    student_exponent=_finite_or_none(
                        student_fit.exponent if student_fit is not None else np.nan
                    ),
                    student_r_squared=_finite_or_none(
                        student_fit.r_squared if student_fit is not None else np.nan
                    ),
                    compact_scale_x=_finite_or_none(
                        compact_fit.scale_x if compact_fit is not None else np.nan
                    ),
                    compact_scale_y=_finite_or_none(
                        compact_fit.scale_y if compact_fit is not None else np.nan
                    ),
                    compact_exponent=_finite_or_none(
                        compact_fit.exponent if compact_fit is not None else np.nan
                    ),
                    compact_cutoff_radius=_finite_or_none(
                        compact_fit.cutoff_radius
                        if compact_fit is not None and compact_fit.cutoff_radius is not None
                        else np.nan
                    ),
                    compact_r_squared=_finite_or_none(
                        compact_fit.r_squared if compact_fit is not None else np.nan
                    ),
                )
            )

            point_sets: list[tuple[np.ndarray, np.ndarray]] = []
            if "contour" in annotations and prepared.boundary_points:
                point_sets.append(tuple(np.array(points) for points in zip(*prepared.boundary_points)))
            if "center_line" in annotations:
                point_sets.append(prepared.origin_point)
            if "fitted_center_line" in annotations:
                point_sets.append(fitted_point)

            display_images.append(
                synthesize_distribution_image(
                    display_name=display_name,
                    image_mode=image_mode,
                    show_bent=show_bent,
                    need_symmetric_restore=need_symmetric_restore,
                    norm_img=norm_img,
                    source_total=source.total,
                    image_shape=image.shape,
                    center_x=cx,
                    center_y=cy,
                    rotate_angle=prepared.rotate_angle,
                    fitted_point=fitted_point,
                    symmetric_fitted_point=symmetric_fitted_point,
                    bent_img=bent_img,
                    surface_prediction=surface_prediction,
                    point_sets=point_sets,
                    map_center_line_points=map_center_line_points,
                    unstraighten_center_line_image=unstraighten_center_line_image,
                )
            )
            summary_rows.append(
                build_summary_row(
                    display_name=display_name,
                    contour_value=prepared.contour_value,
                    has_contour="contour" in annotations,
                    center=prepared.center_presentation,
                    horizontal=prepared.horizontal_presentation,
                    vertical=prepared.vertical_presentation,
                    surface=surface_presentation,
                )
            )

        result = DistributionFitResult(
            fit_text=fit_text,
            surface_text=surface_text,
            images=display_images,
            numeric_results=tuple(numeric_results),
            curve_data=curve_data,
            warnings=tuple(warnings),
            summary_rows=tuple(summary_rows),
        )
        with self._cache_lock:
            self._result_cache_key = result_cache_key
            self._result_cache = result
        return result

    def _result_cache_signature(
        self,
        settings: DistributionSettings,
        annotations: frozenset[str],
        image_mode: str,
        show_bent: bool,
    ) -> tuple:
        return (
            self._processed_cache_key,
            tuple(sorted(annotations)),
            image_mode,
            show_bent,
            settings.thres_contour,
            settings.center_xf,
            settings.center_yf,
            settings.center_xo,
            settings.center_yo,
            settings.center_xfo,
            settings.center_yfo,
            settings.q_center_up_f,
            settings.q_center_down_f,
            settings.q_center_up_o,
            settings.q_center_down_o,
            settings.q_center_up_fo,
            settings.q_center_down_fo,
            settings.max_k,
            settings.rotate_angle_f,
            settings.rotate_angle_o,
            settings.rotate_angle_fo,
        )

    def _surface_fit_cache_signature(
        self,
        settings: DistributionSettings,
        source: DistributionImageData,
        index: int,
    ) -> tuple:
        """Identify inputs that change one image's expensive two-dimensional fit."""
        image_names = ("f_img", "o_img", "fo_img")
        experiment_numbers = (settings.f_exp_no, settings.o_exp_no, settings.fo_exp_no)
        shifts = (settings.f_shift, settings.o_shift, 0)
        centers = (
            (settings.center_xf, settings.center_yf),
            (settings.center_xo, settings.center_yo),
            (settings.center_xfo, settings.center_yfo),
        )
        center_line_bounds = (
            (settings.q_center_up_f, settings.q_center_down_f),
            (settings.q_center_up_o, settings.q_center_down_o),
            (settings.q_center_up_fo, settings.q_center_down_fo),
        )
        center_x, center_y = centers[index]
        q_center_up, q_center_down = center_line_bounds[index]
        return (
            image_names[index],
            experiment_npy_path(settings.image_folder, experiment_numbers[index]).resolve(),
            _array_content_signature(source.denoised),
            shifts[index],
            settings.denoise_ratio,
            settings.max_k,
            center_x,
            center_y,
            q_center_up,
            q_center_down,
            (settings.rotate_angle_f, settings.rotate_angle_o, settings.rotate_angle_fo)[index],
        )

    def _fit_surface_models_batch(self, tasks: list[SurfaceFitTask]) -> list[_SurfaceModelFits]:
        """Run the three independent surface fits in a persistent process pool."""
        if not tasks:
            return []
        try:
            process_pool = self._get_surface_process_pool()
            worker_results = list(process_pool.map(fit_surface_models_worker, tasks))
        except BrokenProcessPool:
            self._reset_surface_process_pool()
            worker_results = [fit_surface_models_worker(task) for task in tasks]
        return [self._surface_model_fits_from_worker(result) for result in worker_results]

    def _get_surface_process_pool(self) -> ProcessPoolExecutor:
        with self._cache_lock:
            if self._surface_process_pool is None:
                self._surface_process_pool = ProcessPoolExecutor(
                    max_workers=_SURFACE_PROCESS_COUNT,
                    mp_context=mp.get_context("spawn"),
                )
            return self._surface_process_pool

    def _reset_surface_process_pool(self) -> None:
        with self._cache_lock:
            process_pool = self._surface_process_pool
            self._surface_process_pool = None
        if process_pool is not None:
            process_pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _surface_model_fits_from_worker(result: SurfaceFitWorkerResult) -> _SurfaceModelFits:
        return _SurfaceModelFits(
            result.student_fit,
            result.compact_fit,
            result.selected_prediction,
            result.student_error,
            result.compact_error,
            result.warnings,
        )

    def estimate_registration(self, settings: DistributionSettings) -> DistributionRegistrationResult:
        source_images = self._distribution_images(settings)
        bounds = (
            (settings.q_center_up_f, settings.q_center_down_f),
            (settings.q_center_up_o, settings.q_center_down_o),
            (settings.q_center_up_fo, settings.q_center_down_fo),
        )
        estimates = tuple(
            estimate_distribution_registration(
                source.denoised,
                up_limit,
                down_limit,
                settings.max_k,
            )
            for source, (up_limit, down_limit) in zip(source_images, bounds)
        )
        return DistributionRegistrationResult(*estimates)

    def _distribution_images(self, settings: DistributionSettings) -> tuple[DistributionImageData, ...]:
        triplet = self.image_cache.load(
            settings.image_folder,
            settings.f_exp_no,
            settings.o_exp_no,
            settings.fo_exp_no,
        )
        key = (
            id(triplet.f),
            id(triplet.o),
            id(triplet.fo),
            settings.f_shift,
            settings.o_shift,
            settings.denoise_ratio,
        )
        with self._cache_lock:
            if key != self._processed_cache_key or self._processed_cache is None:
                self._processed_cache = _prepare_distribution_images(triplet, settings)
                self._processed_cache_key = key
            return self._processed_cache

    def export(self, settings: DistributionSettings, result: DistributionFitResult) -> Path:
        return export_distribution_result(settings, result)

    def write_fit_results_to_excel(self, settings: DistributionSettings, result: DistributionFitResult) -> Path:
        return write_fit_results_to_excel(settings, result)


def _prepare_distribution_images(
    triplet: ExperimentImageTriplet,
    settings: DistributionSettings,
) -> tuple[DistributionImageData, ...]:
    shifted_images = (
        shift_image(triplet.f, settings.f_shift),
        shift_image(triplet.o, settings.o_shift),
        triplet.fo,
    )
    return tuple(
        _denoise_and_normalize_distribution(image, settings.denoise_ratio, label)
        for image, label in zip(shifted_images, ("F", "O", "FO"))
    )


def _denoise_and_normalize_distribution(
    image: np.ndarray,
    denoise_ratio: float,
    label: str,
) -> DistributionImageData:
    if not np.isfinite(denoise_ratio) or not 0 <= denoise_ratio <= 1:
        raise ValueError("相对去噪阈值必须位于 0 到 1 之间。")
    maximum = float(np.max(image))
    if maximum <= 0:
        raise ValueError(f"{label} 图像没有可用于拟合的正值。")
    denoised = image.copy()
    denoised[denoised < maximum * denoise_ratio] = 0
    total = float(np.sum(denoised))
    if total <= 0:
        raise ValueError(f"{label} 图像去噪后总量为 0，请降低相对去噪阈值。")
    return DistributionImageData(
        denoised=denoised,
        normalized=denoised / total,
        total=total,
    )


def _array_content_signature(array: np.ndarray) -> tuple[tuple[int, ...], str, str]:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()
    return tuple(int(size) for size in contiguous.shape), contiguous.dtype.str, digest


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None
