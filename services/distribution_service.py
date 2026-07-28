from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import numpy as np
import openpyxl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from PLIFv2.algorithms.distribution import (
    DistributionModelFitError,
    DistributionRegistrationEstimate,
    estimate_distribution_registration,
    fit_compact_elliptical_student,
    fit_elliptical_student,
)
from PLIFv2.algorithms.mixing import (
    FitConvergenceError,
    calc_flowrate_contour,
    fit_center_line,
    func_fit_3,
    rotate_points_clockwise,
    shift_image,
    straighten_center_line_image,
)
from PLIFv2.core.experiment_images import ExperimentImageTriplet, ExperimentTripletCache
from PLIFv2.core.paths import path_is_set
from PLIFv2.domain.settings import DistributionSettings
from PLIFv2.services.display_types import DisplayImage

@dataclass
class DistributionFitResult:
    fit_text: str
    surface_text: str
    images: list[DisplayImage]
    fit_rows: list[list[float | None]]
    curve_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    warnings: tuple[str, ...]
    summary_rows: tuple["DistributionSummaryRow", ...]


@dataclass(frozen=True)
class DistributionSummaryRow:
    image_name: str
    contour: str
    center_line: str
    center_a: str
    center_b: str
    center_r_squared: str
    horizontal_equation: str
    horizontal_a: str
    horizontal_b: str
    horizontal_c: str
    horizontal_r_squared: str
    vertical_equation: str
    vertical_a: str
    vertical_b: str
    vertical_c: str
    vertical_r_squared: str
    student_equation: str
    student_scale_x: str
    student_scale_y: str
    student_exponent: str
    student_r_squared: str
    compact_equation: str
    compact_scale_x: str
    compact_scale_y: str
    compact_exponent: str
    compact_cutoff_radius: str
    compact_r_squared: str


@dataclass(frozen=True)
class DistributionImageData:
    denoised: np.ndarray
    normalized: np.ndarray
    total: float


@dataclass(frozen=True)
class DistributionRegistrationResult:
    f: DistributionRegistrationEstimate
    o: DistributionRegistrationEstimate
    fo: DistributionRegistrationEstimate


class DistributionFitService:
    """Fit independently normalized F, O, and FO mass distributions."""

    def __init__(self, image_cache: ExperimentTripletCache | None = None):
        self.image_cache = image_cache or ExperimentTripletCache()
        self._processed_cache_key: tuple | None = None
        self._processed_cache: tuple[DistributionImageData, ...] | None = None
        self._cache_lock = RLock()

    def run(
        self,
        settings: DistributionSettings,
        annotations: frozenset[str] = frozenset(),
        image_mode: str = "ori_img",
    ) -> DistributionFitResult:
        source_images = self._distribution_images(settings)

        fit_text = ""
        surface_text = ""
        display_images: list[DisplayImage] = []
        fit_rows: list[list[float | None]] = []
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
        grid_x = grid_y = None
        if need_surface_fit:
            grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")

        for index, source in enumerate(source_images):
            image = source.denoised
            norm_img = source.normalized
            cx, cy = centers[index]
            q_center_up, q_center_down = center_line_bounds[index]
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
                center_equation = "x=a(√(y²+b²)-b)"
                center_a = f"{a:.4f}"
                center_b = f"{b:.4f}"
                center_r_squared = f"{r_squared:.4f}"
                center_text = (
                    f"{center_equation}; a={center_a}; b={center_b}; R²={center_r_squared}"
                )
            except FitConvergenceError as exc:
                a = b = r_squared = np.nan
                origin_point = fitted_point = empty_points
                verti_px_rotated = np.array([], dtype=np.float64)
                center_equation = f"拟合失败：{exc}"
                center_a = center_b = center_r_squared = "-"
                center_text = f"拟合失败：{exc}"
                warnings.append(f"{image_names[index]} {center_text}")
            fit_text += f"{image_names[index]}流量轮廓值:{contour_value};{center_text}     "

            hori_px = np.array(range(-settings.hori_fit_width, settings.hori_fit_width))
            hori_py = np.zeros_like(hori_px)
            hori_px_rotated, hori_py_rotated = rotate_points_clockwise(hori_px, hori_py, -rotate_angles[index])
            sample_y = np.clip(np.round(-hori_py_rotated + cy).astype(np.int32), 0, height - 1)
            sample_x = np.clip(np.round(hori_px_rotated + cx).astype(np.int32), 0, width - 1)
            hori_intensity = norm_img[sample_y, sample_x]

            def fit_func(x: np.ndarray, aa: float, bb: float, cc: float) -> np.ndarray:
                return aa / (x**2 * bb**2 + 1) ** cc

            try:
                ah, bh, ch, r_squared_h, hori_predicted = func_fit_3(hori_px, hori_intensity, fit_func)
                curve_data[f"横向分布_{image_names[index]}"] = (hori_px, hori_intensity, hori_predicted)
                horizontal_equation = "p=a/(x²b²+1)^c"
                horizontal_a = f"{ah:.4e}"
                horizontal_b = f"{bh:.4f}"
                horizontal_c = f"{ch:.4f}"
                horizontal_r_squared = f"{r_squared_h:.4f}"
                horizontal_text = (
                    f"y={ah * 1e4:.4f}E-4/"
                    f"(x^2*{bh:.4f}^2+1)^{ch:.4f}; R²={r_squared_h:.4f}"
                )
            except FitConvergenceError as exc:
                ah = bh = ch = r_squared_h = np.nan
                horizontal_equation = f"拟合失败：{exc}"
                horizontal_a = horizontal_b = horizontal_c = horizontal_r_squared = "-"
                horizontal_text = f"拟合失败：{exc}"
                warnings.append(f"{image_names[index]} {horizontal_text}")
            fit_text += f"{image_names[index]}{horizontal_text}     "

            verti_py, verti_px = origin_point
            if len(verti_py):
                verti_intensity = norm_img[verti_py, verti_px]
                try:
                    av, bv, cv, r_squared_v, vertical_predicted = func_fit_3(
                        verti_px_rotated,
                        verti_intensity,
                        fit_func,
                    )
                    curve_data[f"纵向分布_{image_names[index]}"] = (
                        verti_px_rotated,
                        verti_intensity,
                        vertical_predicted,
                    )
                    vertical_equation = "p=a/(y²b²+1)^c"
                    vertical_a = f"{av:.4e}"
                    vertical_b = f"{bv:.4f}"
                    vertical_c = f"{cv:.4f}"
                    vertical_r_squared = f"{r_squared_v:.4f}"
                    vertical_text = (
                        f"y={av * 1e4:.4f}E-4/"
                        f"(x^2*{bv:.4f}^2+1)^{cv:.4f}; R²={r_squared_v:.4f}"
                    )
                except FitConvergenceError as exc:
                    av = bv = cv = r_squared_v = np.nan
                    vertical_equation = f"拟合失败：{exc}"
                    vertical_a = vertical_b = vertical_c = vertical_r_squared = "-"
                    vertical_text = f"拟合失败：{exc}"
                    warnings.append(f"{image_names[index]} {vertical_text}")
            else:
                av = bv = cv = r_squared_v = np.nan
                vertical_equation = "未执行：中心线拟合没有有效数据"
                vertical_a = vertical_b = vertical_c = vertical_r_squared = "-"
                vertical_text = "未执行：中心线拟合没有有效数据"
            fit_text += f"{image_names[index]}{vertical_text}\n"

            bent_img: np.ndarray | None = None
            surface_prediction: np.ndarray | None = None
            student_fit = None
            compact_fit = None
            if image_mode != "ori_img":
                if not len(fitted_point):
                    raise FitConvergenceError(f"{image_names[index]} 无有效拟合中心线，不能生成掰直图像。")
                bent_img = straighten_center_line_image(norm_img, fitted_point, cx)

            if need_surface_fit:
                assert grid_x is not None and grid_y is not None and bent_img is not None
                if not np.all(np.isfinite([ah, bh, ch, av, bv, cv])):
                    raise FitConvergenceError(f"{image_names[index]} 横向或纵向拟合失败，不能执行二维 Student 拟合。")
                pixel_x = grid_x - cx
                pixel_y = grid_y - cy
                initial_scale_x = 1.0 / max(abs(bh), 1e-12)
                initial_scale_y = 1.0 / max(abs(bv), 1e-12)
                initial_exponent = (ch + cv) / 2.0

                try:
                    student_fit = fit_elliptical_student(
                        bent_img,
                        pixel_x,
                        pixel_y,
                        initial_scale_x,
                        initial_scale_y,
                        initial_exponent,
                    )
                    student_equation = "p∝[1+(x/sₓ)²+(y/sᵧ)²]⁻ᵐ"
                    student_scale_x = f"{student_fit.scale_x:.4f}"
                    student_scale_y = f"{student_fit.scale_y:.4f}"
                    student_exponent = f"{student_fit.exponent:.4f}"
                    student_r_squared = f"{student_fit.r_squared:.4f}"
                    surface_text += (
                        f"{image_names[index]}纯椭圆 Student："
                        f"sₓ={student_scale_x}; sᵧ={student_scale_y}; "
                        f"m={student_exponent}; R²={student_r_squared}\n"
                    )
                except DistributionModelFitError as exc:
                    student_equation = f"拟合失败：{exc}"
                    student_scale_x = student_scale_y = student_exponent = student_r_squared = "-"
                    warnings.append(f"{image_names[index]} 纯椭圆 Student 拟合失败：{exc}")
                    surface_text += f"{image_names[index]}纯椭圆 Student：拟合失败\n"

                compact_initial = student_fit
                try:
                    compact_fit = fit_compact_elliptical_student(
                        bent_img,
                        pixel_x,
                        pixel_y,
                        compact_initial.scale_x if compact_initial is not None else initial_scale_x,
                        compact_initial.scale_y if compact_initial is not None else initial_scale_y,
                        compact_initial.exponent if compact_initial is not None else initial_exponent,
                    )
                    compact_equation = "p∝(1+rₑ²)⁻ᵐ(1-rₑ²/R²)₊²"
                    compact_scale_x = f"{compact_fit.scale_x:.4f}"
                    compact_scale_y = f"{compact_fit.scale_y:.4f}"
                    compact_exponent = f"{compact_fit.exponent:.4f}"
                    assert compact_fit.cutoff_radius is not None
                    compact_cutoff_radius = f"{compact_fit.cutoff_radius:.4f}"
                    compact_r_squared = f"{compact_fit.r_squared:.4f}"
                    surface_text += (
                        f"{image_names[index]}紧支撑 Student："
                        f"sₓ={compact_scale_x}; sᵧ={compact_scale_y}; "
                        f"m={compact_exponent}; R={compact_cutoff_radius}; "
                        f"R²={compact_r_squared}\n"
                    )
                except DistributionModelFitError as exc:
                    compact_equation = f"拟合失败：{exc}"
                    compact_scale_x = compact_scale_y = compact_exponent = compact_cutoff_radius = "-"
                    compact_r_squared = "-"
                    warnings.append(f"{image_names[index]} 紧支撑 Student 拟合失败：{exc}")
                    surface_text += f"{image_names[index]}紧支撑 Student：拟合失败\n"

                selected_fit = compact_fit if compact_fit is not None else student_fit
                if selected_fit is None:
                    raise FitConvergenceError(f"{image_names[index]} 两种二维 Student 模型均拟合失败。")
                surface_prediction = selected_fit.prediction
            else:
                student_equation = compact_equation = "未执行"
                student_scale_x = student_scale_y = student_exponent = student_r_squared = "-"
                compact_scale_x = compact_scale_y = compact_exponent = compact_cutoff_radius = "-"
                compact_r_squared = "-"
                surface_text += f"{image_names[index]}二维 Student 拟合：未执行\n"
            fit_rows.append(
                _finite_values_or_none([
                    cx,
                    cy,
                    rotate_angles[index],
                    a,
                    b,
                    r_squared,
                    ah,
                    bh,
                    ch,
                    r_squared_h,
                    av,
                    bv,
                    cv,
                    r_squared_v,
                    student_fit.scale_x if student_fit is not None else np.nan,
                    student_fit.scale_y if student_fit is not None else np.nan,
                    student_fit.exponent if student_fit is not None else np.nan,
                    student_fit.r_squared if student_fit is not None else np.nan,
                    compact_fit.scale_x if compact_fit is not None else np.nan,
                    compact_fit.scale_y if compact_fit is not None else np.nan,
                    compact_fit.exponent if compact_fit is not None else np.nan,
                    (
                        compact_fit.cutoff_radius
                        if compact_fit is not None and compact_fit.cutoff_radius is not None
                        else np.nan
                    ),
                    compact_fit.r_squared if compact_fit is not None else np.nan,
                ])
            )

            point_sets: list[tuple[np.ndarray, np.ndarray]] = []
            if "contour" in annotations and boundary_points:
                point_sets.append(tuple(np.array(points) for points in zip(*boundary_points)))
            if "center_line" in annotations:
                point_sets.append(origin_point)
            if "fitted_center_line" in annotations:
                point_sets.append(fitted_point)
            special_points = _merge_special_points(point_sets)

            if image_mode == "bent_img":
                assert bent_img is not None
                shown_normalized = bent_img
            elif image_mode == "fit_img":
                assert surface_prediction is not None
                shown_normalized = surface_prediction
            elif image_mode == "delta_img":
                assert surface_prediction is not None and bent_img is not None
                shown_normalized = np.abs(surface_prediction - bent_img)
            else:
                shown_normalized = norm_img
            display_images.append(
                DisplayImage(
                    display_names[index],
                    shown_normalized * source.total,
                    "jet",
                    special_points,
                )
            )
            summary_rows.append(
                DistributionSummaryRow(
                    image_name=display_names[index].upper(),
                    contour=f"{contour_value:g}" if "contour" in annotations else "未计算",
                    center_line=center_equation,
                    center_a=center_a,
                    center_b=center_b,
                    center_r_squared=center_r_squared,
                    horizontal_equation=horizontal_equation,
                    horizontal_a=horizontal_a,
                    horizontal_b=horizontal_b,
                    horizontal_c=horizontal_c,
                    horizontal_r_squared=horizontal_r_squared,
                    vertical_equation=vertical_equation,
                    vertical_a=vertical_a,
                    vertical_b=vertical_b,
                    vertical_c=vertical_c,
                    vertical_r_squared=vertical_r_squared,
                    student_equation=student_equation,
                    student_scale_x=student_scale_x,
                    student_scale_y=student_scale_y,
                    student_exponent=student_exponent,
                    student_r_squared=student_r_squared,
                    compact_equation=compact_equation,
                    compact_scale_x=compact_scale_x,
                    compact_scale_y=compact_scale_y,
                    compact_exponent=compact_exponent,
                    compact_cutoff_radius=compact_cutoff_radius,
                    compact_r_squared=compact_r_squared,
                )
            )

        return DistributionFitResult(
            fit_text,
            surface_text,
            display_images,
            fit_rows,
            curve_data,
            tuple(warnings),
            tuple(summary_rows),
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
        output = Path(settings.output_folder) if path_is_set(settings.output_folder) else Path(settings.image_folder).parent / "distribution_results"
        output.mkdir(parents=True, exist_ok=True)
        name = settings.save_name or f"F{settings.f_exp_no}_FO{settings.fo_exp_no}_O{settings.o_exp_no}"

        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "fit_parameters"
        headers = [
            "center_x",
            "center_y",
            "rotate_angle",
            "center_a",
            "center_b",
            "center_r2",
            "hori_a",
            "hori_b",
            "hori_c",
            "hori_r2",
            "vert_a",
            "vert_b",
            "vert_c",
            "vert_r2",
            "student_scale_x",
            "student_scale_y",
            "student_exponent",
            "student_r2",
            "compact_scale_x",
            "compact_scale_y",
            "compact_exponent",
            "compact_cutoff_radius",
            "compact_r2",
        ]
        sheet.append(["image", *headers])
        for image_name, row in zip(["f_img", "o_img", "fo_img"], result.fit_rows):
            sheet.append([image_name, *row])
        workbook.save(output / f"{name}-fit-results.xlsx")
        workbook.close()

        for curve_name, (x_values, values, predicted) in result.curve_data.items():
            figure = Figure(figsize=(8, 5), dpi=150)
            axis = figure.add_subplot(111)
            axis.scatter(x_values, values, s=6, label="Origin")
            axis.plot(x_values, predicted, color="red", label="Fit")
            axis.set_xlabel("dimensionless coordinate")
            axis.set_ylabel("normalized intensity")
            axis.grid(True, alpha=0.3)
            axis.legend()
            FigureCanvasAgg(figure)
            figure.savefig(output / f"{name}-{curve_name}.png", bbox_inches="tight")
        return output


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


def _merge_special_points(point_sets: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | list:
    valid_sets = [points for points in point_sets if len(points) == 2 and len(points[0])]
    if not valid_sets:
        return []
    return np.concatenate([points[0] for points in valid_sets]), np.concatenate([points[1] for points in valid_sets])


def _finite_values_or_none(values: list[float]) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]
