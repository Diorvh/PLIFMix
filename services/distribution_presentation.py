from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from PLIFMix.core.image_rendering import rotate_image_about_point, rotate_points_about_point
from PLIFMix.services.display_types import DisplayImage
from PLIFMix.services.distribution_results import DistributionSummaryRow
from PLIFMix.services.distribution_workers import SurfaceFitParameters


@dataclass(frozen=True)
class CenterLinePresentation:
    equation: str
    a_text: str
    b_text: str
    r_squared_text: str
    text: str


@dataclass(frozen=True)
class ProfilePresentation:
    equation: str
    a_text: str
    b_text: str
    c_text: str
    r_squared_text: str
    text: str


@dataclass(frozen=True)
class SurfacePresentation:
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
    text: str


def format_center_line_fit(
    a: float,
    b: float,
    r_squared: float,
) -> CenterLinePresentation:
    equation = "x=a(√(y²+b²)-b)"
    a_text = f"{a:.4f}"
    b_text = f"{b:.4f}"
    r_squared_text = f"{r_squared:.4f}"
    text = f"{equation}; a={a_text}; b={b_text}; R²={r_squared_text}"
    return CenterLinePresentation(equation, a_text, b_text, r_squared_text, text)


def format_center_line_failure(error: str) -> CenterLinePresentation:
    text = f"拟合失败：{error}"
    return CenterLinePresentation(text, "-", "-", "-", text)


def format_profile_fit(
    direction: str,
    a: float,
    b: float,
    c: float,
    r_squared: float,
) -> ProfilePresentation:
    axis = "x" if direction == "horizontal" else "y"
    equation = f"p=a/({axis}²b²+1)^c"
    a_text = f"{a:.4e}"
    b_text = f"{b:.4f}"
    c_text = f"{c:.4f}"
    r_squared_text = f"{r_squared:.4f}"
    text = (
        f"y={a * 1e4:.4f}E-4/"
        f"(x^2*{b:.4f}^2+1)^{c:.4f}; R²={r_squared:.4f}"
    )
    return ProfilePresentation(
        equation,
        a_text,
        b_text,
        c_text,
        r_squared_text,
        text,
    )


def format_profile_failure(error: str) -> ProfilePresentation:
    text = f"拟合失败：{error}"
    return ProfilePresentation(text, "-", "-", "-", "-", text)


def format_profile_not_executed() -> ProfilePresentation:
    text = "未执行：中心线拟合没有有效数据"
    return ProfilePresentation(text, "-", "-", "-", "-", text)


def format_surface_fits(
    image_name: str,
    student_fit: SurfaceFitParameters | None,
    compact_fit: SurfaceFitParameters | None,
    student_error: str | None,
    compact_error: str | None,
    executed: bool,
) -> SurfacePresentation:
    if executed:
        if student_fit is not None:
            student_equation = "p∝[1+(x/sₓ)²+(y/sᵧ)²]⁻ᵐ"
            student_scale_x = f"{student_fit.scale_x:.4f}"
            student_scale_y = f"{student_fit.scale_y:.4f}"
            student_exponent = f"{student_fit.exponent:.4f}"
            student_r_squared = f"{student_fit.r_squared:.4f}"
            surface_text = (
                f"{image_name}纯椭圆 Student："
                f"sₓ={student_scale_x}; sᵧ={student_scale_y}; "
                f"m={student_exponent}; R²={student_r_squared}\n"
            )
        else:
            student_error = student_error or "未知错误"
            student_equation = f"拟合失败：{student_error}"
            student_scale_x = student_scale_y = student_exponent = student_r_squared = "-"
            surface_text = f"{image_name}纯椭圆 Student：拟合失败\n"

        if compact_fit is not None:
            compact_equation = "p∝(1+rₑ²)⁻ᵐ(1-rₑ²/R²)₊²"
            compact_scale_x = f"{compact_fit.scale_x:.4f}"
            compact_scale_y = f"{compact_fit.scale_y:.4f}"
            compact_exponent = f"{compact_fit.exponent:.4f}"
            assert compact_fit.cutoff_radius is not None
            compact_cutoff_radius = f"{compact_fit.cutoff_radius:.4f}"
            compact_r_squared = f"{compact_fit.r_squared:.4f}"
            surface_text += (
                f"{image_name}紧支撑 Student："
                f"sₓ={compact_scale_x}; sᵧ={compact_scale_y}; "
                f"m={compact_exponent}; R={compact_cutoff_radius}; "
                f"R²={compact_r_squared}\n"
            )
        else:
            compact_error = compact_error or "未知错误"
            compact_equation = f"拟合失败：{compact_error}"
            compact_scale_x = compact_scale_y = compact_exponent = compact_cutoff_radius = "-"
            compact_r_squared = "-"
            surface_text += f"{image_name}紧支撑 Student：拟合失败\n"
    else:
        student_equation = compact_equation = "未执行"
        student_scale_x = student_scale_y = student_exponent = student_r_squared = "-"
        compact_scale_x = compact_scale_y = compact_exponent = compact_cutoff_radius = "-"
        compact_r_squared = "-"
        surface_text = f"{image_name}二维 Student 拟合：未执行\n"

    return SurfacePresentation(
        student_equation,
        student_scale_x,
        student_scale_y,
        student_exponent,
        student_r_squared,
        compact_equation,
        compact_scale_x,
        compact_scale_y,
        compact_exponent,
        compact_cutoff_radius,
        compact_r_squared,
        surface_text,
    )


def synthesize_distribution_image(
    *,
    display_name: str,
    image_mode: str,
    show_bent: bool,
    need_symmetric_restore: bool,
    norm_img: np.ndarray,
    source_total: float,
    image_shape: tuple[int, ...],
    center_x: int,
    center_y: int,
    rotate_angle: float,
    fitted_point: tuple[np.ndarray, np.ndarray],
    symmetric_fitted_point: tuple[np.ndarray, np.ndarray],
    bent_img: np.ndarray | None,
    surface_prediction: np.ndarray | None,
    point_sets: list[tuple[np.ndarray, np.ndarray]],
    map_center_line_points: Callable,
    unstraighten_center_line_image: Callable,
) -> DisplayImage:
    # The fitted frame rotates sampled points by +rotate_angle, so the matching
    # display of that frame turns the image the other way about the same origin.
    display_rotation = -float(rotate_angle)
    if image_mode == "rot_img":
        point_sets = [
            rotate_points_about_point(points, (center_x, center_y), display_rotation, image_shape)
            for points in point_sets
        ]
    elif show_bent:
        point_sets = [
            map_center_line_points(points, fitted_point, center_x, image_shape, inverse=False)
            for points in point_sets
        ]
    elif need_symmetric_restore:
        restored_point_sets = []
        for points in point_sets:
            bent_points = map_center_line_points(
                points,
                fitted_point,
                center_x,
                image_shape,
                inverse=False,
            )
            restored_point_sets.append(
                map_center_line_points(
                    bent_points,
                    symmetric_fitted_point,
                    center_x,
                    image_shape,
                    inverse=True,
                )
            )
        point_sets = restored_point_sets
    special_points = _merge_special_points(point_sets)

    if image_mode == "rot_img":
        shown_normalized = rotate_image_about_point(
            norm_img,
            (center_x, center_y),
            display_rotation,
        )
    elif image_mode == "ori_img":
        assert not show_bent or bent_img is not None
        shown_normalized = bent_img if show_bent else norm_img
    elif image_mode == "fit_img":
        assert surface_prediction is not None
        shown_normalized = (
            surface_prediction
            if show_bent
            else unstraighten_center_line_image(
                surface_prediction,
                symmetric_fitted_point,
                center_x,
            )
        )
    else:
        assert surface_prediction is not None and bent_img is not None
        bent_difference = np.abs(surface_prediction - bent_img)
        shown_normalized = (
            bent_difference
            if show_bent
            else unstraighten_center_line_image(
                bent_difference,
                symmetric_fitted_point,
                center_x,
            )
        )
    return DisplayImage(
        display_name,
        shown_normalized * source_total,
        "jet",
        special_points,
    )


def build_summary_row(
    *,
    display_name: str,
    contour_value: float,
    has_contour: bool,
    center: CenterLinePresentation,
    horizontal: ProfilePresentation,
    vertical: ProfilePresentation,
    surface: SurfacePresentation,
) -> DistributionSummaryRow:
    return DistributionSummaryRow(
        image_name=display_name.upper(),
        contour=f"{contour_value:g}" if has_contour else "未计算",
        center_line=center.equation,
        center_a=center.a_text,
        center_b=center.b_text,
        center_r_squared=center.r_squared_text,
        horizontal_equation=horizontal.equation,
        horizontal_a=horizontal.a_text,
        horizontal_b=horizontal.b_text,
        horizontal_c=horizontal.c_text,
        horizontal_r_squared=horizontal.r_squared_text,
        vertical_equation=vertical.equation,
        vertical_a=vertical.a_text,
        vertical_b=vertical.b_text,
        vertical_c=vertical.c_text,
        vertical_r_squared=vertical.r_squared_text,
        student_equation=surface.student_equation,
        student_scale_x=surface.student_scale_x,
        student_scale_y=surface.student_scale_y,
        student_exponent=surface.student_exponent,
        student_r_squared=surface.student_r_squared,
        compact_equation=surface.compact_equation,
        compact_scale_x=surface.compact_scale_x,
        compact_scale_y=surface.compact_scale_y,
        compact_exponent=surface.compact_exponent,
        compact_cutoff_radius=surface.compact_cutoff_radius,
        compact_r_squared=surface.compact_r_squared,
    )


def _merge_special_points(
    point_sets: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray] | list:
    valid_sets = [points for points in point_sets if len(points) == 2 and len(points[0])]
    if not valid_sets:
        return []
    return np.concatenate([points[0] for points in valid_sets]), np.concatenate([points[1] for points in valid_sets])


__all__ = [
    "CenterLinePresentation",
    "ProfilePresentation",
    "SurfacePresentation",
    "build_summary_row",
    "format_center_line_failure",
    "format_center_line_fit",
    "format_profile_failure",
    "format_profile_fit",
    "format_profile_not_executed",
    "format_surface_fits",
    "synthesize_distribution_image",
]
