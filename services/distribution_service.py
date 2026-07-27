from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openpyxl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from PLIFv2.algorithms.mixing import (
    calc_flowrate_contour,
    fit_center_line,
    func_fit_3,
    rotate_points_clockwise,
    shift_image,
    straighten_center_line_image,
)
from PLIFv2.core.image_io import read_image
from PLIFv2.core.paths import path_is_set
from PLIFv2.domain.settings import DistributionSettings
from PLIFv2.services.display_types import DisplayImage

@dataclass
class DistributionFitResult:
    fit_text: str
    surface_text: str
    images: list[DisplayImage]
    raw_images: list[DisplayImage]
    fit_rows: list[list[float]]
    curve_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]


class DistributionFitService:
    """Fit calibrated individual F, O, and FO mass distributions without mixing metrics."""

    def run(
        self,
        settings: DistributionSettings,
        annotations: frozenset[str] = frozenset(),
        image_mode: str = "ori_img",
    ) -> DistributionFitResult:
        image_folder = Path(settings.image_folder)
        f_img = read_image(_exp_path(image_folder, settings.f_exp_no)).astype(np.float64)
        o_img = read_image(_exp_path(image_folder, settings.o_exp_no)).astype(np.float64)
        fo_img = read_image(_exp_path(image_folder, settings.fo_exp_no)).astype(np.float64)
        f_img = shift_image(f_img, settings.f_shift)
        o_img = shift_image(o_img, settings.o_shift)

        mask = (f_img < (np.max(f_img) * settings.denoise_ratio)) & (o_img < (np.max(o_img) * settings.denoise_ratio))
        f_img[mask] = 0
        o_img[mask] = 0
        fo_img[mask] = 0
        raw_source_images = (f_img, o_img, fo_img)
        f_img, o_img, fo_img = _calibrate_total_images(f_img, o_img, fo_img, settings.global_of)

        fit_text = ""
        surface_text = ""
        display_images: list[DisplayImage] = []
        raw_display_images: list[DisplayImage] = []
        fit_rows: list[list[float]] = []
        curve_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
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
        height, width = f_img.shape
        need_surface_fit = image_mode in {"fit_img", "delta_img"}
        grid_x = grid_y = None
        if need_surface_fit:
            grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")

        for index, (image, raw_image) in enumerate(zip([f_img, o_img, fo_img], raw_source_images)):
            cx, cy = centers[index]
            q_center_up, q_center_down = center_line_bounds[index]
            contour_value = 0.0
            boundary_points: set[tuple[int, int]] = set()
            if "contour" in annotations:
                contour_value, boundary_points = calc_flowrate_contour(image, settings.thres_contour)

            a, b, r_squared, origin_point, fitted_point, verti_px_rotated = fit_center_line(
                image,
                q_center_up,
                q_center_down,
                settings.max_k,
                cx,
                cy,
                rotate_angles[index],
            )
            fit_text += (
                f"{image_names[index]}流量轮廓值:{contour_value};"
                f"流量中心线方程：y={a:.4f}*sqrt(x^2+{b * b:.4f})-{a:.4f}*{b:.4f}; R2={r_squared:.4f}     "
            )

            norm_coeff = np.sum(image)
            norm_img = image / norm_coeff if norm_coeff else image
            hori_px = np.array(range(-settings.hori_fit_width, settings.hori_fit_width))
            hori_py = np.zeros_like(hori_px)
            hori_px_rotated, hori_py_rotated = rotate_points_clockwise(hori_px, hori_py, -rotate_angles[index])
            sample_y = np.clip(np.round(-hori_py_rotated + cy).astype(np.int32), 0, height - 1)
            sample_x = np.clip(np.round(hori_px_rotated + cx).astype(np.int32), 0, width - 1)
            hori_intensity = norm_img[sample_y, sample_x]

            def fit_func(x: np.ndarray, aa: float, bb: float, cc: float) -> np.ndarray:
                return aa / (x**2 * bb**2 + 1) ** cc

            ah, bh, ch, r_squared_h, hori_predicted = func_fit_3(hori_px, hori_intensity, fit_func)
            curve_data[f"横向分布_{image_names[index]}"] = (hori_px, hori_intensity, hori_predicted)
            fit_text += f"{image_names[index]}横向强度分布拟合：y={ah * 1e4:.4f}E-4/(x^2*{bh:.4f}^2+1)^{ch:.4f}; R2={r_squared_h:.4f}     "

            verti_py, verti_px = origin_point
            if len(verti_py):
                verti_intensity = norm_img[verti_py, verti_px]
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
            else:
                av, bv, cv, r_squared_v = 0.0, 0.0, 0.0, 0.0
            fit_text += f"{image_names[index]}纵向强度分布拟合：y={av * 1e4:.4f}E-4/(x^2*{bv:.4f}^2+1)^{cv:.4f}; R2={r_squared_v:.4f}\n"

            best_rho = 0.0
            r_squared_p = 0.0
            bent_img: np.ndarray | None = None
            surface_prediction: np.ndarray | None = None
            if image_mode != "ori_img":
                bent_img = straighten_center_line_image(norm_img, fitted_point, cx)

            if need_surface_fit:
                assert grid_x is not None and grid_y is not None and bent_img is not None
                pixel_x = grid_x - cx
                pixel_y = grid_y - cy
                aa = (ah + av) / 2
                static_surface = ((pixel_x * pixel_x * bh * bh + 1) ** ch - 1) + ((pixel_y * pixel_y * bv * bv + 1) ** cv - 1)
                coupling_base = np.abs(pixel_x * bh) * np.abs(pixel_y * bv)
                objective_buffer = np.empty_like(static_surface)

                def predict_img_func(rho: float) -> np.ndarray:
                    denominator = static_surface + np.power(coupling_base, rho)
                    denominator += 1
                    pred = aa / denominator
                    pred_sum = np.sum(pred)
                    return pred / pred_sum if pred_sum else pred

                try:
                    from scipy import optimize

                    def object_loss(rho: float) -> float:
                        np.power(coupling_base, rho, out=objective_buffer)
                        np.add(static_surface, objective_buffer, out=objective_buffer)
                        np.add(objective_buffer, 1, out=objective_buffer)
                        np.divide(aa, objective_buffer, out=objective_buffer)
                        np.divide(objective_buffer, np.sum(objective_buffer), out=objective_buffer)
                        np.subtract(bent_img, objective_buffer, out=objective_buffer)
                        np.square(objective_buffer, out=objective_buffer)
                        return float(np.sum(objective_buffer))

                    res = optimize.minimize_scalar(object_loss, bounds=(0.01, 100), method="bounded")
                    ss_tot = np.sum((bent_img - np.mean(bent_img)) ** 2)
                    best_rho = float(res.x)
                    r_squared_p = float(1 - (res.fun / ss_tot)) if ss_tot != 0 else 0.0
                except Exception:
                    best_rho = 0.0
                    r_squared_p = 0.0
                surface_prediction = predict_img_func(best_rho)
            surface_text += f"{image_names[index]}横纵向联合分布拟合：rho={best_rho:.4f}; R2={r_squared_p:.4f}\n"
            fit_rows.append(
                [
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
                    best_rho,
                    r_squared_p,
                ]
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
            display_images.append(DisplayImage(display_names[index], shown_normalized * norm_coeff, "jet", special_points))
            raw_display_images.append(
                DisplayImage(display_names[index], shown_normalized * np.sum(raw_image), "jet", special_points)
            )

        return DistributionFitResult(fit_text, surface_text, display_images, raw_display_images, fit_rows, curve_data)

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
            "rho",
            "surface_r2",
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


def _calibrate_total_images(
    f_img: np.ndarray,
    o_img: np.ndarray,
    fo_img: np.ndarray,
    global_of: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    record_of = float(np.sum(o_img) / np.sum(f_img)) if np.sum(f_img) else 0.0
    calibrated_o = o_img * global_of / record_of if record_of else o_img
    combined = f_img + calibrated_o
    fo_sum = float(np.sum(fo_img))
    calibrated_fo = fo_img / fo_sum * np.sum(combined) if fo_sum else fo_img
    return f_img, calibrated_o, calibrated_fo


def _merge_special_points(point_sets: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | list:
    valid_sets = [points for points in point_sets if len(points) == 2 and len(points[0])]
    if not valid_sets:
        return []
    return np.concatenate([points[0] for points in valid_sets]), np.concatenate([points[1] for points in valid_sets])


def _exp_path(folder: Path, exp_no: str) -> Path:
    if not str(exp_no).strip():
        raise ValueError("质量分布拟合需要输入 F、O 和 FO 图像编号。")
    stem = str(exp_no) if str(exp_no).startswith("实验编号") else f"实验编号{exp_no}"
    return folder / f"{stem}.npy"
