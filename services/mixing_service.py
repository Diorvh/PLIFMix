from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openpyxl

from PLIFv2.algorithms.mixing import (
    ShiftSearchResult,
    calc_mixing_efficiency,
    calculate_shift_search,
    generate_mask_coords,
    shift_image,
)
from PLIFv2.core.image_io import read_image
from PLIFv2.domain.settings import ManualMixingSettings, ShiftSearchSettings
from PLIFv2.services.display_types import DisplayImage


@dataclass
class ManualMixingResult:
    abs_value: float
    me_o_fo: float
    me_f_fo: float
    me_o_f: float
    average_me: float
    record_of: float
    raw_me_o_fo: float
    raw_me_f_fo: float
    raw_me_o_f: float
    raw_average_me: float


@dataclass
class ManualDisplayResult:
    summary: ManualMixingResult
    images: list[DisplayImage]
    mix_curve: tuple[np.ndarray, np.ndarray]
    raw_images: list[DisplayImage]
    raw_mix_curve: tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class _ImagePair:
    source: np.ndarray
    total: np.ndarray


@dataclass(frozen=True)
class _MixingImages:
    f: np.ndarray
    o: np.ndarray
    fo: np.ndarray
    o_f: _ImagePair
    f_fo: _ImagePair
    o_fo: _ImagePair


@dataclass(frozen=True)
class _MixingMetrics:
    abs_value: float
    me_o_fo: float
    me_f_fo: float
    me_o_f: float


class MixingService:
    def run_shift_search(self, settings: ShiftSearchSettings) -> ShiftSearchResult:
        image_folder = Path(settings.image_folder)
        images = (
            read_image(_exp_path(image_folder, settings.f_exp_no)).astype(np.float64),
            read_image(_exp_path(image_folder, settings.o_exp_no)).astype(np.float64),
            read_image(_exp_path(image_folder, settings.fo_exp_no)).astype(np.float64),
        )
        global_mix_ratio = settings.global_of / (1 + settings.global_of)
        return calculate_shift_search(
            images=images,
            f_range=(settings.f_min, settings.f_max + 1),
            o_range=(settings.o_min, settings.o_max + 1),
            ellipse=(settings.ellipse_x, settings.ellipse_y, settings.major_axis, settings.minor_axis),
            global_mix_ratio=global_mix_ratio,
            process_count=settings.process_count,
        )

    def run_manual(self, settings: ManualMixingSettings) -> ManualMixingResult:
        return self.run_manual_display(settings).summary

    def run_manual_display(self, settings: ManualMixingSettings) -> ManualDisplayResult:
        image_folder = Path(settings.image_folder)
        if not str(settings.fo_exp_no).strip():
            raise ValueError("手动调参计算混合特性需要输入 F、O 和 FO 图像编号。")

        f_img = read_image(_exp_path(image_folder, settings.f_exp_no)).astype(np.float64)
        o_img = read_image(_exp_path(image_folder, settings.o_exp_no)).astype(np.float64)
        fo_img = read_image(_exp_path(image_folder, settings.fo_exp_no)).astype(np.float64)
        f_img = shift_image(f_img, settings.f_shift)
        o_img = shift_image(o_img, settings.o_shift)

        mask = (f_img < (np.max(f_img) * settings.denoise_ratio)) & (o_img < (np.max(o_img) * settings.denoise_ratio))
        f_img[mask] = 0
        o_img[mask] = 0
        fo_img[mask] = 0

        raw_images = _create_raw_mixing_images(f_img, o_img, fo_img)

        if settings.use_ellipse_region:
            coords = generate_mask_coords(
                settings.center_xfo,
                settings.center_yfo,
                settings.major_axis,
                settings.minor_axis,
                f_img.shape,
            )
        else:
            ys, xs = np.indices(f_img.shape)
            coords = xs.ravel(), ys.ravel()
        global_mix_ratio = settings.global_of / (1 + settings.global_of)
        raw_metrics = _calculate_mixing_metrics(raw_images, coords, global_mix_ratio)
        record_of = float(np.sum(o_img) / np.sum(f_img)) if np.sum(f_img) else 0.0
        calibrated_images = _create_calibrated_mixing_images(f_img, o_img, fo_img, settings.global_of)
        calibrated_metrics = _calculate_mixing_metrics(calibrated_images, coords, global_mix_ratio)
        summary = ManualMixingResult(
            abs_value=calibrated_metrics.abs_value,
            me_o_fo=calibrated_metrics.me_o_fo,
            me_f_fo=calibrated_metrics.me_f_fo,
            me_o_f=calibrated_metrics.me_o_f,
            average_me=(calibrated_metrics.me_o_fo + calibrated_metrics.me_f_fo + calibrated_metrics.me_o_f) / 3,
            record_of=record_of,
            raw_me_o_fo=raw_metrics.me_o_fo,
            raw_me_f_fo=raw_metrics.me_f_fo,
            raw_me_o_f=raw_metrics.me_o_f,
            raw_average_me=(raw_metrics.me_o_fo + raw_metrics.me_f_fo + raw_metrics.me_o_f) / 3,
        )
        display_images, mix_curve = _build_display_outputs(
            calibrated_images,
            settings.global_of,
            settings.center_yfo,
            global_mix_ratio,
        )
        raw_display_images, raw_mix_curve = _build_display_outputs(
            raw_images,
            settings.global_of,
            settings.center_yfo,
            global_mix_ratio,
        )
        return ManualDisplayResult(summary, display_images, mix_curve, raw_display_images, raw_mix_curve)

    def export_mix_curve(
        self,
        settings: ManualMixingSettings,
        result: ManualDisplayResult,
        use_calibrated: bool = True,
        output_name: str | None = None,
    ) -> Path:
        output = Path(settings.image_folder).parent / "mix_curve"
        output.mkdir(parents=True, exist_ok=True)
        name = output_name or settings.save_name or f"F{settings.f_exp_no}_O{settings.o_exp_no}_FO{settings.fo_exp_no}"

        curve_workbook = openpyxl.Workbook()
        curve_sheet = curve_workbook.active
        curve_sheet.title = "mix_fraction"
        curve_sheet.append(["x", "mix_fraction"])
        mix_curve = result.mix_curve if use_calibrated else result.raw_mix_curve
        for x_value, y_value in zip(*mix_curve):
            curve_sheet.append([float(x_value), float(y_value)])
        curve_path = output / f"{name}.xlsx"
        curve_workbook.save(curve_path)
        curve_workbook.close()
        return curve_path


def _build_display_outputs(
    images: _MixingImages,
    global_of: float,
    center_y: int,
    global_mix_ratio: float,
) -> tuple[list[DisplayImage], tuple[np.ndarray, np.ndarray]]:
    f_img = images.f
    o_img = images.o
    fo_img = images.fo
    combined = f_img + o_img
    display_images = [
        DisplayImage("f", f_img),
        DisplayImage("o", o_img),
        DisplayImage("fo", fo_img),
        DisplayImage("f+o-fo", combined - fo_img),
        DisplayImage("fo-f-o", fo_img - combined),
    ]

    with np.errstate(divide="ignore", invalid="ignore"):
        mix_ratio_o_f = np.where(
            (f_img > 0) | (o_img > 0),
            (o_img / (o_img + global_of * f_img) * 254 + 1).astype(np.uint8),
            0,
        )
        mix_ratio_f_fo = np.where(
            (images.f_fo.source > 0) | ((images.f_fo.total - images.f_fo.source) > 0),
            (
                (images.f_fo.total - images.f_fo.source)
                / ((images.f_fo.total - images.f_fo.source) + global_of * images.f_fo.source)
                * 254
                + 1
            ).astype(np.uint8),
            0,
        )
        mix_ratio_o_fo = np.where(
            ((images.o_fo.total - images.o_fo.source) > 0) | (images.o_fo.source > 0),
            (
                images.o_fo.source
                / (images.o_fo.source + global_of * (images.o_fo.total - images.o_fo.source))
                * 254
                + 1
            ).astype(np.uint8),
            0,
        )
    mix_ratio_f_fo[images.f_fo.total - images.f_fo.source < 0] = 1
    mix_ratio_o_fo[images.o_fo.total - images.o_fo.source < 0] = 255

    _, mix_efficiency_o_f = calc_mixing_efficiency(images.o_f.source, images.o_f.total, global_mix_ratio)
    _, mix_efficiency_f_fo = calc_mixing_efficiency(images.f_fo.source, images.f_fo.total, 1 - global_mix_ratio)
    _, mix_efficiency_o_fo = calc_mixing_efficiency(images.o_fo.source, images.o_fo.total, global_mix_ratio)
    max_eff = max(
        float(np.max(mix_efficiency_o_f)),
        float(np.max(mix_efficiency_f_fo)),
        float(np.max(mix_efficiency_o_fo)),
    )
    if max_eff > 0:
        mix_efficiency_o_f = np.array(mix_efficiency_o_f / max_eff * 255).astype(np.uint8)
        mix_efficiency_f_fo = np.array(mix_efficiency_f_fo / max_eff * 255).astype(np.uint8)
        mix_efficiency_o_fo = np.array(mix_efficiency_o_fo / max_eff * 255).astype(np.uint8)
    else:
        mix_efficiency_o_f = np.zeros_like(mix_efficiency_o_f, dtype=np.uint8)
        mix_efficiency_f_fo = np.zeros_like(mix_efficiency_f_fo, dtype=np.uint8)
        mix_efficiency_o_fo = np.zeros_like(mix_efficiency_o_fo, dtype=np.uint8)
    display_images.extend(
        [
            DisplayImage("oDf+o", mix_ratio_o_f, "bwr"),
            DisplayImage("fDfo", mix_ratio_f_fo, "bwr"),
            DisplayImage("oDfo", mix_ratio_o_fo, "bwr"),
            DisplayImage("ME_oDf+o", mix_efficiency_o_f),
            DisplayImage("ME_fDfo", mix_efficiency_f_fo),
            DisplayImage("ME_oDfo", mix_efficiency_o_fo),
        ]
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        mix_curve_origin = np.where(
            (f_img > 0) | (o_img > 0),
            (o_img / (o_img + global_of * f_img)) + 1,
            0,
        )
    row = int(np.clip(center_y, 0, f_img.shape[0] - 1))
    curve_x = np.arange(f_img.shape[1])
    curve_y = mix_curve_origin[row, :]
    mask_curve = curve_y != 0
    return display_images, (curve_x[mask_curve], curve_y[mask_curve] - 1)


def _create_raw_mixing_images(f_img: np.ndarray, o_img: np.ndarray, fo_img: np.ndarray) -> _MixingImages:
    return _MixingImages(
        f=f_img,
        o=o_img,
        fo=fo_img,
        o_f=_ImagePair(o_img, f_img + o_img),
        f_fo=_ImagePair(f_img, fo_img),
        o_fo=_ImagePair(o_img, fo_img),
    )


def _create_calibrated_mixing_images(
    f_img: np.ndarray,
    o_img: np.ndarray,
    fo_img: np.ndarray,
    global_of: float,
) -> _MixingImages:
    calibrated_o = _scale_to_total(o_img, np.sum(f_img) * global_of)
    # This target matches the existing F/O-calibrated FO total.
    calibrated_fo = _scale_to_total(fo_img, np.sum(f_img) * (1 + global_of))
    # Keep O unchanged for this pair and set O:FO = O/F:(1 + O/F).
    o_fo_target_sum = np.sum(o_img) * (1 + global_of) / global_of if global_of else 0.0
    o_fo_calibrated = _scale_to_total(fo_img, o_fo_target_sum)
    return _MixingImages(
        f=f_img,
        o=calibrated_o,
        fo=calibrated_fo,
        o_f=_ImagePair(calibrated_o, f_img + calibrated_o),
        f_fo=_ImagePair(f_img, calibrated_fo),
        o_fo=_ImagePair(o_img, o_fo_calibrated),
    )


def _scale_to_total(image: np.ndarray, target_sum: float) -> np.ndarray:
    image_sum = float(np.sum(image))
    if image_sum <= 0:
        return image.copy()
    return image * (target_sum / image_sum)


def _calculate_mixing_metrics(
    images: _MixingImages,
    coords: tuple[np.ndarray, np.ndarray],
    global_mix_ratio: float,
) -> _MixingMetrics:
    xs, ys = coords
    f_values = images.f[ys, xs].astype(np.float64)
    o_values = images.o[ys, xs].astype(np.float64)
    fo_values = images.fo[ys, xs].astype(np.float64)
    o_fo_source = images.o_fo.source[ys, xs].astype(np.float64)
    o_fo_total = images.o_fo.total[ys, xs].astype(np.float64)
    f_fo_source = images.f_fo.source[ys, xs].astype(np.float64)
    f_fo_total = images.f_fo.total[ys, xs].astype(np.float64)
    me_o_fo, _ = calc_mixing_efficiency(o_fo_source, o_fo_total, global_mix_ratio)
    me_f_fo, _ = calc_mixing_efficiency(f_fo_source, f_fo_total, 1 - global_mix_ratio)
    me_o_f, _ = calc_mixing_efficiency(o_values, f_values + o_values, global_mix_ratio)
    abs_value = float(np.sum(np.abs(f_values + o_values - fo_values)))
    return _MixingMetrics(abs_value, me_o_fo, me_f_fo, me_o_f)


def _exp_path(folder: Path, exp_no: str) -> Path:
    if not str(exp_no).strip():
        raise ValueError("需要输入图像编号。")
    stem = str(exp_no) if str(exp_no).startswith("实验编号") else f"实验编号{exp_no}"
    return folder / f"{stem}.npy"
