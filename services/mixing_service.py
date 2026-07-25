from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openpyxl

from PLIFv2.algorithms.mixing import (
    ShiftSearchResult,
    calc_mixing_efficiency,
    calc_result,
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


@dataclass
class ManualDisplayResult:
    summary: ManualMixingResult
    images: list[DisplayImage]
    mix_curve: tuple[np.ndarray, np.ndarray]


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

        record_of = float(np.sum(o_img) / np.sum(f_img)) if np.sum(f_img) else 0.0
        if record_of:
            o_img = o_img * settings.global_of / record_of
        combined = f_img + o_img
        fo_sum = np.sum(fo_img)
        if fo_sum:
            fo_img = fo_img / fo_sum * np.sum(combined)

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
        abs_value, me_o_fo, me_f_fo, me_o_f = calc_result(f_img, o_img, fo_img, coords, global_mix_ratio)
        summary = ManualMixingResult(
            abs_value=abs_value,
            me_o_fo=me_o_fo,
            me_f_fo=me_f_fo,
            me_o_f=me_o_f,
            average_me=(me_o_fo + me_f_fo + me_o_f) / 3,
            record_of=record_of,
        )

        display_images = [
            DisplayImage("f", f_img),
            DisplayImage("o", o_img),
            DisplayImage("fo", fo_img),
        ]
        diff = combined - fo_img
        display_images.extend(
            [
                DisplayImage("f+o-fo", diff),
                DisplayImage("fo-f-o", -diff),
            ]
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            mix_ratio_o_f = np.where(
                (f_img > 0) | (o_img > 0),
                (o_img / (o_img + settings.global_of * f_img) * 254 + 1).astype(np.uint8),
                0,
            )
            mix_ratio_f_fo = np.where(
                (f_img > 0) | ((fo_img - f_img) > 0),
                ((fo_img - f_img) / ((fo_img - f_img) + settings.global_of * f_img) * 254 + 1).astype(np.uint8),
                0,
            )
            mix_ratio_o_fo = np.where(
                ((fo_img - o_img) > 0) | (o_img > 0),
                (o_img / (o_img + settings.global_of * (fo_img - o_img)) * 254 + 1).astype(np.uint8),
                0,
            )
        mix_ratio_f_fo[fo_img - f_img < 0] = 1
        mix_ratio_o_fo[fo_img - o_img < 0] = 255

        _, mix_efficiency_o_f = calc_mixing_efficiency(o_img, combined, global_mix_ratio)
        _, mix_efficiency_f_fo = calc_mixing_efficiency(f_img, fo_img, global_mix_ratio)
        _, mix_efficiency_o_fo = calc_mixing_efficiency(o_img, fo_img, global_mix_ratio)
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
                (o_img / (o_img + settings.global_of * f_img)) + 1,
                0,
            )
        row = int(np.clip(settings.center_yfo, 0, f_img.shape[0] - 1))
        curve_x = np.array(range(f_img.shape[1]))
        curve_y = mix_curve_origin[row, :]
        mask_curve = curve_y != 0
        curve_x = curve_x[mask_curve]
        curve_y = curve_y[mask_curve] - 1
        return ManualDisplayResult(summary, display_images, (curve_x, curve_y))

    def export_mix_curve(self, settings: ManualMixingSettings, result: ManualDisplayResult) -> Path:
        output = Path(settings.image_folder).parent / "mix_curve"
        output.mkdir(parents=True, exist_ok=True)
        name = settings.save_name or f"F{settings.f_exp_no}_O{settings.o_exp_no}_FO{settings.fo_exp_no}"

        curve_workbook = openpyxl.Workbook()
        curve_sheet = curve_workbook.active
        curve_sheet.title = "mix_fraction"
        curve_sheet.append(["x", "mix_fraction"])
        for x_value, y_value in zip(*result.mix_curve):
            curve_sheet.append([float(x_value), float(y_value)])
        curve_path = output / f"{name}.xlsx"
        curve_workbook.save(curve_path)
        curve_workbook.close()
        return curve_path


def _exp_path(folder: Path, exp_no: str) -> Path:
    if not str(exp_no).strip():
        raise ValueError("需要输入图像编号。")
    stem = str(exp_no) if str(exp_no).startswith("实验编号") else f"实验编号{exp_no}"
    return folder / f"{stem}.npy"
