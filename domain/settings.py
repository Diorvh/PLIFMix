from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AverageSettings:
    source_folder: Path = Path()
    output_folder: Path = Path()
    experiment_numbers: str = ""
    use_experiment_numbers: bool = True
    best_process_count: int = 4
    denoise_threshold: float = 2
    sample_frequency: float = 400
    time_start: float = 8
    time_end: float = 10
    use_frame_segments: bool = False
    frame_segments: str = ""


@dataclass
class GainCorrectionSettings:
    image_folder: Path = Path()
    matrix_folder: Path = Path()
    output_folder: Path = Path()
    matrix_path: Path = Path()
    operation: str = "multiply"
    excel_path: Path = Path()
    experiment_column: int = 1
    gain_column: int = 4
    gain_scale: float = 100
    matrix_name_template: str = "Gain{gain}.npy"
    include_tif: bool = False


@dataclass
class DavisSettings:
    corrected_folder: Path = Path()
    davis_input_folder: Path = Path()
    davis_export_folder: Path = Path()
    output_folder: Path = Path()
    restore_gain: bool = True


@dataclass
class CalibrationSettings:
    source_folder: Path = Path()
    output_folder: Path = Path()
    display_vmax: float = 0
    analysis_radius: int = 550
    compare_folder: Path = Path()
    compare_files: str = ""
    compare_center_x: int = 640
    compare_center_y: int = 600
    compare_radius: int = 550
    single_image_path: Path = Path()


@dataclass
class NoiseSettings:
    source_folder: Path = Path()
    frame_start: int = 0
    frame_end: int = 1000
    process_count: int = 4


@dataclass
class DistributionSettings:
    image_folder: Path = Path()
    f_exp_no: str = ""
    o_exp_no: str = ""
    save_name: str = ""
    zoom: float = 3
    gain: float = 0
    center_xf: int = 0
    center_yf: int = 0
    center_xo: int = 0
    center_yo: int = 0
    line_width: int = 2
    denoise_ratio: float = 0.01
    thres_contour: float = 0.9
    q_center_up: int = 0
    q_center_down: int = 0
    max_k: int = 30
    rotate_angle_f: float = 0
    rotate_angle_o: float = 0
    hori_fit_width: int = 200
    output_folder: Path = Path()


@dataclass
class ShiftSearchSettings:
    image_folder: Path = Path()
    f_exp_no: str = ""
    o_exp_no: str = ""
    fo_exp_no: str = ""
    f_min: int = -20
    f_max: int = 20
    o_min: int = -20
    o_max: int = 20
    ellipse_x: int = 0
    ellipse_y: int = 0
    major_axis: int = 1000
    minor_axis: int = 1000
    global_of: float = 1.0
    process_count: int = 4


@dataclass
class ManualMixingSettings:
    image_folder: Path = Path()
    f_exp_no: str = ""
    o_exp_no: str = ""
    fo_exp_no: str = ""
    save_name: str = ""
    zoom: float = 3
    gain: float = 0
    f_shift: int = 0
    o_shift: int = 0
    center_xf: int = 0
    center_yf: int = 0
    center_xo: int = 0
    center_yo: int = 0
    center_xfo: int = 0
    center_yfo: int = 0
    major_axis: int = 1000
    minor_axis: int = 1000
    line_width: int = 2
    global_of: float = 1.0
    denoise_ratio: float = 0.01
    thres_contour: float = 0.9
    q_center_up: int = 0
    q_center_down: int = 0
    max_k: int = 30
    rotate_angle_f: float = 0
    rotate_angle_o: float = 0
    rotate_angle_fo: float = 0
    output_folder: Path = Path()
    hori_fit_width: int = 200


@dataclass
class AppSettings:
    average: AverageSettings = field(default_factory=AverageSettings)
    gain: GainCorrectionSettings = field(default_factory=GainCorrectionSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    davis: DavisSettings = field(default_factory=DavisSettings)
    noise: NoiseSettings = field(default_factory=NoiseSettings)
    shift: ShiftSearchSettings = field(default_factory=ShiftSearchSettings)
    manual: ManualMixingSettings = field(default_factory=ManualMixingSettings)
    distribution: DistributionSettings = field(default_factory=DistributionSettings)


def dataclass_to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: dataclass_to_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {key: dataclass_to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [dataclass_to_jsonable(item) for item in value]
    return value
