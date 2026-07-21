from __future__ import annotations

from copy import deepcopy

from PLIFv2.application.app_state import AppState
from PLIFv2.services.average_service import AverageImageService, AverageResult
from PLIFv2.services.calibration_service import CalibrationService
from PLIFv2.services.davis_service import DavisImportResult, DavisService
from PLIFv2.services.intensifier_service import CorrectionResult, IntensifierCorrectionService, MatrixFitResult
from PLIFv2.services.mixing_service import ManualDisplayResult, ManualMixingResult, MixingService
from PLIFv2.services.noise_service import NoiseAnalysisService
from PLIFv2.algorithms.mixing import ShiftSearchResult


class PLIFActions:
    def __init__(self, state: AppState):
        self.state = state
        self.average_service = AverageImageService()
        self.calibration_service = CalibrationService()
        self.intensifier_service = IntensifierCorrectionService()
        self.davis_service = DavisService()
        self.mixing_service = MixingService()
        self.noise_service = NoiseAnalysisService()

    def run_average(self) -> AverageResult:
        result = self.average_service.run(deepcopy(self.state.settings.average))
        self.state.save()
        return result

    def apply_intensifier_correction(self) -> CorrectionResult:
        result = self.intensifier_service.apply_folder(deepcopy(self.state.settings.gain))
        self.state.save()
        return result

    def apply_selected_intensifier_correction(self, paths) -> CorrectionResult:
        result = self.intensifier_service.apply_paths(deepcopy(self.state.settings.gain), paths)
        self.state.save()
        return result

    def apply_excel_intensifier_correction(self) -> CorrectionResult:
        result = self.intensifier_service.apply_by_excel(deepcopy(self.state.settings.gain))
        self.state.save()
        return result

    def fit_intensifier_matrix(self) -> MatrixFitResult:
        settings = deepcopy(self.state.settings.gain)
        result = self.intensifier_service.fit_matrix_from_folder(settings.image_folder, settings.output_folder)
        self.state.save()
        return result

    def analyze_calibration(self):
        result = self.calibration_service.analyze(deepcopy(self.state.settings.calibration))
        self.state.save()
        return result

    def save_calibration_matrix(self, result):
        outputs = self.calibration_service.save_correction(self.state.settings.calibration, result)
        self.state.save()
        return outputs

    def compare_calibration_matrices(self):
        result = self.calibration_service.compare_matrices(deepcopy(self.state.settings.calibration))
        self.state.save()
        return result

    def create_single_image_matrix(self):
        outputs = self.calibration_service.single_image_matrix(deepcopy(self.state.settings.calibration))
        self.state.save()
        return outputs

    def calibration_pixel_stability(self, folder, x: int, y: int, start: int, end: int):
        return self.calibration_service.pixel_stability(folder, x, y, start, end)

    def import_davis(self) -> DavisImportResult:
        result = self.davis_service.import_export_folder(deepcopy(self.state.settings.davis))
        self.state.save()
        return result

    def prepare_davis_input(self):
        result = self.davis_service.prepare_input_folder(deepcopy(self.state.settings.davis))
        self.state.save()
        return result

    def run_shift_search(self) -> ShiftSearchResult:
        result = self.mixing_service.run_shift_search(deepcopy(self.state.settings.shift))
        self.state.save()
        return result

    def run_noise_analysis(self):
        result = self.noise_service.run(deepcopy(self.state.settings.noise))
        self.state.save()
        return result

    def run_manual_mixing(self) -> ManualMixingResult:
        result = self.mixing_service.run_manual(deepcopy(self.state.settings.manual))
        self.state.save()
        return result

    def run_manual_display(self, show_q_center: str, show_img: str) -> ManualDisplayResult:
        result = self.mixing_service.run_manual_display(deepcopy(self.state.settings.manual), show_q_center, show_img)
        self.state.save()
        return result

    def export_manual_result(self, result: ManualDisplayResult):
        output = self.mixing_service.export_manual(self.state.settings.manual, result)
        self.state.save()
        return output

    def run_distribution_display(self, show_q_center: str, show_img: str):
        result = self.mixing_service.run_distribution(deepcopy(self.state.settings.distribution), show_q_center, show_img)
        self.state.save()
        return result

    def export_distribution_result(self, result):
        output = self.mixing_service.export_distribution(self.state.settings.distribution, result)
        self.state.save()
        return output
