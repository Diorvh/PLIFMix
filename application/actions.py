from __future__ import annotations

from copy import deepcopy

from PLIFv2.application.app_state import AppState
from PLIFv2.core.experiment_images import ExperimentTripletCache, experiment_npy_path
from PLIFv2.core.image_io import list_image_files
from PLIFv2.core.result_snapshot import ResultSnapshot, require_current_result
from PLIFv2.services.average_service import AverageImageService, AverageResult
from PLIFv2.services.calibration_service import CalibrationService
from PLIFv2.services.davis_service import DavisImportResult, DavisService
from PLIFv2.services.intensifier_service import CorrectionResult, IntensifierCorrectionService, MatrixFitResult
from PLIFv2.services.distribution_service import (
    DistributionFitResult,
    DistributionFitService,
    DistributionRegistrationResult,
)
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
        experiment_image_cache = ExperimentTripletCache()
        self.mixing_service = MixingService(experiment_image_cache)
        self.distribution_service = DistributionFitService(experiment_image_cache)
        self.noise_service = NoiseAnalysisService()

    def close(self) -> None:
        self.distribution_service.close()

    def run_average(self) -> AverageResult:
        result = self.average_service.run(deepcopy(self.state.settings.average))
        self.state.save()
        return result

    def run_intensifier_average(self) -> AverageResult:
        result = self.average_service.run_intensifier_average(deepcopy(self.state.settings.intensifier_average))
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

    def analyze_calibration(self, settings=None):
        settings = deepcopy(settings if settings is not None else self.state.settings.calibration)
        files = list_image_files(settings.source_folder, (".png", ".tif", ".tiff", ".bmp", ".npy"))
        snapshot = ResultSnapshot.capture(settings, [settings.source_folder, *files])
        result = self.calibration_service.analyze(settings)
        result.snapshot = snapshot
        self.state.save()
        return result

    def save_calibration_matrix(self, result):
        settings = require_current_result(result, self.state.settings.calibration).export_settings(self.state.settings.calibration)
        outputs = self.calibration_service.save_correction(settings, result)
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

    def run_manual_shift_search(self, manual=None, search=None) -> ShiftSearchResult:
        manual = deepcopy(manual if manual is not None else self.state.settings.manual)
        search = deepcopy(search if search is not None else self.state.settings.shift)
        result = self.mixing_service.run_manual_shift_search(manual, search)
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

    def run_manual_display(self, settings=None) -> ManualDisplayResult:
        settings = deepcopy(settings if settings is not None else self.state.settings.manual)
        snapshot = self._triplet_snapshot(settings)
        result = self.mixing_service.run_manual_display(settings)
        result.snapshot = snapshot
        self.state.save()
        return result

    def export_mix_curve_data(
        self,
        result: ManualDisplayResult,
        use_calibrated: bool = True,
        output_name: str | None = None,
    ):
        output = self.mixing_service.export_mix_curve(
            require_current_result(result, self.state.settings.manual).export_settings(self.state.settings.manual),
            result,
            use_calibrated,
            output_name,
        )
        self.state.save()
        return output

    def run_distribution_display(
        self,
        annotations: frozenset[str],
        image_mode: str,
        show_bent: bool | None = None,
        settings=None,
    ) -> DistributionFitResult:
        settings = deepcopy(settings if settings is not None else self.state.settings.distribution)
        snapshot = self._triplet_snapshot(settings)
        result = self.distribution_service.run(
            settings,
            annotations,
            image_mode,
            show_bent,
        )
        result.snapshot = snapshot
        self.state.save()
        return result

    def estimate_distribution_registration(self, settings=None) -> DistributionRegistrationResult:
        settings = deepcopy(settings if settings is not None else self.state.settings.distribution)
        return self.distribution_service.estimate_registration(settings)

    def export_distribution_result(self, result: DistributionFitResult):
        settings = require_current_result(result, self.state.settings.distribution).export_settings(self.state.settings.distribution)
        output = self.distribution_service.export(settings, result)
        self.state.save()
        return output

    def write_distribution_fit_results(self, result: DistributionFitResult):
        output = self.distribution_service.write_fit_results_to_excel(
            require_current_result(result, self.state.settings.distribution).export_settings(self.state.settings.distribution),
            result,
        )
        self.state.save()
        return output

    @staticmethod
    def _triplet_snapshot(settings) -> ResultSnapshot:
        return ResultSnapshot.capture(settings, (
            experiment_npy_path(settings.image_folder, number)
            for number in (settings.f_exp_no, settings.o_exp_no, settings.fo_exp_no)
        ))
