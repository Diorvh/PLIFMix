from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from PLIFMix.algorithms.distribution import DistributionRegistrationEstimate
from PLIFMix.core.result_snapshot import ResultSnapshot
from PLIFMix.services.display_types import DisplayImage


@dataclass(frozen=True)
class DistributionFitParameters:
    """Numeric fit values for one F/O/FO image."""

    image_name: str
    center_x: float | None
    center_y: float | None
    rotate_angle: float | None
    center_a: float | None
    center_b: float | None
    center_r_squared: float | None
    horizontal_a: float | None
    horizontal_b: float | None
    horizontal_c: float | None
    horizontal_r_squared: float | None
    vertical_a: float | None
    vertical_b: float | None
    vertical_c: float | None
    vertical_r_squared: float | None
    student_scale_x: float | None
    student_scale_y: float | None
    student_exponent: float | None
    student_r_squared: float | None
    compact_scale_x: float | None
    compact_scale_y: float | None
    compact_exponent: float | None
    compact_cutoff_radius: float | None
    compact_r_squared: float | None

    def as_fit_row(self) -> list[float | None]:
        return [
            self.center_x,
            self.center_y,
            self.rotate_angle,
            self.center_a,
            self.center_b,
            self.center_r_squared,
            self.horizontal_a,
            self.horizontal_b,
            self.horizontal_c,
            self.horizontal_r_squared,
            self.vertical_a,
            self.vertical_b,
            self.vertical_c,
            self.vertical_r_squared,
            self.student_scale_x,
            self.student_scale_y,
            self.student_exponent,
            self.student_r_squared,
            self.compact_scale_x,
            self.compact_scale_y,
            self.compact_exponent,
            self.compact_cutoff_radius,
            self.compact_r_squared,
        ]


@dataclass
class DistributionFitResult:
    fit_text: str
    surface_text: str
    images: list[DisplayImage]
    numeric_results: tuple[DistributionFitParameters, ...]
    curve_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    warnings: tuple[str, ...]
    summary_rows: tuple["DistributionSummaryRow", ...]
    snapshot: ResultSnapshot | None = None

    @property
    def fit_rows(self) -> list[list[float | None]]:
        """Read-only compatibility view for existing UI, profiler, and tests."""
        return [item.as_fit_row() for item in self.numeric_results]


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


__all__ = [
    "DistributionFitParameters",
    "DistributionFitResult",
    "DistributionImageData",
    "DistributionRegistrationResult",
    "DistributionSummaryRow",
]
