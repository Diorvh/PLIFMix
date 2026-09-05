from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from operator import index
from typing import Iterable


Coordinate = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ImageCoordinateMapping:
    """Map integer coordinates between a source image and its displayed image."""

    source_shape: tuple[int, int]
    quarter_turns: int = 0
    shift: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_shape", _normalize_shape(self.source_shape))
        object.__setattr__(self, "quarter_turns", index(self.quarter_turns) % 4)
        object.__setattr__(self, "shift", index(self.shift))

    @classmethod
    def from_parameters(
        cls,
        source_shape: Iterable[int],
        rotation_degrees: float = 0,
        shift: int = 0,
    ) -> ImageCoordinateMapping:
        return cls(_normalize_shape(source_shape), quarter_turns(rotation_degrees), index(shift))

    @property
    def display_shape(self) -> tuple[int, int]:
        height, width = self.source_shape
        return (width, height) if self.quarter_turns in (1, 3) else (height, width)

    def contains_display(self, x: int, y: int) -> bool:
        try:
            display_x = index(x)
            display_y = index(y)
        except TypeError:
            return False
        height, width = self.display_shape
        return 0 <= display_x < width and 0 <= display_y < height

    def display_to_source(self, x: int, y: int) -> Coordinate | None:
        """Return source ``(x, y)`` or ``None`` for display padding/out-of-bounds."""
        display_x = index(x)
        display_y = index(y)
        if not self.contains_display(display_x, display_y):
            return None

        rotated_x = display_x + self.shift
        rotated_height, rotated_width = self.display_shape
        if not 0 <= rotated_x < rotated_width:
            return None

        height, width = self.source_shape
        if self.quarter_turns == 1:
            return display_y, height - 1 - rotated_x
        if self.quarter_turns == 2:
            return width - 1 - rotated_x, height - 1 - display_y
        if self.quarter_turns == 3:
            return width - 1 - display_y, rotated_x
        return rotated_x, display_y

    def source_to_display(self, x: int, y: int) -> Coordinate | None:
        """Return display ``(x, y)`` or ``None`` if the source pixel is shifted out."""
        source_x = index(x)
        source_y = index(y)
        height, width = self.source_shape
        if not (0 <= source_x < width and 0 <= source_y < height):
            return None

        if self.quarter_turns == 1:
            rotated_x = height - 1 - source_y
            rotated_y = source_x
        elif self.quarter_turns == 2:
            rotated_x = width - 1 - source_x
            rotated_y = height - 1 - source_y
        elif self.quarter_turns == 3:
            rotated_x = source_y
            rotated_y = width - 1 - source_x
        else:
            rotated_x = source_x
            rotated_y = source_y

        display_x = rotated_x - self.shift
        display_y = rotated_y
        if not self.contains_display(display_x, display_y):
            return None
        return display_x, display_y


def quarter_turns(angle_degrees: float) -> int:
    angle = float(angle_degrees)
    if not isfinite(angle):
        raise ValueError("旋转角度必须是有限数值")
    turns = int(round(angle / 90))
    if not isclose(angle, turns * 90, rel_tol=1e-5, abs_tol=1e-9):
        raise ValueError("目前只支持90°的整数倍旋转")
    return turns % 4


def _normalize_shape(shape: Iterable[int]) -> tuple[int, int]:
    dimensions = tuple(index(dimension) for dimension in shape)
    if len(dimensions) != 2 or any(dimension < 0 for dimension in dimensions):
        raise ValueError("source_shape must contain two non-negative dimensions")
    return dimensions
