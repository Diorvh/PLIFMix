from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DisplayImage:
    name: str
    image: np.ndarray
    colormap: str = "jet"
    special_points: tuple[np.ndarray, np.ndarray] | list | None = None
