from __future__ import annotations

from pathlib import Path

import numpy as np

from PLIFMix.ui.components.viewer_panel import ViewerPanel


class ImageDisplay(ViewerPanel):
    """Legacy manual-result API backed by the shared image viewport."""

    def __init__(self, master, name: str, show_save_button: bool = False):
        super().__init__(master, path=None, show_histogram=False, name=name, show_save_button=show_save_button)

    def update(
        self,
        image: np.ndarray,
        gain: float = 1,
        colormap: str = "jet",
        zoom: float = 1,
        crosshair: tuple[int, int, int] | None = None,
        ellipse: tuple[int, int, int, int, int] | None = None,
        rotate: float = 0,
        rotation_center: tuple[int, int] | None = None,
        save_name_suffix: str = "",
        special_points=None,
        save_folder: str | Path | None = None,
    ) -> None:
        self.update_image(
            image,
            gain=gain,
            colormap=colormap,
            zoom=zoom,
            crosshair=crosshair,
            ellipse=ellipse,
            rotate=rotate,
            rotation_center=rotation_center,
            save_name_suffix=save_name_suffix,
            special_points=special_points,
            save_folder=save_folder,
        )

    def update_mouse_info(self, event) -> None:
        self.mouse_info(event)
