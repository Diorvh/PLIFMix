from __future__ import annotations

import tkinter as tk
from PLIFMix.ui import localized_widgets as ttk


class HorizontalDisplayPane(ttk.PanedWindow):
    """Resizable primary and secondary display areas for image-oriented pages."""

    def __init__(self, master, primary_weight: int = 3, secondary_weight: int = 1):
        super().__init__(master, orient=tk.HORIZONTAL)
        self.primary_host = ttk.Frame(self)
        self.secondary_host = ttk.Frame(self)
        self.add(self.primary_host, weight=primary_weight)
        self.add(self.secondary_host, weight=secondary_weight)
