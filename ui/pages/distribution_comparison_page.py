from __future__ import annotations

from PLIFMix.ui import localized_widgets as ttk

import numpy as np
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure

from PLIFMix.ui.mpl_canvas import DebouncedFigureCanvasTkAgg

class DistributionComparisonPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        ttk.Button(self, text="绘制常见中心对称分布", command=self.draw).pack(anchor="w", padx=8, pady=6)
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self.canvas = DebouncedFigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8)
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=8)
        NavigationToolbar2Tk(self.canvas, toolbar)

    def draw(self) -> None:
        from scipy import stats
        x = np.linspace(-10, 10, 1000)
        distributions = {"Normal": stats.norm(), "Laplace": stats.laplace(), "Logistic": stats.logistic(), "t (df=5)": stats.t(5), "Cauchy": stats.cauchy()}
        self.axis.clear()
        for name, distribution in distributions.items():
            self.axis.plot(x, distribution.pdf(x), label=name)
        self.axis.legend()
        self.axis.set_title("中心对称分布比较")
        self.axis.grid(True, alpha=0.3)
        self.canvas.request_draw()
