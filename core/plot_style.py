from __future__ import annotations

from matplotlib import rcParams


def configure_matplotlib() -> None:
    rcParams["font.sans-serif"] = [
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    rcParams["axes.unicode_minus"] = False
