from __future__ import annotations

from multiprocessing import freeze_support
from pathlib import Path
import sys
import tkinter as tk

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PLIFv2.ui.app_window import PLIFAppWindow


def main() -> None:
    freeze_support()
    try:
        app = PLIFAppWindow()
    except tk.TclError as exc:
        raise SystemExit(
            "PLIFv2 无法创建窗口。请修复或重新安装带 Tcl/Tk 组件的 Python 3.11-3.13。\n"
            f"原始错误: {exc}"
        ) from exc
    app.mainloop()


if __name__ == "__main__":
    main()
