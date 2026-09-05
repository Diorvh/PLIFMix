from __future__ import annotations

from multiprocessing import freeze_support
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    freeze_support()
    import tkinter as tk

    from PLIFv2.ui.app_window import PLIFAppWindow

    try:
        app = PLIFAppWindow()
    except tk.TclError as exc:
        raise SystemExit(
            "PLIFv2 无法创建窗口。请修复或重新安装带 Tcl/Tk 组件的 Python 3.11 或更高版本。\n"
            f"原始错误: {exc}"
        ) from exc
    app.mainloop()


if __name__ == "__main__":
    main()
