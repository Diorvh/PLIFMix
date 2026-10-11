from __future__ import annotations

import tkinter as tk
from tkinter import filedialog as _dialogs

from PLIFMix.ui.localization import language_for


def _call(name, options):
    parent = options.get("parent") or tk._default_root
    if parent is not None:
        language = language_for(parent)
        default_title = "选择文件夹" if name == "askdirectory" else "保存文件" if name == "asksaveasfilename" else "选择文件"
        options["title"] = language.text(options.get("title", default_title))
        if "filetypes" in options:
            options["filetypes"] = [(language.text(label), *rest) for label, *rest in options["filetypes"]]
    return getattr(_dialogs, name)(**options)


def askdirectory(**options):
    return _call("askdirectory", options)


def askopenfilename(**options):
    return _call("askopenfilename", options)


def askopenfilenames(**options):
    return _call("askopenfilenames", options)


def asksaveasfilename(**options):
    return _call("asksaveasfilename", options)
