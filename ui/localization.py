from __future__ import annotations

import json
import re
import tkinter as tk
from functools import lru_cache
from pathlib import Path
from weakref import WeakSet


LANGUAGES = ("zh", "en")
_CATALOG = json.loads((Path(__file__).with_name("locales") / "en.json").read_text(encoding="utf-8"))
_FIELD = re.compile(r"\{(\d+)\}")
_TEMPLATES = []
for _source, _target in _CATALOG.items():
    if not _FIELD.search(_source):
        continue
    _parts = _FIELD.split(_source)
    _pattern = "".join(re.escape(part) if index % 2 == 0 else "(.*?)" for index, part in enumerate(_parts))
    _TEMPLATES.append((_parts[0], re.compile(_pattern, re.DOTALL), _target, sum(len(part) for part in _parts[::2])))
_TEMPLATES.sort(key=lambda item: item[3], reverse=True)


@lru_cache(maxsize=2048)
def translate(source: str, language: str = "zh", _depth: int = 0) -> str:
    """Translate presentation text only; captured paths and numbers stay intact."""
    if language != "en" or not source:
        return source
    if source in _CATALOG:
        return _CATALOG[source]
    if not re.search(r"[\u3400-\u9fff]", source):
        return source
    for prefix, pattern, target, _specificity in _TEMPLATES:
        if not source.startswith(prefix):
            continue
        match = pattern.fullmatch(source)
        if match is not None:
            values = match.groups()
            if _depth < 3:
                values = tuple(translate(value, language, _depth + 1) for value in values)
            return target.format(*values)
    if "\n" in source:
        return "\n".join(translate(line, language, _depth) for line in source.split("\n"))
    stripped = source.strip()
    if stripped and stripped != source:
        start = len(source) - len(source.lstrip())
        return source[:start] + translate(stripped, language, _depth) + source[start + len(stripped):]
    return source


class LanguageController:
    def __init__(self, language: str = "zh"):
        self.language = language if language in LANGUAGES else "zh"
        self._targets = WeakSet()

    def text(self, source) -> str:
        return translate(str(source), self.language)

    def register(self, target) -> None:
        self._targets.add(target)

    def set_language(self, language: str) -> None:
        if language not in LANGUAGES:
            raise ValueError(f"Unsupported interface language: {language}")
        self.language = language
        for target in tuple(self._targets):
            try:
                target.refresh_language()
            except tk.TclError:
                self._targets.discard(target)


def language_for(widget) -> LanguageController:
    root = widget._root()
    controller = getattr(root, "_ui_language", None)
    if controller is None:
        controller = LanguageController()
        root._ui_language = controller
    return controller
