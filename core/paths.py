from __future__ import annotations

from pathlib import Path


def path_is_set(path: str | Path) -> bool:
    return str(path).strip() not in {"", "."}

