from __future__ import annotations

from pathlib import Path


def path_is_set(path: str | Path) -> bool:
    return str(path).strip() not in {"", "."}


def bundled_sample_folder() -> Path:
    return Path(__file__).resolve().parents[1] / "样例数据"

