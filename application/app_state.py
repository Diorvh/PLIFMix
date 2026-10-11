from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

from PLIFMix.core.config_io import load_settings, save_settings
from PLIFMix.domain.settings import AppSettings


@dataclass
class AppState:
    settings: AppSettings = field(default_factory=load_settings)
    status: str = "Ready"
    _save_lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def save(self, path: Path | None = None) -> Path:
        with self._save_lock:
            return save_settings(self.settings, path)
