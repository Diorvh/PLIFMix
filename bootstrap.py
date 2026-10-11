"""Load the local application package independently of its checkout name."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType


def load_local_package(root: Path | None = None) -> ModuleType:
    package_root = (root or Path(__file__).parent).resolve()
    initializer = package_root / "__init__.py"
    existing = sys.modules.get("PLIFMix")
    if existing is not None:
        if Path(existing.__file__ or "").resolve() != initializer:
            raise ImportError("A different PLIFMix checkout is already loaded.")
        return existing

    spec = spec_from_file_location(
        "PLIFMix", initializer, submodule_search_locations=[str(package_root)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load PLIFMix from {package_root}")
    package = module_from_spec(spec)
    sys.modules[spec.name] = package
    try:
        spec.loader.exec_module(package)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return package
