from __future__ import annotations

import errno
import os
import shutil
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np


_LOCK_ERRNOS = {errno.EACCES, errno.EBUSY, errno.EPERM}


class FileOccupiedError(PermissionError):
    """Raised when an external application keeps the target file locked."""

    def __init__(self, path: Path, operation: str):
        super().__init__(
            f"Cannot {operation}: {path}. The file is in use or write access is denied. "
            "Save and close it in the other application, check write permissions, and try again."
        )
        self.path = path
        self.operation = operation


def atomic_write(path: str | Path, writer: Callable[[Path], None], operation: str = "write file") -> Path:
    """Write a closed temporary file, then atomically replace the target."""
    target = Path(path)
    temporary = _temporary_path(target)
    try:
        writer(temporary)
        return _replace_temporary(temporary, target, operation)
    finally:
        _remove_temporary(temporary)


def atomic_write_bytes(path: str | Path, data: bytes, operation: str = "write file") -> Path:
    return atomic_write(path, lambda temporary: temporary.write_bytes(data), operation)


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    return atomic_write(path, lambda temporary: temporary.write_text(text, encoding=encoding), "write text file")


def atomic_save_npy(path: str | Path, array: np.ndarray) -> Path:
    def write(temporary: Path) -> None:
        with temporary.open("wb") as stream:
            np.save(stream, array)

    return atomic_write(path, write, "save NPY file")


def atomic_copy2(source: str | Path, destination: str | Path) -> Path:
    return atomic_write(destination, lambda temporary: shutil.copy2(source, temporary), "copy file")


def atomic_save_workbook(workbook: Any, path: str | Path) -> Path:
    target = Path(path)
    temporary = _temporary_path(target)
    try:
        try:
            workbook.save(temporary)
        finally:
            workbook.close()
        return _replace_temporary(temporary, target, "save Excel workbook")
    finally:
        _remove_temporary(temporary)


def rename_path(source: str | Path, destination: str | Path) -> Path:
    source_path = Path(source)
    destination_path = Path(destination)
    try:
        return source_path.rename(destination_path)
    except OSError as exc:
        if _is_lock_error(exc):
            raise _occupied_error(source_path, "rename file", exc) from exc
        raise


@contextmanager
def excel_write_session(path: str | Path) -> Iterator[None]:
    """Check an existing workbook without altering it or controlling Excel."""
    target = Path(path)
    try:
        with target.open("r+b"):
            pass
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise _occupied_error(target, "write Excel workbook", exc) from exc
    # A later lock can still appear; atomic replacement handles that separately.
    yield


def _is_lock_error(exc: OSError) -> bool:
    return isinstance(exc, PermissionError) or exc.errno in _LOCK_ERRNOS


def _occupied_error(path: Path, operation: str, exc: OSError) -> OSError:
    if _is_lock_error(exc):
        return FileOccupiedError(path, operation)
    return exc


def _remove_temporary(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _temporary_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")


def _replace_temporary(temporary: Path, target: Path, operation: str) -> Path:
    try:
        os.replace(temporary, target)
    except OSError as exc:
        raise _occupied_error(target, operation, exc) from exc
    return target
