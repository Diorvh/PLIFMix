from __future__ import annotations

import errno
import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


_EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xltx", ".xltm", ".xlsb", ".xls"}
_LOCK_ERRNOS = {errno.EACCES, errno.EBUSY, errno.EPERM}


class FileOccupiedError(PermissionError):
    """Raised when an external application keeps the target file locked."""

    def __init__(self, path: Path, operation: str):
        super().__init__(
            f"Cannot {operation}: {path}. The file is locked by another application. "
            "Close the application that is using this file and try again."
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


def close_open_excel_workbook(path: str | Path) -> bool:
    return _close_open_excel_workbook(Path(path))


def reopen_external_file(path: str | Path) -> None:
    _reopen_excel_workbook(Path(path))


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
    reopened_excel = False
    try:
        os.replace(temporary, target)
    except OSError as exc:
        if not _is_lock_error(exc) or not _close_open_excel_workbook(target):
            raise _occupied_error(target, operation, exc) from exc
        reopened_excel = True
        try:
            os.replace(temporary, target)
        except OSError as retry_exc:
            raise _occupied_error(target, operation, retry_exc) from retry_exc
    if reopened_excel:
        _reopen_excel_workbook(target)
    return target


def _close_open_excel_workbook(target: Path) -> bool:
    if target.suffix.lower() not in _EXCEL_SUFFIXES:
        return False
    try:
        from comtypes.client import GetActiveObject

        excel = GetActiveObject("Excel.Application")
        target_key = _path_key(target)
        for index in range(1, int(excel.Workbooks.Count) + 1):
            workbook = excel.Workbooks.Item(index)
            if _path_key(Path(str(workbook.FullName))) == target_key:
                workbook.Close(SaveChanges=False)
                return True
    except Exception:
        return False
    return False


def _reopen_excel_workbook(path: Path) -> None:
    try:
        os.startfile(path)  # type: ignore[attr-defined]
    except OSError:
        pass


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))
