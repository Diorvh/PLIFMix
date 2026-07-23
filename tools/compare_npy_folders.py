from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_TOLERANCE = 1 / 65535


@dataclass(frozen=True)
class ComparisonResult:
    relative_path: Path
    reference_shape: tuple[int, ...] | None
    candidate_shape: tuple[int, ...] | None
    exact: bool
    max_abs_difference: float | None
    above_tolerance_count: int | None
    passed: bool


def compare_npy_folders(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    tolerance: float = DEFAULT_TOLERANCE,
    recursive: bool = False,
) -> tuple[list[ComparisonResult], list[Path], list[Path]]:
    """Compare matching NumPy arrays by their paths relative to two roots."""
    reference_files = _npy_files(Path(reference_dir), recursive)
    candidate_files = _npy_files(Path(candidate_dir), recursive)
    common_paths = sorted(reference_files.keys() & candidate_files.keys())
    results = [
        _compare_arrays(relative_path, reference_files[relative_path], candidate_files[relative_path], tolerance)
        for relative_path in common_paths
    ]
    reference_only = sorted(reference_files.keys() - candidate_files.keys())
    candidate_only = sorted(candidate_files.keys() - reference_files.keys())
    return results, reference_only, candidate_only


def _npy_files(folder: Path, recursive: bool) -> dict[Path, Path]:
    if not folder.is_dir():
        raise NotADirectoryError(f"Directory does not exist: {folder}")
    paths = folder.rglob("*.npy") if recursive else folder.glob("*.npy")
    return {path.relative_to(folder): path for path in paths if path.is_file()}


def _compare_arrays(relative_path: Path, reference_path: Path, candidate_path: Path, tolerance: float) -> ComparisonResult:
    reference = np.load(reference_path)
    candidate = np.load(candidate_path)
    if reference.shape != candidate.shape:
        return ComparisonResult(relative_path, reference.shape, candidate.shape, False, None, None, False)

    difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    same_nan_mask = np.array_equal(np.isnan(reference), np.isnan(candidate))
    finite_difference = difference[np.isfinite(difference)]
    max_abs_difference = float(np.max(finite_difference)) if finite_difference.size else 0.0
    above_tolerance_count = int(np.count_nonzero(finite_difference > tolerance))
    exact = np.array_equal(reference, candidate, equal_nan=True)
    passed = same_nan_mask and above_tolerance_count == 0
    return ComparisonResult(
        relative_path,
        reference.shape,
        candidate.shape,
        exact,
        max_abs_difference,
        above_tolerance_count,
        passed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare matching .npy arrays in two directories.")
    parser.add_argument("reference_dir", type=Path, help="Reference directory")
    parser.add_argument("candidate_dir", type=Path, help="Directory to compare with the reference")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="Maximum allowed absolute difference")
    parser.add_argument("--recursive", action="store_true", help="Include .npy files in subdirectories")
    args = parser.parse_args()

    results, reference_only, candidate_only = compare_npy_folders(
        args.reference_dir,
        args.candidate_dir,
        args.tolerance,
        args.recursive,
    )
    if not results and not reference_only and not candidate_only:
        print("No .npy files found.")
        return 1

    print(f"Tolerance: {args.tolerance:.12g}")
    for result in results:
        if result.reference_shape != result.candidate_shape:
            print(f"{result.relative_path}: shape mismatch {result.reference_shape} vs {result.candidate_shape}")
            continue
        outcome = "PASS" if result.passed else "FAIL"
        print(
            f"{result.relative_path}: shape={result.reference_shape}, exact={result.exact}, "
            f"max_abs={result.max_abs_difference:.12g}, above_tol={result.above_tolerance_count}, {outcome}"
        )
    for relative_path in reference_only:
        print(f"{relative_path}: only in reference directory")
    for relative_path in candidate_only:
        print(f"{relative_path}: only in candidate directory")

    passed = (
        bool(results)
        and all(result.passed for result in results)
        and not reference_only
        and not candidate_only
    )
    print(f"OVERALL: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
