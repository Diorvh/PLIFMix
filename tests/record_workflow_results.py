"""Record numeric outputs from an isolated checkout for refactoring comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--settings", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--denoise", type=float)
parser.add_argument("--annotations", default="")
args = parser.parse_args()
sys.path.insert(0, str(args.repo.parent))

import numpy as np
from PLIFMix.core.config_io import load_settings
from PLIFMix.services.distribution_service import DistributionFitService
from PLIFMix.services.mixing_service import MixingService


def main() -> None:
    settings = load_settings(args.settings)
    if args.denoise is not None:
        settings.distribution.denoise_ratio = args.denoise
        settings.manual.denoise_ratio = args.denoise
    annotations = frozenset(filter(None, args.annotations.split(",")))
    arrays = {}
    metadata = {}
    service = DistributionFitService()
    try:
        for mode in ("ori_img", "fit_img", "delta_img"):
            for bent in (False, True):
                key = f"distribution_{mode}_{bent}"
                result = service.run(settings.distribution, annotations, mode, bent)
                for item in result.images:
                    arrays[f"{key}_{item.name}"] = item.image
                    if item.special_points is not None and len(item.special_points):
                        for index, values in enumerate(item.special_points):
                            arrays[f"{key}_{item.name}_points_{index}"] = values
                for name, curve in result.curve_data.items():
                    for index, values in enumerate(curve):
                        arrays[f"{key}_curve_{name}_{index}"] = values
                metadata[key] = result.fit_rows
        mixing = MixingService().run_manual_display(settings.manual)
        metadata["mixing_summary"] = vars(mixing.summary)
        for prefix, images in (("mixing", mixing.images), ("mixing_raw", mixing.raw_images)):
            for item in images:
                arrays[f"{prefix}_{item.name}"] = item.image
        for prefix, curve in (("mixing_curve", mixing.mix_curve), ("mixing_raw_curve", mixing.raw_mix_curve)):
            for index, values in enumerate(curve):
                arrays[f"{prefix}_{index}"] = values
    finally:
        service.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    print(f"Recorded {len(arrays)} arrays and {len(metadata)} result groups: {args.output}")


if __name__ == "__main__":
    main()
