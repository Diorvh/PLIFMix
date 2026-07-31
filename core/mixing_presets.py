from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import openpyxl

from PLIFv2.core.safe_io import atomic_write_text


PRESET_FORMAT = "plifv2.manual-mixing-presets"
PRESET_VERSION = 1


@dataclass(frozen=True)
class MixingPreset:
    dr: str
    u: str
    f_exp_no: str
    fo_exp_no: str
    o_exp_no: str
    global_of: float


@dataclass(frozen=True)
class MixingPresetCatalog:
    name: str
    groups: tuple[MixingPreset, ...]
    source_excel: str = ""


def default_mixing_preset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "runtime" / "presets" / "20250820孔径比影响-正式-混合数据.json"


def load_mixing_presets(path: str | Path) -> MixingPresetCatalog:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"无法读取预设文件: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"预设文件不是有效 JSON: {source}") from exc

    if data.get("format") != PRESET_FORMAT:
        raise ValueError(f"不是 PLIFv2 混合预设文件: {source}")
    if data.get("version") != PRESET_VERSION:
        raise ValueError(f"不支持的预设文件版本: {data.get('version')}")

    groups: list[MixingPreset] = []
    for index, item in enumerate(data.get("groups", []), start=1):
        try:
            groups.append(
                MixingPreset(
                    dr=str(item["dr"]),
                    u=str(item["u"]),
                    f_exp_no=str(item["f_exp_no"]),
                    fo_exp_no=str(item["fo_exp_no"]),
                    o_exp_no=str(item["o_exp_no"]),
                    global_of=float(item["global_of"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"预设文件第 {index} 组数据无效") from exc
    if not groups:
        raise ValueError("预设文件没有实验组")
    return MixingPresetCatalog(
        name=str(data.get("name") or source.stem),
        groups=tuple(groups),
        source_excel=str(data.get("source_excel") or ""),
    )


def create_mixing_presets_from_excel(path: str | Path) -> MixingPresetCatalog:
    source = Path(path)
    try:
        workbook = openpyxl.load_workbook(source, read_only=True, data_only=True)
    except OSError as exc:
        raise ValueError(f"无法读取 Excel: {source}") from exc

    try:
        sheet = workbook.worksheets[0]
        records = [
            row
            for row in sheet.iter_rows(values_only=True)
            if isinstance(row[0], (int, float)) and not isinstance(row[0], bool)
        ]
    finally:
        workbook.close()

    if len(records) % 3:
        raise ValueError("Excel 中的实验编号记录不是 3 的倍数")

    groups: list[MixingPreset] = []
    for start in range(0, len(records), 3):
        rows = records[start:start + 3]
        by_role = {_record_role(row): row for row in rows}
        if set(by_role) != {"f", "fo", "o"}:
            raise ValueError(f"Excel 第 {start // 3 + 1} 组缺少 F、FO 或 O 编号")

        f_row = by_role["f"]
        df = _number_at(f_row, 1, "Df")
        do = _number_at(f_row, 2, "Do")
        speed = _number_at(f_row, 12, "Uf")
        global_of = _number_at(f_row, 20, "全局 O/F")
        groups.append(
            MixingPreset(
                dr=f"{do / df:.1f}",
                u=str(int(round(speed))),
                f_exp_no=_experiment_text(by_role["f"][0]),
                fo_exp_no=_experiment_text(by_role["fo"][0]),
                o_exp_no=_experiment_text(by_role["o"][0]),
                global_of=global_of,
            )
        )

    return MixingPresetCatalog(name=source.stem, groups=tuple(groups), source_excel=source.name)


def save_mixing_presets(catalog: MixingPresetCatalog, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "format": PRESET_FORMAT,
        "version": PRESET_VERSION,
        "name": catalog.name,
        "source_excel": catalog.source_excel,
        "groups": [asdict(group) for group in catalog.groups],
    }
    return atomic_write_text(output, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _record_role(row: tuple[Any, ...]) -> str:
    text = str(row[4] or "").upper()
    if "FO" in text:
        return "fo"
    if "F" in text:
        return "f"
    if "O" in text:
        return "o"
    raise ValueError(f"无法识别实验编号 {row[0]} 的 F/FO/O 类型")


def _number_at(row: tuple[Any, ...], index: int, name: str) -> float:
    try:
        return float(row[index])
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"Excel 缺少有效的 {name} 数据") from exc


def _experiment_text(value: Any) -> str:
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
