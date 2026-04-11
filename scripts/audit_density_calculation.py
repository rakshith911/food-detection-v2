#!/usr/bin/env python3
"""
Audit stored density values in the unified dataset.

Examples:
  python3 scripts/audit_density_calculation.py --id usda:174527
  python3 scripts/audit_density_calculation.py --query "yellow rice, cooked, no added fat"
  python3 scripts/audit_density_calculation.py --query "yorkshire pudding"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
UNIFIED_FULL = ROOT / "unified_data" / "unified_foods_full.json"
USDA_DIR = ROOT / "FoodData_Central_csv_2025-12-18"
COFID_XLSX = ROOT / "cofid_data" / "CoFID.xlsx"


def safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", dest="row_id", help="Exact unified row id, e.g. usda:174527")
    parser.add_argument("--query", help="Case-insensitive substring search on description")
    return parser.parse_args()


def load_unified_rows() -> list[dict]:
    return json.loads(UNIFIED_FULL.read_text())


def find_rows(rows: list[dict], row_id: Optional[str], query: Optional[str]) -> list[dict]:
    if row_id:
        return [row for row in rows if row.get("id") == row_id]
    needle = (query or "").strip().lower()
    return [row for row in rows if needle in (row.get("description") or "").lower()]


def load_measure_units() -> dict[str, str]:
    path = USDA_DIR / "measure_unit.csv"
    units: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            unit_id = (row.get("id") or "").strip()
            if unit_id:
                units[unit_id] = (row.get("name") or "").strip()
    return units


def parse_portion_amount(text: str) -> Optional[float]:
    import re

    lowered = (text or "").strip().lower()
    if not lowered:
        return None

    mixed_match = re.match(r"^(\d+)\s+(\d+)/(\d+)\b", lowered)
    if mixed_match:
        whole = float(mixed_match.group(1))
        numerator = float(mixed_match.group(2))
        denominator = float(mixed_match.group(3))
        if denominator:
            return whole + (numerator / denominator)

    fraction_match = re.match(r"^(\d+)/(\d+)\b", lowered)
    if fraction_match:
        numerator = float(fraction_match.group(1))
        denominator = float(fraction_match.group(2))
        if denominator:
            return numerator / denominator

    numeric_match = re.match(r"^(\d+(?:\.\d+)?)\b", lowered)
    if numeric_match:
        return float(numeric_match.group(1))

    return None


def parse_portion_unit(text: str) -> Optional[tuple[str, float]]:
    lowered = f" {text.lower()} "
    aliases = {
        "cup": ("cup", 240.0),
        "tablespoon": ("tbsp", 15.0),
        "tbsp": ("tbsp", 15.0),
        "teaspoon": ("tsp", 5.0),
        "tsp": ("tsp", 5.0),
        "fl oz": ("fl_oz", 29.5735),
        "fluid ounce": ("fl_oz", 29.5735),
        "milliliter": ("ml", 1.0),
        "millilitre": ("ml", 1.0),
        "ml": ("ml", 1.0),
    }
    for needle, result in aliases.items():
        if f" {needle} " in lowered:
            return result
    if "cup" in lowered:
        return aliases["cup"]
    return None


def audit_usda_row(row: dict) -> None:
    fdc_id = row["source_id"]
    units = load_measure_units()
    portions_path = USDA_DIR / "food_portion.csv"
    usable = []
    with portions_path.open(newline="", encoding="utf-8") as handle:
        for portion in csv.DictReader(handle):
            if (portion.get("fdc_id") or "").strip() != fdc_id:
                continue
            gram_weight = safe_float(portion.get("gram_weight"))
            amount = safe_float(portion.get("amount"))
            if amount is None or amount <= 0:
                amount = parse_portion_amount(portion.get("portion_description") or "")
            if gram_weight is None or gram_weight <= 0 or amount is None or amount <= 0:
                continue
            measure_name = units.get((portion.get("measure_unit_id") or "").strip(), "")
            descriptor = " ".join(
                part for part in (
                    measure_name,
                    portion.get("modifier") or "",
                    portion.get("portion_description") or "",
                )
                if part
            ).strip()
            parsed = parse_portion_unit(descriptor)
            if parsed is None:
                continue
            unit_label, base_ml = parsed
            grams_per_unit = gram_weight / amount
            density = grams_per_unit / base_ml
            usable.append({
                "unit_label": unit_label,
                "base_ml": base_ml,
                "amount": amount,
                "gram_weight": gram_weight,
                "grams_per_unit": round(grams_per_unit, 4),
                "density_g_ml": round(density, 4),
                "measure_name": measure_name,
                "modifier": portion.get("modifier"),
                "portion_description": portion.get("portion_description"),
                "data_points": safe_float(portion.get("data_points")) or 0.0,
            })

    usable.sort(key=lambda item: (item["data_points"], -abs(item["amount"] - 1.0)), reverse=True)
    print("\nUSDA portion audit")
    if not usable:
        print("  No usable USDA volumetric portion rows found.")
        return
    for item in usable[:10]:
        print(json.dumps(item, indent=2))
    chosen = usable[0]
    print("\nChosen density")
    print(
        f"  density = grams_per_unit / base_ml = "
        f"{chosen['grams_per_unit']} / {chosen['base_ml']} = {chosen['density_g_ml']} g/ml"
    )


def audit_non_usda_row(row: dict) -> None:
    print("\nStored density audit")
    print(f"  source: {row.get('source')}")
    print(f"  density_g_ml: {row.get('density_g_ml')}")
    print(f"  density_method: {row.get('density_method')}")
    print(f"  density_source: {row.get('density_source')}")
    if row.get("source") == "cofid":
        print("  CoFID density comes from Specific gravity in sheet '1.2 Factors'.")
        if row.get("density_g_ml") is not None:
            print("  Formula: density_g_ml = specific_gravity (relative to water).")
    elif row.get("source") == "fao":
        print("  FAO density is copied directly from fao_density.json.")


def main() -> int:
    args = parse_args()
    if not args.row_id and not args.query:
        raise SystemExit("Pass --id or --query")

    rows = load_unified_rows()
    matches = find_rows(rows, args.row_id, args.query)
    if not matches:
        print("No matching rows found.")
        return 1

    print(f"Found {len(matches)} matching row(s).")
    for row in matches[:10]:
        print("\n" + "=" * 80)
        print(json.dumps({
            "id": row.get("id"),
            "description": row.get("description"),
            "source": row.get("source"),
            "source_table": row.get("source_table"),
            "density_g_ml": row.get("density_g_ml"),
            "density_method": row.get("density_method"),
            "density_source": row.get("density_source"),
            "cup_g": row.get("cup_g"),
            "tbsp_g": row.get("tbsp_g"),
            "tsp_g": row.get("tsp_g"),
            "fl_oz_g": row.get("fl_oz_g"),
        }, indent=2))
        if row.get("source") == "usda":
            audit_usda_row(row)
        else:
            audit_non_usda_row(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
