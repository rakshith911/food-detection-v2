#!/usr/bin/env python3
"""
Build nutrition datasets for the RAG pipeline from USDA FoodData Central, CoFID,
and FAO.

Outputs:
  - unified_data/unified_foods_full.json
      Canonical USDA non-branded + CoFID + FAO dataset with rich fields.
  - unified_data/unified_foods.json
      Slim retrieval dataset for the current RAG runtime.
  - unified_data/unified_food_names.json
      Parallel names list for FAISS.
  - branded_data/usda_branded_foods.json
      Separate USDA branded dataset.
  - fao_data/fao_foods.json
      Normalized FAO density dataset for separate fallback use.

This script intentionally keeps USDA branded foods out of the generic unified
dataset so retrieval quality does not get swamped by branded rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USDA_DIR = ROOT / "FoodData_Central_csv_2025-12-18"
DEFAULT_COFID_XLSX = ROOT / "cofid_data" / "CoFID.xlsx"
DEFAULT_FAO_DENSITY = ROOT / "fao_data" / "fao_density.json"
DEFAULT_FAO_NAMES = ROOT / "fao_data" / "fao_food_names.json"
UNIFIED_OUT_DIR = ROOT / "unified_data"
BRANDED_OUT_DIR = ROOT / "branded_data"
FAO_OUT_DIR = ROOT / "fao_data"
REPORT_OUT = UNIFIED_OUT_DIR / "build_report.json"

USDA_GENERIC_TYPES = {
    "foundation_food",
    "sr_legacy_food",
    "survey_fndds_food",
    "sample_food",
    "sub_sample_food",
    "agricultural_acquisition",
    "market_acquistion",
    "experimental_food",
}
USDA_BRANDED_TYPE = "branded_food"

USDA_NUTRIENT_IDS = {
    "calories_per_100g": "1008",
    "protein_g": "1003",
    "fat_g": "1004",
    "carbs_g": "1005",
}

PORTION_UNIT_ALIASES = {
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

XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

# Cross-regional / cross-dialect synonym map used to enrich retrieval_text.
# Both directions must be listed explicitly so queries in either dialect match.
FOOD_SYNONYMS: dict[str, list[str]] = {
    # Baked goods
    "popover":           ["yorkshire pudding"],
    "yorkshire pudding": ["popover"],
    # Vegetables (UK ↔ US)
    "aubergine":   ["eggplant"],
    "eggplant":    ["aubergine"],
    "courgette":   ["zucchini", "summer squash"],
    "zucchini":    ["courgette"],
    "rocket":      ["arugula"],
    "arugula":     ["rocket"],
    "spring onion": ["scallion", "green onion"],
    "scallion":    ["spring onion", "green onion"],
    "green onion": ["spring onion", "scallion"],
    "beetroot":    ["beet"],
    "beet":        ["beetroot"],
    "mangetout":   ["snow pea", "snap pea"],
    "mange tout":  ["snow pea", "snap pea"],
    "snow pea":    ["mangetout", "mange tout"],
    "pak choi":    ["bok choy", "chinese cabbage"],
    "bok choy":    ["pak choi", "chinese cabbage"],
    "swede":       ["rutabaga"],
    "rutabaga":    ["swede"],
    "capsicum":    ["bell pepper", "sweet pepper"],
    "bell pepper": ["capsicum", "sweet pepper"],
    "sweet pepper":["capsicum", "bell pepper"],
    "coriander":   ["cilantro"],
    "cilantro":    ["coriander"],
    # Legumes
    "chickpea":    ["garbanzo", "garbanzo bean"],
    "garbanzo":    ["chickpea"],
    "falafel":     ["chickpea patty"],
    # Meat / seafood
    "mince":       ["ground beef", "minced meat"],
    "ground beef": ["mince", "minced beef"],
    "minced beef": ["ground beef", "mince"],
    "prawn":       ["shrimp"],
    "shrimp":      ["prawn"],
    # Dairy
    "double cream":   ["heavy cream", "whipping cream"],
    "heavy cream":    ["double cream", "whipping cream"],
    "whipping cream": ["double cream", "heavy cream"],
    "single cream":   ["half and half", "light cream"],
    "half and half":  ["single cream", "light cream"],
    # Snacks / sides
    "crisps":       ["potato chips"],
    "potato chips": ["crisps"],
    "french fries": ["chips", "fries"],
    "fries":        ["french fries", "chips"],
    # Pantry / other
    "sultana":            ["golden raisin"],
    "golden raisin":      ["sultana"],
    "treacle":            ["molasses"],
    "molasses":           ["treacle"],
    "desiccated coconut": ["shredded coconut"],
    "shredded coconut":   ["desiccated coconut"],
    # Indian cuisine
    "paneer":       ["cottage cheese", "indian cheese", "fresh cheese"],
    "palak paneer": ["saag paneer", "spinach paneer", "spinach with cheese"],
    "saag paneer":  ["palak paneer", "spinach paneer"],
    "naan":         ["flatbread", "indian bread"],
    "chapati":      ["roti", "flatbread", "indian bread"],
    "roti":         ["chapati", "flatbread"],
}


def safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"N", "n", "—", "None"}:
        return None
    if text.lower() == "tr":
        return 0.0
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def normalize_food_name(text: str) -> str:
    cleaned = re.sub(r"\([^)]*\)", " ", text or "")
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_portion_amount(text: str) -> Optional[float]:
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


@dataclass
class PortionChoice:
    grams_per_unit: float
    amount: float
    data_points: float
    method: str


class XlsxReader:
    """Minimal XLSX reader using stdlib so the builder does not require openpyxl."""

    def __init__(self, path: Path):
        self.path = path
        self._zip = zipfile.ZipFile(path)
        self._shared_strings = self._load_shared_strings()
        self._sheet_targets = self._load_sheet_targets()

    def close(self) -> None:
        self._zip.close()

    def _load_shared_strings(self) -> list[str]:
        if "xl/sharedStrings.xml" not in self._zip.namelist():
            return []
        root = ET.fromstring(self._zip.read("xl/sharedStrings.xml"))
        values: list[str] = []
        for si in root:
            values.append("".join(t.text or "" for t in si.iter("{%s}t" % XLSX_NS["a"])))
        return values

    def _load_sheet_targets(self) -> dict[str, str]:
        workbook = ET.fromstring(self._zip.read("xl/workbook.xml"))
        rels = ET.fromstring(self._zip.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        mapping: dict[str, str] = {}
        for sheet in workbook.find("a:sheets", XLSX_NS):
            name = sheet.attrib["name"]
            rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            mapping[name] = "xl/" + rel_map[rel_id]
        return mapping

    def iter_rows(self, sheet_name: str, min_row: int = 1) -> Iterator[list[Optional[str]]]:
        target = self._sheet_targets[sheet_name]
        root = ET.fromstring(self._zip.read(target))
        for idx, row in enumerate(root.findall("a:sheetData/a:row", XLSX_NS), start=1):
            if idx < min_row:
                continue
            values: list[Optional[str]] = []
            for cell in row.findall("a:c", XLSX_NS):
                ref = cell.attrib.get("r", "")
                column_index = self._column_index(ref)
                while len(values) < column_index:
                    values.append(None)
                cell_type = cell.attrib.get("t")
                inline = cell.find("a:is", XLSX_NS)
                value_node = cell.find("a:v", XLSX_NS)
                if inline is not None:
                    value = "".join(t.text or "" for t in inline.iter("{%s}t" % XLSX_NS["a"]))
                elif value_node is None:
                    value = None
                else:
                    raw = value_node.text
                    if cell_type == "s" and raw is not None:
                        value = self._shared_strings[int(raw)]
                    else:
                        value = raw
                if len(values) == column_index:
                    values.append(value)
                else:
                    values[column_index] = value
            yield values

    @staticmethod
    def _column_index(cell_ref: str) -> int:
        letters = "".join(ch for ch in cell_ref if ch.isalpha()).upper()
        if not letters:
            return 0
        value = 0
        for ch in letters:
            value = (value * 26) + (ord(ch) - ord("A") + 1)
        return value - 1


def load_usda_food_map(usda_dir: Path) -> dict[str, dict]:
    food_map: dict[str, dict] = {}
    with (usda_dir / "food.csv").open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            fdc_id = (row.get("fdc_id") or "").strip()
            if not fdc_id:
                continue
            food_map[fdc_id] = {
                "fdc_id": fdc_id,
                "data_type": (row.get("data_type") or "").strip(),
                "description": (row.get("description") or "").strip(),
                "food_category": (row.get("food_category_id") or "").strip(),
                "publication_date": (row.get("publication_date") or "").strip(),
            }
            if index % 250000 == 0:
                print(f"Loaded food.csv rows: {index}")
    return food_map


def load_usda_branded_map(usda_dir: Path) -> dict[str, dict]:
    branded: dict[str, dict] = {}
    path = usda_dir / "branded_food.csv"
    if not path.exists():
        return branded
    with path.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            fdc_id = (row.get("fdc_id") or "").strip()
            if not fdc_id:
                continue
            branded[fdc_id] = {
                "brand_owner": (row.get("brand_owner") or "").strip() or None,
                "brand_name": (row.get("brand_name") or "").strip() or None,
                "ingredients_text": (row.get("ingredients") or "").strip() or None,
                "household_serving_fulltext": (row.get("household_serving_fulltext") or "").strip() or None,
                "serving_size": safe_float(row.get("serving_size")),
                "serving_size_unit": (row.get("serving_size_unit") or "").strip() or None,
                "branded_food_category": (row.get("branded_food_category") or "").strip() or None,
            }
            if index % 250000 == 0:
                print(f"Loaded branded_food.csv rows: {index}")
    return branded


def load_usda_nutrients(usda_dir: Path, valid_ids: set[str]) -> dict[str, dict[str, float]]:
    per_food: dict[str, dict[str, float]] = defaultdict(dict)
    reverse_ids = {nutrient_id: field for field, nutrient_id in USDA_NUTRIENT_IDS.items()}
    with (usda_dir / "food_nutrient.csv").open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            fdc_id = (row.get("fdc_id") or "").strip()
            nutrient_id = (row.get("nutrient_id") or "").strip()
            if fdc_id not in valid_ids or nutrient_id not in reverse_ids:
                if index % 1000000 == 0:
                    print(f"Scanned food_nutrient.csv rows: {index}")
                continue
            amount = safe_float(row.get("amount"))
            if amount is None:
                continue
            field = reverse_ids[nutrient_id]
            current = per_food[fdc_id].get(field)
            if current is None:
                per_food[fdc_id][field] = round(float(amount), 4)
            if index % 1000000 == 0:
                print(f"Scanned food_nutrient.csv rows: {index}")
    return per_food


def portion_sort_key(choice: PortionChoice) -> tuple[float, float]:
    return (choice.data_points, -abs(choice.amount - 1.0))


def parse_portion_unit(text: str) -> Optional[tuple[str, float]]:
    lowered = f" {text.lower()} "
    for needle, result in PORTION_UNIT_ALIASES.items():
        if f" {needle} " in lowered:
            return result
    if "cup" in lowered:
        return PORTION_UNIT_ALIASES["cup"]
    return None


def load_usda_portions(usda_dir: Path, valid_ids: set[str]) -> dict[str, dict[str, PortionChoice]]:
    units: dict[str, str] = {}
    with (usda_dir / "measure_unit.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            unit_id = (row.get("id") or "").strip()
            if unit_id:
                units[unit_id] = (row.get("name") or "").strip()

    per_food: dict[str, dict[str, PortionChoice]] = defaultdict(dict)
    with (usda_dir / "food_portion.csv").open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            fdc_id = (row.get("fdc_id") or "").strip()
            if fdc_id not in valid_ids:
                if index % 250000 == 0:
                    print(f"Scanned food_portion.csv rows: {index}")
                continue
            gram_weight = safe_float(row.get("gram_weight"))
            amount = safe_float(row.get("amount"))
            if amount is None or amount <= 0:
                amount = parse_portion_amount(row.get("portion_description") or "")
            if gram_weight is None or gram_weight <= 0 or amount is None or amount <= 0:
                continue

            measure_name = units.get((row.get("measure_unit_id") or "").strip(), "")
            descriptor = " ".join(
                part for part in (
                    measure_name,
                    row.get("modifier") or "",
                    row.get("portion_description") or "",
                )
                if part
            ).strip()

            parsed = parse_portion_unit(descriptor)
            if parsed is None:
                continue
            unit_label, base_ml = parsed
            grams_per_unit = gram_weight / amount
            density = grams_per_unit / base_ml
            if not (0.05 <= density <= 3.0):
                continue

            candidate = PortionChoice(
                grams_per_unit=round(grams_per_unit, 4),
                amount=amount,
                data_points=safe_float(row.get("data_points")) or 0.0,
                method=f"usda_portion({unit_label},vol={amount:g})",
            )
            current = per_food[fdc_id].get(unit_label)
            if current is None or portion_sort_key(candidate) > portion_sort_key(current):
                per_food[fdc_id][unit_label] = candidate
            if index % 250000 == 0:
                print(f"Scanned food_portion.csv rows: {index}")
    return per_food


def build_usda_datasets(usda_dir: Path, *, include_branded: bool = True) -> tuple[list[dict], list[dict]]:
    print(f"Loading USDA data from {usda_dir}")
    food_map = load_usda_food_map(usda_dir)
    branded_meta = load_usda_branded_map(usda_dir) if include_branded else {}

    generic_ids = {fdc_id for fdc_id, row in food_map.items() if row["data_type"] in USDA_GENERIC_TYPES}
    branded_ids = (
        {fdc_id for fdc_id, row in food_map.items() if row["data_type"] == USDA_BRANDED_TYPE}
        if include_branded else set()
    )
    nutrient_ids = generic_ids | branded_ids

    nutrients = load_usda_nutrients(usda_dir, nutrient_ids)
    portions = load_usda_portions(usda_dir, generic_ids | branded_ids)
    print(
        "USDA source split: "
        f"{len(generic_ids)} generic ids, {len(branded_ids)} branded ids, "
        f"{len(nutrients)} nutrient rows, {len(portions)} portion rows"
    )

    generic_rows: list[dict] = []
    branded_rows: list[dict] = []

    for fdc_id, base in food_map.items():
        row = {
            "id": f"usda:{fdc_id}",
            "source": "usda",
            "source_id": fdc_id,
            "source_table": base["data_type"],
            "description": base["description"],
            "normalized_description": normalize_food_name(base["description"]),
            "ingredients_text": None,
            "calories_per_100g": None,
            "protein_g": None,
            "carbs_g": None,
            "fat_g": None,
            "density_g_ml": None,
            "density_method": None,
            "density_source": None,
            "cup_g": None,
            "tbsp_g": None,
            "tsp_g": None,
            "fl_oz_g": None,
        }
        row.update(nutrients.get(fdc_id, {}))

        portion_map = portions.get(fdc_id, {})
        for unit_name, field_name, ml in (
            ("cup", "cup_g", 240.0),
            ("tbsp", "tbsp_g", 15.0),
            ("tsp", "tsp_g", 5.0),
            ("fl_oz", "fl_oz_g", 29.5735),
        ):
            choice = portion_map.get(unit_name)
            if choice is not None:
                row[field_name] = choice.grams_per_unit

        best_density_choice: Optional[tuple[str, PortionChoice]] = None
        for unit_name, choice in portion_map.items():
            if best_density_choice is None or portion_sort_key(choice) > portion_sort_key(best_density_choice[1]):
                best_density_choice = (unit_name, choice)
        if best_density_choice is not None:
            unit_name, choice = best_density_choice
            base_ml = {
                "cup": 240.0,
                "tbsp": 15.0,
                "tsp": 5.0,
                "fl_oz": 29.5735,
                "ml": 1.0,
            }.get(unit_name, 1000.0 if unit_name == "liter" else None)
            if base_ml:
                row["density_g_ml"] = round(float(choice.grams_per_unit) / base_ml, 4)
                row["density_method"] = choice.method
                row["density_source"] = "usda_food_portion"

        usable_generic = (
            row["calories_per_100g"] is not None
            or row["density_g_ml"] is not None
        )
        if base["data_type"] in USDA_GENERIC_TYPES and usable_generic:
            generic_rows.append(row)
        elif include_branded and base["data_type"] == USDA_BRANDED_TYPE:
            meta = branded_meta.get(fdc_id, {})
            branded_row = dict(row)
            branded_row["source"] = "usda_branded"
            branded_row["id"] = f"usda_branded:{fdc_id}"
            branded_row["brand_owner"] = meta.get("brand_owner")
            branded_row["brand_name"] = meta.get("brand_name")
            branded_row["ingredients_text"] = meta.get("ingredients_text")
            branded_row["household_serving_fulltext"] = meta.get("household_serving_fulltext")
            branded_row["serving_size"] = meta.get("serving_size")
            branded_row["serving_size_unit"] = meta.get("serving_size_unit")
            branded_row["branded_food_category"] = meta.get("branded_food_category")
            branded_rows.append(branded_row)

    return generic_rows, branded_rows


def build_cofid_dataset(xlsx_path: Path) -> list[dict]:
    print(f"Loading CoFID data from {xlsx_path}")
    reader = XlsxReader(xlsx_path)
    try:
        gravity_by_code: dict[str, float] = {}
        for row in reader.iter_rows("1.2 Factors", min_row=4):
            if len(row) < 9 or not row[0]:
                continue
            gravity = safe_float(row[8])
            if gravity is None or not (0.05 <= gravity <= 3.0):
                continue
            gravity_by_code[str(row[0]).strip()] = round(float(gravity), 4)

        foods: list[dict] = []
        for row in reader.iter_rows("1.3 Proximates", min_row=4):
            if len(row) < 13:
                continue
            code = (row[0] or "").strip() if row[0] else ""
            name = (row[1] or "").strip() if row[1] else ""
            if not code or not name:
                continue
            kcal = safe_float(row[12])
            if kcal is None:
                continue
            density = gravity_by_code.get(code)
            foods.append({
                "id": f"cofid:{code}",
                "source": "cofid",
                "source_id": code,
                "source_table": "cofid",
                "description": name,
                "normalized_description": normalize_food_name(name),
                "ingredients_text": None,
                "calories_per_100g": round(float(kcal), 4),
                "protein_g": safe_float(row[9]),
                "carbs_g": safe_float(row[11]),
                "fat_g": safe_float(row[10]),
                "density_g_ml": density,
                "density_method": "cofid_specific_gravity" if density is not None else None,
                "density_source": "cofid_factors_specific_gravity" if density is not None else None,
                "cup_g": None,
                "tbsp_g": None,
                "tsp_g": None,
                "fl_oz_g": None,
            })
        return foods
    finally:
        reader.close()


def build_fao_dataset(density_path: Path, names_path: Path) -> list[dict]:
    print(f"Loading FAO data from {density_path} and {names_path}")
    density_rows = json.loads(density_path.read_text())
    names = json.loads(names_path.read_text())
    rows: list[dict] = []
    for idx, (density_row, name) in enumerate(zip(density_rows, names)):
        rows.append({
            "id": f"fao:{idx}",
            "source": "fao",
            "source_id": str(idx),
            "source_table": "fao_density",
            "description": name,
            "normalized_description": normalize_food_name(name),
            "ingredients_text": None,
            "calories_per_100g": None,
            "protein_g": None,
            "carbs_g": None,
            "fat_g": None,
            "density_g_ml": density_row.get("density_g_ml"),
            "density_method": "fao_measured",
            "density_source": "fao_density",
            "cup_g": None,
            "tbsp_g": None,
            "tsp_g": None,
            "fl_oz_g": None,
        })
    return rows


def build_slim_rag_rows(unified_full_rows: list[dict]) -> list[dict]:
    def build_retrieval_text(row: dict) -> str:
        desc = (row.get("description") or "").strip()
        desc_lower = desc.lower()

        # Collect cross-dialect synonym aliases
        aliases: set[str] = set()
        for keyword, alts in FOOD_SYNONYMS.items():
            if keyword in desc_lower:
                aliases.update(alts)

        parts: list[str] = []

        def add_part(value: object) -> None:
            if value is None:
                return
            text = str(value).strip()
            if not text:
                return
            if text.lower() not in {p.lower() for p in parts}:
                parts.append(text)

        add_part(desc)
        add_part(row.get("household_serving_fulltext"))
        add_part(row.get("ingredients_text"))
        if aliases:
            parts.append("also known as: " + ", ".join(sorted(aliases)))
        return " | ".join(parts)

    slim_rows: list[dict] = []
    for row in unified_full_rows:
        slim_rows.append({
            "id": row["id"],
            "fdc_id": row["source_id"] if row["source"] == "usda" else None,
            "description": row["description"],
            "retrieval_text": build_retrieval_text(row),
            "calories_per_100g": row.get("calories_per_100g"),
            "protein_g": row.get("protein_g"),
            "carbs_g": row.get("carbs_g"),
            "fat_g": row.get("fat_g"),
            "density_g_ml": row.get("density_g_ml"),
            "density_method": row.get("density_method"),
            "density_source": row.get("density_source"),
            "cup_g": row.get("cup_g"),
            "tbsp_g": row.get("tbsp_g"),
            "tsp_g": row.get("tsp_g"),
            "fl_oz_g": row.get("fl_oz_g"),
            "source": row["source"],
        })
    return slim_rows


def build_report(
    generic_usda_rows: list[dict],
    cofid_rows: list[dict],
    fao_rows: list[dict],
    branded_rows: list[dict],
    unified_full_rows: list[dict],
) -> dict:
    def summarize_rows(rows: list[dict]) -> dict:
        density_methods = Counter(row.get("density_method") or "none" for row in rows)
        density_sources = Counter(row.get("density_source") or "none" for row in rows)
        source_tables = Counter(row.get("source_table") or "none" for row in rows)
        return {
            "rows": len(rows),
            "with_kcal": sum(1 for row in rows if row.get("calories_per_100g") is not None),
            "with_density": sum(1 for row in rows if row.get("density_g_ml") is not None),
            "density_methods": dict(sorted(density_methods.items())),
            "density_sources": dict(sorted(density_sources.items())),
            "source_tables": dict(sorted(source_tables.items())),
        }

    return {
        "usda_generic": summarize_rows(generic_usda_rows),
        "cofid": summarize_rows(cofid_rows),
        "fao": summarize_rows(fao_rows),
        "usda_branded": summarize_rows(branded_rows),
        "unified_non_branded": summarize_rows(unified_full_rows),
    }


def maybe_build_faiss(names: list[str], output_path: Path) -> bool:
    try:
        import faiss  # type: ignore
        import numpy as np  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"Skipping FAISS build: {exc}")
        return False

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(names, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
    array = np.array(vectors, dtype=np.float32)
    index = faiss.IndexFlatIP(array.shape[1])
    index.add(array)
    faiss.write_index(index, str(output_path))
    return True


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def summarize(rows: Iterable[dict], label: str) -> None:
    rows = list(rows)
    with_density = sum(1 for row in rows if row.get("density_g_ml") is not None)
    with_kcal = sum(1 for row in rows if row.get("calories_per_100g") is not None)
    print(f"{label}: {len(rows)} rows, {with_kcal} with kcal, {with_density} with density")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usda-dir", type=Path, default=DEFAULT_USDA_DIR)
    parser.add_argument("--cofid-xlsx", type=Path, default=DEFAULT_COFID_XLSX)
    parser.add_argument("--fao-density", type=Path, default=DEFAULT_FAO_DENSITY)
    parser.add_argument("--fao-names", type=Path, default=DEFAULT_FAO_NAMES)
    parser.add_argument("--skip-faiss", action="store_true")
    parser.add_argument("--skip-branded", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    generic_usda_rows, branded_rows = build_usda_datasets(args.usda_dir, include_branded=not args.skip_branded)
    cofid_rows = build_cofid_dataset(args.cofid_xlsx)
    fao_rows = build_fao_dataset(args.fao_density, args.fao_names)

    unified_full_rows = generic_usda_rows + cofid_rows + fao_rows
    unified_full_rows.sort(key=lambda row: (row["description"].lower(), row["source"], row["source_id"]))
    branded_rows.sort(key=lambda row: row["description"].lower())
    fao_rows.sort(key=lambda row: row["description"].lower())

    slim_rows = build_slim_rag_rows(unified_full_rows)
    names = [row.get("retrieval_text") or row["description"] for row in slim_rows]
    report = build_report(generic_usda_rows, cofid_rows, fao_rows, branded_rows, unified_full_rows)

    write_json(UNIFIED_OUT_DIR / "unified_foods_full.json", unified_full_rows)
    write_json(UNIFIED_OUT_DIR / "unified_foods.json", slim_rows)
    write_json(UNIFIED_OUT_DIR / "unified_food_names.json", names)
    write_json(FAO_OUT_DIR / "fao_foods.json", fao_rows)
    write_json(REPORT_OUT, report)
    if not args.skip_branded:
        write_json(BRANDED_OUT_DIR / "usda_branded_foods.json", branded_rows)

    if not args.skip_faiss:
        maybe_build_faiss(names, UNIFIED_OUT_DIR / "unified_faiss.index")

    summarize(generic_usda_rows, "USDA generic")
    summarize(cofid_rows, "CoFID")
    if not args.skip_branded:
        summarize(branded_rows, "USDA branded")
    summarize(fao_rows, "FAO")
    summarize(unified_full_rows, "Unified non-branded")
    print(f"Wrote {UNIFIED_OUT_DIR / 'unified_foods_full.json'}")
    print(f"Wrote {UNIFIED_OUT_DIR / 'unified_foods.json'}")
    if not args.skip_branded:
        print(f"Wrote {BRANDED_OUT_DIR / 'usda_branded_foods.json'}")
    print(f"Wrote {FAO_OUT_DIR / 'fao_foods.json'}")
    print(f"Wrote {REPORT_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
