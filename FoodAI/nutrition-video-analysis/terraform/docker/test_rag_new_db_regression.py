#!/usr/bin/env python3
"""
Run pass/fail retrieval checks against the new USDA+CoFID DB.

Examples:
  python3 test_rag_new_db_regression.py
  python3 test_rag_new_db_regression.py --list
  python3 test_rag_new_db_regression.py "yellow rice" "tomato" "onion"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DOCKER_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCKER_DIR.parent.parent.parent.parent
APP_DIR = DOCKER_DIR / "app"

if str(DOCKER_DIR) not in sys.path:
    sys.path.insert(0, str(DOCKER_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.models import load_nutrition_rag  # noqa: E402


DEFAULT_CASES: list[dict] = [
    {
        "ingredient": "yellow rice",
        "expect_cal_contains": ["yellow rice", "no added fat"],
        "expect_density_contains": ["yellow rice", "no added fat"],
        "forbid_cal_contains": ["dry packet", "unprepared", "wild rice"],
        "forbid_density_contains": ["wild rice", "with fat"],
        "allow_density_sources": ["usda_"],
    },
    {
        "ingredient": "tomato",
        "expect_cal_contains": ["tomatoes", "raw"],
        "forbid_cal_contains": ["ketchup", "relish", "puree", "sauce"],
        "allow_density_sources": ["hardcoded_override", "usda_"],
    },
    {
        "ingredient": "onion",
        "expect_cal_contains": ["onions", "raw"],
        "expect_density_contains": ["onions", "raw"],
        "forbid_density_contains": ["tops only", "green onion", "scallion"],
        "allow_density_sources": ["usda_"],
    },
    {
        "ingredient": "falafel",
        "expect_cal_contains": ["falafel"],
        "allow_density_sources": ["hardcoded_override", "usda_"],
    },
    {
        "ingredient": "hot sauce",
        "expect_cal_contains": ["hot", "sauce"],
        "forbid_cal_contains": ["hot dog"],
        "allow_density_sources": ["hardcoded_override", "usda_"],
    },
    {
        "ingredient": "shredded carrot",
        "expect_cal_contains": ["carrots", "raw"],
        "expect_density_contains": ["carrots", "raw"],
        "allow_density_sources": ["usda_"],
    },
    {
        "ingredient": "yorkshire pudding",
        "expect_cal_contains": ["yorkshire", "pudding"],
        "forbid_density_contains": ["pudding, bread"],
    },
    {
        "ingredient": "white sauce",
        "expect_cal_contains": ["white", "sauce"],
        "forbid_cal_contains": ["sweet"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ingredients", nargs="*", help="Only run these ingredients from the default regression set")
    parser.add_argument("--volume-ml", type=float, default=100.0, help="Volume for final nutrition calculation")
    parser.add_argument("--list", action="store_true", help="List the default regression ingredients and exit")
    return parser.parse_args()


def _lower(value: str | None) -> str:
    return (value or "").strip().lower()


def _contains_all(haystack: str, needles: list[str]) -> bool:
    haystack = _lower(haystack)
    return all(_lower(needle) in haystack for needle in needles)


def _contains_any(haystack: str, needles: list[str]) -> bool:
    haystack = _lower(haystack)
    return any(_lower(needle) in haystack for needle in needles)


def _check_case(case: dict, rag, volume_ml: float) -> tuple[bool, list[str], dict]:
    ingredient = case["ingredient"]
    kcal_value, calorie_matched, calorie_source, _kcal_entry = rag._lookup_unified(  # type: ignore[attr-defined]
        ingredient, "calories_per_100g", top_k=10, crop_image=None
    )
    density_value, density_matched, density_source, _density_entry = rag._get_density_with_match(  # type: ignore[attr-defined]
        ingredient, top_k=10, crop_image=None
    )
    final_result = rag.get_nutrition_for_food(ingredient, volume_ml=volume_ml)

    summary = {
        "ingredient": ingredient,
        "calorie_matched": calorie_matched or final_result.get("calorie_matched") or "",
        "density_matched": density_matched or final_result.get("density_matched") or "",
        "calorie_source": final_result.get("calorie_source") or calorie_source or "",
        "density_source": final_result.get("density_source") or density_source or "",
        "calories_per_100g": final_result.get("calories_per_100g"),
        "density_g_per_ml": final_result.get("density_g_per_ml"),
        "total_calories": final_result.get("total_calories"),
        "raw_lookup_kcal": kcal_value,
        "raw_lookup_density": density_value,
    }

    failures: list[str] = []

    if case.get("expect_cal_contains"):
        if not _contains_all(summary["calorie_matched"], case["expect_cal_contains"]):
            failures.append(
                f"calorie match '{summary['calorie_matched']}' missing expected terms {case['expect_cal_contains']}"
            )

    if case.get("expect_density_contains"):
        if not _contains_all(summary["density_matched"], case["expect_density_contains"]):
            failures.append(
                f"density match '{summary['density_matched']}' missing expected terms {case['expect_density_contains']}"
            )

    if case.get("forbid_cal_contains"):
        if _contains_any(summary["calorie_matched"], case["forbid_cal_contains"]):
            failures.append(
                f"calorie match '{summary['calorie_matched']}' hit forbidden terms {case['forbid_cal_contains']}"
            )

    if case.get("forbid_density_contains"):
        if _contains_any(summary["density_matched"], case["forbid_density_contains"]):
            failures.append(
                f"density match '{summary['density_matched']}' hit forbidden terms {case['forbid_density_contains']}"
            )

    allowed_sources = case.get("allow_density_sources")
    if allowed_sources:
        density_source_l = _lower(summary["density_source"])
        if not any(_lower(prefix) in density_source_l for prefix in allowed_sources):
            failures.append(
                f"density source '{summary['density_source']}' not in allowed set {allowed_sources}"
            )

    return not failures, failures, summary


def main() -> int:
    args = parse_args()

    if args.list:
        for case in DEFAULT_CASES:
            print(case["ingredient"])
        return 0

    selected = {item.strip().lower() for item in args.ingredients if item.strip()}
    cases = [
        case for case in DEFAULT_CASES
        if not selected or case["ingredient"].lower() in selected
    ]

    if not cases:
        print("No matching regression cases selected.")
        return 1

    unified_dir = REPO_ROOT / "unified_data"
    fao_dir = REPO_ROOT / "fao_data"

    rag = load_nutrition_rag(
        unified_faiss_path=unified_dir / "unified_faiss.index",
        unified_foods_path=unified_dir / "unified_foods.json",
        unified_food_names_path=unified_dir / "unified_food_names.json",
        fao_faiss_path=fao_dir / "fao_faiss.index",
        fao_foods_path=fao_dir / "fao_foods.json",
        fao_food_names_path=fao_dir / "fao_food_names.json",
        gemini_api_key=None,
        gemini_model="gemini-flash-latest",
    )

    failures_total = 0
    print(f"{'Ingredient':<18} {'Result':<6} {'Cal Match':<34} {'Density Match':<34}")
    print("-" * 100)

    for case in cases:
        ok, failures, summary = _check_case(case, rag, args.volume_ml)
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures_total += 1
        print(
            f"{summary['ingredient']:<18} {status:<6} "
            f"{summary['calorie_matched'][:34]:<34} {summary['density_matched'][:34]:<34}"
        )
        print(f"  calorie_source: {summary['calorie_source']}")
        print(f"  density_source: {summary['density_source']}")
        print(
            "  values: "
            f"kcal/100g={summary['calories_per_100g']} "
            f"density={summary['density_g_per_ml']} "
            f"total_kcal={summary['total_calories']}"
        )
        if failures:
            for failure in failures:
                print(f"  failure: {failure}")
        print()

    if failures_total:
        print(f"FAILED: {failures_total} regression case(s) failed.")
        return 1

    print(f"PASS: all {len(cases)} regression case(s) passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
