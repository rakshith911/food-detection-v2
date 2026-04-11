#!/usr/bin/env python3
"""
Smoke-test the newly built USDA+CoFID unified DB plus separate FAO fallback
using the current NutritionRAG pipeline, without replacing production files.

Examples:
  python3 test_rag_new_db_smoke.py
  python3 test_rag_new_db_smoke.py "yellow rice" "hot sauce" "falafel"
  python3 test_rag_new_db_smoke.py --volume-ml 150 "onion" "tomato"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


DOCKER_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCKER_DIR.parent.parent.parent.parent
APP_DIR = DOCKER_DIR / "app"

if str(DOCKER_DIR) not in sys.path:
    sys.path.insert(0, str(DOCKER_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.models import load_nutrition_rag  # noqa: E402


DEFAULT_INGREDIENTS = [
    "yellow rice",
    "falafel",
    "hot sauce",
    "lettuce",
    "tomato",
    "cucumber",
    "onion",
    "pasta",
    "apple pie",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ingredients", nargs="*", help="Ingredients or dish names to look up")
    parser.add_argument("--volume-ml", type=float, default=100.0, help="Volume to use for nutrition calculation")
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Allow Gemini fallback using GEMINI_API_KEY from env or app settings",
    )
    return parser.parse_args()


def _fmt_float(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _macro_summary(entry: Optional[dict], mass_g: Optional[float]) -> tuple[str, str, str]:
    if not entry or mass_g is None or mass_g <= 0:
        return "N/A", "N/A", "N/A"
    protein = entry.get("protein_g")
    carbs = entry.get("carbs_g")
    fat = entry.get("fat_g")
    scale = float(mass_g) / 100.0

    def total_str(value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        return f"{float(value) * scale:.1f}g"

    return total_str(protein), total_str(carbs), total_str(fat)


def main() -> int:
    args = parse_args()
    ingredients = args.ingredients or DEFAULT_INGREDIENTS

    unified_dir = REPO_ROOT / "unified_data"
    fao_dir = REPO_ROOT / "fao_data"

    gemini_api_key = None
    gemini_model = "gemini-flash-latest"
    gemini_density_cache_path = unified_dir / "gemini_density_cache.json"
    if args.gemini:
        try:
            from app.config import settings

            gemini_api_key = settings.GEMINI_API_KEY
            gemini_model = settings.GEMINI_FLASH_MODEL
            gemini_density_cache_path = settings.GEMINI_DENSITY_CACHE_PATH
        except Exception:
            gemini_api_key = None

    rag = load_nutrition_rag(
        unified_faiss_path=unified_dir / "unified_faiss.index",
        unified_foods_path=unified_dir / "unified_foods.json",
        unified_food_names_path=unified_dir / "unified_food_names.json",
        fao_faiss_path=fao_dir / "fao_faiss.index",
        fao_foods_path=fao_dir / "fao_foods.json",
        fao_food_names_path=fao_dir / "fao_food_names.json",
        gemini_density_cache_path=gemini_density_cache_path,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
    )

    print(
        f"{'Ingredient':<18} {'Cal Match':<34} {'Density Match':<34} "
        f"{'kcal/100g':>10} {'dens':>8} {'mass':>8} {'kcal':>8}"
    )
    print("-" * 130)

    for ingredient in ingredients:
        kcal_value, calorie_matched, calorie_source, kcal_entry = rag._lookup_unified(  # type: ignore[attr-defined]
            ingredient, "calories_per_100g", top_k=10, crop_image=None
        )
        density_value, density_matched, density_source, density_entry = rag._get_density_with_match(  # type: ignore[attr-defined]
            ingredient, top_k=10, crop_image=None
        )

        final_result = rag.get_nutrition_for_food(ingredient, volume_ml=args.volume_ml)
        mass_g = final_result.get("mass_g")
        total_kcal = final_result.get("total_calories")

        cal_match_display = (calorie_matched or final_result.get("calorie_matched") or "N/A")[:34]
        dens_match_display = (density_matched or final_result.get("density_matched") or "N/A")[:34]

        print(
            f"{ingredient:<18} {cal_match_display:<34} {dens_match_display:<34} "
            f"{_fmt_float(final_result.get('calories_per_100g'), 1):>10} "
            f"{_fmt_float(final_result.get('density_g_per_ml'), 3):>8} "
            f"{_fmt_float(mass_g, 1):>8} "
            f"{_fmt_float(total_kcal, 1):>8}"
        )

        macro_entry = kcal_entry or density_entry
        protein_total, carbs_total, fat_total = _macro_summary(macro_entry, mass_g)

        print(f"  calorie_source: {final_result.get('calorie_source') or calorie_source}")
        print(f"  density_source: {final_result.get('density_source') or density_source}")
        if kcal_entry:
            print(
                "  db_macros/100g: "
                f"protein={_fmt_float(kcal_entry.get('protein_g'), 1)}g "
                f"carbs={_fmt_float(kcal_entry.get('carbs_g'), 1)}g "
                f"fat={_fmt_float(kcal_entry.get('fat_g'), 1)}g"
            )
        else:
            print("  db_macros/100g: N/A")
        print(f"  estimated_macros: protein={protein_total} carbs={carbs_total} fat={fat_total}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
