#!/usr/bin/env python3
"""
Print a human-readable RAG lookup table using the production wrapper.

This exercises the same loader path the pipeline uses:
  app.models.load_nutrition_rag() -> zoe_nutrition_rag.NutritionRAG

It is useful for verifying that the production app now follows the uploaded
RAG bundle retrieval path and for inspecting candidate traces.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "app"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.config import settings  # noqa: E402
from app.models import load_nutrition_rag  # noqa: E402


DEFAULT_INGREDIENTS = [
    "yellow rice",
    "falafel",
    "white sauce",
    "lettuce",
    "tomato",
    "cucumber",
    "hot sauce",
    "shredded carrot",
    "onion",
]


def _fmt_rerank(value) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):+.2f}"
    except Exception:
        return str(value)


def main() -> int:
    ingredients = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_INGREDIENTS

    rag = load_nutrition_rag(
        unified_faiss_path=settings.UNIFIED_FAISS_PATH,
        unified_foods_path=settings.UNIFIED_FOODS_PATH,
        unified_food_names_path=settings.UNIFIED_FOOD_NAMES_PATH,
        gemini_api_key=settings.GEMINI_API_KEY,
        gemini_model=settings.GEMINI_FLASH_MODEL,
    )

    print(f"{'Ingredient':<18} {'USDA Match':<38} {'Method':<18} {'Rerank':>8}  {'Density':>10}")
    print("-" * 100)

    for ingredient in ingredients:
        result = rag.get_nutrition_for_food(ingredient, volume_ml=100.0)

        match = (result.get("calorie_matched") or "N/A")[:38]
        method = result.get("lookup_method") or "?"
        rerank = _fmt_rerank(result.get("rerank_score"))
        density = float(result.get("density_g_per_ml") or 0.0)

        print(f"{ingredient:<18} {match:<38} {method:<18} {rerank:>8}  {density:>7.3f} g/ml")

        density_match = result.get("density_matched")
        density_source = result.get("density_source")
        calorie_source = result.get("calorie_source")
        faiss_score = result.get("faiss_score")

        print(f"  density_match:  {density_match}")
        print(f"  density_source: {density_source}")
        print(f"  calorie_source: {calorie_source}")
        if faiss_score is not None:
            print(f"  faiss_score:    {faiss_score:.4f}")

        candidates = result.get("rag_candidates") or []
        if candidates:
            print("  top_candidates:")
            for candidate in candidates[:5]:
                rank = candidate.get("rank")
                rerank_score = candidate.get("rerank_score")
                candidate_faiss = candidate.get("faiss_score")
                description = candidate.get("description", "")
                print(
                    f"    #{rank} rerank={rerank_score:+.4f} "
                    f"faiss={candidate_faiss:.4f} {description}"
                )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
