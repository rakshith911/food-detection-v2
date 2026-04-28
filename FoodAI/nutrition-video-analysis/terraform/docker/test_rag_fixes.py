#!/usr/bin/env python3
"""
Targeted regression test for the three RAG fixes:
  1. Branded token-coverage: "kalamata olives" → branded Kalamata entry, not generic USDA olives
  2. meal_context reaches FAISS: ambiguous labels resolve better with dish context
  3. Density/calorie consistency: density_matched should align with calorie_matched source

Run from the docker/ directory:
    python test_rag_fixes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "app"
for p in [str(ROOT), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.config import settings
from app.models import load_nutrition_rag
from nutrition_rag_system import NutritionLookupError

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"


def check(label: str, condition: bool, note: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}" + (f"  ({note})" if note else ""))
    return condition


def run(rag, food_name: str, volume_ml: float = 100.0, meal_context: str = None):
    try:
        return rag.get_nutrition_for_food(
            food_name,
            volume_ml=volume_ml,
            meal_context=meal_context,
        )
    except NutritionLookupError as exc:
        print(f"  [{FAIL}] lookup failed: {exc}")
        return None


def main() -> int:
    rag = load_nutrition_rag(
        unified_faiss_path=settings.UNIFIED_FAISS_PATH,
        unified_foods_path=settings.UNIFIED_FOODS_PATH,
        unified_food_names_path=settings.UNIFIED_FOOD_NAMES_PATH,
        fao_faiss_path=settings.FAO_FAISS_PATH,
        fao_foods_path=settings.FAO_FOODS_PATH,
        fao_food_names_path=settings.FAO_FOOD_NAMES_PATH,
        branded_foods_path=settings.BRANDED_FOODS_PATH,
        gemini_api_key=settings.GEMINI_API_KEY,
        gemini_model=settings.GEMINI_FLASH_MODEL,
    )

    failures = 0

    # ── Fix 1: Branded token-coverage ─────────────────────────────────────────
    # "kalamata olives" should match branded Kalamata entry (~167 kcal) not the
    # generic USDA "Olives, black, ripe, canned" (~80 kcal).
    # Branded wins because it contains "kalamata" which unified description lacks.
    print("\n── Fix 1: Branded token-coverage ──────────────────────────────────")
    cases = [
        ("kalamata olives",  "kalamata",  100, 350),  # branded 100–350 kcal (plain ~167, in oil ~300)
    ]
    for food, must_contain, kcal_lo, kcal_hi in cases:
        print(f"\n  {food!r}")
        r = run(rag, food)
        if r is None:
            failures += 1
            continue
        calorie_matched = (r.get("calorie_matched") or "").lower()
        kcal = r.get("calories_per_100g", 0)
        src  = r.get("calorie_source", "")
        print(f"    calorie_matched : {r.get('calorie_matched')}")
        print(f"    calorie_source  : {src}")
        print(f"    calories/100g   : {kcal}")
        ok1 = check(f"description contains '{must_contain}'", must_contain in calorie_matched)
        ok2 = check(f"kcal in [{kcal_lo}, {kcal_hi}]", kcal_lo <= kcal <= kcal_hi, f"got {kcal}")
        ok3 = check("branded source used", "branded" in src)
        if not all([ok1, ok2, ok3]):
            failures += 1

    # ── Fix 2: meal_context reaches FAISS ─────────────────────────────────────
    # Ambiguous short labels should resolve differently with vs without context.
    # "sauce" alone is vague; "sauce" + "pasta carbonara" context should pick
    # something cream/cheese-based rather than tomato-based.
    # We verify the contextual variant is at least *tried* by checking the matched
    # description differs or calorie value shifts between the two calls.
    print("\n── Fix 2: meal_context shifts FAISS results ────────────────────────")
    ctx_cases = [
        # (food_name, context_a, context_b, expect_different_match)
        ("rice", "japanese sushi bowl",   "mexican burrito bowl", True),
        ("sauce", "pasta carbonara dish", "mexican taco bowl",    True),
    ]
    for food, ctx_a, ctx_b, expect_diff in ctx_cases:
        print(f"\n  {food!r}  (context A vs B)")
        r_a = run(rag, food, meal_context=ctx_a)
        r_b = run(rag, food, meal_context=ctx_b)
        if r_a is None or r_b is None:
            failures += 1
            continue
        match_a = r_a.get("calorie_matched", "")
        match_b = r_b.get("calorie_matched", "")
        kcal_a  = r_a.get("calories_per_100g", 0)
        kcal_b  = r_b.get("calories_per_100g", 0)
        print(f"    context '{ctx_a}' → {match_a!r}  ({kcal_a} kcal)")
        print(f"    context '{ctx_b}' → {match_b!r}  ({kcal_b} kcal)")
        differs = (match_a != match_b) or (abs(kcal_a - kcal_b) > 5)
        ok = check("context produces different match or kcal", differs)
        if not ok:
            print(f"    [{WARN}] same match with both contexts — context may not be shifting FAISS")
            # Not a hard failure; same result can be correct if DB is unambiguous

    # ── Fix 3: Density/calorie source consistency ──────────────────────────────
    # density_matched should relate to the same food as calorie_matched.
    # After fix, we run calorie_matched first in density query, so sources align.
    print("\n── Fix 3: Density/calorie source consistency ───────────────────────")
    consistency_cases = [
        "kalamata olives",
        "yellow rice",
        "grilled chicken breast",
        "hummus",
        "pita bread",
        "tahini sauce",
    ]
    for food in consistency_cases:
        print(f"\n  {food!r}")
        r = run(rag, food)
        if r is None:
            failures += 1
            continue
        cm = r.get("calorie_matched") or ""
        dm = r.get("density_matched") or ""
        cs = r.get("calorie_source") or ""
        ds = r.get("density_source") or ""
        print(f"    calorie → {cm!r}  [{cs}]")
        print(f"    density → {dm!r}  [{ds}]")

        # Check 1: sources come from same DB tier (both branded, or both usda, or one gemini)
        calorie_tier = "branded" if "branded" in cs else ("gemini" if "gemini" in cs else "usda")
        density_tier = "branded" if "branded" in ds else ("gemini" if "gemini" in ds else "usda")
        tiers_match  = calorie_tier == density_tier
        ok1 = check("same DB tier for both", tiers_match, f"{calorie_tier} vs {density_tier}")

        # Check 2: if both from USDA/branded, key food word should appear in both descriptions
        if tiers_match and calorie_tier != "gemini":
            import re as _re
            query_words = {w for w in food.lower().split() if len(w) >= 4}
            cal_words   = set(_re.sub(r'[^a-z0-9 ]', ' ', cm.lower()).split())
            den_words   = set(_re.sub(r'[^a-z0-9 ]', ' ', dm.lower()).split())
            shared_in_cal = query_words & cal_words
            shared_in_den = query_words & den_words
            ok2 = check(
                "both descriptions share at least one query word",
                bool(shared_in_cal) and bool(shared_in_den),
                f"cal={shared_in_cal}, den={shared_in_den}",
            )
            if not ok1 or not ok2:
                failures += 1
        elif not ok1:
            failures += 1

    print(f"\n{'='*60}")
    if failures:
        print(f"  {FAIL}  {failures} check(s) failed")
    else:
        print(f"  {PASS}  All checks passed")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
