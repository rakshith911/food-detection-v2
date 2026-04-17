#!/usr/bin/env python3
"""
Local end-to-end pipeline test.

Usage:
    python run_local.py <image_or_dir> [image2 ...]

Examples:
    python run_local.py food.jpg
    python run_local.py ~/Downloads/lunch.jpg ~/Downloads/dinner.jpg
    python run_local.py ./test_images/
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("USE_PRODUCTION_IMAGE_PIPELINE", "true")

# ── Logging: show pipeline stages as they happen ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
# Silence noisy third-party loggers
for lib in ("urllib3", "httpx", "httpcore", "PIL", "filelock", "transformers",
            "sentence_transformers", "faiss", "torch"):
    logging.getLogger(lib).setLevel(logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────

def _load_pipeline():
    from app.config import Settings
    from app.models import ModelManager
    from app.pipeline import NutritionVideoPipeline

    cfg = Settings()
    cfg.DEVICE = "cpu"
    cfg.DEBUG = True
    cfg.USE_PRODUCTION_IMAGE_PIPELINE = True

    mm = ModelManager(cfg)
    return NutritionVideoPipeline(mm, cfg), cfg


def _print_results(image_path: Path, result: dict, elapsed: float):
    prod = result.get("production_debug") or {}
    nutrition = result.get("nutrition") or {}
    summary = nutrition.get("summary") or {}
    items = nutrition.get("items") or []

    W = 72
    print("\n" + "═" * W)
    print(f"  {image_path.name}")
    if prod.get("meal_name"):
        print(f"  {prod['meal_name']}"
              + (f"  ·  {prod['cuisine_type']}" if prod.get("cuisine_type") else ""))
    print("═" * W)

    if not items:
        print("  No food items detected.")
        print("═" * W)
        return

    # Header
    print(f"  {'Item':<24} {'Vol (ml)':>9} {'Mass (g)':>9} {'kcal/100g':>10} {'kcal':>7}  Source")
    print("  " + "─" * (W - 2))

    total_kcal = 0.0
    for item in items:
        name   = (item.get("food_name") or "?")[:24]
        vol    = item.get("volume_ml") or 0
        mass   = item.get("mass_g") or 0
        k100   = item.get("calories_per_100g") or 0
        kcal   = item.get("total_calories") or 0
        src    = (item.get("calorie_source") or "").replace("usda_", "").replace("cofid_", "").replace("_gemini_verified", "✓")
        total_kcal += kcal
        print(f"  {name:<24} {vol:>9.1f} {mass:>9.1f} {k100:>10.1f} {kcal:>7.1f}  {src}")

    print("  " + "─" * (W - 2))
    print(f"  {'TOTAL':<24} {summary.get('total_food_volume_ml', 0):>9.1f}"
          f" {summary.get('total_mass_g', 0):>9.1f} {'':>10} {total_kcal:>7.1f}")
    print("═" * W)

    conf = (prod.get("overall_confidence") or {})
    if conf:
        print(f"  confidence: {conf.get('overall_confidence', '?')}  "
              f"uncertainty: {conf.get('overall_uncertainty', '?')}")
    print(f"  elapsed: {elapsed:.1f}s")
    print()


def _run(paths: list[Path]):
    print(f"\n{'─'*72}")
    print("  Loading models (SAM3 + ZoeDepth)…")
    print(f"{'─'*72}\n")

    t0 = time.time()
    pipeline, cfg = _load_pipeline()
    print(f"\nModels ready in {time.time()-t0:.1f}s\n")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = SCRIPT_DIR / "outputs" / stamp
    out_root.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(paths, 1):
        job_id = f"local-{stamp}-{i:03d}"
        print(f"\n{'─'*72}")
        print(f"  [{i}/{len(paths)}]  {img_path.name}  →  job {job_id}")
        print(f"{'─'*72}")

        t1 = time.time()
        try:
            result = pipeline.process_image(img_path, job_id)
            elapsed = time.time() - t1

            # The pipeline writes images to data/outputs/production_<job_id>/
            # Save result.json alongside them in the same directory.
            prod_debug = result.get("production_debug") or {}
            pipeline_out_dir = prod_debug.get("output_dir")
            if pipeline_out_dir:
                out_dir = Path(pipeline_out_dir)
            else:
                out_dir = cfg.OUTPUT_DIR / f"production_{job_id}"
                out_dir.mkdir(parents=True, exist_ok=True)

            (out_dir / "result.json").write_text(
                json.dumps(result, indent=2, default=str)
            )

            _print_results(img_path, result, elapsed)

            # Show saved files
            saved = sorted(out_dir.iterdir())
            print(f"  Output dir: {out_dir}")
            print(f"  Saved files ({len(saved)}):")
            for f in saved:
                print(f"    {f.name}  ({f.stat().st_size // 1024} KB)")

        except Exception as exc:
            elapsed = time.time() - t1
            print(f"\n  ✗ FAILED after {elapsed:.1f}s: {exc}")
            import traceback
            traceback.print_exc()


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths: list[Path] = []

    for arg in args:
        p = Path(arg).expanduser().resolve()
        if p.is_dir():
            paths += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            paths.append(p)
        else:
            print(f"Skipping: {arg} (not a recognised image or directory)")

    if not paths:
        print("No valid images found.")
        sys.exit(1)

    _run(paths)


if __name__ == "__main__":
    main()
