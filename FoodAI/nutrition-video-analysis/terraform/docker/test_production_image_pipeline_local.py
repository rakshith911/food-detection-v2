#!/usr/bin/env python3
"""
Local harness for testing the nutrition image pipeline before deployment.

Examples:
  python test_production_image_pipeline_local.py /abs/path/to/image.jpg
  python test_production_image_pipeline_local.py /abs/path/to/images_dir --limit 3
  python test_production_image_pipeline_local.py /abs/path/to/image.jpg --context questionnaire.json
  python test_production_image_pipeline_local.py /abs/path/to/images_dir --context contexts_by_file.json --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from app.config import Settings
from app.models import ModelManager
from app.pipeline import NutritionVideoPipeline


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local nutrition image pipeline on one or more sample images.")
    parser.add_argument("input_path", help="Path to an image file or a directory of images")
    parser.add_argument(
        "--context",
        help=(
            "Optional questionnaire context JSON. "
            "Can be either a single context object or a mapping of filename -> context object."
        ),
    )
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cpu"), help="cpu or cuda")
    parser.add_argument(
        "--output-dir",
        help="Directory where run outputs should be written. Defaults to docker/outputs/<timestamp>",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images when input_path is a directory")
    parser.add_argument(
        "--disable-production",
        action="store_true",
        help="Force the legacy image pipeline instead of the new production image pipeline",
    )
    return parser.parse_args()


def resolve_images(input_path: Path, limit: Optional[int]) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    images = sorted(
        path for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit is not None:
        images = images[:limit]
    return images


def load_context_map(context_path: Optional[str]) -> Any:
    if not context_path:
        return None
    path = Path(context_path)
    if not path.exists():
        raise FileNotFoundError(f"Context JSON not found: {path}")
    return json.loads(path.read_text())


def context_for_image(context_data: Any, image_path: Path) -> Optional[dict]:
    if not context_data:
        return None
    if isinstance(context_data, dict):
        if any(key in context_data for key in ("hidden_ingredients", "extras", "recipe_description")):
            return context_data
        by_name = context_data.get(image_path.name)
        if isinstance(by_name, dict):
            return by_name
        by_stem = context_data.get(image_path.stem)
        if isinstance(by_stem, dict):
            return by_stem
    return None


def build_settings(args: argparse.Namespace, run_output_dir: Path) -> Settings:
    config = Settings()
    config.DEVICE = args.device
    config.USE_PRODUCTION_IMAGE_PIPELINE = not args.disable_production
    config.DEBUG = True
    config.OUTPUT_DIR = run_output_dir
    config.UPLOAD_DIR = SCRIPT_DIR / "data" / "uploads"
    config.MODEL_CACHE_DIR = SCRIPT_DIR / "models"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return config


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def summarize_result(image_path: Path, result: dict) -> dict:
    production_debug = result.get("production_debug") or {}
    overall = production_debug.get("overall_confidence") or {}
    nutrition = result.get("nutrition") or {}
    summary = nutrition.get("summary") or {}
    items = nutrition.get("items") or []
    gemini_outputs = result.get("gemini_outputs") or []
    runtime = result.get("pipeline_runtime") or {}
    return {
        "image": image_path.name,
        "job_id": result.get("job_id"),
        "status": result.get("status"),
        "meal_name": production_debug.get("meal_name") or result.get("meal_name"),
        "cuisine_type": production_debug.get("cuisine_type"),
        "cooking_method": production_debug.get("cooking_method"),
        "total_calories_kcal": summary.get("total_calories_kcal"),
        "num_food_items": summary.get("num_food_items", len(items)),
        "overall_confidence": overall.get("overall_confidence"),
        "overall_uncertainty": overall.get("overall_uncertainty"),
        "gemini_output_count": len(gemini_outputs),
        "gemini_stages": [entry.get("stage") for entry in gemini_outputs],
        "used_production_debug": bool(production_debug),
        "runtime": runtime,
    }


def print_summary(summary: dict) -> None:
    print(f"\n[{summary['image']}]")
    print(f"  status: {summary['status']}")
    print(f"  job_id: {summary['job_id']}")
    if summary.get("meal_name"):
        print(f"  meal: {summary['meal_name']}")
    if summary.get("cuisine_type"):
        print(f"  cuisine: {summary['cuisine_type']}")
    if summary.get("cooking_method"):
        print(f"  cooking_method: {summary['cooking_method']}")
    print(f"  total_kcal: {summary.get('total_calories_kcal')}")
    print(f"  items: {summary.get('num_food_items')}")
    print(f"  overall_confidence: {summary.get('overall_confidence')}")
    print(f"  overall_uncertainty: {summary.get('overall_uncertainty')}")
    print(f"  gemini_outputs: {summary.get('gemini_output_count')}")
    print(f"  pipeline_path: {'production' if summary.get('used_production_debug') else 'legacy_fallback'}")
    runtime = summary.get("runtime") or {}
    if runtime:
        print(f"  runtime: {json.dumps(runtime, default=str)}")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (SCRIPT_DIR / "outputs" / run_stamp)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    context_data = load_context_map(args.context)
    images = resolve_images(input_path, args.limit)
    if not images:
        print("No images found to test.")
        return 1

    print("=" * 80)
    print("Local Nutrition Pipeline Test")
    print("=" * 80)
    print(f"input: {input_path}")
    print(f"images: {len(images)}")
    print(f"device: {args.device}")
    print(f"production_pipeline: {not args.disable_production}")
    print(f"run_output_dir: {run_output_dir}")
    print(f"gemini_api_configured: {bool(Settings().GEMINI_API_KEY)}")

    config = build_settings(args, run_output_dir)
    model_manager = ModelManager(config)
    pipeline = NutritionVideoPipeline(model_manager, config)

    run_summary: list[dict] = []
    failures: list[dict] = []

    for index, image_path in enumerate(images, start=1):
        job_id = f"local-test-{run_stamp}-{index:03d}"
        image_dir = run_output_dir / image_path.stem
        image_dir.mkdir(parents=True, exist_ok=True)
        user_context = context_for_image(context_data, image_path)

        print(f"\nProcessing {index}/{len(images)}: {image_path.name}")
        if user_context:
            print(f"  questionnaire context: yes")
            write_json(image_dir / "user_context.json", user_context)
        else:
            print(f"  questionnaire context: no")

        try:
            result = pipeline.process_image(image_path, job_id, user_context=user_context)
            write_json(image_dir / "result.json", result)
            summary = summarize_result(image_path, result)
            run_summary.append(summary)
            write_json(image_dir / "summary.json", summary)
            print_summary(summary)
        except Exception as exc:
            failure = {
                "image": image_path.name,
                "job_id": job_id,
                "error": str(exc),
            }
            failures.append(failure)
            write_json(image_dir / "error.json", failure)
            print(f"  FAILED: {exc}")

    aggregate = {
        "run_timestamp": run_stamp,
        "input_path": str(input_path),
        "images_processed": len(images),
        "success_count": len(run_summary),
        "failure_count": len(failures),
        "results": run_summary,
        "failures": failures,
    }
    write_json(run_output_dir / "run_summary.json", aggregate)

    print("\n" + "=" * 80)
    print("Run Complete")
    print("=" * 80)
    print(f"success: {len(run_summary)}")
    print(f"failed: {len(failures)}")
    print(f"summary_json: {run_output_dir / 'run_summary.json'}")
    print(f"per_image_results: {run_output_dir}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
