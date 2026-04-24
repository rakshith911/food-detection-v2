#!/usr/bin/env python3
"""
Run the updated Gemini metric-depth image pipeline locally.

This exercises the same production image entrypoint used by the worker:
  raw image -> existing Gemini first pass labels
  -> Gemini full food-only metric depth map
  -> Gemini per-ingredient depth maps
  -> Gemini volume estimate
  -> RAG nutrition results

If the Gemini metric-depth block fails, the local run stops and prints the error.

Examples:
  python run_gemini_metric_depth_local.py /path/to/meal.jpg
  python run_gemini_metric_depth_local.py /path/to/images_dir --limit 3
  python run_gemini_metric_depth_local.py meal.jpg --context questionnaire.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("USE_PRODUCTION_IMAGE_PIPELINE", "true")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for lib in (
        "PIL",
        "urllib3",
        "httpx",
        "httpcore",
        "botocore",
        "boto3",
        "filelock",
        "transformers",
        "sentence_transformers",
        "faiss",
        "torch",
    ):
        logging.getLogger(lib).setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Gemini metric-depth nutrition pipeline locally.")
    parser.add_argument("input_path", help="Image file or directory of images")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images when input_path is a directory")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cpu"), help="cpu or cuda")
    parser.add_argument(
        "--output-dir",
        help="Output root. Defaults to docker/outputs/gemini_metric_depth_<timestamp>",
    )
    parser.add_argument(
        "--context",
        help="Optional questionnaire context JSON object, or mapping of filename/stem to context object",
    )
    parser.add_argument(
        "--depth-image-model",
        default=os.environ.get("GEMINI_DEPTH_IMAGE_MODEL"),
        help="Override GEMINI_DEPTH_IMAGE_MODEL for local testing",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def resolve_images(input_path: Path, limit: int | None) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    images = sorted(
        path for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return images[:limit] if limit else images


def load_json(path: str | None) -> Any:
    if not path:
        return None
    json_path = Path(path).expanduser().resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"Context JSON not found: {json_path}")
    return json.loads(json_path.read_text())


def context_for_image(context_data: Any, image_path: Path) -> dict | None:
    if not context_data:
        return None
    if not isinstance(context_data, dict):
        return None
    if any(key in context_data for key in ("hidden_ingredients", "extras", "recipe_description")):
        return context_data
    by_name = context_data.get(image_path.name)
    if isinstance(by_name, dict):
        return by_name
    by_stem = context_data.get(image_path.stem)
    if isinstance(by_stem, dict):
        return by_stem
    return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def build_pipeline(args: argparse.Namespace, output_root: Path):
    try:
        from app.config import Settings
        from app.models import ModelManager
        from app.pipeline import NutritionVideoPipeline
    except ModuleNotFoundError as exc:
        venv_python = SCRIPT_DIR / "venv311" / "bin" / "python"
        if exc.name == "pydantic_settings" and venv_python.exists():
            raise RuntimeError(
                "Missing local Python dependencies in the current interpreter. "
                f"Run this script with the project venv instead:\n  {venv_python} {Path(__file__).name} <image_path>"
            ) from exc
        raise

    if args.depth_image_model:
        os.environ["GEMINI_DEPTH_IMAGE_MODEL"] = args.depth_image_model

    config = Settings()
    config.DEVICE = args.device
    config.DEBUG = True
    config.USE_PRODUCTION_IMAGE_PIPELINE = True
    config.OUTPUT_DIR = output_root
    config.UPLOAD_DIR = SCRIPT_DIR / "data" / "uploads"
    config.MODEL_CACHE_DIR = SCRIPT_DIR / "models"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required for the Gemini metric-depth local runner")

    model_manager = ModelManager(config)
    return NutritionVideoPipeline(model_manager, config), config


def summarize_result(result: dict) -> dict:
    production_debug = result.get("production_debug") or {}
    nutrition = result.get("nutrition") or {}
    summary = nutrition.get("summary") or {}
    depth_assets = (
        production_debug.get("gemini_depth_assets")
        or production_debug.get("depth_outputs", {}).get("gemini_metric_depth")
        or {}
    )
    volume_estimate = production_debug.get("gemini_metric_depth_volume") or {}
    runtime = result.get("pipeline_runtime") or production_debug.get("runtime") or {}
    return {
        "job_id": result.get("job_id"),
        "status": result.get("status"),
        "pipeline_runtime": runtime,
        "meal_name": result.get("meal_name") or production_debug.get("meal_name"),
        "visible_items": production_debug.get("visible_items") or [],
        "total_volume_ml": volume_estimate.get("total_volume_ml") or summary.get("total_food_volume_ml"),
        "total_calories_kcal": summary.get("total_calories_kcal"),
        "total_mass_g": summary.get("total_mass_g"),
        "num_food_items": summary.get("num_food_items"),
        "full_depth_path": depth_assets.get("full_depth_path"),
        "ingredient_depths": depth_assets.get("ingredients") or [],
        "depth_asset_latency_s": depth_assets.get("latency_s"),
        "volume_map": production_debug.get("gemini_pass_2_volume") or {},
        "gemini_stages": [
            entry.get("stage")
            for entry in production_debug.get("gemini_outputs") or []
        ],
        "nutrition_items": nutrition.get("items") or [],
    }


def print_summary(image_path: Path, elapsed_s: float, summary: dict) -> None:
    print("\n" + "=" * 88)
    print(f"{image_path.name}  job={summary.get('job_id')}  elapsed={elapsed_s:.1f}s")
    print("=" * 88)
    print(f"Pipeline: {json.dumps(summary.get('pipeline_runtime') or {}, default=str)}")
    print(f"Meal: {summary.get('meal_name')}")
    print(f"Total volume: {summary.get('total_volume_ml')} ml")
    print(f"Total mass: {summary.get('total_mass_g')} g")
    print(f"Total calories: {summary.get('total_calories_kcal')} kcal")
    print(f"Full Gemini depth: {summary.get('full_depth_path')}")
    print(
        "Ingredient depth maps: "
        f"{len(summary.get('ingredient_depths') or [])}/{len(summary.get('visible_items') or [])}"
    )
    if summary.get("depth_asset_latency_s") is not None:
        print(f"Depth asset latency: {summary['depth_asset_latency_s']}s")

    print("\nVolumes:")
    volume_map = summary.get("volume_map") or {}
    for name, entry in volume_map.items():
        print(
            f"  {name:<28} {float(entry.get('volume_ml') or 0):>7.1f} ml"
            f"  confidence={float(entry.get('confidence') or 0):.2f}"
        )

    print("\nNutrition:")
    for item in summary.get("nutrition_items") or []:
        print(
            f"  {(item.get('food_name') or '?'):<28}"
            f" {float(item.get('mass_g') or 0):>7.1f} g"
            f" {float(item.get('volume_ml') or 0):>7.1f} ml"
            f" {float(item.get('total_calories') or 0):>7.1f} kcal"
        )

    print("\nGemini stages:")
    for stage in summary.get("gemini_stages") or []:
        print(f"  - {stage}")


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    input_path = Path(args.input_path).expanduser().resolve()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else SCRIPT_DIR / "outputs" / f"gemini_metric_depth_{run_stamp}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    images = resolve_images(input_path, args.limit)
    if not images:
        print(f"No images found under {input_path}")
        return 1

    context_data = load_json(args.context)
    pipeline, _config = build_pipeline(args, output_root)

    print("=" * 88)
    print("Gemini Metric-Depth Local Pipeline")
    print("=" * 88)
    print(f"input: {input_path}")
    print(f"images: {len(images)}")
    print(f"output_root: {output_root}")
    print(f"device: {args.device}")
    print(f"gemini_depth_image_model: {os.environ.get('GEMINI_DEPTH_IMAGE_MODEL') or 'pipeline default'}")

    run_results = []
    failures = []
    for index, image_path in enumerate(images, start=1):
        job_id = f"local-gemini-depth-{run_stamp}-{index:03d}"
        user_context = context_for_image(context_data, image_path)
        print(f"\n[{index}/{len(images)}] Running {image_path}")
        start = time.monotonic()
        try:
            result = pipeline.process_image(image_path, job_id, user_context=user_context)
            elapsed_s = time.monotonic() - start
            summary = summarize_result(result)

            out_dir = output_root / f"production_{job_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            write_json(out_dir / "result.json", result)
            write_json(out_dir / "summary.json", summary)
            print_summary(image_path, elapsed_s, summary)
            run_results.append({
                "image": str(image_path),
                "elapsed_s": round(elapsed_s, 3),
                "output_dir": str(out_dir),
                **summary,
            })
        except Exception as exc:
            elapsed_s = time.monotonic() - start
            failure = {
                "image": str(image_path),
                "job_id": job_id,
                "elapsed_s": round(elapsed_s, 3),
                "error": str(exc),
            }
            failures.append(failure)
            write_json(output_root / f"error_{job_id}.json", failure)
            print(f"FAILED {image_path.name} after {elapsed_s:.1f}s: {exc}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    aggregate = {
        "run_timestamp": run_stamp,
        "input_path": str(input_path),
        "output_root": str(output_root),
        "success_count": len(run_results),
        "failure_count": len(failures),
        "results": run_results,
        "failures": failures,
    }
    write_json(output_root / "run_summary.json", aggregate)

    print("\n" + "=" * 88)
    print("Run complete")
    print("=" * 88)
    print(f"success: {len(run_results)}")
    print(f"failed: {len(failures)}")
    print(f"run_summary: {output_root / 'run_summary.json'}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
