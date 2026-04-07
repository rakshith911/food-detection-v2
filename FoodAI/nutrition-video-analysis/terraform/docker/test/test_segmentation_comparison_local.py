#!/usr/bin/env python3
"""
Run a side-by-side local comparison of the production image pipeline using:
1. Standard SAM3 segmentation
2. Gemini segmentation override

Outputs are written under docker/test/compare/<timestamp>/<image_stem>/.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
DOCKER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DOCKER_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SAM3 vs Gemini segmentation on the local production image pipeline.")
    parser.add_argument("image_path", help="Path to an image file")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cpu"), help="cpu or cuda")
    parser.add_argument(
        "--gemini-model",
        default=os.environ.get("GEMINI_SEGMENTATION_MODEL", "gemini-2.5-flash"),
        help="Gemini model to use for segmentation override",
    )
    parser.add_argument("--context", help="Optional questionnaire context JSON path")
    parser.add_argument("--output-dir", help="Optional output root")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def run_command(cmd: list[str], env: dict[str, str], cwd: Path) -> int:
    print(f"\n$ {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(cwd), env=env)
    return completed.returncode


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _font(size: int):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Menlo.ttc", size)
    except Exception:
        return ImageFont.load_default()


def _fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    if not path.exists():
        return canvas
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _draw_text_block(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, *, font, fill="black", line_gap: int = 6) -> int:
    x, y = xy
    for line in text.splitlines():
        draw.text((x, y), line, fill=fill, font=font)
        y += font.size + line_gap
    return y


def _make_side_by_side_panel(
    title: str,
    left_label: str,
    right_label: str,
    left_path: Path,
    right_path: Path,
    out_path: Path,
    *,
    image_size: tuple[int, int] = (720, 720),
) -> None:
    header_h = 90
    gap = 24
    width = image_size[0] * 2 + gap * 3
    height = header_h + image_size[1] + gap * 2
    canvas = Image.new("RGB", (width, height), "#f5f1e8")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(28)
    label_font = _font(20)

    draw.text((gap, 18), title, fill="#111111", font=title_font)
    draw.text((gap, 54), left_label, fill="#444444", font=label_font)
    draw.text((gap * 2 + image_size[0], 54), right_label, fill="#444444", font=label_font)

    left_img = _fit_image(left_path, image_size)
    right_img = _fit_image(right_path, image_size)
    canvas.paste(left_img, (gap, header_h))
    canvas.paste(right_img, (gap * 2 + image_size[0], header_h))
    draw.rectangle((gap - 1, header_h - 1, gap + image_size[0], header_h + image_size[1]), outline="#d0c7b8", width=2)
    draw.rectangle((gap * 2 + image_size[0] - 1, header_h - 1, gap * 2 + image_size[0] + image_size[0], header_h + image_size[1]), outline="#d0c7b8", width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _nutrition_row_map(result: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    rows = {}
    if not result:
        return rows
    items = ((result.get("nutrition") or {}).get("items") or [])
    for item in items:
        name = (item.get("food_name") or "").strip()
        if name:
            rows[name] = item
    return rows


def _format_item_line(name: str, item: dict[str, Any] | None) -> str:
    if not item:
        return f"{name}: missing"
    density = item.get("density_g_per_ml")
    kcal100 = item.get("calories_per_100g")
    total = item.get("total_calories")
    dmatch = item.get("density_matched")
    cmatch = item.get("calorie_matched")
    dsrc = item.get("density_source")
    csrc = item.get("calorie_source")
    return (
        f"{name}\n"
        f"  vol={item.get('volume_ml')}ml  mass={item.get('mass_g')}g  kcal={total}\n"
        f"  density={density} [{dmatch}] ({dsrc})\n"
        f"  kcal/100g={kcal100} [{cmatch}] ({csrc})"
    )


def _make_nutrition_comparison_image(
    image_name: str,
    sam3_result: dict[str, Any] | None,
    gemini_result: dict[str, Any] | None,
    out_path: Path,
) -> None:
    sam3_rows = _nutrition_row_map(sam3_result)
    gemini_rows = _nutrition_row_map(gemini_result)
    all_names = sorted(set(sam3_rows) | set(gemini_rows))

    width = 2200
    header_h = 120
    row_h = 110
    height = header_h + max(1, len(all_names)) * row_h + 80
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(30)
    head_font = _font(22)
    body_font = _font(16)

    draw.text((30, 20), f"Nutrition Comparison: {image_name}", fill="#111111", font=title_font)
    draw.text((30, 70), "Ingredient", fill="#222222", font=head_font)
    draw.text((360, 70), "SAM3", fill="#222222", font=head_font)
    draw.text((1140, 70), "Gemini", fill="#222222", font=head_font)
    draw.line((20, header_h - 10, width - 20, header_h - 10), fill="#bbbbbb", width=2)

    y = header_h
    for idx, name in enumerate(all_names or ["(no items)"]):
        fill = "#faf7f2" if idx % 2 == 0 else "#ffffff"
        draw.rectangle((20, y - 8, width - 20, y + row_h - 12), fill=fill)
        draw.text((30, y), name, fill="#111111", font=head_font)
        _draw_text_block(draw, (360, y), _format_item_line(name, sam3_rows.get(name)), font=body_font, fill="#333333", line_gap=4)
        _draw_text_block(draw, (1140, y), _format_item_line(name, gemini_rows.get(name)), font=body_font, fill="#333333", line_gap=4)
        y += row_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def generate_visual_comparison(
    image_path: Path,
    output_root: Path,
    sam3_output: Path,
    gemini_output: Path,
) -> dict[str, str]:
    sam3_dir = sam3_output / image_path.stem
    gemini_dir = gemini_output / image_path.stem
    comparison_dir = output_root / "panels"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    panel_specs = [
        ("rgb_comparison.png", "RGB Input", sam3_dir / "rgb.png", gemini_dir / "rgb.png"),
        ("segmentation_comparison.png", "Segmentation Overlay", sam3_dir / "sam3_segmentation.png", gemini_dir / "gemini_segmentation.png"),
        ("depth_colored_comparison.png", "ZoeDepth Colored", sam3_dir / "zoedepth_colored.png", gemini_dir / "zoedepth_colored.png"),
        ("masked_depth_comparison.png", "Masked Depth", sam3_dir / "dish_masked_depth.png", gemini_dir / "dish_masked_depth.png"),
    ]
    for filename, title, left, right in panel_specs:
        out_path = comparison_dir / filename
        _make_side_by_side_panel(title, "SAM3", "Gemini", left, right, out_path)
        outputs[filename] = str(out_path)

    sam3_result = load_json_if_exists(sam3_dir / "result.json")
    gemini_result = load_json_if_exists(gemini_dir / "result.json")
    nutrition_path = comparison_dir / "nutrition_comparison.png"
    _make_nutrition_comparison_image(image_path.name, sam3_result, gemini_result, nutrition_path)
    outputs["nutrition_comparison.png"] = str(nutrition_path)
    return outputs


def main() -> int:
    args = parse_args()
    image_path = Path(args.image_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else DOCKER_DIR / "test" / "compare" / run_stamp / image_path.stem
    )
    sam3_output = output_root / "sam3"
    gemini_output = output_root / "gemini"
    output_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("AWS_REGION", "us-east-1")
    env.setdefault("AWS_DEFAULT_REGION", "us-east-1")

    print("=" * 80)
    print("Local Segmentation Comparison")
    print("=" * 80)
    print(f"image: {image_path}")
    print(f"device: {args.device}")
    print(f"output_root: {output_root}")
    print(f"gemini_model: {args.gemini_model}")

    base_args = [str(image_path), "--device", args.device]
    if args.context:
        base_args.extend(["--context", str(Path(args.context).expanduser().resolve())])

    sam3_cmd = [
        sys.executable,
        str(DOCKER_DIR / "test_production_image_pipeline_local.py"),
        *base_args,
        "--output-dir",
        str(sam3_output),
    ]
    gemini_cmd = [
        sys.executable,
        str(DOCKER_DIR / "test" / "test_gemini_segmentation_pipeline_local.py"),
        *base_args,
        "--model",
        args.gemini_model,
        "--output-dir",
        str(gemini_output),
    ]

    sam3_exit = run_command(sam3_cmd, env=env, cwd=DOCKER_DIR)
    gemini_exit = run_command(gemini_cmd, env=env, cwd=DOCKER_DIR)

    sam3_summary = load_json_if_exists(sam3_output / image_path.stem / "summary.json")
    gemini_summary = load_json_if_exists(gemini_output / image_path.stem / "summary.json")
    gemini_backend = load_json_if_exists(gemini_output / image_path.stem / "segmentation_backend.json")

    comparison = {
        "image": image_path.name,
        "run_timestamp": run_stamp,
        "sam3_exit_code": sam3_exit,
        "gemini_exit_code": gemini_exit,
        "sam3_summary": sam3_summary,
        "gemini_summary": gemini_summary,
        "gemini_backend": gemini_backend,
        "paths": {
            "sam3": str(sam3_output),
            "gemini": str(gemini_output),
        },
    }
    if sam3_exit == 0 and gemini_exit == 0:
        comparison["visual_panels"] = generate_visual_comparison(
            image_path,
            output_root,
            sam3_output,
            gemini_output,
        )
    write_json(output_root / "comparison_summary.json", comparison)

    print("\n" + "=" * 80)
    print("Comparison Complete")
    print("=" * 80)
    print(f"sam3_exit: {sam3_exit}")
    print(f"gemini_exit: {gemini_exit}")
    print(f"summary_json: {output_root / 'comparison_summary.json'}")
    if comparison.get("visual_panels"):
        print(f"panels_dir: {output_root / 'panels'}")

    return 0 if sam3_exit == 0 and gemini_exit == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
