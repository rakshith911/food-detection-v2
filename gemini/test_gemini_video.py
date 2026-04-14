"""
Test script for Gemini video understanding: food/meal analysis from video.

Supports three input methods:
1. Inline video: small files (<20MB) passed directly in the request
2. File API upload: larger files or when reusing the same file
3. YouTube URL: public video URLs

Uses food-specific prompts for meal description, ingredients, and nutrition over time.

Usage:
  # Inline (small video <20MB):
  python test_gemini_video.py path/to/meal.mp4

  # File API upload (larger or reusable):
  python test_gemini_video.py path/to/meal.mp4 --upload

  # YouTube URL:
  python test_gemini_video.py "https://www.youtube.com/watch?v=..."

  # Optional: custom FPS for inline, or clip for YouTube:
  python test_gemini_video.py meal.mp4 --fps 0.5
  python test_gemini_video.py "https://youtube.com/watch?v=..." --start 120 --end 300 -o result.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

try:
    from google import genai
    from google.genai import types
    NEW_GENAI_AVAILABLE = True
except ImportError:
    NEW_GENAI_AVAILABLE = False

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageColor
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

# Size threshold: use File API when file is larger than this (bytes)
INLINE_VIDEO_SIZE_LIMIT = 20 * 1024 * 1024  # 20 MB

# Supported video MIME types (from Gemini docs)
VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mov": "video/quicktime",  # video/mov often mapped to quicktime
    ".avi": "video/avi",
    ".flv": "video/x-flv",
    ".mpg": "video/mpeg",
    ".webm": "video/webm",
    ".wmv": "video/x-ms-wmv",
    ".3gp": "video/3gpp",
}


def get_api_key() -> Optional[str]:
    """Get Gemini API key from environment or file."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        test_file = Path(__file__).parent.parent / "FoodAI" / "nutrition-video-analysis" / "terraform" / "docker" / "TEST_OPTIMIZATIONS.md"
        if test_file.exists():
            try:
                content = test_file.read_text()
                for line in content.split("\n"):
                    if "GEMINI_API_KEY=" in line and "export" in line:
                        api_key = line.split('"')[1] if '"' in line else line.split("'")[1]
                        break
            except Exception:
                pass
    return api_key


def get_mime_type(path: Union[str, Path]) -> str:
    """Return MIME type for a video path. Defaults to video/mp4."""
    ext = Path(path).suffix.lower()
    return VIDEO_MIME_TYPES.get(ext, "video/mp4")


def analyze_video_inline(
    client: "genai.Client",
    video_path: Path,
    prompt: str,
    model: str,
    fps: Optional[float] = None,
) -> str:
    """
    Send video as inline bytes. Use for small files (<20MB).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    size = video_path.stat().st_size
    if size > INLINE_VIDEO_SIZE_LIMIT:
        raise ValueError(
            f"Video is {size / (1024*1024):.1f} MB. "
            f"For files >20MB use --upload (File API)."
        )

    video_bytes = video_path.read_bytes()
    mime = get_mime_type(video_path)

    parts: List[types.Part] = []
    inline_part = types.Part(
        inline_data=types.Blob(data=video_bytes, mime_type=mime)
    )
    # Optional: custom FPS (e.g. video_metadata)
    if fps is not None and hasattr(types, "VideoMetadata"):
        try:
            inline_part = types.Part(
                inline_data=types.Blob(data=video_bytes, mime_type=mime),
                video_metadata=types.VideoMetadata(fps=fps),
            )
        except Exception:
            inline_part = types.Part(
                inline_data=types.Blob(data=video_bytes, mime_type=mime)
            )
    parts.append(inline_part)
    parts.append(types.Part(text=prompt))

    response = client.models.generate_content(
        model=model,
        contents=types.Content(parts=parts),
        config=types.GenerateContentConfig(top_p=1, top_k=1, seed=42),
    )
    return response.text


def analyze_video_upload(
    client: "genai.Client",
    video_path: Path,
    prompt: str,
    model: str,
) -> str:
    """
    Upload video via File API, then generate. Use for larger files or reuse.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print("   Uploading video via File API...")
    myfile = client.files.upload(file=str(video_path))
    print("   Waiting for processing...")
    # Poll until file is processed (optional; SDK may block)
    response = client.models.generate_content(
        model=model,
        contents=[myfile, prompt],
        config=types.GenerateContentConfig(top_p=1, top_k=1, seed=42),
    )
    return response.text


def analyze_video_youtube(
    client: "genai.Client",
    youtube_url: str,
    prompt: str,
    model: str,
    start_offset_sec: Optional[float] = None,
    end_offset_sec: Optional[float] = None,
) -> str:
    """
    Send a YouTube URL to Gemini. Public videos only.
    Optional start/end in seconds (Gemini uses e.g. '1250s', '1570s').
    """
    parts: List[types.Part] = []
    kwargs = {}
    if start_offset_sec is not None or end_offset_sec is not None:
        if hasattr(types, "VideoMetadata"):
            md = {}
            if start_offset_sec is not None:
                md["start_offset"] = f"{int(start_offset_sec)}s"
            if end_offset_sec is not None:
                md["end_offset"] = f"{int(end_offset_sec)}s"
            try:
                kwargs["video_metadata"] = types.VideoMetadata(**md)
            except Exception:
                pass
    part = types.Part(
        file_data=types.FileData(file_uri=youtube_url),
        **kwargs
    )
    parts.append(part)
    parts.append(types.Part(text=prompt))

    response = client.models.generate_content(
        model=model,
        contents=types.Content(parts=parts),
        config=types.GenerateContentConfig(top_p=1, top_k=1, seed=42),
    )
    return response.text


# Same JSON keys as image analysis (test_gemini_analysis.py) for consistent output structure.
VIDEO_JSON_KEYS = (
    "main_food_item",
    "cuisine_type",
    "cooking_method",
    "visible_ingredients",
    "ingredient_breakdown",
    "nutritional_info",
    "allergens",
    "dietary_tags",
    "additional_notes",
)


def get_food_video_prompt(
    summarize: bool = True,
    timestamps: bool = True,
    nutrition: bool = True,
    custom: Optional[str] = None,
) -> str:
    """Build a food-focused video analysis prompt that requests the same JSON structure as image analysis."""
    if custom:
        return custom
    parts = [
        "Analyze this video from a food and nutrition perspective. Describe what is shown: meals, dishes, ingredients, and any cooking or eating actions.",
        "",
        "Please format the response as structured JSON with the following keys (same structure as for a single food image):",
        "- main_food_item",
        "- cuisine_type",
        "- cooking_method",
        "- visible_ingredients: list of objects. For each item include: name, estimated_quantity, and bounding_box. Also include when it appears: timestamp_seconds (number) or time_range (string MM:SS-MM:SS).",
        "  Bounding box: use [x_min, y_min, x_max, y_max] in pixels for a representative frame at that timestamp. Assume frame dimensions 1280x720 if unknown; or state actual frame size in additional_notes.",
        "- ingredient_breakdown (detailed list of ingredients)",
        "- nutritional_info (object with calories, macros, micronutrients, fiber_content)",
        "- allergens (list)",
        "- dietary_tags (list)",
        "- additional_notes",
        "",
        "Example format for visible_ingredients (video; same keys as image + timestamp):",
        '[{"name": "pizza slice", "bounding_box": [320, 180, 960, 540], "estimated_quantity": "1 slice", "timestamp_seconds": 12}, {"name": "olives", "bounding_box": [400, 200, 550, 350], "estimated_quantity": "5-6 olives", "time_range": "00:15-00:22"}]',
        "",
        "Output only valid JSON (you may wrap it in a markdown code block with ```json).",
    ]
    return "\n".join(parts)


def parse_video_response_json(response_text: str, source: str) -> dict:
    """
    Extract and parse JSON from the model response to match image analysis output structure.
    Returns a dict with image_path, image_name, raw_response, parsed, and the same keys as image JSON.
    """
    source_path = Path(source) if not source.strip().lower().startswith("http") else source
    source_name = Path(source).name if not source.strip().lower().startswith("http") else source
    result = {
        "image_path": str(source_path),
        "image_name": source_name,
        "raw_response": response_text,
        "parsed": False,
    }
    try:
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        else:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_str = response_text[json_start:json_end] if json_start >= 0 else None
        if json_str:
            parsed_data = json.loads(json_str)
            result.update(parsed_data)
            result["parsed"] = True
    except json.JSONDecodeError:
        pass
    return result


# Assumed reference resolution for video bounding boxes from Gemini (prompt says 1280x720)
VIDEO_BBOX_REF_WIDTH = 1280
VIDEO_BBOX_REF_HEIGHT = 720


def extract_frame_at_sec(video_path: Path, time_sec: float = 0.0) -> "Image.Image":
    """Extract a single frame from the video at the given time (seconds). Returns PIL Image RGB."""
    if not PLOT_AVAILABLE:
        raise ImportError("Plotting requires: pip install opencv-python-headless Pillow")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    frame_idx = int(time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret or frame_bgr is None:
        # Fallback: first frame
        cap = cv2.VideoCapture(str(video_path))
        ret, frame_bgr = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Cannot read any frame from: {video_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def plot_video_result(
    video_path: Union[str, Path],
    result_json_path_or_dict: Union[str, Path, Dict[str, Any]],
    output_path: Optional[Path] = None,
    frame_time_sec: Optional[float] = None,
) -> Path:
    """
    Plot bounding boxes from video analysis JSON onto a frame from the video.
    Uses the same drawing style as image analysis (rectangles + labels).
    If bounding_box coordinates are for 1280x720, they are scaled to the actual frame size.
    """
    if not PLOT_AVAILABLE:
        raise ImportError("Plotting requires: pip install opencv-python-headless Pillow")
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if isinstance(result_json_path_or_dict, (str, Path)):
        with open(result_json_path_or_dict, "r", encoding="utf-8") as f:
            result = json.load(f)
    else:
        result = result_json_path_or_dict

    visible_ingredients = result.get("visible_ingredients") or []
    with_boxes = [x for x in visible_ingredients if x.get("bounding_box") and len(x.get("bounding_box", [])) == 4]
    if not with_boxes:
        print("⚠ No bounding_box in visible_ingredients. Re-run analysis with the updated prompt to get boxes.")
        raise ValueError("No bounding boxes in result")

    # Pick frame time: first timestamp_seconds, or time_range start, or 0
    if frame_time_sec is None:
        for ing in with_boxes:
            if "timestamp_seconds" in ing:
                frame_time_sec = float(ing["timestamp_seconds"])
                break
            if "time_range" in ing:
                part = str(ing["time_range"]).split("-")[0].strip()
                if ":" in part:
                    mm, ss = part.split(":")[:2]
                    frame_time_sec = int(mm) * 60 + float(ss)
                else:
                    frame_time_sec = 0.0
                break
        if frame_time_sec is None:
            frame_time_sec = 0.0

    image = extract_frame_at_sec(video_path, frame_time_sec)
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        try:
            font = ImageFont.load_default()
            font_small = font
        except Exception:
            font = None
            font_small = None

    colors_list = list(ImageColor.colormap.keys())
    for i, ingredient in enumerate(with_boxes):
        name = ingredient.get("name", f"Item {i+1}")
        bbox = ingredient.get("bounding_box", [])
        if len(bbox) != 4:
            continue
        x_min, y_min, x_max, y_max = bbox
        # Scale from reference 1280x720 to actual frame size
        x_min = int(x_min * img_width / VIDEO_BBOX_REF_WIDTH)
        y_min = int(y_min * img_height / VIDEO_BBOX_REF_HEIGHT)
        x_max = int(x_max * img_width / VIDEO_BBOX_REF_WIDTH)
        y_max = int(y_max * img_height / VIDEO_BBOX_REF_HEIGHT)
        x_min = max(0, min(x_min, img_width))
        y_min = max(0, min(y_min, img_height))
        x_max = max(0, min(x_max, img_width))
        y_max = max(0, min(y_max, img_height))

        color_name = colors_list[i % len(colors_list)]
        rgb = ImageColor.getrgb(color_name)
        draw.rectangle([x_min, y_min, x_max, y_max], outline=rgb, width=4)
        label_text = name
        if font_small:
            bbox_text = draw.textbbox((0, 0), label_text, font=font_small)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
        else:
            text_width = len(label_text) * 6
            text_height = 12
        label_bg = [x_min, y_min - text_height - 4, x_min + text_width + 8, y_min]
        draw.rectangle(label_bg, fill=(255, 255, 255))
        draw.text((x_min + 4, y_min - text_height - 2), label_text, fill=(0, 0, 0), font=font_small)
        print(f"   [{i+1}] {name}: [{x_min}, {y_min}, {x_max}, {y_max}]")

    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_annotated.jpg"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"\n💾 Annotated frame saved to: {output_path}")
    return output_path


def dry_run(source: str, use_upload: bool, output_json: Optional[Path]) -> None:
    """Validate source and print what would be sent (no API call)."""
    is_youtube = source.strip().lower().startswith("http")
    if is_youtube:
        print(f"Source: YouTube URL\n  {source}")
        print("Mode: YouTube (file_uri)")
    else:
        path = Path(source)
        if not path.exists():
            print(f"Error: file not found: {path}")
            return
        size_mb = path.stat().st_size / (1024 * 1024)
        mime = get_mime_type(path)
        print(f"Source: local file\n  {path}\n  Size: {size_mb:.2f} MB  MIME: {mime}")
        if use_upload:
            print("Mode: File API upload")
        else:
            if path.stat().st_size > INLINE_VIDEO_SIZE_LIMIT:
                print("Mode: INLINE would FAIL (file >20MB). Use --upload")
            else:
                print("Mode: inline")
    if output_json:
        print(f"Output: {output_json}")
    print("Dry run OK. Run without --dry-run to call the API.")


def run(
    source: str,
    *,
    prompt: Optional[str] = None,
    use_upload: bool = False,
    model: str = "gemini-2.0-flash-exp",
    fps: Optional[float] = None,
    start_offset_sec: Optional[float] = None,
    end_offset_sec: Optional[float] = None,
    summarize: bool = True,
    timestamps: bool = True,
    nutrition: bool = True,
    output_json: Optional[Path] = None,
) -> str:
    """
    Run video analysis.

    source: path to a local video file, or a YouTube URL (must start with http).
    """
    if not NEW_GENAI_AVAILABLE:
        raise ImportError(
            "google-genai is required. Install with: pip install google-genai"
        )

    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not set. Set the env var or add it to TEST_OPTIMIZATIONS.md."
        )

    client = genai.Client(api_key=api_key)
    analysis_prompt = get_food_video_prompt(
        summarize=summarize,
        timestamps=timestamps,
        nutrition=nutrition,
        custom=prompt,
    )

    models_to_try = [
        model,
        "gemini-2.5-flash",
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]

    is_youtube = source.strip().lower().startswith("http")
    result_text = None
    used_model = None

    for m in models_to_try:
        try:
            if is_youtube:
                print(f"   Using model: {m} (YouTube)")
                result_text = analyze_video_youtube(
                    client, source, analysis_prompt, m,
                    start_offset_sec=start_offset_sec,
                    end_offset_sec=end_offset_sec,
                )
            elif use_upload:
                print(f"   Using model: {m} (File API upload)")
                result_text = analyze_video_upload(
                    client, Path(source), analysis_prompt, m
                )
            else:
                print(f"   Using model: {m} (inline)")
                result_text = analyze_video_inline(
                    client, Path(source), analysis_prompt, m, fps=fps
                )
            used_model = m
            break
        except Exception as e:
            print(f"   Model {m} failed: {e}")
            continue

    if result_text is None:
        raise RuntimeError("All models failed for this video.")

    # Parse to same output structure as image analysis
    result = parse_video_response_json(result_text, source)
    result["model"] = used_model

    print(f"\n✓ Result (model: {used_model}):\n")
    if result.get("parsed"):
        # Brief structured summary (same keys as image case)
        if result.get("main_food_item"):
            print(f"  main_food_item: {result['main_food_item']}")
        if result.get("cuisine_type"):
            print(f"  cuisine_type: {result['cuisine_type']}")
        if result.get("visible_ingredients"):
            print(f"  visible_ingredients: {len(result['visible_ingredients'])} items")
        if result.get("additional_notes"):
            notes = result["additional_notes"]
            print(f"  additional_notes: {notes[:300]}{'...' if len(notes) > 300 else ''}")
        print("  (full structured JSON in saved file)")
    else:
        print(result_text)

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        # Save same structure as image analysis JSON (image_path, image_name, raw_response, parsed, + parsed keys)
        output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved JSON to {output_json} (same structure as image analysis)")

    return result_text


def run_with_plot(
    source: str,
    output_json: Path,
    plot_output: Optional[Path] = None,
) -> None:
    """Run analysis then plot bounding boxes on a frame (only for local video)."""
    if source.strip().lower().startswith("http"):
        print("⚠ --plot only works for local video files, not YouTube URLs.")
        return
    # Run analysis (will save to output_json)
    run(source, output_json=output_json)
    # Load result and plot
    with open(output_json, "r", encoding="utf-8") as f:
        result = json.load(f)
    if not result.get("visible_ingredients"):
        print("⚠ No visible_ingredients to plot.")
        return
    if not any(ing.get("bounding_box") for ing in result["visible_ingredients"]):
        print("⚠ No bounding_box in visible_ingredients; cannot plot.")
        return
    if not PLOT_AVAILABLE:
        print("⚠ Plotting requires: pip install opencv-python-headless Pillow")
        return
    plot_video_result(Path(source), result, output_path=plot_output)


def main():
    # Standalone plot: python test_gemini_video.py plot <video> <json> [-o annotated.jpg]
    if len(sys.argv) >= 2 and sys.argv[1] == "plot":
        plot_parser = argparse.ArgumentParser(description="Plot video analysis bounding boxes on a frame")
        plot_parser.add_argument("video", help="Path to video file")
        plot_parser.add_argument("json", help="Path to analysis JSON (e.g. from -o)")
        plot_parser.add_argument("-o", "--output", type=Path, default=None, help="Output image path (default: <video_stem>_annotated.jpg)")
        plot_args = plot_parser.parse_args(sys.argv[2:])
        plot_video_result(plot_args.video, plot_args.json, plot_args.output)
        return

    parser = argparse.ArgumentParser(
        description="Analyze food/meal content in videos using Gemini. Use 'plot <video> <json>' to draw boxes on a frame.",
    )
    parser.add_argument(
        "source",
        help="Path to a local video file, or a YouTube URL (e.g. https://www.youtube.com/watch?v=...)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Use File API upload (for files >20MB or to reuse the file)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt (default: food/nutrition-focused prompt)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash-exp",
        help="Model name (e.g. gemini-2.5-flash, gemini-1.5-flash)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Custom frame rate for inline video (e.g. 0.5 for long videos)",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Start offset in seconds (YouTube only)",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="End offset in seconds (YouTube only)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not ask for a short summary",
    )
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Do not ask for timestamps",
    )
    parser.add_argument(
        "--no-nutrition",
        action="store_true",
        help="Do not ask for nutrition notes",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Write response and metadata to this JSON file",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="After saving JSON, plot bounding boxes on a frame (requires -o and local video)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate file/URL and print config only (no API call)",
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.source, args.upload, args.output)
        return

    run(
        args.source,
        prompt=args.prompt,
        use_upload=args.upload,
        model=args.model,
        fps=args.fps,
        start_offset_sec=args.start,
        end_offset_sec=args.end,
        summarize=not args.no_summary,
        timestamps=not args.no_timestamps,
        nutrition=not args.no_nutrition,
        output_json=args.output,
    )

    if args.plot and args.output and not args.source.strip().lower().startswith("http"):
        if not PLOT_AVAILABLE:
            print("⚠ Plotting requires: pip install opencv-python-headless Pillow")
        else:
            with open(args.output, "r", encoding="utf-8") as f:
                result = json.load(f)
            if result.get("visible_ingredients") and any(ing.get("bounding_box") for ing in result["visible_ingredients"]):
                plot_video_result(Path(args.source), result, output_path=Path(args.source).parent / f"{Path(args.source).stem}_annotated.jpg")
            else:
                print("⚠ No bounding_box in visible_ingredients; skip plot.")


if __name__ == "__main__":
    main()
