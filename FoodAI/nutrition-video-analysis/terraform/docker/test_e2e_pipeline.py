#!/usr/bin/env python3
"""
End-to-end local pipeline test.
Runs the full production path: Gemini detection → SAM3 → ZoeDepth → RAG → calorie totals.
No mocks — uses real models and real Gemini API.

Saved outputs (inside test_e2e_outputs/<job_id>/):
  rgb.png                      - original image
  sam3_segmentation.png        - SAM3 overlay with all ingredient masks
  <ingredient>_mask.png        - individual binary mask per ingredient
  zoedepth_colored.png         - colourised depth map
  zoe_depth_raw_visual.png     - raw ZoeDepth render
  zoe_depth_calibrated_visual.png - calibrated depth render
  dish_masked_depth.png        - depth masked to dish region
  result.json                  - full pipeline output (nutrition, sources, RAG matches)

Usage:
    python3 test_e2e_pipeline.py /path/to/image.jpg
    python3 test_e2e_pipeline.py /path/to/image.jpg '{"hidden_ingredients":[{"name":"butter","quantity":"10g"}]}'
"""

import os
import sys
import json
import logging
import importlib.machinery
import uuid
from pathlib import Path
from datetime import datetime

# ── env must be set before any app import ────────────────────────────────────
os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("USE_PRODUCTION_IMAGE_PIPELINE", "true")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not set. Export it before running.")
    sys.exit(1)

IMAGE_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if not IMAGE_PATH or not IMAGE_PATH.exists():
    print(f"Usage: python3 {sys.argv[0]} /path/to/image.jpg [user_context_json]")
    sys.exit(1)

USER_CONTEXT = {}
if len(sys.argv) > 2:
    try:
        USER_CONTEXT = json.loads(sys.argv[2])
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid user_context JSON: {e}")
        sys.exit(1)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("e2e_test")

# ── real torch first (safetensors checks torch.__version__ at import time) ────
import torch  # noqa: E402

# ── mock only cv2 and boto3 (not used in production image pipeline path) ──────
# cv2 is only used in the legacy fallback; we stay on the production path.
from unittest.mock import MagicMock
for _mod in ["cv2", "boto3"]:
    if _mod not in sys.modules:
        _m = MagicMock()
        object.__setattr__(_m, "__spec__", importlib.machinery.ModuleSpec(_mod, loader=None))
        sys.modules[_mod] = _m

sys.path.insert(0, str(Path(__file__).parent))

# ZoeDepth is not a pip package — add the cloned repo to sys.path
_ZOEDEPTH_REPO = Path("/Users/rakshith911/Documents/food_detection/PRODUCTION/model_assets/zoedepth_repo")
if _ZOEDEPTH_REPO.exists() and str(_ZOEDEPTH_REPO) not in sys.path:
    sys.path.insert(0, str(_ZOEDEPTH_REPO))
os.environ.setdefault("ZOEDEPTH_REPO_DIR", str(_ZOEDEPTH_REPO))

from app.config import Settings
from app.models import ModelManager
from app.pipeline import NutritionVideoPipeline
from PIL import Image
import numpy as np

# ── config ────────────────────────────────────────────────────────────────────
config = Settings()
config.GEMINI_API_KEY = GEMINI_API_KEY
config.DEVICE = "cpu"
config.USE_PRODUCTION_IMAGE_PIPELINE = True
config.USE_GEMINI_DETECTION = True

# Point OUTPUT_DIR to a local test_e2e_outputs/ folder next to this script
OUTPUT_BASE = Path(__file__).parent / "test_e2e_outputs"
OUTPUT_BASE.mkdir(exist_ok=True)
config.OUTPUT_DIR = OUTPUT_BASE

job_id = f"e2e-{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:6]}"

print("\n" + "=" * 64)
print("  END-TO-END PIPELINE TEST")
print("=" * 64)
print(f"  Image      : {IMAGE_PATH}")
print(f"  Job ID     : {job_id}")
print(f"  Device     : {config.DEVICE}")
print(f"  SAM3       : {config.SAM3_MODEL_DIR}")
print(f"  ZoeDepth   : {config.ZOEDEPTH_CHECKPOINT}")
print(f"  MiDaS      : {config.MIDAS_REPO_DIR}")
print(f"  RAG index  : {config.UNIFIED_FAISS_PATH}")
print(f"  Outputs    : {OUTPUT_BASE / ('production_' + job_id)}")
print(f"  User ctx   : {json.dumps(USER_CONTEXT) if USER_CONTEXT else 'none'}")
print("=" * 64 + "\n")

# ── verify assets ─────────────────────────────────────────────────────────────
missing = []
for label, path in [
    ("SAM3 model",    config.SAM3_MODEL_DIR / "model.safetensors"),
    ("ZoeDepth ckpt", config.ZOEDEPTH_CHECKPOINT),
    ("RAG index",     config.UNIFIED_FAISS_PATH),
    ("RAG foods",     config.UNIFIED_FOODS_PATH),
]:
    if not Path(path).exists():
        missing.append(f"  ✗ {label}: {path}")
if missing:
    print("Missing required assets:\n" + "\n".join(missing))
    sys.exit(1)

# ── load models ───────────────────────────────────────────────────────────────
print("⏳ Loading models (one-time cost)...\n")
t0 = datetime.utcnow()
model_manager = ModelManager(config)

print("  [1/3] Loading SAM3...")
_ = model_manager.sam3
print("  [2/3] Loading ZoeDepth...")
_ = model_manager.zoedepth
print("  [3/3] Loading RAG (FAISS + sentence-transformer + CLIP + cross-encoder)...")
_ = model_manager.rag

elapsed = (datetime.utcnow() - t0).total_seconds()
print(f"\n✅ Models loaded in {elapsed:.1f}s\n")

# ── patch cv2.imread / cv2.cvtColor so the production pipeline can load the image ──
# The production pipeline uses cv2 only at the very top of _run_production_image_pipeline.
# We patch those two calls with PIL equivalents so we don't need the real cv2 library.
import cv2 as _cv2_mock  # this is our MagicMock

def _pil_imread(path, *args, **kwargs):
    img = Image.open(path).convert("RGB")
    return np.array(img)[:, :, ::-1]  # RGB → BGR to match cv2 convention

def _pil_cvtColor(img, code, *args, **kwargs):
    # We only need BGR→RGB (code=4 in cv2 is COLOR_BGR2RGB)
    return img[:, :, ::-1]

def _pil_resize(img, size, **kw):
    # size is (width, height) in cv2 convention
    if img.ndim == 2:
        return np.array(Image.fromarray(img).resize(size, Image.BILINEAR))
    else:
        return np.array(Image.fromarray(img[:, :, ::-1]).resize(size, Image.BILINEAR))[:, :, ::-1]

def _pil_cvtColor_full(img, code, *args, **kwargs):
    # Channel flip works for both BGR→RGB and RGB→BGR
    if img.ndim == 3 and img.shape[2] == 3:
        return img[:, :, ::-1].copy()
    return img

_cv2_mock.imread = _pil_imread
_cv2_mock.cvtColor = _pil_cvtColor_full
_cv2_mock.resize = _pil_resize

# Constants
_cv2_mock.COLOR_BGR2RGB = 4
_cv2_mock.COLOR_RGB2BGR = 4
_cv2_mock.COLOR_BGR2GRAY = 6
_cv2_mock.INTER_LINEAR = 1
_cv2_mock.RETR_EXTERNAL = 0
_cv2_mock.CHAIN_APPROX_SIMPLE = 1
_cv2_mock.LINE_AA = 16
_cv2_mock.FONT_HERSHEY_SIMPLEX = 0

_cv2_mock.findContours = lambda *a, **kw: ([], None)
_cv2_mock.drawContours = lambda *a, **kw: None

# Scale font up so labels are readable on high-res (4032×3024) iPhone photos.
# cv2 font_scale=0.5 → ~15px on 640px image → 96px equivalent on 4032px image.
# We apply a 6× boost so text is legible at screen zoom.
_FONT_SCALE_BOOST = 6

def _cv2_getTextSize(text, fontFace, fontScale, thickness, **kw):
    fs = max(14, int(14 * fontScale * _FONT_SCALE_BOOST))
    char_w = int(fs * 0.62)
    return ((len(text) * char_w, fs), int(fs * 0.25))

def _cv2_rectangle(img, pt1, pt2, color, thickness=-1, **kw):
    """In-place filled or outlined rectangle."""
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    h, w = img.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return img
    c = tuple(int(v) for v in color)
    if thickness < 0:
        img[y1:y2, x1:x2] = c
    else:
        t = max(1, thickness)
        img[y1:min(y1+t, h), x1:x2] = c
        img[max(0, y2-t):y2, x1:x2] = c
        img[y1:y2, x1:min(x1+t, w)] = c
        img[y1:y2, max(0, x2-t):x2] = c
    return img

def _cv2_putText(img, text, org, fontFace, fontScale, color, thickness=1, lineType=None, **kw):
    """In-place text rendering via PIL. img is a BGR numpy array."""
    from PIL import ImageDraw, ImageFont
    x, y = int(org[0]), int(org[1])
    fs = max(14, int(14 * fontScale * _FONT_SCALE_BOOST))
    # img is BGR — convert to RGB for PIL, draw, convert back
    pil_img = Image.fromarray(img[:, :, ::-1])
    draw = ImageDraw.Draw(pil_img)
    rgb_color = (int(color[2]), int(color[1]), int(color[0]))  # BGR → RGB
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", fs)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", fs)
        except Exception:
            font = ImageFont.load_default()
    # org in cv2 is the text baseline (bottom-left), so draw starts at y - fs
    draw.text((x, y - fs), text, fill=rgb_color, font=font)
    img[:] = np.array(pil_img)[:, :, ::-1]
    return img

_cv2_mock.getTextSize = _cv2_getTextSize
_cv2_mock.rectangle = _cv2_rectangle
_cv2_mock.putText = _cv2_putText

# ── run pipeline ──────────────────────────────────────────────────────────────
pipeline = NutritionVideoPipeline(model_manager, config)

print(f"🚀 Running pipeline  job_id={job_id}\n")
t1 = datetime.utcnow()

result = pipeline.process_image(IMAGE_PATH, job_id, user_context=USER_CONTEXT or None)

elapsed2 = (datetime.utcnow() - t1).total_seconds()
print(f"\n✅ Pipeline finished in {elapsed2:.1f}s\n")

# ── save result JSON ──────────────────────────────────────────────────────────
out_dir = OUTPUT_BASE / f"production_{job_id}"
out_dir.mkdir(parents=True, exist_ok=True)
result_path = out_dir / "result.json"
with open(result_path, "w") as f:
    json.dump(result, f, indent=2, default=str)

# ── summary ───────────────────────────────────────────────────────────────────
nutrition = result.get("nutrition") or {}
summary   = nutrition.get("summary") or {}
items     = nutrition.get("items") or []

print("=" * 64)
print("  RESULT SUMMARY")
print("=" * 64)
total_kcal = (
    summary.get("total_calories_kcal")
    or summary.get("base_dish_total_kcal")
    or summary.get("total_kcal")
    or "?"
)
total_volume_ml = sum(item.get("volume_ml") or 0.0 for item in items)
total_mass_g    = sum(item.get("mass_g") or 0.0 for item in items)
print(f"  Total kcal    : {total_kcal}")
print(f"  Total volume  : {total_volume_ml:.1f} mL")
print(f"  Total mass    : {total_mass_g:.1f} g")
print(f"  Items         : {len(items)}\n")
print(f"  {'Label':<28} {'kcal':>6}  {'vol(mL)':>8}  {'mass(g)':>8}  {'density':>8}  density_src / calorie_src / rag_match")
print(f"  {'-'*28} {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*40}")
for item in items:
    label    = (item.get("food_name") or item.get("label") or "?")[:27]
    kcal     = item.get("total_calories", "?")
    vol      = item.get("volume_ml", "?")
    mass     = item.get("mass_g", "?")
    density  = item.get("density_g_per_ml", "?")
    dsrc     = item.get("density_source") or "?"
    csrc     = item.get("calorie_source") or "?"
    dmatch   = item.get("density_matched") or ""
    cmatch   = item.get("calorie_matched") or ""
    vol_str  = f"{vol:.1f}" if isinstance(vol, float) else str(vol)
    mass_str = f"{mass:.1f}" if isinstance(mass, float) else str(mass)
    print(f"  {label:<28} {str(kcal):>6}  {vol_str:>8}  {mass_str:>8}  {str(density):>8}  d={dsrc} ({dmatch})")
    print(f"  {'':<28} {'':>6}  {'':>8}  {'':>8}  {'':>8}  c={csrc} ({cmatch})")

print()
print(f"  Output dir : {out_dir}")
print(f"  Images     : rgb.png, sam3_segmentation.png, zoedepth_colored.png,")
print(f"               zoe_depth_raw_visual.png, zoe_depth_calibrated_visual.png,")
print(f"               dish_masked_depth.png, <ingredient>_mask.png")
print(f"  JSON       : result.json")
print("=" * 64 + "\n")
