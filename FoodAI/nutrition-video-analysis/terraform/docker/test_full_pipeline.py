"""
End-to-end pipeline test on a local image.
- Real Gemini API (detection + volume estimation)
- Real Depth Anything V2 (depth map)
- Real RAG (FAO density + USDA calories)
- SAM2 is mocked with a simple bounding-box mask (no checkpoint needed)
Usage:
    GEMINI_API_KEY=your-key python3 test_full_pipeline.py /path/to/image.jpg
"""
import os, sys, json, numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).parent))

# Mock modules not installed locally (imported at module level in pipeline.py)
# Note: timm must NOT be mocked — transformers uses importlib.util.find_spec on it
for mod in ['boto3', 'botocore', 'hydra', 'hydra.core', 'hydra.core.global_hydra',
            'iopath', 'iopath.common', 'iopath.common.file_io',
            'sam2', 'sam2.build_sam', 'sam2.utils', 'sam2.utils.misc',
            'einops']:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# google.generativeai — use real package for Gemini API calls
import importlib
try:
    import google.generativeai as _real_genai
    sys.modules['google.generativeai'] = _real_genai
except ImportError:
    sys.modules['google.generativeai'] = MagicMock()

# google.genai — mock (not used in current pipeline calls)
sys.modules['google.genai'] = MagicMock()

IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else \
    '/Users/rakshithmahishi/Documents/food-detection/dish_1556575327_masked_depth.png'

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
if not GEMINI_API_KEY:
    # Try reading from TEST_OPTIMIZATIONS.md
    for candidate in [Path(__file__).parent, Path(__file__).parent.parent]:
        f = candidate / 'TEST_OPTIMIZATIONS.md'
        if f.exists():
            for line in f.read_text().splitlines():
                if 'GEMINI_API_KEY=' in line and '"' in line:
                    GEMINI_API_KEY = line.split('"')[1].strip()
                    break

if not GEMINI_API_KEY:
    print("ERROR: Set GEMINI_API_KEY env var")
    sys.exit(1)

if not Path(IMAGE_PATH).exists():
    print(f"ERROR: Image not found: {IMAGE_PATH}")
    sys.exit(1)

print(f"Testing on: {IMAGE_PATH}")
os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY

# ── Mock SAM2 to return one big mask per detected box ──────────────────────────
class FakeSAM2:
    """Returns full-frame masks for each detected object (correct frame size)."""
    def __init__(self):
        self._obj_ids = []
        self._frame_h = 100
        self._frame_w = 100

    def init_state(self, video_path=None, **kwargs):
        # Read actual frame size so masks match depth map dimensions
        if video_path is not None:
            import glob as _glob
            frames = sorted(_glob.glob(str(video_path) + '/*.jpg') + _glob.glob(str(video_path) + '/*.png'))
            if frames:
                img = Image.open(frames[0])
                self._frame_w, self._frame_h = img.size
        return {}

    def add_new_points_or_box(self, inference_state, box, frame_idx, obj_id, **kw):
        if obj_id not in self._obj_ids:
            self._obj_ids.append(obj_id)

    def infer_single_frame(self, inference_state, frame_idx):
        obj_ids = self._obj_ids if self._obj_ids else [1]
        # Return torch tensors (shape: N_obj, 1, H, W) so pipeline can call .cpu().numpy()
        logits = torch.ones(len(obj_ids), 1, self._frame_h, self._frame_w, dtype=torch.float32)
        return frame_idx, list(obj_ids), logits

    def reset_state(self, inference_state):
        self._obj_ids = []

# ── Load config ────────────────────────────────────────────────────────────────
from app.config import Settings
config = Settings()
config.GEMINI_API_KEY = GEMINI_API_KEY
config.USE_GEMINI_DETECTION = True
config.USE_GEMINI_VIDEO_DETECTION = False
config.DEVICE = 'cpu'
config.OUTPUT_DIR = Path('/tmp/nutrition_test_output')
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load real RAG ──────────────────────────────────────────────────────────────
print("\n[1/4] Loading RAG...")
from nutrition_rag_system import NutritionRAG
base = '/Users/rakshithmahishi/Documents/food-detection'
rag = NutritionRAG(
    fao_faiss_path=f'{base}/fao_data/fao_faiss.index',
    fao_density_path=f'{base}/fao_data/fao_density.json',
    fao_names_path=f'{base}/fao_data/fao_food_names.json',
    usda_faiss_path=f'{base}/usda_data/usda_faiss.index',
    usda_foods_path=f'{base}/usda_data/usda_foods.json',
    usda_names_path=f'{base}/usda_data/usda_food_names.json',
    usda_density_faiss_path=f'{base}/usda_data/usda_density_faiss.index',
    usda_density_path=f'{base}/usda_data/usda_density.json',
    usda_density_names_path=f'{base}/usda_data/usda_density_names.json',
    gemini_api_key=GEMINI_API_KEY,
)
rag.load()

# ── Load Depth Anything V2 ─────────────────────────────────────────────────────
print("\n[2/4] Loading Depth Anything V2 Small...")
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
da_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
da_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
da_model.eval()
print("   Depth Anything V2 loaded")

# ── Build mock ModelManager ────────────────────────────────────────────────────
mock_models = MagicMock()
mock_models.sam2 = FakeSAM2()
mock_models.florence2 = (MagicMock(), MagicMock())
mock_models.depth_anything = (da_processor, da_model)
mock_models.rag = rag

# ── Load pipeline ──────────────────────────────────────────────────────────────
from app.pipeline import NutritionVideoPipeline
pipeline = NutritionVideoPipeline(mock_models, config)

# ── Intercept depth map to save locally ────────────────────────────────────────
SAVE_DIR = Path('/tmp/nutrition_test_output')
SAVE_DIR.mkdir(parents=True, exist_ok=True)
_original_upload = pipeline._upload_depth_map_to_s3

def _save_depth_map_locally(job_id, frame_idx, depth_image):
    out_path = SAVE_DIR / f'masked_depth_{job_id}_frame{frame_idx}.png'
    depth_image.save(str(out_path))
    print(f"   Masked depth map saved -> {out_path}")
    return _original_upload(job_id, frame_idx, depth_image)

pipeline._upload_depth_map_to_s3 = _save_depth_map_locally

# ── Run on image ───────────────────────────────────────────────────────────────
print(f"\n[3/4] Running pipeline on {Path(IMAGE_PATH).name}...")
user_context = {
    'hidden_ingredients': [],
    'extras': [],
    'recipe_description': ''
}
result = pipeline.process_image(Path(IMAGE_PATH), job_id='test_001', user_context=user_context)

# ── Print results ──────────────────────────────────────────────────────────────
print("\n[4/4] Results:")
nutrition = result.get('nutrition', {})
items = nutrition.get('items', [])
summary = nutrition.get('summary', {})

if not items:
    print("  No food items detected.")
else:
    print(f"\n{'Food Item':<25} {'Volume':>8} {'Density':>8} {'Mass':>8} {'kcal/100g':>10} {'Total kcal':>11}")
    print("-" * 80)
    for item in items:
        print(
            f"{item.get('food_name','?'):<25} "
            f"{item.get('volume_ml',0):>7.0f}ml "
            f"{item.get('density_g_per_ml',0):>7.2f}  "
            f"{item.get('mass_g',0):>7.0f}g "
            f"{item.get('calories_per_100g',0):>9.0f} "
            f"{item.get('total_calories',0):>10.0f}"
        )
        from nutrition_rag_system import NutritionRAG
        food_name = item.get('food_name', '?')
        normalized = NutritionRAG._normalize_food_name(food_name)
        volume = item.get('volume_ml', 0)
        density = item.get('density_g_per_ml', 0)
        mass = item.get('mass_g', 0)
        fao = item.get('density_matched', '?')
        fao_src = item.get('density_source', '?')
        usda = item.get('calorie_matched', '?')
        usda_src = item.get('calorie_source', '?')
        norm_note = f" → '{normalized}'" if normalized != food_name.lower().strip() else ""
        print(f"  {'':25}  searched as:     '{food_name}'{norm_note}")
        print(f"  {'':25}  mass:            {volume:.0f}ml × {density:.2f} g/ml = {mass:.0f}g")
        print(f"  {'':25}  density matched: {fao!r:<40} [{fao_src}]")
        print(f"  {'':25}  calorie matched: {usda!r:<40} [{usda_src}]")
    print("-" * 80)
    print(f"{'TOTAL':<25} {'':>8} {'':>8} {summary.get('total_mass_g',0):>7.0f}g {'':>9} {summary.get('total_calories_kcal',0):>10.0f}")
