#!/usr/bin/env python3
"""
Tests _detect_objects_gemini() from the actual pipeline.py with real Gemini API
and a real food image, using mocked SAM2/Florence so no checkpoints are needed.
This verifies that user_context from the questionnaire reaches the Gemini prompt.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not set")
    sys.exit(1)

IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else None
if not IMAGE_PATH or not os.path.exists(IMAGE_PATH):
    print(f"Usage: python3 {sys.argv[0]} /path/to/food.jpg [user_context_json]")
    sys.exit(1)

USER_CONTEXT = {}
if len(sys.argv) > 2:
    try:
        USER_CONTEXT = json.loads(sys.argv[2])
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid user_context JSON: {e}")
        sys.exit(1)

# --- Mock heavy dependencies so pipeline.py imports without errors ---
from unittest.mock import MagicMock
import types

# Import torch for real first — new transformers/safetensors checks torch.__version__
# at import time, which breaks if torch is a MagicMock.
import torch  # noqa: F401  (real torch 2.4.1 is installed in this venv)

# Mock cv2, boto3 if not installed
import importlib.machinery
for mod in ['cv2', 'boto3']:
    if mod not in sys.modules:
        m = MagicMock()
        # Python 3.9's importlib.util.find_spec requires __spec__ to be set on any
        # module present in sys.modules — bypass MagicMock's dunder interception.
        object.__setattr__(m, '__spec__', importlib.machinery.ModuleSpec(mod, loader=None))
        sys.modules[mod] = m

# Mock numpy properly (needs to be real numpy for PIL)
import numpy as np

# Mock SAM2/Florence/Depth Anything model manager
mock_models = MagicMock()
mock_models.sam2 = MagicMock()
mock_models.florence2 = (MagicMock(), MagicMock())
mock_models.depth_anything = (MagicMock(), MagicMock())

# Load config with real GEMINI_API_KEY
os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY

from app.config import Settings
config = Settings()
config.GEMINI_API_KEY = GEMINI_API_KEY
config.USE_GEMINI_DETECTION = True
config.DEVICE = 'cpu'

# Import pipeline (now safe since heavy libs are mocked)
from app.pipeline import NutritionVideoPipeline
pipeline = NutritionVideoPipeline(mock_models, config)

# Load image as PIL
from PIL import Image
image_pil = Image.open(IMAGE_PATH).convert('RGB')

print(f"\n--- IMAGE: {IMAGE_PATH}")
print(f"--- USER CONTEXT: {json.dumps(USER_CONTEXT, indent=2) if USER_CONTEXT else 'none'}")
print("\nCalling pipeline._detect_objects_gemini() with real Gemini API...\n")

result = pipeline._detect_objects_gemini(image_pil, 'test-job', user_context=USER_CONTEXT)

print("\n--- RESULT FROM PIPELINE GEMINI CALL ---")
print(json.dumps(result, indent=2, default=str))
