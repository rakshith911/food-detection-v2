"""
Model Loading and Caching
Handles initialization of Florence-2, SAM2, Depth Anything V2, and RAG system
"""
# CRITICAL: Patch transformers.utils BEFORE importing transformers
# Florence-2's custom code executes at import time and needs this function
# We must import transformers first to get the utils module, then patch it
# This MUST happen before ANY transformers imports, including AutoProcessor/AutoModelForCausalLM
import sys
import importlib

# Import transformers to get access to utils module
import transformers
import transformers.utils as transformers_utils

# Apply patch immediately - this MUST happen before Florence-2 custom code executes
if not hasattr(transformers_utils, 'is_flash_attn_greater_or_equal_2_10'):
    def is_flash_attn_greater_or_equal_2_10():
        """Check if flash_attn version >= 2.10. Returns False for CPU-only environments."""
        return False  # Always False for CPU, which is what we want
    transformers_utils.is_flash_attn_greater_or_equal_2_10 = is_flash_attn_greater_or_equal_2_10
    # Also patch it in the module's __dict__ to ensure it's available
    setattr(transformers_utils, 'is_flash_attn_greater_or_equal_2_10', is_flash_attn_greater_or_equal_2_10)
    print("✅ Applied monkey patch for is_flash_attn_greater_or_equal_2_10 in models.py")
    print(f"✅ Verified: hasattr check = {hasattr(transformers_utils, 'is_flash_attn_greater_or_equal_2_10')}")
else:
    print("✅ is_flash_attn_greater_or_equal_2_10 already exists in transformers.utils")

# CRITICAL: Patch transformers import checking BEFORE importing transformers
# Florence-2's modeling file checks for flash_attn imports
# We'll patch transformers.dynamic_module_utils.check_imports in load_florence2
print("✅ Will patch transformers import checking for flash_attn (CPU mode)")

# CRITICAL: Import NumPy BEFORE PyTorch to ensure PyTorch can detect it
# PyTorch's torch.from_numpy() requires NumPy to be imported first
import numpy as np
print(f"✅ NumPy {np.__version__} imported before PyTorch in models.py")

import torch
from pathlib import Path
from typing import Optional, Dict, Any
from functools import lru_cache
import logging

from transformers import AutoProcessor, AutoModelForCausalLM
from sam2.build_sam import build_sam2_video_predictor
from app.production_models import load_sam3, load_zoedepth

logger = logging.getLogger(__name__)


class ModelCache:
    """Singleton class to cache loaded models"""
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get(self, key: str):
        return self._models.get(key)
    
    def set(self, key: str, model: Any):
        self._models[key] = model
    
    def clear(self):
        """Clear all cached models to free memory"""
        self._models.clear()
        torch.cuda.empty_cache()


# Global cache instance
model_cache = ModelCache()


def load_florence2(model_name: str = "microsoft/Florence-2-base-ft", device: str = "cuda") -> tuple:
    """
    Load Florence-2 model and processor
    
    Args:
        model_name: HuggingFace model name
        device: Device to load model on
        
    Returns:
        (processor, model) tuple
    """
    cache_key = f"florence2_{model_name}_{device}"
    cached = model_cache.get(cache_key)
    if cached:
        logger.info(f"Using cached Florence-2 model")
        return cached
    
    logger.info(f"Loading Florence-2 model: {model_name}...")
    print(f"⏳ Loading Florence-2 model '{model_name}' from HuggingFace...")
    print("   This may take 1-2 minutes (downloading ~1GB)...")
    import sys
    sys.stdout.flush()
    
    # CRITICAL: Patch transformers' check_imports to skip flash_attn requirement
    # This is called when loading models with trust_remote_code=True
    try:
        from transformers.dynamic_module_utils import check_imports as _original_check_imports
        def _patched_check_imports(module_file):
            # Call original check_imports but catch flash_attn ImportError
            try:
                return _original_check_imports(module_file)
            except ImportError as e:
                error_msg = str(e)
                if 'flash_attn' in error_msg:
                    # Suppress flash_attn requirement - we'll use eager attention
                    logger.info("Suppressing flash_attn requirement (using CPU eager attention)")
                    print("✓ Suppressing flash_attn requirement (CPU mode)")
                    sys.stdout.flush()
                    return []  # Return empty list (no missing packages)
                raise  # Re-raise if it's a different ImportError
        
        import transformers.dynamic_module_utils
        transformers.dynamic_module_utils.check_imports = _patched_check_imports
        print("✓ Patched transformers.check_imports to skip flash_attn")
        sys.stdout.flush()
    except Exception as e:
        logger.warning(f"Could not patch check_imports: {e}")
        print(f"⚠ Warning: Could not patch check_imports: {e}")
        sys.stdout.flush()
    
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=None  # Use default HF cache
    )
    print("✓ Florence-2 processor loaded")
    sys.stdout.flush()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=None,
        attn_implementation="eager"  # Use eager attention instead of flash_attn for CPU
    ).to(device)
    print("✓ Florence-2 model loaded")
    sys.stdout.flush()
    
    model.eval()
    
    result = (processor, model)
    model_cache.set(cache_key, result)
    
    logger.info(f"✓ Florence-2 loaded successfully")
    return result


def load_flan_t5(model_name: str = "google/flan-t5-small", device: str = "cpu") -> tuple:
    """
    Load FLAN-T5 model for text formatting tasks
    
    Args:
        model_name: HuggingFace model name (default: flan-t5-small ~300MB)
        device: Device to load model on (default: cpu)
        
    Returns:
        (tokenizer, model) tuple
    """
    cache_key = f"flan_t5_{model_name}_{device}"
    cached = model_cache.get(cache_key)
    if cached:
        logger.info(f"Using cached FLAN-T5 model")
        return cached
    
    logger.info(f"Loading FLAN-T5 model: {model_name}...")
    print(f"⏳ Loading FLAN-T5 model '{model_name}' from HuggingFace...")
    print("   This is a small model (~300MB) for text formatting...")
    sys.stdout.flush()
    
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    print("✓ FLAN-T5 model loaded")
    sys.stdout.flush()
    
    result = (tokenizer, model)
    model_cache.set(cache_key, result)
    
    logger.info(f"✓ FLAN-T5 loaded successfully")
    return result


def load_sam2(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """
    Load SAM2 video predictor
    
    Args:
        config_path: Path to SAM2 config YAML
        checkpoint_path: Path to SAM2 checkpoint
        device: Device to load model on
        
    Returns:
        SAM2 video predictor
    """
    cache_key = f"sam2_{checkpoint_path}_{device}"
    cached = model_cache.get(cache_key)
    if cached:
        logger.info("Using cached SAM2 model")
        return cached
    
    logger.info(f"Loading SAM2 model: {checkpoint_path}...")
    print(f"⏳ Loading SAM2 model from checkpoint...")
    print(f"   Config: {config_path}")
    print(f"   Checkpoint: {checkpoint_path}")
    import sys
    import os
    sys.stdout.flush()
    
    # Convert Path objects to strings and resolve relative to sam2 package
    # Hydra expects config_name as a relative path from the sam2 package root (e.g., "configs/sam2.1/sam2.1_hiera_b+.yaml")
    from pathlib import Path as PathLib
    
    # Get the absolute config path
    if isinstance(config_path, PathLib):
        config_abs = config_path
    else:
        config_abs = PathLib(config_path)
    
    # Find sam2 package root (sam2.__path__[0] points to the sam2 package directory)
    import sam2
    sam2_root = PathLib(sam2.__path__[0])  # This is the sam2 package directory itself
    
    # Convert absolute path to relative path from sam2 root
    try:
        config_file_str = str(config_abs.relative_to(sam2_root))
        # Normalize path separators for Hydra (use forward slashes)
        config_file_str = config_file_str.replace(os.sep, '/')
    except ValueError:
        # If not relative to sam2_root, try to extract relative part
        # Config path should be like: .../sam2/configs/.../file.yaml
        config_str = str(config_abs.resolve())
        sam2_str = str(sam2_root.resolve())
        if sam2_str in config_str:
            config_file_str = config_str.split(sam2_str + os.sep, 1)[1]
            # Normalize path separators for Hydra (use forward slashes)
            config_file_str = config_file_str.replace(os.sep, '/')
        else:
            # Fallback: use as-is if we can't resolve
            config_file_str = str(config_abs)
            logger.warning(f"Could not resolve config path relative to sam2 root, using: {config_file_str}")
    
    if isinstance(checkpoint_path, (PathLib, str)):
        checkpoint_path_str = str(checkpoint_path)
    else:
        checkpoint_path_str = checkpoint_path
    
    print(f"   Using Hydra config path: {config_file_str}")
    sys.stdout.flush()
    
    predictor = build_sam2_video_predictor(
        config_file=config_file_str,
        ckpt_path=checkpoint_path_str,
        device=device
    )
    print("✓ SAM2 model loaded")
    sys.stdout.flush()
    
    model_cache.set(cache_key, predictor)
    
    logger.info("✓ SAM2 loaded successfully")
    return predictor


def load_depth_anything(model_name: str = "depth-anything/Depth-Anything-V2-Small-hf", device: str = "cpu"):
    """
    Load Depth Anything V2 (Small) depth estimation model via HuggingFace.
    Returns (processor, model) tuple. Runs well on CPU.
    """
    cache_key = f"depth_anything_{model_name}_{device}"
    cached = model_cache.get(cache_key)
    if cached:
        logger.info("Using cached Depth Anything V2 model")
        return cached

    logger.info(f"Loading Depth Anything V2 model: {model_name}...")
    print(f"Loading Depth Anything V2 '{model_name}' from HuggingFace...")
    import sys
    sys.stdout.flush()

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    print("Depth Anything V2 model loaded")
    sys.stdout.flush()

    result = (processor, model)
    model_cache.set(cache_key, result)
    logger.info("Depth Anything V2 loaded successfully")
    return result


def load_nutrition_rag(
    unified_faiss_path: Path,
    unified_foods_path: Path,
    unified_food_names_path: Path,
    fao_faiss_path: Path,
    fao_foods_path: Path,
    fao_food_names_path: Path,
    branded_foods_path: Path = None,
    gemini_density_cache_path: Path = None,
    gemini_api_key: str = None,
    gemini_model: str = "gemini-flash-latest",
):
    """
    Load and initialize Nutrition RAG system using a unified USDA+CoFID index
    plus a separate FAO density fallback.

    Returns:
        Initialized NutritionRAG instance
    """
    cache_key = "nutrition_rag"
    cached = model_cache.get(cache_key)
    if cached:
        logger.info("Using cached NutritionRAG system")
        return cached

    logger.info("Loading Nutrition RAG system...")

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from nutrition_rag_system import NutritionRAG

    rag = NutritionRAG(
        unified_faiss_path=unified_faiss_path,
        unified_foods_path=unified_foods_path,
        unified_food_names_path=unified_food_names_path,
        fao_faiss_path=fao_faiss_path,
        fao_density_path=fao_foods_path,
        fao_names_path=fao_food_names_path,
        branded_foods_path=branded_foods_path,
        gemini_density_cache_path=gemini_density_cache_path,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
    )
    rag.load()

    model_cache.set(cache_key, rag)
    logger.info("NutritionRAG loaded successfully")
    return rag


class ModelManager:
    """High-level model management interface"""
    
    def __init__(self, config):
        self.config = config
        # Use CPU when CUDA is requested but not available (e.g. on Mac / CPU-only PyTorch)
        if config.DEVICE == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            logger.info("CUDA not available (e.g. Torch not compiled with CUDA) - using CPU")
        else:
            self.device = config.DEVICE

        logger.info(
            "[ModelManager] Runtime summary: "
            f"device={self.device}, "
            f"use_production_image_pipeline={getattr(config, 'USE_PRODUCTION_IMAGE_PIPELINE', None)}, "
            f"production_root={getattr(config, 'PRODUCTION_ROOT', None)}, "
            f"sam3_model_dir={getattr(config, 'SAM3_MODEL_DIR', None)}, "
            f"zoedepth_checkpoint={getattr(config, 'ZOEDEPTH_CHECKPOINT', None)}, "
            f"midas_repo_dir={getattr(config, 'MIDAS_REPO_DIR', None)}"
        )
        
        # Models will be loaded on demand
        self._florence2 = None
        self._flan_t5 = None
        self._sam2 = None
        self._sam3 = None
        self._depth_anything = None
        self._zoedepth = None
        self._rag = None
    
    @property
    def florence2(self):
        """Lazy load Florence-2"""
        if self._florence2 is None:
            self._florence2 = load_florence2(
                model_name=self.config.FLORENCE2_MODEL,
                device=self.device
            )
        return self._florence2
    
    @property
    def flan_t5(self):
        """Lazy load FLAN-T5 for text formatting"""
        if self._flan_t5 is None:
            self._flan_t5 = load_flan_t5(
                model_name=self.config.FLAN_T5_MODEL,
                device="cpu"  # Always use CPU for this small model
            )
        return self._flan_t5
    
    @property
    def sam2(self):
        """Lazy load SAM2"""
        if self._sam2 is None:
            self._sam2 = load_sam2(
                config_path=self.config.SAM2_CONFIG,
                checkpoint_path=self.config.SAM2_CHECKPOINT,
                device=self.device
            )
        return self._sam2
    
    @property
    def depth_anything(self):
        """Lazy load Depth Anything V2 Small"""
        if self._depth_anything is None:
            self._depth_anything = load_depth_anything(
                model_name=self.config.DEPTH_ANYTHING_MODEL,
                device=self.device
            )
        return self._depth_anything

    @property
    def sam3(self):
        """Lazy load SAM3"""
        if self._sam3 is None:
            logger.info(f"[ModelManager] Loading SAM3 for production image pipeline from {self.config.SAM3_MODEL_DIR}")
            self._sam3 = load_sam3(
                model_dir=self.config.SAM3_MODEL_DIR,
                device=self.device
            )
        return self._sam3

    @property
    def zoedepth(self):
        """Lazy load ZoeDepth"""
        if self._zoedepth is None:
            logger.info(
                f"[ModelManager] Loading ZoeDepth for production image pipeline from "
                f"{self.config.ZOEDEPTH_CHECKPOINT} with MiDaS repo {self.config.MIDAS_REPO_DIR}"
            )
            self._zoedepth = load_zoedepth(
                checkpoint_path=self.config.ZOEDEPTH_CHECKPOINT,
                midas_repo_dir=self.config.MIDAS_REPO_DIR,
                device=self.device
            )
        return self._zoedepth

    @property
    def rag(self):
        """Lazy load NutritionRAG"""
        if self._rag is None:
            self._rag = load_nutrition_rag(
                unified_faiss_path=self.config.UNIFIED_FAISS_PATH,
                unified_foods_path=self.config.UNIFIED_FOODS_PATH,
                unified_food_names_path=self.config.UNIFIED_FOOD_NAMES_PATH,
                fao_faiss_path=self.config.FAO_FAISS_PATH,
                fao_foods_path=self.config.FAO_FOODS_PATH,
                fao_food_names_path=self.config.FAO_FOOD_NAMES_PATH,
                branded_foods_path=self.config.BRANDED_FOODS_PATH,
                gemini_density_cache_path=self.config.GEMINI_DENSITY_CACHE_PATH,
                gemini_api_key=self.config.GEMINI_API_KEY,
                gemini_model=self.config.GEMINI_FLASH_MODEL,
            )
        return self._rag

    def preload_all(self):
        """Preload all models (useful for container warmup)"""
        logger.info("Preloading all models...")
        _ = self.florence2
        if not getattr(self.config, "USE_PRODUCTION_IMAGE_PIPELINE", True):
            _ = self.sam2
        _ = self.sam3
        _ = self.depth_anything
        _ = self.zoedepth
        _ = self.rag
        logger.info("All models preloaded")

    def clear_cache(self):
        """Clear all cached models"""
        self._florence2 = None
        self._sam2 = None
        self._sam3 = None
        self._depth_anything = None
        self._zoedepth = None
        self._rag = None
        model_cache.clear()
        logger.info("Model cache cleared")


# Test function
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from config import settings
    
    logging.basicConfig(level=logging.INFO)
    
    print("Testing model loading...")
    
    manager = ModelManager(settings)
    
    # Test lazy loading
    print("\n1. Testing Florence-2 lazy load...")
    proc, model = manager.florence2
    print(f"   ✓ Florence-2: {type(model).__name__}")
    
    print("\n2. Testing SAM2 lazy load...")
    sam2 = manager.sam2
    print(f"   ✓ SAM2: {type(sam2).__name__}")
    
    print("\nDone!")
