"""
Production Configuration Management
Supports environment variables and different deployment modes
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings

# Root for data paths: parent of app/ (docker dir when local, /app when in container)
_CONFIG_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_PRODUCTION_ROOT = _CONFIG_ROOT.parent.parent.parent.parent / "PRODUCTION"
_DOCKER_PRODUCTION_ROOT = Path("/app/PRODUCTION")


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # API Settings
    API_TITLE: str = "Nutrition Video Analysis API"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False

    # File Storage (resolved in validator: /app in Docker, project dir when local)
    UPLOAD_DIR: Path = Path("/app/data/uploads")
    OUTPUT_DIR: Path = Path("/app/data/outputs")
    MODEL_CACHE_DIR: Path = Path("/app/models")
    
    # Video Processing
    MAX_VIDEO_SIZE_MB: int = 500
    ALLOWED_FORMATS: list = [".mp4", ".avi", ".mov", ".mkv"]
    FRAME_SKIP: int = 20  # Process every 20th frame (faster for 5 seconds: 60fps × 5sec / 20 = 15 frames)
    MAX_FRAMES: Optional[int] = 15  # Limit to ~5 seconds of video at 60fps
    RESIZE_WIDTH: int = 800
    # Strict 5-second video: only videos up to this duration; extract exactly this many frames for Gemini multi-image
    VIDEO_MAX_DURATION_SECONDS: float = 5.0
    VIDEO_NUM_FRAMES: int = 5  # Extract 5 frames (one per second) for multi-image prompt; do not count duplicates
    
    # General Calibration (fallback when no reference object detected)
    DEFAULT_PIXELS_PER_CM: float = 16.0  # Default: 800px image ≈ 50cm scene width (800/50 = 16 px/cm)
    DEFAULT_REFERENCE_PLANE_DEPTH_M: float = 0.5  # Default reference plane depth: 50cm (0.5m)

    # Model Settings
    USE_PRODUCTION_IMAGE_PIPELINE: bool = True
    SAM2_CHECKPOINT: str = "checkpoints/sam2.1_hiera_base_plus.pt"
    SAM2_CONFIG: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    PRODUCTION_ROOT: Path = _DOCKER_PRODUCTION_ROOT if Path("/app").exists() else _LOCAL_PRODUCTION_ROOT
    # Use Gemini for detection (image/video understanding) instead of Florence-2 when True
    USE_GEMINI_DETECTION: bool = True  # Set False to use Florence-2 for object detection
    # When True and media is video, call Gemini video API once for the whole clip; when False, use Gemini image per frame
    USE_GEMINI_VIDEO_DETECTION: bool = True
    FLORENCE2_MODEL: str = "microsoft/Florence-2-large-ft"  # Use large model for better accuracy (heavier: ~3GB vs ~1GB)
    DEPTH_ANYTHING_MODEL: str = "depth-anything/Depth-Anything-V2-Small-hf"  # CPU-friendly small variant
    FLAN_T5_MODEL: str = "google/flan-t5-small"  # Small LLM for text formatting (~300MB)
    caption_type: str = "vqa"  # Florence-2 task type: "caption", "detailed_caption", "more_detailed_caption", "object_detection", "hybrid_detection", "detailed_od", or "vqa" (Visual Question Answering - asks questions about food items)
    
    # VQA Configuration (if using VQA mode)
    # Simple conversational question - works best with Florence-2
    VQA_QUESTIONS: list = [
        "What foods are here?"
    ]  # Simple and natural - Gemini reformats the answer
    
    # Florence-2 Generation Parameters
    FLORENCE2_MAX_NEW_TOKENS: int = 1024  # Maximum tokens (1024 is max for Florence-2)
    FLORENCE2_NUM_BEAMS: int = 5  # Beam search for better quality (1-5, higher = better but slower) - increased to reduce hallucinations
    FLORENCE2_DO_SAMPLE: bool = False  # Set to True for more diverse outputs (with temperature) - keep False to reduce hallucinations
    FLORENCE2_TEMPERATURE: float = 0.7  # Only used if do_sample=True (higher = more creative)
    FLORENCE2_MIN_LENGTH: int = 50  # Minimum caption length to encourage longer outputs
    FLORENCE2_VQA_MIN_LENGTH: int = 20  # Minimum length for VQA answers (reduced to allow shorter, more accurate answers)

    # Tracking Settings
    DETECTION_INTERVAL: int = 5  # Re-detect every 5 frames (more frequent for fewer total frames)
    IOU_MATCH_THRESHOLD: float = 0.20
    CENTER_DISTANCE_THRESHOLD: float = 200.0
    LABEL_SIMILARITY_BOOST: float = 0.20
    
    # Object Filtering
    GENERIC_OBJECTS: list = ['table', 'tablecloth', 'menu card', 'background', 'setting', 'surface']
    MIN_BOX_AREA: int = 500
    
    # Nutrition Analysis
    REFERENCE_PLATE_DIAMETER_CM: float = 25.0
    REFERENCE_BOWL_DIAMETER_CM: float = 20.0  # Typical bowl diameter (can vary)
    REFERENCE_OBJECTS: list = ['plate', 'bowl', 'platter', 'dish']  # Objects that can be used for calibration
    CALORIE_SIMILARITY_THRESHOLD: float = 0.5
    # Unified FAISS index for generic nutrition retrieval (USDA + CoFID)
    UNIFIED_FAISS_PATH: Path = Path("/app/data/rag/unified_faiss.index")
    UNIFIED_FOODS_PATH: Path = Path("/app/data/rag/unified_foods.json")
    UNIFIED_FOOD_NAMES_PATH: Path = Path("/app/data/rag/unified_food_names.json")
    # Separate FAO density fallback
    FAO_FAISS_PATH: Path = Path("/app/data/rag/fao_faiss.index")
    FAO_FOODS_PATH: Path = Path("/app/data/rag/fao_foods.json")
    FAO_FOOD_NAMES_PATH: Path = Path("/app/data/rag/fao_food_names.json")
    BRANDED_FOODS_PATH: Path = Path("/app/data/rag/branded_foods.json")
    GEMINI_DENSITY_CACHE_PATH: Path = Path("/app/data/rag/gemini_density_cache.json")
    # Number of frames to run Depth Anything V2 on for video (averaged)
    DEPTH_NUM_FRAMES: int = 3

    # TRELLIS GPU (v2 pipeline)
    ENABLE_TRELLIS: bool = False
    TRELLIS_ECS_CLUSTER: str = "food-detection-v2-cluster"
    TRELLIS_TASK_DEFINITION: str = "food-detection-v2-trellis-gpu"
    TRELLIS_GPU_INSTANCE_ID: str = "i-0b1f38c1962e885fd"
    TRELLIS_INPUT_BUCKET: str = "nutrition-video-analysis-dev-results-dbenpoj2"
    TRELLIS_OUTPUT_BUCKET: str = "nutrition-video-analysis-dev-results-dbenpoj2"
    TRELLIS_INPUT_PREFIX: str = "v2/trellis/inputs"
    TRELLIS_OUTPUT_PREFIX: str = "v2/trellis/outputs"
    TRELLIS_AWS_REGION: str = "us-east-1"
    TRELLIS_TASK_TIMEOUT_S: int = 1800  # 30-min hard timeout per GPU task
    TRELLIS_PREVIEW_SECONDS: int = 8
    TRELLIS_PREVIEW_FPS: int = 15

    # External APIs
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_FLASH_MODEL: str = "gemini-flash-latest"
    
    # Database
    DATABASE_URL: str = "sqlite:///./data/nutrition.db"  # PostgreSQL: postgresql://user:pass@host/db
    
    # Cache
    REDIS_URL: Optional[str] = None  # e.g., "redis://localhost:6379"
    CACHE_MODELS: bool = True
    
    # GPU/Compute
    DEVICE: str = "cuda"  # "cuda" or "cpu"
    USE_FP16: bool = True  # Half precision for 2x speedup
    BATCH_SIZE: int = 1  # Increase for more GPU memory
    
    # Job Queue
    QUEUE_TYPE: str = "memory"  # "memory", "redis", or "sqs"
    SQS_QUEUE_URL: Optional[str] = None
    MAX_CONCURRENT_JOBS: int = 3
    JOB_TIMEOUT_SECONDS: int = 3600  # 1 hour
    
    # Security
    CORS_ORIGINS: list = ["*"]  # In production: ["https://yourdomain.com"]
    MAX_REQUESTS_PER_MINUTE: int = 10
    API_KEY_HEADER: str = "X-API-Key"
    REQUIRE_API_KEY: bool = False
    
    # Logging
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    LOG_FORMAT: str = "json"  # "json" or "text"

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug_flag(cls, value):
        """Accept common deployment strings like 'release' and 'debug'."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "prod", "production"}:
                return False
            if normalized in {"debug", "dev", "development"}:
                return True
        return value

    @model_validator(mode="after")
    def use_local_paths_when_not_in_docker(self):
        """When /app doesn't exist (running locally), use project dir for data paths."""
        if not Path("/app").exists():
            root = _CONFIG_ROOT
            self.UPLOAD_DIR = root / "data" / "uploads"
            self.OUTPUT_DIR = root / "data" / "outputs"
            self.MODEL_CACHE_DIR = root / "models"
            self.PRODUCTION_ROOT = _LOCAL_PRODUCTION_ROOT
            unified_data = root.parent.parent.parent.parent / "unified_data"
            fao_data = root.parent.parent.parent.parent / "fao_data"
            branded_data = root.parent.parent.parent.parent / "branded_data"
            self.UNIFIED_FAISS_PATH = unified_data / "unified_faiss.index"
            self.UNIFIED_FOODS_PATH = unified_data / "unified_foods.json"
            self.UNIFIED_FOOD_NAMES_PATH = unified_data / "unified_food_names.json"
            self.FAO_FAISS_PATH = fao_data / "fao_faiss.index"
            self.FAO_FOODS_PATH = fao_data / "fao_foods.json"
            self.FAO_FOOD_NAMES_PATH = fao_data / "fao_food_names.json"
            self.BRANDED_FOODS_PATH = branded_data / "usda_branded_foods.json"
            self.GEMINI_DENSITY_CACHE_PATH = unified_data / "gemini_density_cache.json"
        else:
            self.PRODUCTION_ROOT = _DOCKER_PRODUCTION_ROOT
        return self

    @model_validator(mode="after")
    def gemini_api_key_fallback(self):
        """If GEMINI_API_KEY not set, try reading from TEST_OPTIMIZATIONS.md in FoodAI/docker (same as gemini scripts)."""
        if self.GEMINI_API_KEY and str(self.GEMINI_API_KEY).strip():
            return self
        # Check docker dir and parent (terraform) so we find the key from any run location
        for candidate in [_CONFIG_ROOT, _CONFIG_ROOT.parent]:
            fallback_file = candidate / "TEST_OPTIMIZATIONS.md"
            if not fallback_file.exists():
                continue
            try:
                content = fallback_file.read_text(encoding="utf-8")
                for line in content.split("\n"):
                    if "GEMINI_API_KEY=" not in line:
                        continue
                    # Support: export GEMINI_API_KEY="..." or GEMINI_API_KEY="..."
                    if '"' in line:
                        key = line.split('"')[1].strip()
                    elif "'" in line:
                        key = line.split("'")[1].strip()
                    else:
                        continue
                    if key and len(key) > 10:
                        self.GEMINI_API_KEY = key
                        break
                if self.GEMINI_API_KEY:
                    break
            except Exception:
                pass
        return self

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


def init_directories():
    """Create required directories if they don't exist"""
    dirs = [
        settings.UPLOAD_DIR,
        settings.OUTPUT_DIR,
        settings.MODEL_CACHE_DIR,
        settings.UNIFIED_FAISS_PATH.parent,
        settings.FAO_FAISS_PATH.parent,
        settings.BRANDED_FOODS_PATH.parent,
        settings.GEMINI_DENSITY_CACHE_PATH.parent,
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_device():
    """Get compute device (cuda/cpu) with error handling"""
    import torch
    if settings.DEVICE == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def validate_config():
    """Validate configuration before starting"""
    errors = []
    
    # Check Gemini API key if required
    if not settings.GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY not set - calorie fallback will be limited")
    
    # Check model files exist (in production)
    if not settings.DEBUG:
        model_path = Path(settings.SAM2_CHECKPOINT)
        if not model_path.exists() and not model_path.is_absolute():
            errors.append(f"SAM2 checkpoint not found: {settings.SAM2_CHECKPOINT}")
    
    # Check GPU availability
    import torch
    if settings.DEVICE == "cuda" and not torch.cuda.is_available():
        errors.append("CUDA requested but not available - falling back to CPU")
        settings.DEVICE = "cpu"
    
    if errors:
        print("⚠️  Configuration warnings:")
        for error in errors:
            print(f"  - {error}")
    
    return len([e for e in errors if "not found" in e]) == 0  # Only fail on critical errors


if __name__ == "__main__":
    print("Current Configuration:")
    print(f"  Upload Dir: {settings.UPLOAD_DIR}")
    print(f"  Output Dir: {settings.OUTPUT_DIR}")
    print(f"  Device: {get_device()}")
    print(f"  Database: {settings.DATABASE_URL}")
    print(f"  Gemini API: {'✓ Configured' if settings.GEMINI_API_KEY else '✗ Not set'}")
    print(f"\nValidation: {'✓ Passed' if validate_config() else '✗ Failed'}")
