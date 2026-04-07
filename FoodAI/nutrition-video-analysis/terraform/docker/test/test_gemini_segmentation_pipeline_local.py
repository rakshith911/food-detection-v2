#!/usr/bin/env python3
"""
Local comparison harness for the production image pipeline using Gemini
segmentation masks instead of SAM3.

This keeps the rest of the pipeline intact:
  Gemini first pass -> Gemini segmentation masks -> ZoeDepth calibration ->
  Gemini volume estimation -> RAG nutrition lookup -> calorie totals

Outputs are written under this test/ folder so we can compare against the
regular local production pipeline without touching deployable code paths.

Reference notebook:
  zero_shot_object_detection_and_segmentation_with_google_gamini_2_5.ipynb
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import re

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
DOCKER_DIR = SCRIPT_DIR.parent
RECOVERED_REPO_ROOT = DOCKER_DIR.parent.parent.parent.parent
ORIGINAL_REPO_ROOT = Path("/Users/rakshithmahishi/Documents/food-detection")
sys.path.insert(0, str(DOCKER_DIR))

from app.config import Settings


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the production image pipeline locally using Gemini segmentation masks."
    )
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
        "--model",
        default=os.environ.get("GEMINI_SEGMENTATION_MODEL", "gemini-2.5-flash"),
        help="Gemini model to use for segmentation masks",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory where test outputs should be written. Defaults to docker/test/<timestamp>",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images when input_path is a directory")
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


def build_settings(device: str, output_dir: Path) -> Settings:
    config = Settings()
    config.DEVICE = device
    config.USE_PRODUCTION_IMAGE_PIPELINE = True
    config.DEBUG = True
    config.OUTPUT_DIR = output_dir
    config.UPLOAD_DIR = DOCKER_DIR / "data" / "uploads"
    config.MODEL_CACHE_DIR = DOCKER_DIR / "models"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    recovered_production_root = RECOVERED_REPO_ROOT / "PRODUCTION"
    original_production_root = ORIGINAL_REPO_ROOT / "PRODUCTION"
    if not config.ZOEDEPTH_CHECKPOINT.exists() and original_production_root.exists():
        config.PRODUCTION_ROOT = original_production_root
        config.SAM3_MODEL_DIR = config.PRODUCTION_ROOT / "model_assets" / "sam3_foodseg_final"
        config.ZOEDEPTH_CHECKPOINT = config.PRODUCTION_ROOT / "model_assets" / "zoedepth" / "ZoeD_M12_N.pt"
        config.MIDAS_REPO_DIR = config.PRODUCTION_ROOT / "model_assets" / "midas_repo"
        print(f"Using production assets from: {config.PRODUCTION_ROOT}")
    elif recovered_production_root.exists():
        print(f"Using production assets from: {recovered_production_root}")

    # The config.py validator computes unified_data one level too deep (FoodAI/unified_data).
    # Fix to the actual location: food-detection-recovered/unified_data.
    unified_data = RECOVERED_REPO_ROOT / "unified_data"
    if unified_data.exists():
        config.UNIFIED_FAISS_PATH = unified_data / "unified_faiss.index"
        config.UNIFIED_FOODS_PATH = unified_data / "unified_foods.json"
        config.UNIFIED_FOOD_NAMES_PATH = unified_data / "unified_food_names.json"
        print(f"Using unified RAG data from: {unified_data}")

    return config


class ComparisonModelManager:
    """Local shim so this comparison harness does not import the SAM2/Hydra stack."""

    def __init__(self, config: Settings) -> None:
        self.config = config
        self.device = config.DEVICE
        try:
            import torch

            if self.device == "cuda" and not torch.cuda.is_available():
                self.device = "cpu"
        except Exception:
            self.device = "cpu"

        self._sam3 = (None, None)
        self._zoedepth = None
        self._rag = None

    @property
    def sam3(self):
        return self._sam3

    @property
    def zoedepth(self):
        if self._zoedepth is None:
            from app.production_models import load_zoedepth

            self._zoedepth = load_zoedepth(
                checkpoint_path=self.config.ZOEDEPTH_CHECKPOINT,
                midas_repo_dir=self.config.MIDAS_REPO_DIR,
                device=self.device,
            )
        return self._zoedepth

    @property
    def rag(self):
        if self._rag is None:
            from nutrition_rag_system import NutritionRAG

            self._rag = NutritionRAG(
                unified_faiss_path=self.config.UNIFIED_FAISS_PATH,
                unified_foods_path=self.config.UNIFIED_FOODS_PATH,
                unified_food_names_path=self.config.UNIFIED_FOOD_NAMES_PATH,
                gemini_api_key=self.config.GEMINI_API_KEY,
                gemini_model=self.config.GEMINI_FLASH_MODEL,
            )
            self._rag.load()
        return self._rag


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


class GeminiSegmentationProvider:
    def __init__(self, api_key: str, model_name: str) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.last_prompt: Optional[str] = None
        self.last_response_text: Optional[str] = None
        self.last_items: list[dict[str, Any]] = []
        self.last_mask_summary: dict[str, Any] = {}
        self.last_debug_payloads: list[dict[str, Any]] = []
        # Stores {label: {"mask": np.ndarray bool, "soft_mask": np.ndarray float32, "score": float}}
        # for notebook-style re-rendering after the call completes.
        self.last_results: dict[str, dict[str, Any]] = {}
        # Stores {label: soft_float_mask} — grayscale [0,1] mask before binary threshold.
        self.last_soft_masks: dict[str, np.ndarray] = {}
        # supervision Detections object and original image for notebook-style rendering.
        self.last_sv_detections: Any = None
        self.last_pil_image: Optional[Image.Image] = None

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(
            "".join(ch.lower() if ch.isalnum() else " " for ch in (text or "")).split()
        )

    def _match_label(self, label: str, prompts: list[str]) -> Optional[str]:
        normalized_label = self._normalize(label)
        normalized_prompts = {prompt: self._normalize(prompt) for prompt in prompts}

        for prompt, normalized_prompt in normalized_prompts.items():
            if normalized_label == normalized_prompt:
                return prompt

        for prompt, normalized_prompt in normalized_prompts.items():
            if normalized_label in normalized_prompt or normalized_prompt in normalized_label:
                return prompt

        label_tokens = set(normalized_label.split())
        best_prompt = None
        best_overlap = 0
        for prompt, normalized_prompt in normalized_prompts.items():
            overlap = len(label_tokens & set(normalized_prompt.split()))
            if overlap > best_overlap:
                best_prompt = prompt
                best_overlap = overlap
        return best_prompt if best_overlap > 0 else None

    @staticmethod
    def _sanitize_seg_tokens(text: str) -> str:
        """Remove or repair <start_of_mask><seg_N> token fragments in mask values.

        Gemini 2.5 Flash occasionally returns segmentation tokens instead of (or
        mixed with) the base64 PNG format. Three observed patterns:

        A) "mask": "<start_of_mask><seg_N>...data:image/png;base64,..."
           → strip the token prefix, keep the base64 payload.
        B) "mask": "<start_of_mask><seg_N>... some text"  (valid JSON string, no PNG)
        C) "mask": "<start_of_mask><seg_N>... (coords), "label": ...  (broken JSON)

        Cases B and C produce unparseable JSON. We remove the entire item so the
        remaining items can still be decoded.
        """
        # Pattern A: rescue items where base64 data follows the token prefix.
        text = re.sub(
            r'"<start_of_mask>(?:<seg_\d+>\s*)*(data:image/png;base64,)',
            r'"\1',
            text,
        )
        # Patterns B & C: remove entire JSON objects that still contain seg tokens.
        # The comma+whitespace before the item is also consumed so we don't leave
        # a trailing comma before ] or between items.
        text = re.sub(r',?\s*\{[^{}]*<start_of_mask>[^{}]*\}', '', text)
        # Clean up any trailing comma before ] that removal may have produced.
        text = re.sub(r',(\s*\])', r'\1', text)
        return text

    @staticmethod
    def _parse_json_payload(response_text: str) -> list[dict[str, Any]]:
        text = (response_text or "").strip()
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()
        else:
            start = text.find("[")
            if start == -1:
                raise ValueError(f"Could not parse Gemini segmentation JSON: {text[:500]}")
            # Walk forward with bracket depth tracking so we find the matching
            # closing ] rather than rfind which picks up ] inside nested arrays
            # or inside string values like base64 data with newlines.
            depth = 0
            in_str = False
            esc = False
            end = start
            for i, ch in enumerate(text[start:], start):
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == '[':
                        depth += 1
                    elif ch == ']':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
            text = text[start:end] if end > start else text[start:]

        # Remove/repair <start_of_mask><seg_N> token fragments before JSON parsing.
        if "<start_of_mask>" in text:
            text = GeminiSegmentationProvider._sanitize_seg_tokens(text)

        def _balance_json(candidate: str) -> str:
            stack = []
            in_string = False
            escape = False
            for ch in candidate:
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == "\"":
                        in_string = False
                    continue
                if ch == "\"":
                    in_string = True
                elif ch in "[{":
                    stack.append(ch)
                elif ch == "]" and stack and stack[-1] == "[":
                    stack.pop()
                elif ch == "}" and stack and stack[-1] == "{":
                    stack.pop()
            closers = []
            while stack:
                opener = stack.pop()
                closers.append("]" if opener == "[" else "}")
            return candidate + "".join(closers)

        def _repair_json(candidate: str) -> str:
            repaired = candidate
            repaired = repaired.replace("\r", " ").replace("\n", " ")
            repaired = re.sub(r'(\]|\}|\")\s*(\")', r'\1, \2', repaired)
            repaired = re.sub(r'(\]|\})\s*([A-Za-z0-9_])', r'\1, \2', repaired)
            repaired = re.sub(r'(\d)\s*(\")', r'\1, \2', repaired)
            repaired = re.sub(r'"\s*([}\]])', r'"\1', repaired)
            repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
            repaired = _balance_json(repaired.strip())
            return repaired

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            pass

        # Gemini 2.5 often wraps base64 mask data with literal newlines inside
        # JSON string values (invalid JSON). Strip newlines only while inside
        # strings, leaving structural whitespace intact so json.loads can parse.
        def _strip_string_newlines(s: str) -> str:
            out = []
            in_str = False
            esc = False
            for ch in s:
                if in_str:
                    if esc:
                        esc = False
                        out.append(ch)
                    elif ch == '\\':
                        esc = True
                        out.append(ch)
                    elif ch == '"':
                        in_str = False
                        out.append(ch)
                    elif ch in ('\n', '\r'):
                        out.append(' ')  # replace with space, base64 decoder tolerates it
                    else:
                        out.append(ch)
                else:
                    if ch == '"':
                        in_str = True
                    out.append(ch)
            return ''.join(out)

        try:
            parsed = json.loads(_strip_string_newlines(text))
            return parsed if isinstance(parsed, list) else []
        except Exception:
            pass

        try:
            repaired = _repair_json(text)
            parsed = json.loads(repaired)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            pass

        try:
            pythonish = (
                _repair_json(text)
                .replace("null", "None")
                .replace("true", "True")
                .replace("false", "False")
            )
            parsed = ast.literal_eval(pythonish)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            pass

        raise ValueError(f"Could not parse Gemini segmentation JSON: {text[:500]}")

    @staticmethod
    def _decode_mask(
        mask_value: str,
        box_2d: list[float],
        image_size: tuple[int, int],
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Decode a Gemini base64 mask thumbnail into full-image masks.

        Returns (binary_bool_mask, soft_float32_mask) both shaped (image_h, image_w).
        soft_float32_mask holds [0, 1] alpha weights for notebook-style blending.
        Returns (None, None) on failure.
        """
        if not mask_value or len(box_2d) != 4:
            return None, None

        encoded = mask_value
        if encoded.startswith("data:image/png;base64,"):
            encoded = encoded.removeprefix("data:image/png;base64,")
        # Model may wrap base64 with whitespace/newlines — strip them before decoding
        encoded = "".join(encoded.split())

        try:
            mask_image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("L")
        except Exception:
            return None, None

        image_w, image_h = image_size
        y0 = max(0, min(int(round(float(box_2d[0]) / 1000.0 * image_h)), image_h))
        x0 = max(0, min(int(round(float(box_2d[1]) / 1000.0 * image_w)), image_w))
        y1 = max(0, min(int(round(float(box_2d[2]) / 1000.0 * image_h)), image_h))
        x1 = max(0, min(int(round(float(box_2d[3]) / 1000.0 * image_w)), image_w))
        if x1 <= x0 or y1 <= y0:
            return None, None

        resized_mask = mask_image.resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)
        gray_array = np.array(resized_mask, dtype=np.float32) / 255.0
        binary_array = gray_array > 0.5

        full_binary = np.zeros((image_h, image_w), dtype=bool)
        full_binary[y0:y1, x0:x1] = binary_array[: y1 - y0, : x1 - x0]

        full_soft = np.zeros((image_h, image_w), dtype=np.float32)
        full_soft[y0:y1, x0:x1] = gray_array[: y1 - y0, : x1 - x0]

        return full_binary, full_soft

    @staticmethod
    def _resize_mask_to_image(mask: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        target_w, target_h = target_size
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        resized = mask_img.resize((target_w, target_h), Image.Resampling.NEAREST)
        return np.array(resized, dtype=np.uint8) > 127

    @staticmethod
    def _resize_soft_mask_to_image(soft_mask: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        target_w, target_h = target_size
        mask_img = Image.fromarray((soft_mask * 255).clip(0, 255).astype(np.uint8), mode="L")
        resized = mask_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        return np.array(resized, dtype=np.float32) / 255.0

    @staticmethod
    def _prepare_segmentation_image(pil_image: Image.Image, max_width: int = 512) -> Image.Image:
        width, height = pil_image.size
        if width <= max_width:
            return pil_image
        target_height = int(max_width * height / width)
        return pil_image.resize((max_width, target_height), Image.Resampling.LANCZOS)

    def _generate_response(self, prompt: str, pil_image: Image.Image) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                from google import genai as genai_new
                from google.genai import types

                client = genai_new.Client(api_key=self.api_key)
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=[pil_image, prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.5,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                return response.text or ""
            except Exception as exc:
                last_error = exc
                try:
                    import google.generativeai as genai

                    genai.configure(api_key=self.api_key)
                    model = genai.GenerativeModel(
                        self.model_name,
                        generation_config={"temperature": 0.5},
                    )
                    response = model.generate_content([pil_image, prompt])
                    return response.text or ""
                except Exception as fallback_exc:
                    last_error = fallback_exc
                    print(f"  Gemini segmentation attempt {attempt}/3 failed: {fallback_exc}")
                    if attempt < 3:
                        time.sleep(2 * attempt)
        if last_error:
            raise last_error
        raise RuntimeError("Gemini segmentation failed without an error")

    @staticmethod
    def _simplify_prompt_label(label: str) -> str:
        text = (label or "").strip().lower()
        replacements = [
            ("shredded ", ""),
            ("diced ", ""),
            ("sliced ", ""),
            ("chunks", ""),
            ("chunk", ""),
            ("slices", ""),
            ("slice", ""),
            ("yellow ", ""),
            ("rectangular ", ""),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return " ".join(text.split()).strip()

    def __call__(
        self,
        sam3_model: Any,
        sam3_processor: Any,
        pil_image: Image.Image,
        prompts: list[str],
        device: str = "cpu",
        min_coverage: float = 0.002,
        max_coverage: float = 0.60,
    ) -> dict[str, dict[str, Any]]:
        del sam3_model, sam3_processor, device

        import supervision as sv

        # Filter out non-food container/packaging items exactly as the reference
        # notebook does (it never includes trays or containers in the prompt).
        _CONTAINER_KEYWORDS = {
            "tray", "container", "foil", "aluminum", "aluminium", "tin",
            "bowl", "plate", "cup", "dish", "wrapper", "packaging", "utensil",
        }
        food_prompts = [
            p for p in prompts
            if not any(kw in p.lower() for kw in _CONTAINER_KEYWORDS)
        ]
        seg_prompts = food_prompts if food_prompts else prompts

        print(f"  [Gemini segmentation] requesting masks for: {', '.join(seg_prompts)}")
        results = {prompt_name: {"mask": None, "score": 0.0} for prompt_name in prompts}

        # Resize to 1024px wide exactly as the reference notebook does
        orig_w, orig_h = pil_image.size
        target_h = int(1024 * orig_h / orig_w)
        segmentation_image = pil_image.resize((1024, target_h), Image.Resampling.LANCZOS)

        ingredients_str = ", ".join(seg_prompts)
        prompt = (
            f"Give the segmentation masks of {ingredients_str}. "
            "Output a JSON list of segmentation masks where each entry contains the 2D bounding box "
            "in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in key "
            "\"label\". Use descriptive labels."
        )

        response_text = self._generate_response(prompt, segmentation_image)

        # Parse with supervision exactly like the notebook:
        # resolution_wh=pil_image.size uses ORIGINAL image dimensions (not 1024px)
        try:
            detections = sv.Detections.from_vlm(
                vlm=sv.VLM.GOOGLE_GEMINI_2_5,
                result=response_text,
                resolution_wh=pil_image.size,
            )
        except Exception as exc:
            preview = (response_text or "")[:300].replace("\n", "\\n")
            print(f"  [Gemini segmentation] supervision parse failed: {exc}")
            print(f"    response_preview={preview}")
            self.last_prompt = prompt
            self.last_response_text = response_text
            self.last_items = []
            self.last_mask_summary = {}
            self.last_debug_payloads = [{"parse_error": str(exc), "response_text": response_text}]
            self.last_sv_detections = None
            self.last_pil_image = pil_image
            return results

        # Store for notebook-style rendering
        self.last_sv_detections = detections
        self.last_pil_image = pil_image

        summary: dict[str, Any] = {}
        class_names = list(detections.data.get("class_name", []))
        image_pixels = orig_w * orig_h

        for i in range(len(detections)):
            returned_label = str(class_names[i]) if i < len(class_names) else f"item_{i}"
            matched_label = self._match_label(returned_label, prompts)
            if matched_label is None:
                continue

            if detections.mask is not None and i < len(detections.mask):
                binary_mask = detections.mask[i].astype(bool)
            else:
                binary_mask = None

            if binary_mask is None:
                continue

            coverage = float(binary_mask.mean())
            if coverage < min_coverage or coverage > max_coverage:
                continue

            score = float(detections.confidence[i]) if detections.confidence is not None else coverage
            if score < float(results[matched_label]["score"]):
                continue

            results[matched_label] = {"mask": binary_mask, "score": score}

            # Convert xyxy box back to box_2d (0-1000 space) for reporting
            x0, y0, x1, y1 = detections.xyxy[i]
            box_2d = [
                round(y0 / orig_h * 1000),
                round(x0 / orig_w * 1000),
                round(y1 / orig_h * 1000),
                round(x1 / orig_w * 1000),
            ]
            summary[matched_label] = {
                "returned_label": returned_label,
                "coverage": round(coverage, 4),
                "score": round(score, 4),
                "box_2d": box_2d,
            }

        self.last_prompt = prompt
        self.last_response_text = response_text
        self.last_items = [{"label": n} for n in class_names]
        self.last_mask_summary = summary
        self.last_debug_payloads = [{"prompt": prompt, "response_text": response_text}]
        self.last_results = results
        self.last_soft_masks = {}
        print(f"  [Gemini segmentation] matched labels: {list(summary.keys())}")
        return results


def save_notebook_style_overlay(
    image_path: Path,
    provider: GeminiSegmentationProvider,
    output_dir: Path,
) -> None:
    """Render masks using supervision exactly like the reference Jupyter notebook.

    Uses sv.BoxAnnotator + sv.LabelAnnotator(smart_position=True) + sv.MaskAnnotator()
    in the same order as notebook cell 23.
    """
    import supervision as sv

    detections = provider.last_sv_detections
    pil_image = provider.last_pil_image

    if detections is None or pil_image is None or len(detections) == 0:
        return

    image_np = np.array(pil_image.convert("RGB"))

    # Build display labels from class_name data (same as notebook)
    class_names = list(detections.data.get("class_name", []))
    labels = [str(n) for n in class_names] if class_names else None

    # Annotate exactly like the notebook: boxes → labels → masks
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(smart_position=True, text_color=sv.Color.BLACK)
    mask_annotator = sv.MaskAnnotator()

    annotated = image_np.copy()
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    if labels:
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    else:
        annotated = label_annotator.annotate(scene=annotated, detections=detections)
    annotated = mask_annotator.annotate(scene=annotated, detections=detections)

    out_path = output_dir / "notebook_style_segmentation.png"
    Image.fromarray(annotated).save(out_path)
    print(f"  [notebook overlay] saved: {out_path.name}")


def save_provider_artifacts(
    provider: GeminiSegmentationProvider,
    output_dir: Path,
    job_id: str,
) -> None:
    payload = {
        "job_id": job_id,
        "model": provider.model_name,
        "prompt": provider.last_prompt,
        "parsed_items": provider.last_items,
        "mask_summary": provider.last_mask_summary,
        "response_text": provider.last_response_text,
        "debug_payloads": provider.last_debug_payloads,
    }
    write_json(output_dir / "gemini_segmentation_response.json", payload)
    write_json(
        output_dir / "segmentation_backend.json",
        {
            "job_id": job_id,
            "segmentation_backend": "gemini",
            "segmentation_model": provider.model_name,
            "matched_labels": list((provider.last_mask_summary or {}).keys()),
        },
    )
    raw_dir = output_dir / "gemini_raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for idx, entry in enumerate(provider.last_debug_payloads or [], start=1):
        label = str(entry.get("target_label") or f"label_{idx}")
        safe_label = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_") or f"label_{idx}"
        (raw_dir / f"{idx:02d}_{safe_label}.txt").write_text(str(entry.get("response_text") or ""))
        write_json(raw_dir / f"{idx:02d}_{safe_label}.json", entry)


def copy_debug_artifacts(output_root: Path, image_dir: Path, job_id: str) -> None:
    debug_dir = output_root / f"production_{job_id}"
    if not debug_dir.exists():
        return

    for path in sorted(debug_dir.glob("*.png")):
        target = image_dir / path.name
        if target.exists():
            continue
        shutil.copy2(path, target)
        if path.name == "sam3_segmentation.png":
            gemini_target = image_dir / "gemini_segmentation.png"
            if not gemini_target.exists():
                shutil.copy2(path, gemini_target)

    for path in sorted(debug_dir.glob("*.npy")):
        target = image_dir / path.name
        if target.exists():
            continue
        shutil.copy2(path, target)


def summarize_result(image_path: Path, result: dict, provider: GeminiSegmentationProvider) -> dict:
    production_debug = result.get("production_debug") or {}
    overall = production_debug.get("overall_confidence") or {}
    nutrition = result.get("nutrition") or {}
    summary = nutrition.get("summary") or {}
    items = nutrition.get("items") or []
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
        "segmentation_backend": "gemini",
        "segmentation_model": provider.model_name,
        "segmentation_items": provider.last_mask_summary,
    }


def print_summary(summary: dict) -> None:
    print(f"\n[{summary['image']}]")
    print(f"  status: {summary['status']}")
    print(f"  job_id: {summary['job_id']}")
    print(f"  meal: {summary.get('meal_name')}")
    print(f"  total_kcal: {summary.get('total_calories_kcal')}")
    print(f"  items: {summary.get('num_food_items')}")
    print(f"  overall_confidence: {summary.get('overall_confidence')}")
    print(f"  segmentation_backend: {summary.get('segmentation_backend')}")
    print(f"  segmentation_model: {summary.get('segmentation_model')}")
    print(f"  segmentation_labels: {list((summary.get('segmentation_items') or {}).keys())}")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (SCRIPT_DIR / run_stamp)
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)

    context_data = load_context_map(args.context)
    images = resolve_images(input_path, args.limit)
    if not images:
        print("No images found to test.")
        return 1

    print("Starting local Gemini segmentation comparison harness...")
    print("Importing production image pipeline modules...")
    import app.pipeline as pipeline_module
    from app.pipeline import NutritionVideoPipeline

    config = build_settings(args.device, run_output_dir)
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required for Gemini segmentation local testing")

    model_manager = ComparisonModelManager(config)
    pipeline = NutritionVideoPipeline(model_manager, config)
    provider = GeminiSegmentationProvider(config.GEMINI_API_KEY, args.model)

    original_run_sam3_image = pipeline_module.run_sam3_image
    pipeline_module.run_sam3_image = provider

    run_summary: list[dict] = []
    failures: list[dict] = []

    try:
        print("=" * 80)
        print("Local Production Pipeline Test (Gemini Segmentation Comparison)")
        print("=" * 80)
        print(f"input: {input_path}")
        print(f"images: {len(images)}")
        print(f"device: {args.device}")
        print(f"output_dir: {run_output_dir}")
        print(f"segmentation_model: {args.model}")

        for index, image_path in enumerate(images, start=1):
            job_id = f"gemini-seg-test-{run_stamp}-{index:03d}"
            image_dir = run_output_dir / image_path.stem
            image_dir.mkdir(parents=True, exist_ok=True)
            user_context = context_for_image(context_data, image_path)

            print(f"\nProcessing {index}/{len(images)}: {image_path.name}")
            if user_context:
                write_json(image_dir / "user_context.json", user_context)

            try:
                # For this comparison harness, fail fast on the production image path.
                # Falling back to the legacy image/video pipeline would pull in SAM2 and
                # invalidate the Gemini-vs-SAM3 comparison we are trying to test.
                print("  [Pipeline] starting production image pipeline with Gemini segmentation override")
                result = pipeline._run_production_image_pipeline(
                    image_path,
                    job_id,
                    user_context=user_context,
                )
                print("  [Pipeline] production image pipeline completed")
                write_json(image_dir / "result.json", result)
                save_provider_artifacts(provider, image_dir, job_id)
                save_notebook_style_overlay(image_path, provider, image_dir)
                copy_debug_artifacts(run_output_dir, image_dir, job_id)

                summary = summarize_result(image_path, result, provider)
                write_json(image_dir / "summary.json", summary)
                run_summary.append(summary)
                print_summary(summary)
            except Exception as exc:
                failure = {
                    "image": image_path.name,
                    "job_id": job_id,
                    "error": str(exc),
                }
                failures.append(failure)
                save_provider_artifacts(provider, image_dir, job_id)
                save_notebook_style_overlay(image_path, provider, image_dir)
                write_json(image_dir / "error.json", failure)
                print(f"  FAILED: {exc}")
    finally:
        pipeline_module.run_sam3_image = original_run_sam3_image

    aggregate = {
        "run_timestamp": run_stamp,
        "input_path": str(input_path),
        "images_processed": len(images),
        "success_count": len(run_summary),
        "failure_count": len(failures),
        "segmentation_model": args.model,
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
    print(f"artifacts_root: {run_output_dir}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
