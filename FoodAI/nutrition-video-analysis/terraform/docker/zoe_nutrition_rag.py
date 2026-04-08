"""
Thin production wrapper around the uploaded RAG bundle.

This keeps the production app interface (`NutritionRAG.get_nutrition_for_food`)
while delegating retrieval to the uploaded files in /RAG:
  - RAG/nutrition_lookup.py
  - RAG/pipeline/clip_rag_module.py
  - RAG/density_db/*
  - RAG/usda_data/*

Only one intentional behavior change is applied: Gemini fallback is restricted to
grounded web-search lookup rather than falling back again to plain Gemini.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class NutritionRAG:
    def __init__(
        self,
        unified_faiss_path: Optional[Path] = None,
        unified_foods_path: Optional[Path] = None,
        unified_food_names_path: Optional[Path] = None,
        fao_faiss_path: Optional[Path] = None,
        fao_density_path: Optional[Path] = None,
        fao_names_path: Optional[Path] = None,
        usda_faiss_path: Optional[Path] = None,
        usda_foods_path: Optional[Path] = None,
        usda_names_path: Optional[Path] = None,
        usda_density_faiss_path: Optional[Path] = None,
        usda_density_path: Optional[Path] = None,
        usda_density_names_path: Optional[Path] = None,
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-flash-latest",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        clip_model_name: str = "openai/clip-vit-base-patch32",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self._repo_root = Path(__file__).resolve().parents[4]
        self._rag_root = self._repo_root / "RAG"
        self._gemini_api_key = gemini_api_key
        self._gemini_model = gemini_model
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._rag_config = None
        self._rag_gemini = None
        self._rag_clip = None
        self._rag_lookup = None

    def _load_module(self, module_name: str, module_path: Path):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module {module_name} from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _grounded_lookup_only(self, ingredient_name: str, retries: int = 2):
        """Use the uploaded Gemini helper, but disable the plain-Gemini fallback."""
        gemini = self._rag_gemini
        prompt = gemini._DENSITY_PROMPT.format(ingredient=ingredient_name)
        api_key = getattr(gemini, "GEMINI_API_KEY", None) or self._gemini_api_key
        model_name = getattr(gemini, "GEMINI_MODEL", None) or self._gemini_model

        if not api_key:
            return None

        def _extract_grounding_metadata(response):
            def _as_plain(obj):
                if obj is None or isinstance(obj, (str, int, float, bool)):
                    return obj
                if isinstance(obj, list):
                    return [_as_plain(item) for item in obj]
                if isinstance(obj, dict):
                    return {key: _as_plain(value) for key, value in obj.items()}
                if hasattr(obj, "model_dump"):
                    try:
                        return _as_plain(obj.model_dump())
                    except Exception:
                        pass
                if hasattr(obj, "to_dict"):
                    try:
                        return _as_plain(obj.to_dict())
                    except Exception:
                        pass
                if hasattr(obj, "__dict__"):
                    try:
                        return _as_plain(vars(obj))
                    except Exception:
                        pass
                return str(obj)

            data = _as_plain(response)
            if not isinstance(data, dict):
                return None

            candidates = data.get("candidates") or []
            candidate = candidates[0] if candidates else {}
            grounding = candidate.get("grounding_metadata") or candidate.get("groundingMetadata") or {}
            if not isinstance(grounding, dict):
                return None

            chunks = grounding.get("grounding_chunks") or grounding.get("groundingChunks") or []
            queries = grounding.get("web_search_queries") or grounding.get("webSearchQueries") or []
            supports = grounding.get("grounding_supports") or grounding.get("groundingSupports") or []

            sources = []
            seen = set()
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                web = chunk.get("web") or {}
                if not isinstance(web, dict):
                    continue
                url = (web.get("uri") or web.get("url") or "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                sources.append(
                    {
                        "title": (web.get("title") or "").strip() or None,
                        "url": url,
                        "domain": urlparse(url).netloc or None,
                    }
                )

            if not (sources or queries or supports):
                return None
            return {
                "grounded": True,
                "queries": [q for q in queries if isinstance(q, str) and q.strip()],
                "sources": sources,
            }

        try:
            from google import genai as genai_new
            from google.genai import types

            client = genai_new.Client(api_key=api_key)
            tools = None
            if hasattr(types, "Tool"):
                if hasattr(types, "GoogleSearch"):
                    tools = [types.Tool(google_search=types.GoogleSearch())]
                elif hasattr(types, "GoogleSearchRetrieval"):
                    tools = [types.Tool(google_search_retrieval=types.GoogleSearchRetrieval())]

            if not tools:
                logger.warning(f"[Gemini grounding only] {ingredient_name}: Google Search tool unavailable in SDK")
                return None

            for _attempt in range(retries):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.0, tools=tools),
                    )
                    raw = gemini._strip_json(response.text or "")
                    result = json.loads(raw)
                    if "density_g_ml" in result and "calories_per_100g" in result:
                        result["lookup_method"] = "gemini_grounding"
                        metadata = _extract_grounding_metadata(response)
                        if metadata:
                            result["_grounding_metadata"] = metadata
                        return result
                except Exception:
                    continue
        except Exception as err:
            logger.warning(f"[Gemini grounding only] {ingredient_name}: {err}")
        return None

    def load(self):
        if not self._rag_root.exists():
            raise FileNotFoundError(f"Expected uploaded RAG folder at {self._rag_root}")

        logger.info(f"Loading uploaded RAG bundle from {self._rag_root}")

        # Ensure the uploaded bundle wins module resolution for its own absolute imports.
        sys.path.insert(0, str(self._rag_root))

        self._rag_config = self._load_module("config", self._rag_root / "config.py")
        self._rag_gemini = self._load_module("gemini_labeler", self._rag_root / "gemini_labeler.py")
        # Allow `import pipeline.clip_rag_module` inside nutrition_lookup.py
        rag_pipeline = self._load_module("pipeline", self._rag_root / "pipeline" / "__init__.py")
        rag_pipeline.__path__ = [str((self._rag_root / "pipeline").resolve())]
        self._rag_clip = self._load_module(
            "pipeline.clip_rag_module",
            self._rag_root / "pipeline" / "clip_rag_module.py",
        )
        self._rag_lookup = self._load_module("nutrition_lookup", self._rag_root / "nutrition_lookup.py")

        # Override runtime Gemini config to use the app-provided key/model.
        if self._gemini_api_key:
            self._rag_config.GEMINI_API_KEY = self._gemini_api_key
            self._rag_gemini.GEMINI_API_KEY = self._gemini_api_key
        if self._gemini_model:
            self._rag_config.GEMINI_MODEL = self._gemini_model
            self._rag_gemini.GEMINI_MODEL = self._gemini_model

        # Keep uploaded retrieval logic, but limit fallback to grounded search only.
        self._rag_lookup.lookup_density_grounded = self._grounded_lookup_only

        self._rag_lookup._load_models(self._device)
        logger.info("Uploaded RAG bundle loaded successfully")

    def _clip_image_vec(self, crop_image: Optional[Image.Image]) -> Optional[np.ndarray]:
        if crop_image is None:
            return None
        if self._rag_lookup is None:
            raise RuntimeError("RAG bundle not loaded")

        clip_model = self._rag_lookup._clip_model
        clip_processor = self._rag_lookup._clip_processor
        if clip_model is None or clip_processor is None:
            self._rag_lookup._load_models(self._device)
            clip_model = self._rag_lookup._clip_model
            clip_processor = self._rag_lookup._clip_processor

        vecs = self._rag_clip.encode_image_clip(
            [crop_image.convert("RGB")],
            clip_model,
            clip_processor,
            self._device,
        )
        if vecs.shape[0] == 0:
            return None
        return vecs[0]

    def get_grounding_metadata(self, food_name: str, field: str):
        cache_key = f"{food_name}::{field}"
        metadata = getattr(self, "_grounding_metadata", {}).get(cache_key)
        return json.loads(json.dumps(metadata)) if metadata else None

    def _lookup_ingredient(self, ingredient_name: str, crop_image: Optional[Image.Image]):
        clip_image_vec = self._clip_image_vec(crop_image)
        return self._rag_lookup.lookup_ingredient(
            ingredient_name,
            clip_image_vec=clip_image_vec,
            device=self._device,
        )

    def _find_usda_entry_by_description(self, description: Optional[str]) -> Optional[dict]:
        if not description or self._rag_lookup is None:
            return None
        foods = getattr(self._rag_lookup, "_usda_foods", None) or []
        for food in foods:
            if (food.get("description") or "").strip().lower() == description.strip().lower():
                return food
        return None

    @staticmethod
    def _score_usda_description(query: str, description: str) -> float:
        query_text = (query or "").strip().lower()
        desc_text = (description or "").strip().lower()
        if not query_text or not desc_text:
            return float("-inf")

        query_words = [w for w in re.findall(r"[a-z0-9]+", query_text) if len(w) > 1]
        desc_words = set(re.findall(r"[a-z0-9]+", desc_text))
        score = 0.0
        if query_text == desc_text:
            score += 20.0
        if query_text in desc_text:
            score += 8.0
        for word in query_words:
            if word in desc_words:
                score += 2.5
        if "powder" in desc_text:
            score -= 8.0
        if "sauce" in desc_text and "sauce" not in query_text:
            score -= 4.0
        if "frozen" in desc_text and "frozen" not in query_text:
            score -= 2.0
        return score

    def get_verifier_candidates(
        self,
        food_name: str,
        chosen_description: Optional[str],
        rag_candidates: Optional[list[dict]],
        limit: int = 8,
    ) -> list[dict]:
        foods = getattr(self._rag_lookup, "_usda_foods", None) or []
        descriptions: list[str] = []

        def add_description(text: Optional[str]):
            normalized = (text or "").strip()
            if normalized and normalized not in descriptions:
                descriptions.append(normalized)

        add_description(chosen_description)
        for candidate in rag_candidates or []:
            add_description(candidate.get("description"))

        lexical_ranked = sorted(
            foods,
            key=lambda food: self._score_usda_description(food_name, food.get("description") or ""),
            reverse=True,
        )
        for food in lexical_ranked[: limit * 3]:
            add_description(food.get("description"))
            if len(descriptions) >= limit:
                break

        verifier_candidates = []
        for description in descriptions[:limit]:
            entry = self._find_usda_entry_by_description(description)
            if not entry:
                continue
            verifier_candidates.append(
                {
                    "description": description,
                    "calories_per_100g": float(entry.get("calories_per_100g", 0.0)),
                    "fat_per_100g": float(entry.get("fat_per_100g", 0.0)),
                    "carb_per_100g": float(entry.get("carb_per_100g", 0.0)),
                    "protein_per_100g": float(entry.get("protein_per_100g", 0.0)),
                }
            )
        return verifier_candidates

    def get_usda_candidate_by_description(self, description: str) -> Optional[dict]:
        entry = self._find_usda_entry_by_description(description)
        if not entry:
            return None
        return {
            "description": entry.get("description"),
            "calories_per_100g": float(entry.get("calories_per_100g", 0.0)),
            "fat_per_100g": float(entry.get("fat_per_100g", 0.0)),
            "carb_per_100g": float(entry.get("carb_per_100g", 0.0)),
            "protein_per_100g": float(entry.get("protein_per_100g", 0.0)),
        }

    def get_density(self, food_name: str, crop_image: Optional[Image.Image] = None) -> float:
        result = self._lookup_ingredient(food_name, crop_image)
        return float(result.get("density_g_ml", 1.0))

    def get_calories_per_100g(self, food_name: str, crop_image: Optional[Image.Image] = None) -> float:
        result = self._lookup_ingredient(food_name, crop_image)
        return float(result.get("calories_per_100g", 100.0))

    def get_nutrition(self, food_name: str, volume_ml: float, crop_image: Optional[Image.Image] = None) -> dict:
        return self.get_nutrition_for_food(food_name, volume_ml, crop_image=crop_image)

    def get_nutrition_for_food(
        self,
        food_name: str,
        volume_ml: float,
        mass_g: Optional[float] = None,
        quantity: int = 1,
        crop_image: Optional[Image.Image] = None,
    ) -> dict:
        result = self._lookup_ingredient(food_name, crop_image)
        if result.get("_grounding_metadata"):
            if not hasattr(self, "_grounding_metadata"):
                self._grounding_metadata = {}
            self._grounding_metadata[f"{food_name}::density_g_ml"] = result["_grounding_metadata"]
            self._grounding_metadata[f"{food_name}::calories_per_100g"] = result["_grounding_metadata"]

        density = float(result.get("density_g_ml", 1.0))
        calories_per_100g = float(result.get("calories_per_100g", 100.0))
        if mass_g is None or mass_g <= 0:
            mass_g = volume_ml * density
        total_calories = (float(mass_g) / 100.0) * calories_per_100g

        lookup_method = result.get("lookup_method", "default_fallback")
        if lookup_method == "clip_rag":
            calorie_source = "usda_clip_rag"
        elif lookup_method == "lexical_usda":
            calorie_source = "usda_lexical_override"
        elif lookup_method in {"gemini_grounding", "gemini_google_search"}:
            calorie_source = "gemini_grounding"
        elif lookup_method == "clip_rag_low_conf":
            calorie_source = "usda_clip_rag_low_conf"
        else:
            calorie_source = "fallback_default"

        density_match = result.get("density_match")
        gemini_density_used = density_match is not None and density_match == result.get("source")
        if density_match and "heuristic:" in str(density_match).lower():
            density_source = "heuristic_density"
        elif gemini_density_used:
            density_source = "gemini_grounding"
        elif density_match:
            density_source = "density_faiss"
        else:
            density_source = "fallback_default"

        return {
            "food_name": food_name,
            "quantity": quantity,
            "volume_ml": volume_ml,
            "density_g_per_ml": density,
            "mass_g": round(float(mass_g), 1),
            "calories_per_100g": round(calories_per_100g, 1),
            "total_calories": round(total_calories, 1),
            "density_matched": density_match,
            "density_source": density_source,
            "calorie_matched": result.get("description"),
            "calorie_source": calorie_source,
            "lookup_method": lookup_method,
            "rerank_score": result.get("rerank_score"),
            "faiss_score": result.get("faiss_score"),
            "rag_candidates": result.get("rag_candidates"),
            "density_grounding_metadata": None,
            "calorie_grounding_metadata": None,
        }
