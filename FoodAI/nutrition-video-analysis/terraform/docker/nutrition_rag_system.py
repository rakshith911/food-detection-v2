"""
Nutrition RAG System
Unified FAISS index (USDA + CoFID) with cross-encoder re-ranking, plus a
separate FAO density fallback.
Google Search grounding (via Gemini) is the only web fallback.
"""
import csv
import json
import logging
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import faiss
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

# Minimum cosine similarity from FAISS to consider a candidate.
_MIN_FAISS_SIM = 0.45
_MIN_RERANK_SCORE = -5.0

# Per-source minimum similarity — FAO is a small density-only fallback set and
# needs a stricter threshold than the main USDA/CoFID retrieval DB.
_MIN_SIM_BY_SOURCE = {'fao': 0.65, 'usda': 0.50, 'cofid': 0.60}


class NutritionLookupError(LookupError):
    """Raised when nutrition lookup cannot resolve required data."""

    def __init__(self, food_name: str, missing_fields: list[str]):
        self.food_name = food_name
        self.missing_fields = missing_fields
        joined = ", ".join(missing_fields)
        super().__init__(f"Could not resolve {joined} for '{food_name}'")

class NutritionRAG:
    """
    Unified nutrition lookup:
      1. FAISS search over unified USDA + CoFID for kcal and density
      2. Cross-encoder re-ranking of top candidates
      3. Separate FAO fallback for density only
      4. Gemini grounding (Google Search) as sole web fallback
    """

    def __init__(
        self,
        # Unified index (preferred)
        unified_faiss_path: Optional[Path] = None,
        unified_foods_path: Optional[Path] = None,
        unified_food_names_path: Optional[Path] = None,
        # Legacy separate indexes (used if unified not provided)
        fao_faiss_path: Optional[Path] = None,
        fao_density_path: Optional[Path] = None,
        fao_names_path: Optional[Path] = None,
        branded_foods_path: Optional[Path] = None,
        usda_faiss_path: Optional[Path] = None,
        usda_foods_path: Optional[Path] = None,
        usda_names_path: Optional[Path] = None,
        usda_density_faiss_path: Optional[Path] = None,
        usda_density_path: Optional[Path] = None,
        usda_density_names_path: Optional[Path] = None,
        gemini_density_cache_path: Optional[Path] = None,
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-flash-latest",
        embedding_model: str = "all-MiniLM-L6-v2",
        clip_model_name: str = "openai/clip-vit-base-patch32",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self._unified_faiss_path   = Path(unified_faiss_path)   if unified_faiss_path   else None
        self._unified_foods_path   = Path(unified_foods_path)   if unified_foods_path   else None
        self._unified_names_path   = Path(unified_food_names_path) if unified_food_names_path else None

        # Legacy paths kept for backward compat
        self._fao_faiss_path    = Path(fao_faiss_path)    if fao_faiss_path    else None
        self._fao_density_path  = Path(fao_density_path)  if fao_density_path  else None
        self._fao_names_path    = Path(fao_names_path)    if fao_names_path    else None
        self._branded_foods_path = Path(branded_foods_path) if branded_foods_path else None
        self._usda_faiss_path   = Path(usda_faiss_path)   if usda_faiss_path   else None
        self._usda_foods_path   = Path(usda_foods_path)   if usda_foods_path   else None
        self._usda_names_path   = Path(usda_names_path)   if usda_names_path   else None
        self._gemini_density_cache_path = (
            Path(gemini_density_cache_path) if gemini_density_cache_path else None
        )

        self._gemini_api_key    = gemini_api_key
        self._gemini_model      = gemini_model
        self._embedding_model   = embedding_model
        self._clip_model_name   = clip_model_name
        self._cross_encoder_model = cross_encoder_model

        # Runtime objects (populated in load())
        self._embedder: Optional[SentenceTransformer] = None
        self._clip_model: Optional[CLIPModel] = None
        self._clip_processor: Optional[CLIPProcessor] = None
        self._cross_encoder: Optional[CrossEncoder] = None
        self._clip_index = None
        self._clip_name_embeddings: Optional[np.ndarray] = None
        self._unified_index = None
        self._unified_foods: list = []
        self._unified_names: list = []

        # Legacy objects (populated if unified not available)
        self._fao_index   = None
        self._fao_density = []
        self._fao_names   = []
        self._branded_foods: Optional[list[dict]] = None
        self._branded_token_index: dict[str, list[int]] = {}  # token → entry indices
        self._branded_load_lock = threading.Lock()
        self._usda_index  = None
        self._usda_foods  = []
        self._usda_names  = []

        self._use_unified = False
        self._gemini_cache: dict = {}  # food_name+field -> value
        self._gemini_grounding_metadata: dict = {}  # food_name+field -> grounding metadata
        self._gemini_persistent_cache: dict = {}
        self._usda_desc_to_fdc_id: Optional[dict[str, str]] = None
        self._usda_portion_index: Optional[dict[str, list[dict[str, str]]]] = None
        self._usda_measure_units: Optional[dict[str, str]] = None
        self._usda_portion_data_loaded = False

    # ──────────────────────────────────────────────────────────────────────────
    # Loading
    # ──────────────────────────────────────────────────────────────────────────

    def load(self):
        """Load indexes and retrieval/reranking models."""
        logger.info("Loading NutritionRAG...")
        self._load_persistent_gemini_cache()

        self._embedder = SentenceTransformer(self._embedding_model)
        logger.info(f"  Embedding model: {self._embedding_model}")
        self._cross_encoder = CrossEncoder(self._cross_encoder_model)
        logger.info(f"  Cross-encoder:   {self._cross_encoder_model}")
        self._clip_processor = CLIPProcessor.from_pretrained(self._clip_model_name)
        self._clip_model = CLIPModel.from_pretrained(self._clip_model_name, use_safetensors=True)
        self._clip_model.eval()
        logger.info(f"  CLIP model:      {self._clip_model_name}")

        # Prefer unified index
        if (self._unified_faiss_path and self._unified_faiss_path.exists()
                and self._unified_foods_path and self._unified_foods_path.exists()):
            self._unified_index = faiss.read_index(str(self._unified_faiss_path))
            with open(self._unified_foods_path) as f:
                self._unified_foods = json.load(f)
            with open(self._unified_names_path) as f:
                self._unified_names = json.load(f)
            self._use_unified = True
            logger.info(f"  Unified index:   {len(self._unified_foods)} entries (USDA+CoFID)")
            self._build_clip_index()
            if self._fao_faiss_path and self._fao_faiss_path.exists():
                self._fao_index = faiss.read_index(str(self._fao_faiss_path))
                with open(self._fao_density_path) as f:
                    self._fao_density = json.load(f)
                with open(self._fao_names_path) as f:
                    self._fao_names = json.load(f)
                logger.info(f"  FAO fallback:   {len(self._fao_density)} density entries")
        else:
            # Fall back to separate FAO + USDA indexes
            logger.info("  Unified index not found — falling back to separate FAO + USDA indexes")
            if self._fao_faiss_path and self._fao_faiss_path.exists():
                self._fao_index = faiss.read_index(str(self._fao_faiss_path))
                with open(self._fao_density_path) as f:
                    self._fao_density = json.load(f)
                with open(self._fao_names_path) as f:
                    self._fao_names = json.load(f)
                logger.info(f"  FAO index:       {len(self._fao_density)} entries")
            if self._usda_faiss_path and self._usda_faiss_path.exists():
                self._usda_index = faiss.read_index(str(self._usda_faiss_path))
                with open(self._usda_foods_path) as f:
                    self._usda_foods = json.load(f)
                with open(self._usda_names_path) as f:
                    self._usda_names = json.load(f)
                logger.info(f"  USDA index:      {len(self._usda_foods)} entries")

        logger.info("NutritionRAG ready")

    def _ensure_branded_foods_loaded(self) -> None:
        if self._branded_foods is not None:
            return
        with self._branded_load_lock:
            if self._branded_foods is not None:  # double-check after acquiring lock
                return
            self._branded_foods = []
            if not self._branded_foods_path or not self._branded_foods_path.exists():
                return
            try:
                with open(self._branded_foods_path, encoding="utf-8") as f:
                    self._branded_foods = json.load(f)
                logger.info("  Branded fallback: %s entries", len(self._branded_foods))
                self._build_branded_token_index()
            except Exception as exc:
                logger.warning("Failed to load branded fallback dataset: %s", exc)
                self._branded_foods = []

    def _build_branded_token_index(self) -> None:
        """
        Build an inverted token index over branded food descriptions.
        Maps each meaningful token → list of entry indices, enabling O(candidates)
        lookup instead of O(1.99M) linear scan.  Includes both exact tokens and
        singular/plural variants so near-token-match (apple↔apples) is preserved.
        """
        idx: dict[str, list[int]] = {}
        signal_words = self._SIGNAL_WORDS
        generic_words = self._GENERIC_WORDS
        for i, entry in enumerate(self._branded_foods):
            desc = (entry.get("description") or "").lower()
            tokens = set(re.sub(r'[^\w ]', ' ', desc).split())
            for tok in tokens:
                if not (len(tok) >= 4 or tok in signal_words):
                    continue
                if tok in generic_words:
                    continue
                idx.setdefault(tok, []).append(i)
                # Also index singular (strip trailing 's') so "apple" finds "apples" entries
                if tok.endswith('s') and len(tok) >= 5:
                    idx.setdefault(tok[:-1], []).append(i)
        self._branded_token_index = idx
        logger.info("  Branded token index: %d unique tokens", len(idx))

    def _lookup_branded_fallback(
        self,
        food_name: str,
        field: str,
    ):
        self._ensure_branded_foods_loaded()
        if not self._branded_foods:
            return None, None, None, None

        normalized = self._normalize_food_name(food_name)
        query_tokens = self._tokenize_words(normalized, keep_generic=True)

        # Resolve candidate indices via inverted token index (O(k) instead of O(1.99M)).
        # Use the same token-quality criteria as _words_overlap so no candidates are missed.
        index_query_tokens: set[str] = set()
        for tok in query_tokens:
            if not (len(tok) >= 4 or tok in self._SIGNAL_WORDS):
                continue
            if tok in self._GENERIC_WORDS:
                continue
            index_query_tokens.add(tok)
            # Also look up plural form so "apple" retrieves "apples" entries
            if tok.endswith('s') and len(tok) >= 5:
                index_query_tokens.add(tok[:-1])
            else:
                index_query_tokens.add(tok + 's')

        if self._branded_token_index and index_query_tokens:
            candidate_indices: set[int] = set()
            for tok in index_query_tokens:
                candidate_indices.update(self._branded_token_index.get(tok, []))
        else:
            # Index not built (e.g. file absent) — fall back to full scan
            candidate_indices = set(range(len(self._branded_foods)))

        candidates = []

        for i in candidate_indices:
            entry = self._branded_foods[i]
            value = entry.get(field)
            if value is None:
                continue
            if field == "density_g_ml" and not self._is_reliable_density_entry(entry):
                continue

            matched = entry.get("description") or ""
            if not matched:
                continue
            matched_lower = matched.lower()
            matched_norm = self._normalize_food_name(matched_lower)
            if not self._words_overlap(normalized, matched_lower):
                continue

            text_blob = " | ".join(
                str(entry.get(key) or "")
                for key in ("description", "ingredients_text", "household_serving_fulltext", "brand_name")
            ).lower()
            matched_tokens = self._tokenize_words(text_blob, keep_generic=True)

            score = 0.0
            if matched_norm == normalized:
                score += 120.0
            if normalized and normalized in matched_norm:
                score += 50.0

            overlap = len(query_tokens & matched_tokens)
            missing = len(query_tokens - matched_tokens)
            score += overlap * 8.0
            score -= missing * 6.0
            score += self._compatibility_adjustment(normalized, matched_lower, field=field)

            if self._is_mixed_dish_candidate_for_condiment_query(normalized, matched_lower):
                score -= 40.0

            if score < 20.0:
                continue

            candidates.append((score, float(value), matched, entry))

        if not candidates:
            return None, None, None, None

        candidates.sort(reverse=True, key=lambda item: item[0])
        best_score, value, matched, entry = candidates[0]
        source = "branded_lexical"
        if field == "density_g_ml":
            source += f",{entry.get('density_method', 'unknown')}"
        logger.info(
            f"[{field}] '{food_name}' branded fallback → '{matched}' "
            f"({source}, score={best_score:.2f}): {value}"
        )
        return value, matched, source, entry

    @staticmethod
    def _normalize_cache_key(food_name: str, field: str) -> str:
        return f"{food_name.strip().lower()}::{field}"

    def _load_persistent_gemini_cache(self) -> None:
        self._gemini_persistent_cache = {}
        if not self._gemini_density_cache_path or not self._gemini_density_cache_path.exists():
            return
        try:
            with open(self._gemini_density_cache_path, encoding="utf-8") as f:
                payload = json.load(f)
            entries = payload.get("entries", []) if isinstance(payload, dict) else payload
            if not isinstance(entries, list):
                logger.warning(
                    "Gemini density cache file had unexpected format: %s",
                    self._gemini_density_cache_path,
                )
                return
            loaded = 0
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                query = (entry.get("normalized_query") or entry.get("query") or "").strip().lower()
                field = (entry.get("field") or "").strip()
                value = entry.get("value")
                if not query or not field or value is None:
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                key = self._normalize_cache_key(query, field)
                self._gemini_persistent_cache[key] = {
                    **entry,
                    "normalized_query": query,
                    "field": field,
                    "value": numeric_value,
                }
                if entry.get("grounding_metadata"):
                    self._gemini_grounding_metadata[key] = json.loads(
                        json.dumps(entry["grounding_metadata"])
                    )
                loaded += 1
            logger.info("  Gemini density cache: %s entries", loaded)
        except Exception as exc:
            logger.warning("Failed to load Gemini density cache: %s", exc)

    def _persist_gemini_cache_entry(
        self,
        food_name: str,
        field: str,
        value: float,
        grounding_metadata: Optional[dict] = None,
        source_type: str = "gemini_grounding",
    ) -> None:
        if not self._gemini_density_cache_path:
            return
        key = self._normalize_cache_key(food_name, field)
        entry = {
            "query": food_name,
            "normalized_query": food_name.strip().lower(),
            "field": field,
            "value": float(value),
            "source_type": source_type,
            "model": self._gemini_model,
            "grounding_metadata": json.loads(json.dumps(grounding_metadata)) if grounding_metadata else None,
        }
        self._gemini_persistent_cache[key] = entry
        try:
            self._gemini_density_cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "entries": sorted(
                    self._gemini_persistent_cache.values(),
                    key=lambda item: (item.get("field") or "", item.get("normalized_query") or ""),
                ),
            }
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=str(self._gemini_density_cache_path.parent),
                delete=False,
            ) as tmp:
                json.dump(payload, tmp, indent=2, ensure_ascii=True)
                tmp.write("\n")
                temp_path = Path(tmp.name)
            temp_path.replace(self._gemini_density_cache_path)
        except Exception as exc:
            logger.warning("Failed to persist Gemini density cache entry for '%s': %s", food_name, exc)
            try:
                if 'temp_path' in locals() and temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass

    def _get_cached_gemini_value(self, food_name: str, field: str) -> Optional[float]:
        runtime_key = f"{food_name}::{field}"
        if runtime_key in self._gemini_cache:
            value = self._gemini_cache[runtime_key]
            metadata = self._gemini_grounding_metadata.get(runtime_key)
            if self._is_reasonable_gemini_value(food_name, field, value, metadata):
                return value
            self._gemini_cache.pop(runtime_key, None)
            self._gemini_grounding_metadata.pop(runtime_key, None)
        persistent_key = self._normalize_cache_key(food_name, field)
        cached = self._gemini_persistent_cache.get(persistent_key)
        if not cached:
            return None
        value = cached.get("value")
        if value is None:
            return None
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        metadata = cached.get("grounding_metadata")
        if not self._is_reasonable_gemini_value(food_name, field, numeric_value, metadata):
            return None
        self._gemini_cache[runtime_key] = numeric_value
        if metadata:
            self._gemini_grounding_metadata[runtime_key] = json.loads(json.dumps(metadata))
        return numeric_value

    @staticmethod
    def _to_numpy(features) -> np.ndarray:
        """Extract a numpy array from either a raw tensor or a ModelOutput object (transformers 5.x)."""
        import torch as _torch
        if isinstance(features, _torch.Tensor):
            return features.detach().cpu().numpy().astype(np.float32)
        # transformers 5.x returns BaseModelOutputWithPooling / similar dataclasses
        for attr in ("pooler_output", "last_hidden_state", "text_embeds", "image_embeds"):
            val = getattr(features, attr, None)
            if val is not None and isinstance(val, _torch.Tensor):
                t = val.detach().cpu().numpy().astype(np.float32)
                # pooler_output is (batch, hidden); last_hidden_state is (batch, seq, hidden) — mean-pool
                return t.mean(axis=1) if t.ndim == 3 else t
        raise ValueError(f"Cannot extract tensor from CLIP output: {type(features)}")

    def _build_clip_index(self) -> None:
        if not self._unified_names or self._clip_model is None or self._clip_processor is None:
            return
        logger.info("  Building CLIP retrieval index from unified food names...")
        batch_size = 256
        embeddings = []
        for start in range(0, len(self._unified_names), batch_size):
            batch_names = self._unified_names[start:start + batch_size]
            inputs = self._clip_processor(text=batch_names, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                text_features = self._clip_model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
            batch_emb = self._to_numpy(text_features)
            norms = np.linalg.norm(batch_emb, axis=1, keepdims=True)
            batch_emb = batch_emb / np.clip(norms, 1e-8, None)
            embeddings.append(batch_emb)

        self._clip_name_embeddings = np.vstack(embeddings).astype(np.float32)
        dim = int(self._clip_name_embeddings.shape[1])
        self._clip_index = faiss.IndexFlatIP(dim)
        self._clip_index.add(self._clip_name_embeddings)
        logger.info(f"  CLIP index:      {self._clip_index.ntotal} entries ({dim} dims)")

    # ──────────────────────────────────────────────────────────────────────────
    # Label normalisation
    # ──────────────────────────────────────────────────────────────────────────

    _LABEL_ALIASES = [
        (r'\bcrumb\s+bar\b', 'crumble'),
        (r'\bcrumb\b',       'crumble'),
        (r'\bcrisp\b',       'crumble'),
        (r'\bfritter\b',     'pastry'),
        (r'\bshortbread\b',  'cookie'),
        # Cooking oils — normalise spray/brand variants to plain oil
        (r'\bcooking\s+oil\b',          'vegetable oil'),
        (r'\bcooking\s+spray\b',        'vegetable oil'),
        (r'\bpam\b',                    'vegetable oil'),
        (r'\bolive\s+oil\b',            'olive oil'),
        (r'\bsunflower\s+oil\b',        'sunflower oil'),
        # Hot/chili sauce — normalise so "hot" is preserved as signal
        (r'\bred\s+hot\s+sauce\b',      'hot chili sauce'),
        (r'\bchili\s+sauce\b',          'hot chili sauce'),
        # Fresh carrots (not canned)
        (r'\bshredded\s+carrots?\b',    'carrots raw'),
        (r'\bgrated\s+carrots?\b',      'carrots raw'),
    ]
    _FORM_WORDS_RE = (
        r'\b(bar|slice|slices|piece|pieces|wedge|wedges|portion|portions|'
        r'serving|servings|ball|balls|patty|patties|log|logs|chunk|chunks)\b'
    )

    @classmethod
    def _normalize_food_name(cls, name: str) -> str:
        name = re.sub(r'\([^)]*\)', '', name)
        name = name.replace(',', ' ')
        for pattern, replacement in cls._LABEL_ALIASES:
            name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
        qualifiers = (
            r'\b(chopped|sliced|diced|cooked|raw|fresh|dried|grilled|fried|'
            r'boiled|steamed|baked|roasted|sautéed|sauteed|mashed|whole|'
            r'halved|quartered|shredded|grated|minced|pitted|peeled|'
            r'crumbled|crushed|mixed|assorted)\b'
        )
        name = re.sub(qualifiers, '', name, flags=re.IGNORECASE)
        name = re.sub(cls._FORM_WORDS_RE, '', name, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', name).strip().lower()

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value in (None, ""):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_portion_amount(text: str) -> Optional[float]:
        text = (text or '').strip().lower()
        if not text:
            return None

        mixed = re.match(r'^(\d+)\s+(\d+)/(\d+)\b', text)
        if mixed:
            whole = float(mixed.group(1))
            num = float(mixed.group(2))
            den = float(mixed.group(3))
            if den:
                return whole + (num / den)

        fraction = re.match(r'^(\d+)/(\d+)\b', text)
        if fraction:
            num = float(fraction.group(1))
            den = float(fraction.group(2))
            if den:
                return num / den

        numeric = re.match(r'^(\d+(?:\.\d+)?)\b', text)
        if numeric:
            return float(numeric.group(1))

        return None

    @classmethod
    def _portion_volume_ml(cls, row: dict[str, str], measure_name: str) -> Optional[tuple[float, str]]:
        text = " ".join(
            part for part in (
                measure_name,
                row.get('modifier') or '',
                row.get('portion_description') or '',
            )
            if part
        ).lower()

        amount = cls._safe_float(row.get('amount'))
        if amount is None or amount <= 0:
            amount = cls._parse_portion_amount(row.get('portion_description') or '')
        if amount is None or amount <= 0:
            amount = cls._parse_portion_amount(text)
        if amount is None or amount <= 0:
            return None

        unit_candidates = (
            ('milliliter', 1.0, 'ml'),
            ('millilitre', 1.0, 'ml'),
            (' ml', 1.0, 'ml'),
            ('cup', 236.588, 'cup'),           # USDA: 1 cup = 236.588 mL exactly
            ('tablespoon', 14.78675, 'tbsp'),   # USDA: 1 cup / 16 = 14.78675 mL
            ('tbsp', 14.78675, 'tbsp'),
            ('teaspoon', 4.92892, 'tsp'),        # USDA: 1 cup / 48 = 4.92892 mL
            ('tsp', 4.92892, 'tsp'),
            ('fluid ounce', 29.5735, 'fl_oz'),  # USDA: 1 cup / 8 = 29.5735 mL
            ('fl oz', 29.5735, 'fl_oz'),
            ('liter', 1000.0, 'liter'),
            ('litre', 1000.0, 'liter'),
            (' l', 1000.0, 'liter'),
        )
        for needle, multiplier, unit_label in unit_candidates:
            if needle in text:
                return amount * multiplier, unit_label
        return None

    @staticmethod
    def _usda_portion_sort_key(candidate: tuple[float, str, dict[str, str]]) -> tuple[int, float, float]:
        _density, unit_label, row = candidate
        priority = {
            'ml': 5,
            'tbsp': 4,
            'tsp': 4,
            'fl_oz': 4,
            'cup': 3,
            'liter': 2,
        }.get(unit_label, 0)
        amount = NutritionRAG._safe_float(row.get('amount')) or 0.0
        data_points = NutritionRAG._safe_float(row.get('data_points')) or 0.0
        return priority, data_points, -abs(amount - 1.0)

    def _load_usda_portion_data(self) -> None:
        if self._usda_portion_data_loaded:
            return

        self._usda_portion_data_loaded = True
        self._usda_desc_to_fdc_id = {}
        self._usda_portion_index = {}
        self._usda_measure_units = {}

        _file_resolved = Path(__file__).resolve()
        _ancestors = list(_file_resolved.parents)
        raw_dir_candidates = []
        if len(_ancestors) > 4:
            raw_dir_candidates += [
                _ancestors[4] / 'FoodData_Central_csv_2025-12-18',
                _ancestors[4] / 'usda_data' / 'usda_raw',
            ]
        raw_dir_candidates.append(_file_resolved.parent / 'data' / 'usda_raw')
        raw_dir = next((path for path in raw_dir_candidates if path.exists()), None)
        if raw_dir is None:
            logger.info("USDA raw portion tables not found; skipping USDA portion density fallback")
            return

        food_csv = raw_dir / 'food.csv'
        portion_csv = raw_dir / 'food_portion.csv'
        measure_csv = raw_dir / 'measure_unit.csv'
        if not (food_csv.exists() and portion_csv.exists() and measure_csv.exists()):
            logger.info("USDA raw portion tables incomplete; skipping USDA portion density fallback")
            return

        with measure_csv.open() as f:
            for row in csv.DictReader(f):
                unit_id = (row.get('id') or '').strip()
                if unit_id:
                    self._usda_measure_units[unit_id] = (row.get('name') or '').strip()

        with food_csv.open() as f:
            for row in csv.DictReader(f):
                desc = (row.get('description') or '').strip()
                fdc_id = (row.get('fdc_id') or '').strip()
                if desc and fdc_id:
                    self._usda_desc_to_fdc_id.setdefault(desc, fdc_id)

        with portion_csv.open() as f:
            for row in csv.DictReader(f):
                fdc_id = (row.get('fdc_id') or '').strip()
                if not fdc_id:
                    continue
                self._usda_portion_index.setdefault(fdc_id, []).append(row)

    def _derive_usda_density_entry(self, entry: Optional[dict]) -> Optional[dict]:
        if not entry or (entry.get('source') or '').strip().lower() != 'usda':
            return entry
        density_method = (entry.get('density_method') or '').strip().lower()
        # Recompute cup-based USDA densities so they stay aligned with the
        # canonical USDA cup volume used by _portion_volume_ml().
        if entry.get('density_g_ml') is not None and not density_method.startswith('usda_portion(cup'):
            return entry

        self._load_usda_portion_data()
        if not self._usda_desc_to_fdc_id or not self._usda_portion_index:
            return entry

        description = (entry.get('description') or '').strip()
        if not description:
            return entry

        fdc_id = str(entry.get('fdc_id') or self._usda_desc_to_fdc_id.get(description) or '').strip()
        if not fdc_id:
            return entry

        candidates: list[tuple[float, str, dict[str, str]]] = []
        for row in self._usda_portion_index.get(fdc_id, []):
            gram_weight = self._safe_float(row.get('gram_weight'))
            if gram_weight is None or gram_weight <= 0:
                continue
            unit_name = self._usda_measure_units.get((row.get('measure_unit_id') or '').strip(), '')
            volume = self._portion_volume_ml(row, unit_name)
            if volume is None:
                continue
            volume_ml, unit_label = volume
            if volume_ml <= 0:
                continue
            density = gram_weight / volume_ml
            if not (0.05 <= density <= 3.0):
                continue
            candidates.append((density, unit_label, row))

        if not candidates:
            return entry

        density, unit_label, row = max(candidates, key=self._usda_portion_sort_key)
        amount = self._safe_float(row.get('amount')) or 1.0
        enriched = dict(entry)
        enriched['fdc_id'] = fdc_id
        enriched['density_g_ml'] = round(float(density), 4)
        enriched['density_method'] = f"usda_portion({unit_label},vol={amount:g})"
        return enriched

    @staticmethod
    def _is_reliable_density_entry(entry: dict) -> bool:
        density = entry.get('density_g_ml')
        if density is None:
            return False

        source = (entry.get('source') or '').strip().lower()
        method = (entry.get('density_method') or '').strip().lower()
        if source == 'fao':
            return method == 'fao_measured'
        if source == 'usda':
            return method.startswith('cup(') or method.startswith('usda_portion(')
        if source == 'cofid':
            return method == 'cofid_specific_gravity'
        return False

    # ──────────────────────────────────────────────────────────────────────────
    # FAISS helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        return self._embedder.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

    def _faiss_search(self, index, labels: list, top_k: int) -> list:
        """Search with multiple label variants; return merged (sim, idx) sorted by sim desc."""
        hits = {}
        for label in labels:
            if not label.strip():
                continue
            vec = self._embed(label)
            dists, idxs = index.search(vec, top_k)
            for i in range(top_k):
                idx = int(idxs[0][i])
                sim = float(dists[0][i])
                if sim < _MIN_FAISS_SIM:
                    break
                if idx not in hits or sim > hits[idx]:
                    hits[idx] = sim
        return sorted(hits.items(), key=lambda x: -x[1])

    def _clip_text_embedding(self, text: str) -> np.ndarray:
        inputs = self._clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_features = self._clip_model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        vec = self._to_numpy(text_features)
        vec = vec / np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-8, None)
        return vec

    def _clip_image_embedding(self, crop_image: Optional[Image.Image]) -> Optional[np.ndarray]:
        if crop_image is None:
            return None
        inputs = self._clip_processor(images=[crop_image.convert("RGB")], return_tensors="pt")
        with torch.no_grad():
            image_features = self._clip_model.get_image_features(pixel_values=inputs["pixel_values"])
        vec = self._to_numpy(image_features)
        vec = vec / np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-8, None)
        return vec

    def _clip_fused_search(
        self,
        food_name: str,
        crop_image: Optional[Image.Image],
        top_k: int,
    ) -> list:
        if self._clip_index is None or self._clip_model is None or self._clip_processor is None:
            return []

        text_vec = self._clip_text_embedding(food_name)
        image_vec = self._clip_image_embedding(crop_image)
        if image_vec is not None:
            fused = (text_vec + image_vec) / 2.0
            fused = fused / np.clip(np.linalg.norm(fused, axis=1, keepdims=True), 1e-8, None)
        else:
            fused = text_vec

        sims, idxs = self._clip_index.search(fused.astype(np.float32), top_k)
        hits = []
        for i in range(top_k):
            idx = int(idxs[0][i])
            sim = float(sims[0][i])
            if idx < 0:
                continue
            hits.append((idx, sim))
        return hits

    # ──────────────────────────────────────────────────────────────────────────
    # Word-overlap plausibility guard
    # ──────────────────────────────────────────────────────────────────────────

    # Generic cooking/serving words that don't identify a food — excluded from word overlap check
    _GENERIC_WORDS = {
        'sauce', 'soup', 'food', 'dish', 'meal', 'item', 'product', 'type',
        'style', 'with', 'from', 'made', 'home', 'store', 'brand', 'kind',
        'extra', 'additional', 'added',  # strip modifier prefixes from queries
        'raw', 'cooked', 'roasted', 'fried', 'grilled', 'baked', 'steamed',
        'boiled', 'pickled', 'prepared', 'homemade', 'fresh', 'plain',
        'diced', 'sliced', 'slice', 'shredded', 'grated', 'chopped', 'wedge',
        'wedges', 'sauteed', 'sautéed', 'mashed',
    }
    # Short but semantically meaningful words — treated as specific identifiers
    # even though they are <4 chars, so word-overlap check doesn't treat them as generic
    _SIGNAL_WORDS = {
        'red', 'tan', 'yam', 'white', 'green', 'brown', 'pink',
        'hot', 'soy', 'oil', 'egg', 'ham', 'cod', 'rye', 'nut',
        'raw', 'dry', 'gel', 'fat', 'ice',
    }
    _UNPREPARED_TERMS = {
        'raw', 'uncooked', 'dry', 'unprepared', 'packet', 'mix', 'concentrate', 'powder',
        'seasoning', 'instant', 'flavor', 'flavoured', 'flavored',
    }
    _PREPARED_TERMS = {
        'prepared', 'cooked', 'boiled', 'steamed', 'fried', 'grilled',
        'baked', 'roasted', 'restaurant', 'homemade', 'entree',
        'pilaf', 'spanish',
    }
    _PICKLED_TERMS = {'pickled', 'pickle', 'pickles', 'brined', 'fermented'}
    _COOKED_TERMS = {'cooked', 'roasted', 'fried', 'grilled', 'baked', 'steamed', 'boiled', 'sauteed', 'sautéed'}
    _RAWISH_TERMS = {'raw', 'fresh'}
    _SWEET_TERMS = {'sweet', 'dessert', 'sugared', 'sugary', 'syrup', 'candied'}
    _DIP_SPREAD_TERMS = {
        'hummus', 'tzatziki', 'ganoush', 'baba', 'dip', 'spread', 'puree', 'purée',
        'aioli', 'mayo', 'mayonnaise',
    }
    _SAUCE_TERMS = {
        'sauce', 'gravy', 'salsa', 'dressing', 'ketchup', 'bbq', 'barbecue',
        'marinara', 'enchilada', 'chili', 'hot',
    }
    _OIL_FAT_TERMS = {
        'oil', 'oils', 'butter', 'ghee', 'lard', 'fat', 'grease', 'shortening',
    }
    _PROTEIN_TERMS = {
        'chicken', 'beef', 'pork', 'turkey', 'ham', 'fish', 'salmon', 'tuna',
        'shrimp', 'prawn', 'meat', 'falafel', 'egg', 'eggs', 'sausage',
    }
    _STARCH_TERMS = {
        'rice', 'potato', 'potatoes', 'fries', 'bread', 'pita', 'pasta', 'noodle',
        'noodles', 'yorkshire', 'pudding', 'stuffing',
    }
    _VEGETABLE_TERMS = {
        'lettuce', 'cucumber', 'carrot', 'carrots', 'tomato', 'tomatoes',
        'pepper', 'peppers', 'onion', 'onions', 'cabbage', 'broccoli',
        'cauliflower', 'peas', 'jalapeno', 'jalapeño', 'turnip', 'turnips',
        'eggplant',
    }
    _DAIRY_TERMS = {
        'feta', 'cheese', 'yogurt', 'yoghurt', 'tzatziki', 'cream', 'creamy',
        'milk',
    }
    _WHOLE_DISH_TERMS = {
        'pie', 'platter', 'bowl', 'dinner', 'casserole', 'lasagna', 'burger',
        'sandwich', 'entree', 'meal',
    }
    _PROCESSED_PRODUCE_TERMS = {
        'puree', 'purée', 'paste', 'concentrate', 'canned', 'stewed',
        'sauce', 'soup', 'juice', 'dried', 'sun-dried', 'powder',
    }
    _VARIETY_TERMS = {
        'grape', 'cherry', 'plum', 'roma', 'pear', 'heirloom',
        'green', 'spring', 'scallion', 'tops', 'top', 'only',
        'wild', 'brown', 'red', 'black',
    }
    _RAW_QUERY_TERMS = {
        'raw', 'uncooked', 'dry', 'unprepared', 'powder', 'mix', 'packet', 'instant',
        'concentrate', 'seasoning',
    }
    _FORM_GROUPS = {
        # 'raw' is handled separately via _FRESH_PRODUCE_TERMS below — do NOT add it here.
        # Only block truly unusable dry/unready states.
        'dry_form': {
            'uncooked', 'dry', 'dehydrated', 'powder', 'powdered', 'mix',
            'packet', 'instant', 'concentrate', 'seasoning', 'unprepared',
        },
        # Preserved/pickled forms should not match fresh produce queries.
        # e.g. query "cucumber" must not match "Pickles, cucumber, sour"
        # Keep narrow — 'salted', 'sour', 'cured' are too broad and cause
        # false rejections (salted butter, sour cream, cured salmon).
        'preserved': {
            'pickle', 'pickled', 'pickles', 'brined', 'brine', 'fermented',
        },
        'flour': {'flour', 'meal', 'starch'},
        'juice': {'juice', 'nectar', 'smoothie'},
        'soup': {'soup', 'broth', 'bisque', 'chowder'},
        'sauce_family': {
            'sauce', 'dressing', 'dip', 'gravy', 'salsa', 'chutney',
            'ketchup', 'bbq', 'barbecue', 'mayonnaise', 'mayo', 'aioli',
        },
    }

    # Fresh produce that is always served raw — 'raw' DB entries ARE the right match.
    # e.g. "shredded carrots" → "Carrots, raw" ✓  / "yellow rice" → block "Rice, red, raw" ✓
    _FRESH_PRODUCE_TERMS = {
        'lettuce', 'cucumber', 'carrot', 'carrots', 'tomato', 'tomatoes',
        'celery', 'pepper', 'peppers', 'spinach', 'kale', 'cabbage',
        'broccoli', 'cauliflower', 'zucchini', 'avocado', 'radish',
        'parsley', 'cilantro', 'mint', 'basil', 'scallion', 'arugula',
        'onion', 'onions', 'beet', 'beets', 'fennel', 'endive',
        'watercress', 'radicchio', 'chard', 'leek', 'leeks',
    }
    _PRODUCE_TOKEN_ALIASES = {
        'tomato': {'tomato', 'tomatoes'},
        'tomatoes': {'tomato', 'tomatoes'},
        'onion': {'onion', 'onions'},
        'onions': {'onion', 'onions'},
        'cucumber': {'cucumber', 'cucumbers'},
        'cucumbers': {'cucumber', 'cucumbers'},
        'pepper': {'pepper', 'peppers'},
        'peppers': {'pepper', 'peppers'},
        'carrot': {'carrot', 'carrots'},
        'carrots': {'carrot', 'carrots'},
        'beet': {'beet', 'beets'},
        'beets': {'beet', 'beets'},
        'leek': {'leek', 'leeks'},
        'leeks': {'leek', 'leeks'},
    }
    _CANONICAL_PRODUCE_MODIFIERS = {
        'red', 'ripe', 'year', 'round', 'average', 'white', 'yellow', 'green',
        'whole', 'mature', 'baby', 'sweet', 'large', 'small', 'medium',
    }
    _DISH_QUERY_EXCLUDED = {'with', 'made', 'and', 'or', 'style', 'homemade', 'home'}

    @classmethod
    def _tokenize_words(cls, text: str, *, keep_generic: bool = False) -> set[str]:
        tokens = set(re.sub(r'[^a-z0-9 ]', ' ', (text or '').lower()).split())
        if keep_generic:
            return tokens
        return {token for token in tokens if token not in cls._GENERIC_WORDS}

    @classmethod
    def _words_overlap(cls, query_lower: str, matched_lower: str) -> bool:
        """
        At least one specific (non-generic) ≥4-char word from the query must appear
        as a substring in the matched name, to block false positives like
        'caramel sauce' → 'tomato chili sauce' (only 'sauce' matches, which is generic).
        """
        q_words = {
            w for w in re.sub(r'[^\w ]', ' ', query_lower).split()
            if (len(w) >= 4 or w in cls._SIGNAL_WORDS) and w not in cls._GENERIC_WORDS
        }
        m_words = {
            w for w in re.sub(r'[^\w ]', ' ', matched_lower).split()
            if len(w) >= 4 or w in cls._SIGNAL_WORDS
        }
        if not q_words:
            # All query words are generic — fall back to any overlap
            q_words = {
                w for w in re.sub(r'[^\w ]', ' ', query_lower).split()
                if len(w) >= 4 or w in cls._SIGNAL_WORDS
            }
        if not q_words:
            return True
        if q_words & m_words:
            return True

        def _near_token_match(qw: str, mw: str) -> bool:
            if min(len(qw), len(mw)) < 5:
                return False
            shorter, longer = sorted((qw, mw), key=len)
            if not longer.startswith(shorter):
                return False
            return (len(shorter) / len(longer)) >= 0.8

        return any(_near_token_match(qw, mw) for qw in q_words for mw in m_words)

    @classmethod
    def _classify_tokens(cls, text: str) -> dict[str, bool]:
        tokens = cls._tokenize_words(text, keep_generic=True)

        def has_any(term_set: set[str]) -> bool:
            return bool(tokens & term_set)

        return {
            "pickled": has_any(cls._PICKLED_TERMS),
            "cooked": has_any(cls._COOKED_TERMS),
            "rawish": has_any(cls._RAWISH_TERMS),
            "dip_or_spread": has_any(cls._DIP_SPREAD_TERMS),
            "sauce": has_any(cls._SAUCE_TERMS),
            "oil_or_fat": has_any(cls._OIL_FAT_TERMS),
            "protein": has_any(cls._PROTEIN_TERMS),
            "starch": has_any(cls._STARCH_TERMS),
            "vegetable": has_any(cls._VEGETABLE_TERMS),
            "dairy": has_any(cls._DAIRY_TERMS),
            "whole_dish": has_any(cls._WHOLE_DISH_TERMS),
            "sweet": has_any(cls._SWEET_TERMS),
        }

    @classmethod
    def _compatibility_adjustment(
        cls,
        query_lower: str,
        matched_lower: str,
        *,
        field: str,
    ) -> float:
        query_profile = cls._classify_tokens(query_lower)
        matched_profile = cls._classify_tokens(matched_lower)
        score = 0.0

        if query_profile["pickled"]:
            score += 10.0 if matched_profile["pickled"] else -22.0
        elif matched_profile["pickled"]:
            score -= 12.0

        if query_profile["dip_or_spread"]:
            if matched_profile["dip_or_spread"] or matched_profile["sauce"]:
                score += 16.0
            elif matched_profile["whole_dish"] or matched_profile["vegetable"]:
                score -= 24.0
            query_tokens = cls._tokenize_words(query_lower, keep_generic=True)
            matched_tokens = cls._tokenize_words(matched_lower, keep_generic=True)
            extra_protein = (matched_tokens & cls._PROTEIN_TERMS) - query_tokens
            extra_starch = (matched_tokens & cls._STARCH_TERMS) - query_tokens
            if extra_protein:
                score -= 26.0 * len(extra_protein)
            if extra_starch:
                score -= 20.0 * len(extra_starch)

        if query_profile["sauce"]:
            if matched_profile["sauce"] or matched_profile["dip_or_spread"]:
                score += 12.0
            elif matched_profile["whole_dish"] or matched_profile["vegetable"]:
                score -= 18.0
            query_tokens = cls._tokenize_words(query_lower, keep_generic=True)
            matched_tokens = cls._tokenize_words(matched_lower, keep_generic=True)
            extra_protein = (matched_tokens & cls._PROTEIN_TERMS) - query_tokens
            extra_starch = (matched_tokens & cls._STARCH_TERMS) - query_tokens
            if extra_protein:
                score -= 30.0 * len(extra_protein)
            if extra_starch:
                score -= 24.0 * len(extra_starch)
        elif matched_profile["sauce"] and not query_profile["dip_or_spread"]:
            score -= 10.0

        if query_profile["protein"] and not matched_profile["protein"]:
            if matched_profile["vegetable"] or matched_profile["oil_or_fat"]:
                score -= 26.0

        if query_profile["starch"] and matched_profile["oil_or_fat"]:
            score -= 34.0

        if query_profile["vegetable"] and matched_profile["whole_dish"]:
            score -= 22.0

        if query_profile["dairy"] and not matched_profile["dairy"] and matched_profile["vegetable"]:
            score -= 18.0

        if query_profile["cooked"] and 'raw' in cls._tokenize_words(matched_lower, keep_generic=True):
            if not (query_profile["vegetable"] and not query_profile["starch"]):
                score -= 16.0

        if query_profile["rawish"] and matched_profile["sauce"]:
            score -= 18.0

        if matched_profile["sweet"] and not query_profile["sweet"]:
            score -= 22.0

        query_tokens = cls._tokenize_words(query_lower, keep_generic=True)
        matched_tokens = cls._tokenize_words(matched_lower, keep_generic=True)

        specific_query_tokens = {
            token for token in query_tokens
            if token not in cls._GENERIC_WORDS and len(token) >= 4
        }
        if len(specific_query_tokens) == 1:
            extra_variety = (matched_tokens & cls._VARIETY_TERMS) - query_tokens
            if extra_variety:
                score -= 12.0 * len(extra_variety)

            if not (query_tokens & cls._COOKED_TERMS) and (matched_tokens & cls._COOKED_TERMS):
                score -= 14.0

            if query_profile["vegetable"] and ('raw' in matched_tokens or 'fresh' in matched_tokens):
                score += 8.0

            if query_profile["vegetable"]:
                processed_mismatch = (matched_tokens & cls._PROCESSED_PRODUCE_TERMS) - query_tokens
                if processed_mismatch:
                    score -= 18.0 * len(processed_mismatch)

        if 'rice' in query_tokens:
            rice_variant_terms = {'wild', 'brown', 'red', 'black'}
            mismatch = (matched_tokens & rice_variant_terms) - query_tokens
            if mismatch:
                score -= 18.0 * len(mismatch)

        if field == 'density_g_ml' and matched_profile["oil_or_fat"] and not query_profile["oil_or_fat"]:
            score -= 40.0

        return score

    @classmethod
    def _is_standalone_condiment_query(cls, query_lower: str) -> bool:
        tokens = cls._tokenize_words(query_lower, keep_generic=True)
        profile = cls._classify_tokens(query_lower)
        if not (profile["sauce"] or profile["dip_or_spread"]):
            return False
        if profile["protein"] or profile["whole_dish"]:
            return False
        return len(tokens) <= 3

    @classmethod
    def _is_mixed_dish_candidate_for_condiment_query(cls, query_lower: str, matched_lower: str) -> bool:
        if not cls._is_standalone_condiment_query(query_lower):
            return False
        query_tokens = cls._tokenize_words(query_lower, keep_generic=True)
        matched_tokens = cls._tokenize_words(matched_lower, keep_generic=True)
        extra_protein = (matched_tokens & cls._PROTEIN_TERMS) - query_tokens
        extra_starch = (matched_tokens & cls._STARCH_TERMS) - query_tokens
        return bool(extra_protein or extra_starch)

    @classmethod
    def _has_conflicting_form(cls, query_lower: str, matched_lower: str) -> bool:
        query_tokens = cls._tokenize_words(query_lower, keep_generic=True)
        matched_tokens = cls._tokenize_words(matched_lower, keep_generic=True)
        if not query_tokens or not matched_tokens:
            return False

        for terms in cls._FORM_GROUPS.values():
            if matched_tokens & terms and not query_tokens & terms:
                return True

        # Block 'raw' DB entries for cooked/processed food queries UNLESS the query
        # is asking about fresh produce (which IS naturally served raw).
        # Example blocks:  "yellow rice" → "Rice, red, raw" (356 kcal) ✗
        #                  "falafel"     → "Falafel, raw" ✗
        # Example allows:  "shredded carrots" → "Carrots, raw" ✓
        #                  "sliced cucumber"  → "Cucumber, peeled, raw" ✓
        if 'raw' in matched_tokens and not (query_tokens & cls._FRESH_PRODUCE_TERMS):
            return True

        return False

    @classmethod
    def _tokenize_for_rank(cls, text: str) -> list[str]:
        cleaned = re.sub(r'[^a-z0-9 ]', ' ', (text or '').lower())
        return [
            token for token in cleaned.split()
            if len(token) >= 3 and token not in cls._GENERIC_WORDS
        ]

    @classmethod
    def _query_implies_served_dish(cls, query_lower: str) -> bool:
        query_tokens = set(cls._tokenize_for_rank(query_lower))
        return not bool(query_tokens & cls._RAW_QUERY_TERMS)

    @classmethod
    def _dish_query_tokens(cls, text: str) -> set[str]:
        return {
            token
            for token in cls._tokenize_words(text, keep_generic=True)
            if len(token) >= 3 and token not in cls._DISH_QUERY_EXCLUDED
        }

    @classmethod
    def _candidate_rank_score(
        cls,
        query_lower: str,
        matched_lower: str,
        entry: dict,
        sim: float,
    ) -> float:
        """
        Final ranking score after FAISS recall.
        Rewards exact/near-exact lexical matches and prepared dish candidates.
        Penalizes generic commodity hits and dry/unprepared mix entries for plated dishes.
        """
        score = float(sim) * 100.0

        normalized_query = cls._normalize_food_name(query_lower)
        normalized_matched = cls._normalize_food_name(matched_lower)
        query_tokens = set(cls._tokenize_for_rank(normalized_query))
        matched_tokens = set(cls._tokenize_for_rank(normalized_matched))

        if normalized_query and normalized_query == normalized_matched:
            score += 20.0
        elif normalized_query and normalized_query in normalized_matched:
            score += 12.0

        overlap_count = len(query_tokens & matched_tokens)
        missing_count = len(query_tokens - matched_tokens)
        score += overlap_count * 6.0
        score -= missing_count * 5.0

        if len(query_tokens) >= 2 and len(matched_tokens) <= 1 and missing_count > 0:
            score -= 14.0

        query_served = cls._query_implies_served_dish(query_lower)
        matched_all_text = re.sub(r'[^a-z0-9 ]', ' ', matched_lower)
        matched_all_tokens = set(matched_all_text.split())

        has_unprepared_marker = bool(matched_all_tokens & cls._UNPREPARED_TERMS)
        has_prepared_marker = bool(matched_all_tokens & cls._PREPARED_TERMS)

        if query_served:
            # Any plated-dish query should prefer prepared/cooked entries and avoid dry ones.
            # This applies universally — not just to specific food categories.
            if has_prepared_marker:
                score += 8.0
            if has_unprepared_marker:
                score -= 25.0
            if len(query_tokens) >= 2 and overlap_count >= 2:
                score -= 12.0 * len(matched_all_tokens & cls._UNPREPARED_TERMS)
            # Extra reward for explicitly cooked/boiled/steamed forms — physically correct
            # for any plated food (right water content, right kcal/100g).
            if matched_all_tokens & cls._COOKED_TERMS:
                score += 8.0

        # Strongly discourage falling back to a generic single-food label like "Rice"
        # when the query is more specific, e.g. "yellow rice".
        if len(query_tokens) >= 2 and normalized_matched in matched_tokens and len(matched_tokens) == 1:
            score -= 16.0

        # Favor richer USDA/CoFID dish descriptions over generic FAO commodity labels
        # when the query itself is a multi-word plated dish.
        if len(query_tokens) >= 2 and entry.get('source') == 'fao' and len(matched_tokens) <= 2:
            score -= 6.0

        if {'yellow', 'rice'} <= query_tokens:
            if {'yellow', 'rice'} <= matched_all_tokens:
                score += 18.0
            if 'no' in matched_all_tokens and 'fat' in matched_all_tokens:
                score += 10.0

        score += cls._compatibility_adjustment(query_lower, matched_lower, field='calories_per_100g')

        return score

    @classmethod
    def _lexical_override_score(cls, query_lower: str, matched_lower: str) -> float:
        score = 0.0
        normalized_query = cls._normalize_food_name(query_lower)
        normalized_matched = cls._normalize_food_name(matched_lower)
        query_tokens = set(cls._tokenize_for_rank(normalized_query))
        matched_tokens = set(cls._tokenize_for_rank(normalized_matched))

        if normalized_query == normalized_matched:
            score += 40.0
        elif normalized_matched.startswith(normalized_query):
            score += 28.0
        elif normalized_query in normalized_matched:
            score += 18.0

        overlap = len(query_tokens & matched_tokens)
        score += overlap * 8.0
        score -= len(query_tokens - matched_tokens) * 6.0

        processed_penalties = {"mix", "packet", "instant", "powder", "concentrate", "flavor", "flavored", "flavoured"}
        matched_all_tokens = set(re.sub(r'[^a-z0-9 ]', ' ', matched_lower).split())
        score -= len(processed_penalties & matched_all_tokens) * 7.0
        score += cls._compatibility_adjustment(query_lower, matched_lower, field='calories_per_100g') * 0.4
        return score

    def _retrieve_candidates(
        self,
        food_name: str,
        crop_image: Optional[Image.Image],
        top_k: int = 10,
        meal_context: Optional[str] = None,
    ) -> list[dict]:
        if not self._use_unified:
            return []

        normalized = self._normalize_food_name(food_name)

        # Sentence-transformer FAISS search (primary recall — same embedder used to build the index)
        st_hits: dict[int, float] = {}
        if self._unified_index is not None and self._embedder is not None:
            variants = list(dict.fromkeys([normalized, food_name.strip().lower()]))
            # Add a contextual variant when meal_context is available (e.g. "falafel bowl").
            # For generic labels like "red sauce", appending the meal name shifts the
            # embedding toward the correct cuisine region — away from false positives
            # like "Barbecue sauce" and toward "chili sauce" / "hot sauce".
            if meal_context:
                ctx_short = meal_context.strip().lower()[:40]
                contextual = f"{normalized} {ctx_short}".strip()
                if contextual not in variants:
                    variants.append(contextual)
            for variant in variants:
                if not variant.strip():
                    continue
                vec = self._embed(variant)
                dists, idxs = self._unified_index.search(vec, top_k)
                for i in range(top_k):
                    idx = int(idxs[0][i])
                    sim = float(dists[0][i])
                    if sim < _MIN_FAISS_SIM:
                        break
                    if idx not in st_hits or sim > st_hits[idx]:
                        st_hits[idx] = sim

        # CLIP fused search (adds image-grounded recall when a crop is available)
        clip_hits = self._clip_fused_search(food_name, crop_image, top_k)

        # Merge: keep best sim per index across both retrievers
        merged_hits: dict[int, float] = dict(st_hits)
        for idx, clip_sim in clip_hits:
            if idx not in merged_hits or clip_sim > merged_hits[idx]:
                merged_hits[idx] = clip_sim

        candidates = []
        for idx, sim in merged_hits.items():
            if not (0 <= idx < len(self._unified_foods)):
                continue
            entry = self._unified_foods[idx]
            retrieval_text = entry.get("retrieval_text") or (
                self._unified_names[idx] if idx < len(self._unified_names) else ""
            )
            matched = entry.get("description") or retrieval_text
            if not retrieval_text or not matched:
                continue
            cross_score = (
                float(self._cross_encoder.predict([(food_name, retrieval_text)])[0])
                if self._cross_encoder else float("-inf")
            )
            lexical_score = self._lexical_override_score(normalized, matched.lower())
            rank_score = self._candidate_rank_score(normalized, matched.lower(), entry, sim)
            compatibility_score = self._compatibility_adjustment(normalized, matched.lower(), field='calories_per_100g')
            selection_score = (cross_score * 12.0) + lexical_score + rank_score + compatibility_score
            candidates.append({
                "idx": idx,
                "entry": entry,
                "matched": matched,
                "retrieval_text": retrieval_text,
                "clip_sim": sim,
                "cross_score": cross_score,
                "lexical_score": lexical_score,
                "rank_score": rank_score,
                "compatibility_score": compatibility_score,
                "selection_score": selection_score,
            })

        candidates.sort(
            key=lambda item: (
                -item["selection_score"],
                -item["cross_score"],
                -item["lexical_score"],
                -item["rank_score"],
                -item["clip_sim"],
            )
        )
        return candidates

    def _apply_lexical_override(self, food_name: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return candidates
        best = candidates[0]
        lexical_best = max(candidates, key=lambda item: item["lexical_score"])
        if lexical_best["idx"] != best["idx"]:
            lexical_gap = lexical_best["lexical_score"] - best["lexical_score"]
            bad_best = (
                lexical_gap >= 10.0
                and (
                    best["cross_score"] < -2.0
                    or lexical_best["cross_score"] >= best["cross_score"] - 1.0
                )
            )
            if bad_best:
                reordered = [lexical_best] + [item for item in candidates if item["idx"] != lexical_best["idx"]]
                return reordered
        return candidates

    def _find_entry_by_description(self, description: str) -> Optional[dict]:
        target = (description or "").strip().lower()
        if not target:
            return None

        for entry in self._unified_foods:
            if (entry.get("description") or "").strip().lower() == target:
                return self._derive_usda_density_entry(entry)

        for entry in self._usda_foods:
            if (entry.get("description") or "").strip().lower() == target:
                return self._derive_usda_density_entry(entry)

        return None

    @classmethod
    def _score_verifier_description(cls, query: str, description: str) -> float:
        query_text = (query or "").strip().lower()
        desc_text = (description or "").strip().lower()
        if not query_text or not desc_text:
            return float("-inf")

        query_words = [word for word in re.findall(r"[a-z0-9]+", query_text) if len(word) > 1]
        desc_words = set(re.findall(r"[a-z0-9]+", desc_text))
        score = 0.0
        if query_text == desc_text:
            score += 20.0
        if query_text in desc_text:
            score += 8.0
        for word in query_words:
            if word in desc_words:
                score += 2.5

        # Strongly reward cooked/prepared forms — verifier candidates are for plated food
        _COOKED_BONUS = {'cooked', 'prepared', 'boiled', 'steamed', 'baked',
                         'roasted', 'fried', 'grilled', 'restaurant', 'homemade'}
        _DRY_PENALTY  = {'unprepared', 'uncooked', 'dry', 'dried', 'dehydrated',
                         'powder', 'powdered', 'mix', 'packet', 'instant',
                         'concentrate', 'seasoning', 'raw'}
        if desc_words & _COOKED_BONUS:
            score += 6.0
        if desc_words & _DRY_PENALTY:
            score -= 15.0

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
        normalized_query = self._normalize_food_name(food_name)

        def _is_valid_for_query(desc: Optional[str]) -> bool:
            """Return False if the description is a conflicting form for this food query.

            This ensures the Gemini verifier never sees dry/raw/unprepared candidates
            for a query that doesn't explicitly ask for that form — the same rule applied
            in _lookup_unified for FAISS retrieval.
            """
            if not desc:
                return False
            return not self._has_conflicting_form(normalized_query, desc.lower())

        descriptions: list[str] = []

        def add_description(text: Optional[str]) -> None:
            normalized = (text or "").strip()
            if normalized and normalized not in descriptions and _is_valid_for_query(normalized):
                descriptions.append(normalized)

        add_description(chosen_description)
        for candidate in rag_candidates or []:
            add_description(candidate.get("description"))

        candidate_pool = self._unified_foods or self._usda_foods
        lexical_ranked = sorted(
            candidate_pool,
            key=lambda food: self._score_verifier_description(food_name, food.get("description") or ""),
            reverse=True,
        )
        for food in lexical_ranked[: limit * 3]:
            add_description(food.get("description"))
            if len(descriptions) >= limit:
                break

        verifier_candidates = []
        for description in descriptions[:limit]:
            entry = self._find_entry_by_description(description)
            if not entry:
                continue
            verifier_candidates.append(
                {
                    "description": description,
                    "calories_per_100g": float(entry.get("calories_per_100g") or 0.0),
                    "fat_per_100g": float(entry.get("fat_g") or 0.0),
                    "carb_per_100g": float(entry.get("carbs_g") or 0.0),
                    "protein_per_100g": float(entry.get("protein_g") or 0.0),
                }
            )
        return verifier_candidates

    def get_usda_candidate_by_description(self, description: str) -> Optional[dict]:
        entry = self._find_entry_by_description(description)
        if not entry:
            return None
        return {
            "description": entry.get("description"),
            "calories_per_100g": float(entry.get("calories_per_100g") or 0.0),
            "fat_per_100g": float(entry.get("fat_g") or 0.0),
            "carb_per_100g": float(entry.get("carbs_g") or 0.0),
            "protein_per_100g": float(entry.get("protein_g") or 0.0),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Unified lookup (density or calories)
    # ──────────────────────────────────────────────────────────────────────────

    _SKIP_WORDS = {'oil', 'fat', 'extract', 'powder', 'concentrate', 'flavoring', 'supplement'}

    def _lookup_served_dish_lexical(
        self,
        food_name: str,
        field: str,
    ):
        if not self._use_unified or not self._unified_foods or not self._unified_names:
            return None, None, None, None

        normalized = self._normalize_food_name(food_name)
        query_tokens = self._dish_query_tokens(normalized)
        if len(query_tokens) < 2:
            return None, None, None, None

        query_profile = self._classify_tokens(normalized)
        if query_profile["vegetable"] and not query_profile["starch"] and not query_profile["protein"]:
            return None, None, None, None
        if not self._query_implies_served_dish(normalized):
            return None, None, None, None

        candidates = []
        for idx, (entry, matched) in enumerate(zip(self._unified_foods, self._unified_names)):
            candidate_entry = entry
            if field == 'density_g_ml':
                candidate_entry = self._derive_usda_density_entry(candidate_entry)
                if not self._is_reliable_density_entry(candidate_entry):
                    continue

            value = candidate_entry.get(field)
            if value is None:
                continue

            matched_lower = matched.lower()
            if self._is_mixed_dish_candidate_for_condiment_query(normalized, matched_lower):
                continue
            if not self._words_overlap(normalized, matched_lower):
                continue
            if self._has_conflicting_form(normalized, matched_lower):
                continue

            matched_all_tokens = set(re.sub(r'[^a-z0-9 ]', ' ', matched_lower).split())
            overlap_count = len(query_tokens & matched_all_tokens)
            if overlap_count < min(2, len(query_tokens)):
                continue

            score = 40.0
            if query_tokens <= matched_all_tokens:
                score += 50.0
            normalized_matched = self._normalize_food_name(matched)
            if normalized in normalized_matched:
                score += 20.0
            if normalized_matched.startswith(normalized):
                score += 12.0

            prepared_markers = matched_all_tokens & self._PREPARED_TERMS
            unprepared_markers = matched_all_tokens & self._UNPREPARED_TERMS
            score += 12.0 * len(prepared_markers)
            score -= 16.0 * len(unprepared_markers)

            if 'no' in matched_all_tokens and 'fat' in matched_all_tokens:
                score += 10.0
            if 'cooked' in matched_all_tokens:
                score += 10.0

            extra_tokens = query_tokens.symmetric_difference(query_tokens & matched_all_tokens)
            score -= 4.0 * max(0, len(extra_tokens) - max(0, overlap_count - 2))
            score += self._compatibility_adjustment(normalized, matched_lower, field=field)
            score -= 0.05 * len(matched_lower)

            candidates.append((
                score,
                overlap_count,
                idx,
                float(value),
                matched,
                candidate_entry,
            ))

        if not candidates:
            return None, None, None, None

        candidates.sort(reverse=True)
        best_score, _overlap_count, _idx, value, matched, entry = candidates[0]
        if best_score < 55.0:
            return None, None, None, None

        source = f"{entry.get('source', 'unified')}_served_lexical"
        if field == 'density_g_ml':
            source += f",{entry.get('density_method', 'unknown')}"
        logger.info(
            f"[{field}] '{food_name}' served lexical → '{matched}' "
            f"({source}, score={best_score:.2f}): {value}"
        )
        return value, matched, source, entry

    def _lookup_single_ingredient_lexical(
        self,
        food_name: str,
        field: str,
    ):
        if not self._use_unified or not self._unified_foods or not self._unified_names:
            return None, None, None, None

        normalized = self._normalize_food_name(food_name)
        all_query_tokens = [
            token for token in self._tokenize_words(normalized, keep_generic=True)
            if token
        ]
        query_tokens = [
            token for token in self._tokenize_words(normalized, keep_generic=True)
            if token not in self._GENERIC_WORDS
        ]
        if len(all_query_tokens) != 1 or len(query_tokens) != 1:
            return None, None, None, None

        query_token = query_tokens[0]
        query_profile = self._classify_tokens(normalized)
        candidates = []

        for idx, (entry, matched) in enumerate(zip(self._unified_foods, self._unified_names)):
            value = entry.get(field)
            if value is None:
                continue
            if field == 'density_g_ml' and not self._is_reliable_density_entry(entry):
                continue

            matched_lower = matched.lower()
            if self._is_mixed_dish_candidate_for_condiment_query(normalized, matched_lower):
                continue
            if not self._words_overlap(normalized, matched_lower):
                continue

            matched_tokens = self._tokenize_words(matched_lower, keep_generic=True)
            normalized_matched = self._normalize_food_name(matched)
            if any(needle in matched_lower for needle in (' and ', ' with ', ' made with ')):
                continue
            matched_specific = {
                token for token in matched_tokens
                if token not in self._GENERIC_WORDS
            }
            matched_specific_order = [
                token for token in re.sub(r'[^a-z0-9 ]', ' ', matched_lower).split()
                if token not in self._GENERIC_WORDS
            ]
            if matched_specific_order:
                lead_token = matched_specific_order[0]
                if not (
                    lead_token == query_token
                    or lead_token.startswith(query_token)
                    or query_token.startswith(lead_token)
                ):
                    continue

            score = 0.0
            if normalized_matched == normalized:
                score += 120.0
            if any(token == query_token or token.startswith(query_token) or query_token.startswith(token) for token in matched_specific):
                score += 45.0
            if normalized in normalized_matched:
                score += 25.0

            score += self._compatibility_adjustment(normalized, matched_lower, field=field)

            if query_profile["vegetable"] and ('raw' in matched_tokens or 'fresh' in matched_tokens):
                score += 28.0

            if query_profile["vegetable"]:
                if 'raw' not in matched_tokens and 'fresh' not in matched_tokens:
                    continue
                processed_mismatch = (matched_tokens & self._PROCESSED_PRODUCE_TERMS) - set(query_tokens)
                if processed_mismatch:
                    continue
                variety_mismatch = (matched_tokens & self._VARIETY_TERMS) - set(query_tokens)
                if variety_mismatch:
                    score -= 14.0 * len(variety_mismatch)
                if not (set(query_tokens) & self._COOKED_TERMS) and (matched_tokens & self._COOKED_TERMS):
                    score -= 18.0

            candidates.append((
                score,
                -len(normalized_matched),
                idx,
                float(value),
                matched,
                entry,
            ))

        if not candidates:
            return None, None, None, None

        candidates.sort(reverse=True)
        best_score, _neg_len, _idx, value, matched, entry = candidates[0]
        if best_score < 35.0:
            return None, None, None, None

        src = entry.get('source', 'unified')
        source = f"{src}_lexical"
        if field == 'density_g_ml':
            source += f",{entry.get('density_method', 'unknown')}"
        logger.info(f"[{field}] '{food_name}' lexical pre-pass → '{matched}' ({source}, score={best_score:.2f}): {value}")
        return value, matched, source, entry

    def _lookup_plain_raw_produce(
        self,
        food_name: str,
        field: str,
    ):
        if not self._use_unified or not self._unified_foods or not self._unified_names:
            return None, None, None, None

        normalized = self._normalize_food_name(food_name)
        all_query_tokens = [
            token for token in self._tokenize_words(normalized, keep_generic=True)
            if token
        ]
        query_tokens = [
            token for token in self._tokenize_words(normalized, keep_generic=True)
            if token not in self._GENERIC_WORDS
        ]
        if len(all_query_tokens) != 1 or len(query_tokens) != 1:
            return None, None, None, None

        query_token = query_tokens[0]
        query_profile = self._classify_tokens(normalized)
        if not query_profile["vegetable"]:
            return None, None, None, None

        query_variants = self._PRODUCE_TOKEN_ALIASES.get(query_token, {query_token})
        candidates = []

        for idx, (entry, matched) in enumerate(zip(self._unified_foods, self._unified_names)):
            value = entry.get(field)
            if value is None:
                continue
            if field == 'density_g_ml' and not self._is_reliable_density_entry(entry):
                continue

            matched_lower = matched.lower()
            if self._is_mixed_dish_candidate_for_condiment_query(normalized, matched_lower):
                continue
            if any(needle in matched_lower for needle in (' and ', ' with ', ' made with ')):
                continue

            matched_tokens = self._tokenize_words(matched_lower, keep_generic=True)
            if not (matched_tokens & query_variants):
                continue
            if 'raw' not in matched_tokens and 'fresh' not in matched_tokens:
                continue

            matched_profile = self._classify_tokens(matched_lower)
            if matched_profile["sauce"] or matched_profile["whole_dish"] or matched_profile["dip_or_spread"]:
                continue

            processed_tokens = (matched_tokens & self._PROCESSED_PRODUCE_TERMS) - query_variants
            if processed_tokens:
                continue

            variety_tokens = (matched_tokens & self._VARIETY_TERMS) - query_variants
            if query_token in {'tomato', 'tomatoes'}:
                variety_tokens -= {'red'}
            if query_token in {'onion', 'onions'}:
                variety_tokens -= {'white'}

            normalized_matched = self._normalize_food_name(matched)
            matched_specific = {
                token for token in matched_tokens
                if token not in self._GENERIC_WORDS
            }
            extra_descriptors = matched_specific - query_variants - self._CANONICAL_PRODUCE_MODIFIERS - {'raw', 'fresh'}

            score = 140.0
            if normalized == normalized_matched:
                score += 50.0
            if normalized in normalized_matched:
                score += 30.0
            if 'raw' in matched_tokens:
                score += 20.0
            if 'fresh' in matched_tokens:
                score += 12.0

            score += self._compatibility_adjustment(normalized, matched_lower, field=field)
            score -= 10.0 * len(variety_tokens)
            score -= 4.0 * len(extra_descriptors)
            score -= 0.02 * len(matched_lower)

            candidates.append((
                score,
                idx,
                float(value),
                matched,
                entry,
            ))

        if not candidates:
            return None, None, None, None

        candidates.sort(reverse=True)
        best_score, _idx, value, matched, entry = candidates[0]
        src = entry.get('source', 'unified')
        source = f"{src}_produce_lexical"
        if field == 'density_g_ml':
            source += f",{entry.get('density_method', 'unknown')}"
        logger.info(f"[{field}] '{food_name}' produce lexical → '{matched}' ({source}, score={best_score:.2f}): {value}")
        return value, matched, source, entry

    def _pick_best_result(
        self,
        food_name: str,
        result_a: tuple,
        result_b: tuple,
    ) -> tuple:
        """
        Compare two lookup results (value, matched, source, entry) and return
        the one whose matched description scores higher against food_name via
        CrossEncoder.  This treats unified-FAISS and branded results equally —
        the better semantic match wins regardless of source.

        If only one result has a value it is returned immediately without
        running CrossEncoder.  On CrossEncoder failure, result_a is returned.
        """
        val_a = result_a[0]
        val_b = result_b[0]

        if val_a is None and val_b is None:
            return result_a
        if val_a is None:
            return result_b
        if val_b is None:
            return result_a

        # Both have values — use CrossEncoder as the arbiter
        matched_a = result_a[1] or ""
        matched_b = result_b[1] or ""

        try:
            scores = self._cross_encoder.predict(
                [(food_name, matched_a), (food_name, matched_b)]
            )
            score_a, score_b = float(scores[0]), float(scores[1])
            if score_b > score_a:
                logger.info(
                    "[conflict] '%s': branded '%s' (ce=%.2f) beats unified '%s' (ce=%.2f)",
                    food_name, matched_b, score_b, matched_a, score_a,
                )
                return result_b
            logger.info(
                "[conflict] '%s': unified '%s' (ce=%.2f) beats branded '%s' (ce=%.2f)",
                food_name, matched_a, score_a, matched_b, score_b,
            )
        except Exception as exc:
            logger.warning(
                "[conflict] CrossEncoder scoring failed for '%s' (%s) — keeping unified result",
                food_name, exc,
            )
        return result_a

    def _lookup_unified(
        self,
        food_name: str,
        field: str,
        top_k: int = 10,
        crop_image: Optional[Image.Image] = None,
    ):
        """
        Search unified FAISS index, re-rank with cross-encoder.
        field: 'density_g_ml' or 'calories_per_100g'
        Returns (value, matched_name, source, entry) or (None, None, None, None).
        """
        served_value, served_matched, served_source, served_entry = self._lookup_served_dish_lexical(food_name, field)
        if served_value is not None:
            return served_value, served_matched, served_source, served_entry

        produce_value, produce_matched, produce_source, produce_entry = self._lookup_plain_raw_produce(food_name, field)
        if produce_value is not None:
            return produce_value, produce_matched, produce_source, produce_entry

        lexical_value, lexical_matched, lexical_source, lexical_entry = self._lookup_single_ingredient_lexical(food_name, field)
        if lexical_value is not None:
            return lexical_value, lexical_matched, lexical_source, lexical_entry

        normalized = self._normalize_food_name(food_name)
        dish_query_tokens = self._dish_query_tokens(normalized)
        retrieved_candidates = self._apply_lexical_override(
            normalized,
            self._retrieve_candidates(food_name, crop_image=crop_image, top_k=max(top_k, 50)),
        )

        candidates = []
        for candidate in retrieved_candidates:
            idx = candidate["idx"]
            sim = candidate["clip_sim"]
            entry = candidate["entry"]
            if field == 'density_g_ml':
                entry = self._derive_usda_density_entry(entry)
            value = entry.get(field)
            if value is None:
                continue
            matched = candidate["matched"]
            matched_tokens_all = set(re.sub(r'[^a-z0-9 ]', ' ', matched.lower()).split())

            # Apply per-source minimum similarity thresholds
            src = entry.get('source', 'usda')
            min_sim = _MIN_SIM_BY_SOURCE.get(src, _MIN_FAISS_SIM) - 0.05
            if sim < min_sim:
                continue

            # Skip irrelevant entry types (oil/extract/powder) unless query also is that type
            if any(sk in matched.lower() for sk in self._SKIP_WORDS):
                if not any(sk in normalized for sk in self._SKIP_WORDS):
                    continue

            if field == 'density_g_ml':
                if not self._is_reliable_density_entry(entry):
                    continue
                if len(dish_query_tokens) >= 2:
                    overlap = len(dish_query_tokens & matched_tokens_all)
                    if overlap < len(dish_query_tokens):
                        continue

            # Word-overlap guard against phonetic false positives
            if not self._words_overlap(normalized, matched.lower()):
                logger.info(f"Skip (no word overlap): '{food_name}' vs '{matched}'")
                continue
            if self._is_mixed_dish_candidate_for_condiment_query(normalized, matched.lower()):
                logger.info(f"Skip (mixed dish for condiment query): '{food_name}' vs '{matched}'")
                continue
            # Form-conflict check — applied identically for both density and kcal.
            # Rule: if the query doesn't mention a dry/raw/unprepared form, don't match
            # a dry/raw/unprepared DB entry.  This is a universal property of plated food:
            # a query for "yellow rice", "chicken breast", "pasta", or any other served
            # dish never wants dry-packet/uncooked calorie values regardless of whether
            # the phrase appears verbatim in the description.
            #
            # The old "phrase_in_desc" bypass has been removed — it was the root cause of
            # matches like "yellow rice" → "Yellow rice, dry packet mix, unprepared" (343
            # kcal/100g instead of ~130 for cooked).  _has_conflicting_form already
            # correctly exempts fresh produce (lettuce, cucumber, carrots, etc.) that
            # is genuinely served raw, so those still resolve to "raw" DB entries.
            #
            # Examples:
            #   "yellow rice"    → "Yellow rice … dry packet, unprepared"  ✗ blocked
            #   "chicken breast" → "Chicken, raw"                          ✗ blocked
            #   "pasta"          → "Pasta, dry, unenriched"               ✗ blocked
            #   "shredded carrots"→ "Carrots, raw"                         ✓ allowed (fresh produce)
            #   "yellow rice mix"→ "Yellow rice … dry packet"              ✓ allowed (query has 'mix')
            if self._has_conflicting_form(normalized, matched.lower()):
                logger.info(f"Skip (conflicting form): '{food_name}' vs '{matched}'")
                continue

            rank_score = candidate["rank_score"]
            compatibility_score = self._compatibility_adjustment(normalized, matched.lower(), field=field)
            selection_score = candidate.get("selection_score", 0.0)
            candidates.append((
                selection_score,
                candidate["cross_score"],
                candidate["lexical_score"],
                compatibility_score,
                rank_score,
                idx,
                sim,
                float(value),
                matched,
                entry,
            ))

        if not candidates:
            return None, None, None, None

        candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], -item[4], -item[6]))
        best = candidates[0]

        selection_score, cross_score, lexical_score, compatibility_score, rank_score, idx, sim, value, matched, entry = best
        if cross_score < _MIN_RERANK_SCORE:
            logger.info(f"[{field}] '{food_name}' top rerank score too low ({cross_score:.2f}) - using Gemini fallback path")
            return None, None, None, None
        src = entry.get('source', 'unified')
        source = f"{src}_faiss(sim={sim:.2f})"
        if field == 'density_g_ml':
            source += f",{entry.get('density_method', 'unknown')}"
        logger.info(
            f"[{field}] '{food_name}' → '{matched}' "
            f"({source}, clip={sim:.2f}, cross={cross_score:.2f}, lexical={lexical_score:.2f}, compat={compatibility_score:.2f}, rank={rank_score:.2f}, select={selection_score:.2f}): {value}"
        )
        return value, matched, source, entry

    # ──────────────────────────────────────────────────────────────────────────
    # Legacy lookup (separate FAO + USDA, used if unified not loaded)
    # ──────────────────────────────────────────────────────────────────────────

    def _lookup_legacy_density(self, food_name: str, top_k: int = 10):
        normalized = self._normalize_food_name(food_name)
        variants   = list(dict.fromkeys([normalized, food_name.strip().lower()]))

        if self._fao_index is not None:
            for idx, sim in self._faiss_search(self._fao_index, variants, top_k):
                if sim < 0.65:
                    break
                if not (0 <= idx < len(self._fao_density)):
                    continue
                matched = self._fao_names[idx] if idx < len(self._fao_names) else ""
                density = float(self._fao_density[idx].get('density_g_ml', 0.9))
                if not self._words_overlap(normalized, matched.lower()):
                    continue
                if self._has_conflicting_form(normalized, matched.lower()):
                    continue
                return density, matched, f"fao_faiss(sim={sim:.2f})"

        if self._usda_index is not None:
            for idx, sim in self._faiss_search(self._usda_index, variants, top_k):
                if sim < 0.55:
                    break
                if not (0 <= idx < len(self._usda_foods)):
                    continue
                entry   = self._derive_usda_density_entry(self._usda_foods[idx])
                density = entry.get('density_g_ml')
                if density is None:
                    continue
                matched = self._usda_names[idx] if idx < len(self._usda_names) else ""
                method  = entry.get('density_method', 'usda')
                if not self._is_reliable_density_entry({**entry, 'source': 'usda'}):
                    continue
                if not self._words_overlap(normalized, matched.lower()):
                    continue
                if self._has_conflicting_form(normalized, matched.lower()):
                    continue
                return float(density), matched, f"usda_faiss(sim={sim:.2f},{method})"

        return None, None, None

    def _lookup_legacy_calories(self, food_name: str, top_k: int = 10):
        normalized = self._normalize_food_name(food_name)
        variants   = list(dict.fromkeys([normalized, food_name.strip().lower()]))

        if self._usda_index is not None:
            for idx, sim in self._faiss_search(self._usda_index, variants, top_k):
                if sim < 0.50:
                    break
                if not (0 <= idx < len(self._usda_foods)):
                    continue
                matched = self._usda_names[idx] if idx < len(self._usda_names) else ""
                if any(sk in matched.lower() for sk in self._SKIP_WORDS):
                    if not any(sk in normalized for sk in self._SKIP_WORDS):
                        continue
                if not self._words_overlap(normalized, matched.lower()):
                    continue
                if self._has_conflicting_form(normalized, matched.lower()):
                    food_phrase = food_name.strip().lower()
                    phrase_in_desc = len(food_phrase.split()) >= 2 and food_phrase in matched.lower()
                    if not phrase_in_desc:
                        continue
                entry = self._usda_foods[idx]
                kcal  = float(entry.get('calories_per_100g', 0))
                return kcal, matched, f"usda_faiss(sim={sim:.2f})"

        return None, None, None

    # ──────────────────────────────────────────────────────────────────────────
    # Gemini grounding fallback
    # ──────────────────────────────────────────────────────────────────────────

    def _gemini_lookup(self, food_name: str, field: str, meal_context: Optional[str] = None) -> Optional[float]:
        """
        Query Gemini for density or calories using a strict 3-step grounding order:
          Step 1 — Grounded search targeting USDA FoodData Central + CoFID specifically.
          Step 2 — Grounded broader web search (only if Step 1 yields no parseable value).
          Step 3 — Context-based estimation using dish/meal context (only if Step 2 also fails).
        Raises if the API key is missing or an unrecoverable API error occurs.
        field: 'density_g_ml' or 'calories_per_100g'
        """
        if not self._gemini_api_key:
            raise NutritionLookupError(
                food_name,
                [field],
                reason="Gemini API key not configured — cannot perform grounded lookup",
            )
        cache_key = f"{food_name}::{field}"
        cached_value = self._get_cached_gemini_value(food_name, field)
        if cached_value is not None:
            logger.info("Gemini cache hit [%s] '%s': %s", field, food_name, cached_value)
            return cached_value

        if field == 'density_g_ml':
            # Step 1: USDA/CoFID targeted search
            prompt_usda_cofid = (
                f"What is the density of '{food_name}' in grams per milliliter (g/ml)? "
                f"Search ONLY in USDA FoodData Central (fdc.nal.usda.gov) and the UK CoFID "
                f"(composition of foods) database. "
                f"If found, reply with ONLY a single decimal number, e.g. 0.85. "
                f"If not found in either database, reply with exactly: NOT_FOUND"
            )
            # Step 2: Broader web search
            prompt_web = (
                f"What is the density of '{food_name}' in grams per milliliter (g/ml)? "
                f"Search food science journals, nutrition databases, and authoritative web sources. "
                f"Only report a value if you are highly confident (sourced from a reliable reference). "
                f"Reply with the value and source URL, e.g. '0.85 (source: example.com)'. "
                f"If no reliable value found, reply with exactly: NOT_FOUND"
            )
            # Step 3: Context estimation with dish context
            context_clause = (
                f" This item is part of a dish containing: {meal_context}."
                if meal_context else ""
            )
            prompt_context = (
                f"Estimate the density of '{food_name}' in grams per milliliter (g/ml) "
                f"based on its physical characteristics (water content, fat content, texture).{context_clause} "
                f"Typical food densities range from 0.1 (very airy/dry) to 1.5 (dense/wet). "
                f"Reply with ONLY a single decimal number, e.g. 0.85"
            )
            lo, hi = 0.05, 3.0
        else:
            # Step 1: USDA/CoFID targeted search
            prompt_usda_cofid = (
                f"How many kilocalories (kcal) are in 100 grams of '{food_name}'? "
                f"Search ONLY in USDA FoodData Central (fdc.nal.usda.gov) and the UK CoFID "
                f"(composition of foods) database. "
                f"If found, reply with ONLY a single integer or decimal number, e.g. 250. "
                f"If not found in either database, reply with exactly: NOT_FOUND"
            )
            # Step 2: Broader web search
            prompt_web = (
                f"How many kilocalories (kcal) are in 100 grams of '{food_name}'? "
                f"Search food science journals, nutrition databases, and authoritative web sources. "
                f"Only report a value if you are highly confident (sourced from a reliable reference). "
                f"Reply with the value and source URL, e.g. '250 (source: example.com)'. "
                f"If no reliable value found, reply with exactly: NOT_FOUND"
            )
            # Step 3: Context estimation with dish context
            context_clause = (
                f" This item appears in a dish containing: {meal_context}."
                if meal_context else ""
            )
            prompt_context = (
                f"Estimate how many kilocalories (kcal) are in 100 grams of '{food_name}' "
                f"based on its known ingredients and cooking method.{context_clause} "
                f"Reply with ONLY a single integer or decimal number, e.g. 250"
            )
            lo, hi = 1.0, 900.0

        def _call_with_grounding(prompt: str):
            from google import genai as genai_new
            from google.genai import types
            client = genai_new.Client(api_key=self._gemini_api_key)
            config_kwargs: dict = {"temperature": 0.0, "top_p": 1, "top_k": 1, "seed": 42}
            if hasattr(types, "Tool"):
                if hasattr(types, "GoogleSearch"):
                    config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
                elif hasattr(types, "GoogleSearchRetrieval"):
                    config_kwargs["tools"] = [
                        types.Tool(google_search_retrieval=types.GoogleSearchRetrieval())
                    ]
            resp = client.models.generate_content(
                model=self._gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            return resp

        def _call_context_only(prompt: str):
            from google import genai as genai_new
            from google.genai import types
            client = genai_new.Client(api_key=self._gemini_api_key)
            resp = client.models.generate_content(
                model=self._gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, top_p=1, top_k=1, seed=42),
            )
            return resp

        def _parse_value(text: str) -> Optional[float]:
            if not text or "NOT_FOUND" in text.upper():
                return None
            m = re.search(r'(\d+\.?\d*)', text)
            if m:
                candidate = float(m.group(1))
                if lo <= candidate <= hi:
                    return candidate
            return None

        val = None
        grounding_metadata = None
        grounding_step = None

        # ── Step 1: USDA / CoFID grounded search ────────────────────────────────
        try:
            resp1 = _call_with_grounding(prompt_usda_cofid)
            text1 = (resp1.text or "").strip()
            meta1 = self._extract_grounding_metadata(resp1)
            val = _parse_value(text1)
            if val is not None:
                grounding_metadata = meta1
                grounding_step = "usda_cofid"
                logger.info(
                    f"Gemini [{field}] '{food_name}': found in USDA/CoFID grounding → {val}"
                )
        except Exception as e:
            # API-level failure (auth, quota, network) — propagate so caller can raise
            raise NutritionLookupError(
                food_name, [field],
                reason=f"Gemini API error during USDA/CoFID grounding: {e}",
            ) from e

        # ── Step 2: Broader web search (only if Step 1 found nothing) ───────────
        if val is None:
            logger.info(
                f"Gemini [{field}] '{food_name}': not found in USDA/CoFID, trying web search"
            )
            try:
                resp2 = _call_with_grounding(prompt_web)
                text2 = (resp2.text or "").strip()
                meta2 = self._extract_grounding_metadata(resp2)
                val = _parse_value(text2)
                if val is not None:
                    grounding_metadata = meta2
                    grounding_step = "web_search"
                    # Extract source URL from response text if present
                    src_m = re.search(r'\(source:\s*([^)]+)\)', text2, re.IGNORECASE)
                    if src_m and grounding_metadata:
                        grounding_metadata["web_source"] = src_m.group(1).strip()
                    logger.info(
                        f"Gemini [{field}] '{food_name}': found via web search → {val} "
                        f"(source: {grounding_metadata.get('web_source') if grounding_metadata else 'unknown'})"
                    )
            except Exception as e:
                raise NutritionLookupError(
                    food_name, [field],
                    reason=f"Gemini API error during web search grounding: {e}",
                ) from e

        # ── Step 3: Context-based estimation using dish context ─────────────────
        if val is None:
            logger.info(
                f"Gemini [{field}] '{food_name}': web search also found nothing, "
                f"using context-based estimation (meal_context={meal_context!r})"
            )
            try:
                resp3 = _call_context_only(prompt_context)
                text3 = (resp3.text or "").strip()
                val = _parse_value(text3)
                if val is not None:
                    grounding_metadata = {
                        "grounded": False,
                        "source": "gemini_context_estimation",
                        "meal_context": meal_context,
                    }
                    grounding_step = "context_estimation"
                    logger.info(
                        f"Gemini [{field}] '{food_name}': context estimation → {val}"
                    )
            except Exception as e:
                raise NutritionLookupError(
                    food_name, [field],
                    reason=f"Gemini API error during context estimation: {e}",
                ) from e

        if val is None:
            logger.warning(
                f"Gemini [{field}] '{food_name}': all 3 steps returned no value"
            )
            return None

        if not self._is_reasonable_gemini_value(food_name, field, val, grounding_metadata):
            logger.warning(
                "Gemini [%s] '%s': rejected implausible value %s (step=%s)",
                field, food_name, val, grounding_step,
            )
            return None

        logger.info(f"Gemini [{field}] '{food_name}': {val} (via {grounding_step})")
        self._gemini_cache[cache_key] = val
        if grounding_metadata and grounding_metadata.get("grounded"):
            grounding_metadata.update({
                "food_name": food_name,
                "field": field,
                "value": val,
                "model": self._gemini_model,
                "grounding_step": grounding_step,
            })
            self._gemini_grounding_metadata[cache_key] = grounding_metadata
        elif grounding_metadata and grounding_step in ("web_search", "context_estimation"):
            grounding_metadata.update({
                "food_name": food_name,
                "field": field,
                "value": val,
                "model": self._gemini_model,
                "grounding_step": grounding_step,
            })
            self._gemini_grounding_metadata[cache_key] = grounding_metadata
        self._persist_gemini_cache_entry(
            food_name=food_name,
            field=field,
            value=val,
            grounding_metadata=self._gemini_grounding_metadata.get(cache_key),
        )
        return val

    def _is_reasonable_gemini_value(
        self,
        food_name: str,
        field: str,
        value: float,
        grounding_metadata: Optional[dict],
    ) -> bool:
        if field != 'density_g_ml':
            return True

        normalized = self._normalize_food_name(food_name)
        profile = self._classify_tokens(normalized)

        if (profile["starch"] or profile["protein"] or profile["whole_dish"]) and not profile["vegetable"]:
            if value < 0.35:
                return False

        if profile["sauce"] or profile["dip_or_spread"]:
            if not (0.4 <= value <= 1.6):
                return False

        if grounding_metadata and grounding_metadata.get("sources"):
            preferred_domains = {
                "fdc.nal.usda.gov",
                "food.gov.uk",
                "fao.org",
                "europa.eu",
                "ifis.org",
            }
            source_domains = {
                (source.get("domain") or "").lower()
                for source in grounding_metadata.get("sources", [])
                if isinstance(source, dict)
            }
            if source_domains and not any(
                any(preferred in domain for preferred in preferred_domains)
                for domain in source_domains
            ):
                if (profile["starch"] or profile["protein"] or profile["whole_dish"]) and value < 0.45:
                    return False

        return True

    @staticmethod
    def _extract_grounding_metadata(response: Any) -> Optional[dict]:
        """Extract source URLs/titles from Gemini grounding metadata when available."""
        def _as_plain(obj: Any):
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

        response_data = _as_plain(response)
        if not isinstance(response_data, dict):
            return None

        candidates = response_data.get("candidates") or []
        candidate = candidates[0] if candidates else {}
        grounding = (
            candidate.get("grounding_metadata")
            or candidate.get("groundingMetadata")
            or {}
        )
        if not isinstance(grounding, dict):
            return None

        chunks = grounding.get("grounding_chunks") or grounding.get("groundingChunks") or []
        queries = grounding.get("web_search_queries") or grounding.get("webSearchQueries") or []
        supports = grounding.get("grounding_supports") or grounding.get("groundingSupports") or []

        sources = []
        seen_urls = set()
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            web = chunk.get("web") or {}
            if not isinstance(web, dict):
                continue
            url = (web.get("uri") or web.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            domain = urlparse(url).netloc
            sources.append({
                "title": (web.get("title") or "").strip() or None,
                "url": url,
                "domain": domain or None,
            })

        grounded = bool(sources or queries or supports)
        if not grounded:
            return None

        return {
            "grounded": True,
            "queries": [q for q in queries if isinstance(q, str) and q.strip()],
            "sources": sources,
        }

    def get_grounding_metadata(self, food_name: str, field: str) -> Optional[dict]:
        cache_key = f"{food_name}::{field}"
        metadata = self._gemini_grounding_metadata.get(cache_key)
        if not metadata:
            return None
        return json.loads(json.dumps(metadata))

    # ──────────────────────────────────────────────────────────────────────────
    # Single-entry lookup: one FAISS search, both fields from the same entry
    # ──────────────────────────────────────────────────────────────────────────

    def _lookup_best_entry(
        self,
        food_name: str,
        top_k: int = 10,
        crop_image: Optional[Image.Image] = None,
    ):
        """
        Find the best matching food entry that has BOTH density and calories.
        Returns (entry, matched_name, sim, source_tag) or (None, None, None, None).

        One FAISS search; density and calories are read from the same DB entry so
        they are always nutritionally consistent.
        """
        normalized = self._normalize_food_name(food_name)
        candidates = []
        for candidate in self._apply_lexical_override(
            normalized,
            self._retrieve_candidates(food_name, crop_image=crop_image, top_k=top_k),
        ):
            idx = candidate["idx"]
            sim = candidate["clip_sim"]
            entry = self._derive_usda_density_entry(candidate["entry"])
            density = entry.get('density_g_ml')
            kcal    = entry.get('calories_per_100g')
            # Must have both fields
            if density is None or kcal is None:
                continue
            matched = candidate["matched"]

            src = entry.get('source', 'usda')
            min_sim = _MIN_SIM_BY_SOURCE.get(src, _MIN_FAISS_SIM) - 0.05
            if sim < min_sim:
                continue

            if any(sk in matched.lower() for sk in self._SKIP_WORDS):
                if not any(sk in normalized for sk in self._SKIP_WORDS):
                    continue

            if not self._is_reliable_density_entry(entry):
                continue

            if not self._words_overlap(normalized, matched.lower()):
                logger.info(f"Skip (no word overlap): '{food_name}' vs '{matched}'")
                continue
            if self._has_conflicting_form(normalized, matched.lower()):
                logger.info(f"Skip (conflicting form): '{food_name}' vs '{matched}'")
                continue

            compatibility_score = self._compatibility_adjustment(normalized, matched.lower(), field='density_g_ml')
            selection_score = candidate.get("selection_score", 0.0) + compatibility_score
            candidates.append((
                selection_score,
                candidate["cross_score"],
                candidate["lexical_score"],
                compatibility_score,
                candidate["rank_score"],
                idx,
                sim,
                entry,
                matched,
            ))

        if not candidates:
            return None, None, None, None

        candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], -item[4], -item[6]))
        selection_score, cross_score, lexical_score, compatibility_score, rank_score, idx, sim, entry, matched = candidates[0]
        if cross_score < _MIN_RERANK_SCORE:
            logger.info(f"[shared_lookup] '{food_name}' top rerank score too low ({cross_score:.2f}) - not using shared DB entry")
            return None, None, None, None
        src = entry.get('source', 'unified')
        method = entry.get('density_method', 'unknown')
        source_tag = f"{src}_faiss(sim={sim:.2f}),{method}"
        logger.info(
            f"[shared_lookup] '{food_name}' → '{matched}' "
            f"({source_tag}, clip={sim:.2f}, cross={cross_score:.2f}, lexical={lexical_score:.2f}, compat={compatibility_score:.2f}, rank={rank_score:.2f}, select={selection_score:.2f})"
        )
        return entry, matched, sim, source_tag

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def get_density(self, food_name: str, crop_image: Optional[Image.Image] = None, meal_context: Optional[str] = None) -> float:
        return self._get_density_with_match(food_name, crop_image=crop_image, meal_context=meal_context)[0]

    def _get_density_with_match(self, food_name: str, top_k: int = 10, crop_image: Optional[Image.Image] = None, meal_context: Optional[str] = None):
        """Return (density_g_ml, matched_name, source, entry|None)."""
        # Run unified FAISS and branded lookup in parallel; pick best via CrossEncoder.
        with ThreadPoolExecutor(max_workers=2) as _ex:
            if self._use_unified:
                _unified_fut = _ex.submit(self._lookup_unified, food_name, 'density_g_ml', top_k, crop_image)
            else:
                _unified_fut = _ex.submit(self._lookup_legacy_density, food_name)
            _branded_fut = _ex.submit(self._lookup_branded_fallback, food_name, 'density_g_ml')

        if self._use_unified:
            unified_result = _unified_fut.result()
            # FAO legacy fallback when unified FAISS has no density entry
            if unified_result[0] is None and self._fao_index is not None:
                fao_d, fao_m, fao_s = self._lookup_legacy_density(food_name)
                if fao_d is not None:
                    unified_result = (fao_d, fao_m, fao_s, None)
            branded_result = _branded_fut.result()
            density, matched, source, entry = self._pick_best_result(food_name, unified_result, branded_result)
        else:
            _r = _unified_fut.result()
            density, matched, source = _r[0], _r[1], _r[2]
            entry = None
            if density is None:
                density, matched, source, entry = _branded_fut.result()

        if density is not None:
            return float(density), matched, source, entry

        density = self._gemini_lookup(food_name, 'density_g_ml', meal_context=meal_context)
        if density is not None:
            return density, food_name, "gemini_grounding", None

        logger.warning(f"No density found for '{food_name}'")
        return None, None, None, None

    def get_calories_per_100g(self, food_name: str, crop_image: Optional[Image.Image] = None, meal_context: Optional[str] = None) -> float:
        return self._get_calories_with_match(food_name, crop_image=crop_image, meal_context=meal_context)[0]

    def _get_calories_with_match(self, food_name: str, top_k: int = 10, crop_image: Optional[Image.Image] = None, meal_context: Optional[str] = None):
        """Return (kcal_per_100g, matched_name, source). Entry not needed here."""
        # Run unified FAISS and branded lookup in parallel; pick best via CrossEncoder.
        with ThreadPoolExecutor(max_workers=2) as _ex:
            if self._use_unified:
                _unified_fut = _ex.submit(self._lookup_unified, food_name, 'calories_per_100g', top_k, crop_image)
            else:
                _unified_fut = _ex.submit(self._lookup_legacy_calories, food_name)
            _branded_fut = _ex.submit(self._lookup_branded_fallback, food_name, 'calories_per_100g')

        if self._use_unified:
            kcal, matched, source, _ = self._pick_best_result(
                food_name, _unified_fut.result(), _branded_fut.result()
            )
        else:
            _r = _unified_fut.result()
            kcal, matched, source = _r[0], _r[1], _r[2]
            if kcal is None:
                kcal, matched, source, _ = _branded_fut.result()

        if kcal is not None:
            return float(kcal), matched, source

        kcal = self._gemini_lookup(food_name, 'calories_per_100g', meal_context=meal_context)
        if kcal is not None:
            return kcal, food_name, "gemini_grounding"

        logger.warning(f"No calories found for '{food_name}'")
        return None, None, None

    def get_nutrition(self, food_name: str, volume_ml: float, crop_image: Optional[Image.Image] = None, meal_context: Optional[str] = None) -> dict:
        density, kcal_per_100g, density_matched, density_source, calorie_matched, calorie_source = self._resolve_nutrition(food_name, crop_image=crop_image, meal_context=meal_context)
        mass_g     = volume_ml * density
        total_kcal = (mass_g / 100.0) * kcal_per_100g
        return {
            "food_name":         food_name,
            "volume_ml":         volume_ml,
            "density_g_per_ml":  density,
            "mass_g":            round(mass_g, 1),
            "calories_per_100g": round(kcal_per_100g, 1),
            "total_calories":    round(total_kcal, 1),
            "density_matched":   density_matched,
            "density_source":    density_source,
            "calorie_matched":   calorie_matched,
            "calorie_source":    calorie_source,
        }

    def _resolve_nutrition(self, food_name: str, crop_image: Optional[Image.Image] = None, meal_context: Optional[str] = None):
        """
        ZOE-pipeline-aligned resolution — always kcal-first, then density separately.

        Step 1 — Find the best USDA/unified kcal match via CLIP-FAISS + cross-encoder.
                 Dry/mix/packet USDA entries are intentionally NOT blocked here;
                 the cross-encoder is the sole quality gate.  This lets "yellow rice"
                 correctly reach "Yellow rice with seasoning, dry packet mix, unprepared"
                 (343 kcal/100g) instead of falling back to a generic cooked-rice entry.

        Step 1a — If the winning kcal entry ALSO carries a reliable density reading,
                  reuse it immediately (nutritionally self-consistent, no second search).

        Step 2 — Density lookup — two queries tried in order:
                  (a) The matched food description from Step 1
                      → shares form-tokens (dry/mix/packet) with real density DB entries
                        so _has_conflicting_form does NOT wrongly block them.
                  (b) The raw ingredient name as fallback.
                 This mirrors the ZOE pipeline's
                 `_lookup_density_from_queries(rag_result["description"], ingredient_name)`.

        Note: _lookup_best_entry is NOT used as the primary path.  It can confuse
        cross-entry matches (e.g. "yellow rice" → "Wild rice, cooked" because that
        entry happens to have both fields) when the correct kcal entry has no density.

        Returns (density, kcal_per_100g, density_matched, density_source, calorie_matched, calorie_source).
        """
        # ── Step 1: kcal — unified FAISS and branded run in parallel, best wins ────
        kcal_entry: Optional[dict] = None
        with ThreadPoolExecutor(max_workers=2) as _ex:
            if self._use_unified:
                _kcal_unified_fut = _ex.submit(self._lookup_unified, food_name, 'calories_per_100g', 10, crop_image)
            else:
                _kcal_unified_fut = _ex.submit(self._lookup_legacy_calories, food_name)
            _kcal_branded_fut = _ex.submit(self._lookup_branded_fallback, food_name, 'calories_per_100g')

        if self._use_unified:
            kcal_per_100g, calorie_matched, calorie_source, kcal_entry = self._pick_best_result(
                food_name, _kcal_unified_fut.result(), _kcal_branded_fut.result()
            )
        else:
            _r = _kcal_unified_fut.result()
            kcal_per_100g, calorie_matched, calorie_source = _r[0], _r[1], _r[2]
            if kcal_per_100g is None:
                branded_kcal, branded_matched, branded_source, _ = _kcal_branded_fut.result()
                if branded_kcal is not None:
                    kcal_per_100g, calorie_matched, calorie_source = float(branded_kcal), branded_matched, branded_source

        if kcal_per_100g is None:
            kcal_per_100g_gem = self._gemini_lookup(food_name, 'calories_per_100g', meal_context=meal_context)
            if kcal_per_100g_gem is not None:
                kcal_per_100g   = kcal_per_100g_gem
                calorie_matched = food_name
                calorie_source  = "gemini_grounding"
            else:
                raise NutritionLookupError(food_name, ["calories_per_100g"])

        if kcal_entry is not None:
            kcal_entry = self._derive_usda_density_entry(kcal_entry)

        # ── Step 1a: reuse kcal entry density if reliable ──
        if kcal_entry is not None and self._is_reliable_density_entry(kcal_entry):
            density = float(kcal_entry['density_g_ml'])
            logger.info(
                f"[shared_entry] '{food_name}' → '{calorie_matched}' "
                f"({calorie_source}, reusing entry density={density}, kcal={kcal_per_100g})"
            )
            return density, float(kcal_per_100g), calorie_matched, calorie_source, calorie_matched, calorie_source

        # ── Step 2: density — matched description → raw name ─────────
        density: Optional[float] = None
        density_matched: Optional[str] = None
        density_source: Optional[str] = None

        density_queries = list(dict.fromkeys(
            q for q in [food_name, calorie_matched] if q and q.strip()
        ))

        for dq in density_queries:
            d, dm, ds, _de = self._get_density_with_match(dq, crop_image=crop_image, meal_context=meal_context)
            if d is not None and float(d) > 0:
                density, density_matched, density_source = d, dm, ds
                logger.info(
                    f"[density] '{food_name}' — found via query '{dq}' → '{dm}' "
                    f"({ds}, {d:.3f} g/ml)"
                )
                break

        if density is None:
            raise NutritionLookupError(food_name, ["density_g_ml"])

        return density, float(kcal_per_100g), density_matched, density_source, calorie_matched, calorie_source

    def get_nutrition_for_food(
        self,
        food_name: str,
        volume_ml: float,
        mass_g: Optional[float] = None,
        quantity: int = 1,
        crop_image: Optional[Image.Image] = None,
        meal_context: Optional[str] = None,
    ) -> dict:
        rag_candidates = [
            {
                "description": candidate["matched"],
                "clip_sim": round(float(candidate["clip_sim"]), 4),
                "cross_score": round(float(candidate["cross_score"]), 4),
                "selection_score": round(float(candidate["selection_score"]), 4),
            }
            for candidate in self._apply_lexical_override(
                self._normalize_food_name(food_name),
                self._retrieve_candidates(food_name, crop_image=crop_image, top_k=8, meal_context=meal_context),
            )[:5]
        ] if self._use_unified else []
        density, kcal_per_100g, density_matched, density_source, calorie_matched, calorie_source = self._resolve_nutrition(food_name, crop_image=crop_image, meal_context=meal_context)
        if mass_g is None or mass_g <= 0:
            mass_g = volume_ml * density
        total_kcal = (mass_g / 100.0) * kcal_per_100g
        return {
            "food_name":         food_name,
            "quantity":          quantity,
            "volume_ml":         volume_ml,
            "density_g_per_ml":  density,
            "mass_g":            round(float(mass_g), 1),
            "calories_per_100g": round(kcal_per_100g, 1),
            "total_calories":    round(total_kcal, 1),
            "density_matched":   density_matched,
            "density_source":    density_source,
            "calorie_matched":   calorie_matched,
            "calorie_source":    calorie_source,
            "density_grounding_metadata": (
                self.get_grounding_metadata(food_name, 'density_g_ml')
                if density_source == "gemini_grounding" else None
            ),
            "calorie_grounding_metadata": (
                self.get_grounding_metadata(food_name, 'calories_per_100g')
                if calorie_source == "gemini_grounding" else None
            ),
            "rag_candidates": rag_candidates,
        }
