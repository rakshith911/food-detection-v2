"""
Nutrition RAG System
Single unified FAISS index (FAO + USDA + CoFID) with cross-encoder re-ranking.
Google Search grounding (via Gemini) is the only fallback — no hardcoded keyword dicts.
"""
import json
import logging
import re
import numpy as np
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Minimum cosine similarity from FAISS to consider a candidate.
_MIN_FAISS_SIM = 0.45

# Per-source minimum similarity — FAO has only 634 entries and can match off-target
# at lower similarities; USDA/CoFID are larger and more tolerant.
_MIN_SIM_BY_SOURCE = {'fao': 0.65, 'usda': 0.50, 'cofid': 0.60}

# Higher threshold required before trusting macro_only density values (baked goods
# have air pockets that the formula can't account for).
_MACRO_ONLY_MIN_SIM = 0.70


class NutritionRAG:
    """
    Unified nutrition lookup:
      1. FAISS search over FAO + USDA + CoFID (11k+ foods)
      2. Cross-encoder re-ranking of top candidates
      3. Gemini grounding (Google Search) as sole fallback
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
        usda_faiss_path: Optional[Path] = None,
        usda_foods_path: Optional[Path] = None,
        usda_names_path: Optional[Path] = None,
        usda_density_faiss_path: Optional[Path] = None,
        usda_density_path: Optional[Path] = None,
        usda_density_names_path: Optional[Path] = None,
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-flash-latest",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self._unified_faiss_path   = Path(unified_faiss_path)   if unified_faiss_path   else None
        self._unified_foods_path   = Path(unified_foods_path)   if unified_foods_path   else None
        self._unified_names_path   = Path(unified_food_names_path) if unified_food_names_path else None

        # Legacy paths kept for backward compat
        self._fao_faiss_path    = Path(fao_faiss_path)    if fao_faiss_path    else None
        self._fao_density_path  = Path(fao_density_path)  if fao_density_path  else None
        self._fao_names_path    = Path(fao_names_path)    if fao_names_path    else None
        self._usda_faiss_path   = Path(usda_faiss_path)   if usda_faiss_path   else None
        self._usda_foods_path   = Path(usda_foods_path)   if usda_foods_path   else None
        self._usda_names_path   = Path(usda_names_path)   if usda_names_path   else None

        self._gemini_api_key    = gemini_api_key
        self._gemini_model      = gemini_model
        self._embedding_model   = embedding_model

        # Runtime objects (populated in load())
        self._embedder: Optional[SentenceTransformer] = None
        self._unified_index = None
        self._unified_foods: list = []
        self._unified_names: list = []

        # Legacy objects (populated if unified not available)
        self._fao_index   = None
        self._fao_density = []
        self._fao_names   = []
        self._usda_index  = None
        self._usda_foods  = []
        self._usda_names  = []

        self._use_unified = False
        self._gemini_cache: dict = {}  # food_name+field -> value
        self._gemini_grounding_metadata: dict = {}  # food_name+field -> grounding metadata

    # ──────────────────────────────────────────────────────────────────────────
    # Loading
    # ──────────────────────────────────────────────────────────────────────────

    def load(self):
        """Load indexes, embedding model, and cross-encoder re-ranker."""
        logger.info("Loading NutritionRAG...")

        self._embedder = SentenceTransformer(self._embedding_model)
        logger.info(f"  Embedding model: {self._embedding_model}")

        # Prefer unified index
        if (self._unified_faiss_path and self._unified_faiss_path.exists()
                and self._unified_foods_path and self._unified_foods_path.exists()):
            self._unified_index = faiss.read_index(str(self._unified_faiss_path))
            with open(self._unified_foods_path) as f:
                self._unified_foods = json.load(f)
            with open(self._unified_names_path) as f:
                self._unified_names = json.load(f)
            self._use_unified = True
            logger.info(f"  Unified index:   {len(self._unified_foods)} entries (FAO+USDA+CoFID)")
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

    # ──────────────────────────────────────────────────────────────────────────
    # Label normalisation
    # ──────────────────────────────────────────────────────────────────────────

    _LABEL_ALIASES = [
        (r'\bcrumb\s+bar\b', 'crumble'),
        (r'\bcrumb\b',       'crumble'),
        (r'\bcrisp\b',       'crumble'),
        (r'\bfritter\b',     'pastry'),
        (r'\bshortbread\b',  'cookie'),
    ]
    _FORM_WORDS_RE = (
        r'\b(bar|slice|slices|piece|pieces|wedge|wedges|portion|portions|'
        r'serving|servings|ball|balls|patty|patties|log|logs|chunk|chunks)\b'
    )

    @classmethod
    def _normalize_food_name(cls, name: str) -> str:
        name = re.sub(r'\([^)]*\)', '', name)
        name = re.sub(r',.*', '', name)
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

    # ──────────────────────────────────────────────────────────────────────────
    # Word-overlap plausibility guard
    # ──────────────────────────────────────────────────────────────────────────

    # Generic cooking/serving words that don't identify a food — excluded from word overlap check
    _GENERIC_WORDS = {
        'sauce', 'soup', 'food', 'dish', 'meal', 'item', 'product', 'type',
        'style', 'with', 'from', 'made', 'home', 'store', 'brand', 'kind',
    }

    @classmethod
    def _words_overlap(cls, query_lower: str, matched_lower: str) -> bool:
        """
        At least one specific (non-generic) ≥4-char word from the query must appear
        as a substring in the matched name, to block false positives like
        'caramel sauce' → 'tomato chili sauce' (only 'sauce' matches, which is generic).
        """
        q_words = {w for w in query_lower.split() if len(w) >= 4 and w not in cls._GENERIC_WORDS}
        m_words = set(re.sub(r'[^\w ]', ' ', matched_lower).split())
        if not q_words:
            # All query words are generic — fall back to any overlap
            q_words = {w for w in query_lower.split() if len(w) >= 4}
        if not q_words:
            return True
        return any(
            qw in mw or mw in qw
            for qw in q_words for mw in m_words if len(mw) >= 4
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Unified lookup (density or calories)
    # ──────────────────────────────────────────────────────────────────────────

    _SKIP_WORDS = {'oil', 'fat', 'extract', 'powder', 'concentrate', 'flavoring', 'supplement'}

    def _lookup_unified(self, food_name: str, field: str, top_k: int = 20):
        """
        Search unified FAISS index, re-rank with cross-encoder.
        field: 'density_g_ml' or 'calories_per_100g'
        Returns (value, matched_name, source) or (None, None, None).
        """
        normalized = self._normalize_food_name(food_name)
        raw_lower  = food_name.strip().lower()
        variants   = list(dict.fromkeys([normalized, raw_lower]))

        hits = self._faiss_search(self._unified_index, variants, top_k)

        # Build candidate list filtered to entries that have the field
        candidates = []
        for idx, sim in hits:
            if not (0 <= idx < len(self._unified_foods)):
                continue
            entry = self._unified_foods[idx]
            value = entry.get(field)
            if value is None:
                continue
            matched = self._unified_names[idx] if idx < len(self._unified_names) else ""

            # Apply per-source minimum similarity thresholds
            src = entry.get('source', 'usda')
            min_sim = _MIN_SIM_BY_SOURCE.get(src, _MIN_FAISS_SIM)
            if sim < min_sim:
                continue

            # Skip irrelevant entry types (oil/extract/powder) unless query also is that type
            if any(sk in matched.lower() for sk in self._SKIP_WORDS):
                if not any(sk in normalized for sk in self._SKIP_WORDS):
                    continue

            # For density: macro_only needs high FAISS sim
            if field == 'density_g_ml':
                method = entry.get('density_method', '')
                if method == 'macro_only' and sim < _MACRO_ONLY_MIN_SIM:
                    continue

            # Word-overlap guard against phonetic false positives
            if not self._words_overlap(normalized, matched.lower()):
                logger.info(f"Skip (no word overlap): '{food_name}' vs '{matched}'")
                continue

            candidates.append((idx, sim, float(value), matched, entry))

        if not candidates:
            return None, None, None

        # For density: prefer cup-based measurements over macro_only as a soft tie-breaker.
        # Cup entries capture real serving density (air pockets included); macro_only does not.
        # Only apply the preference when a cup-based entry has a competitive similarity
        # (within 0.15 of the best match) — avoids dropping a clearly better macro_only match.
        if field == 'density_g_ml' and candidates:
            best_sim = candidates[0][1]
            cup_candidates = [
                c for c in candidates
                if c[4].get('density_method', '') != 'macro_only'
                and best_sim - c[1] <= 0.15  # within 0.15 of best similarity
            ]
            if cup_candidates:
                candidates = cup_candidates

        # Pick best by FAISS similarity (already sorted desc)
        best = candidates[0]

        idx, sim, value, matched, entry = best
        src = entry.get('source', 'unified')
        source = f"{src}_faiss(sim={sim:.2f})"
        if field == 'density_g_ml':
            source += f",{entry.get('density_method', 'unknown')}"
        logger.info(f"[{field}] '{food_name}' → '{matched}' ({source}): {value}")
        return value, matched, source

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
                return density, matched, f"fao_faiss(sim={sim:.2f})"

        if self._usda_index is not None:
            for idx, sim in self._faiss_search(self._usda_index, variants, top_k):
                if sim < 0.55:
                    break
                if not (0 <= idx < len(self._usda_foods)):
                    continue
                entry   = self._usda_foods[idx]
                density = entry.get('density_g_ml')
                if density is None:
                    continue
                matched = self._usda_names[idx] if idx < len(self._usda_names) else ""
                method  = entry.get('density_method', 'usda')
                if method == 'macro_only' and sim < _MACRO_ONLY_MIN_SIM:
                    continue
                if not self._words_overlap(normalized, matched.lower()):
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
                entry = self._usda_foods[idx]
                kcal  = float(entry.get('calories_per_100g', 0))
                return kcal, matched, f"usda_faiss(sim={sim:.2f})"

        return None, None, None

    # ──────────────────────────────────────────────────────────────────────────
    # Gemini grounding fallback
    # ──────────────────────────────────────────────────────────────────────────

    def _gemini_lookup(self, food_name: str, field: str) -> Optional[float]:
        """
        Query Gemini with Google Search grounding for density or calories.
        field: 'density_g_ml' or 'calories_per_100g'
        """
        if not self._gemini_api_key:
            return None
        cache_key = f"{food_name}::{field}"
        if cache_key in self._gemini_cache:
            return self._gemini_cache[cache_key]
        try:
            if field == 'density_g_ml':
                prompt = (
                    f"What is the density of '{food_name}' in grams per milliliter (g/ml) "
                    f"as a typical cooked or served food portion? "
                    f"Reply with ONLY a single decimal number like 0.85 — no units, no explanation."
                )
                lo, hi = 0.05, 3.0
            else:  # calories_per_100g
                prompt = (
                    f"How many kilocalories (kcal) are in 100 grams of '{food_name}'? "
                    f"Use standard nutritional data. "
                    f"Reply with ONLY a single integer or decimal number — no units, no explanation."
                )
                lo, hi = 1.0, 900.0

            text = ""
            grounding_metadata = None
            try:
                from google import genai as genai_new
                from google.genai import types

                client = genai_new.Client(api_key=self._gemini_api_key)
                config_kwargs = {"temperature": 0.0}

                # Prefer Google Search grounding when the installed SDK supports it,
                # but degrade cleanly to a plain Gemini lookup if the tool surface
                # differs across local/prod environments.
                if hasattr(types, "Tool"):
                    if hasattr(types, "GoogleSearch"):
                        config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
                    elif hasattr(types, "GoogleSearchRetrieval"):
                        config_kwargs["tools"] = [
                            types.Tool(google_search_retrieval=types.GoogleSearchRetrieval())
                        ]

                response = client.models.generate_content(
                    model=self._gemini_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                text = (response.text or "").strip()
                grounding_metadata = self._extract_grounding_metadata(response)
            except Exception as new_sdk_error:
                logger.warning(
                    f"Gemini grounding [{field}] new SDK failed for '{food_name}', "
                    f"retrying legacy client without grounding tool: {new_sdk_error}"
                )
                import google.generativeai as genai

                genai.configure(api_key=self._gemini_api_key)
                model = genai.GenerativeModel(
                    self._gemini_model,
                    generation_config={"temperature": 0.0},
                )
                response = model.generate_content(prompt)
                text = (response.text or "").strip()

            match = re.search(r'\b(\d+\.?\d*)\b', text)
            if match:
                val = float(match.group(1))
                if lo <= val <= hi:
                    logger.info(f"Gemini grounding [{field}] '{food_name}': {val}")
                    self._gemini_cache[cache_key] = val
                    if grounding_metadata and grounding_metadata.get("grounded"):
                        grounding_metadata.update({
                            "food_name": food_name,
                            "field": field,
                            "value": val,
                            "model": self._gemini_model,
                        })
                        self._gemini_grounding_metadata[cache_key] = grounding_metadata
                    return val
        except Exception as e:
            logger.warning(f"Gemini grounding [{field}] failed for '{food_name}': {e}")
        return None

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

    def _lookup_best_entry(self, food_name: str, top_k: int = 20):
        """
        Find the best matching food entry that has BOTH density and calories.
        Returns (entry, matched_name, sim, source_tag) or (None, None, None, None).

        One FAISS search; density and calories are read from the same DB entry so
        they are always nutritionally consistent.
        """
        normalized = self._normalize_food_name(food_name)
        raw_lower  = food_name.strip().lower()
        variants   = list(dict.fromkeys([normalized, raw_lower]))

        hits = self._faiss_search(self._unified_index, variants, top_k)

        candidates = []
        for idx, sim in hits:
            if not (0 <= idx < len(self._unified_foods)):
                continue
            entry   = self._unified_foods[idx]
            density = entry.get('density_g_ml')
            kcal    = entry.get('calories_per_100g')
            # Must have both fields
            if density is None or kcal is None:
                continue
            matched = self._unified_names[idx] if idx < len(self._unified_names) else ""

            src = entry.get('source', 'usda')
            min_sim = _MIN_SIM_BY_SOURCE.get(src, _MIN_FAISS_SIM)
            if sim < min_sim:
                continue

            if any(sk in matched.lower() for sk in self._SKIP_WORDS):
                if not any(sk in normalized for sk in self._SKIP_WORDS):
                    continue

            method = entry.get('density_method', '')
            if method == 'macro_only' and sim < _MACRO_ONLY_MIN_SIM:
                continue

            if not self._words_overlap(normalized, matched.lower()):
                logger.info(f"Skip (no word overlap): '{food_name}' vs '{matched}'")
                continue

            candidates.append((idx, sim, entry, matched))

        if not candidates:
            return None, None, None, None

        # Prefer cup-based density within 0.15 of the best sim
        best_sim = candidates[0][1]
        cup_candidates = [
            c for c in candidates
            if c[2].get('density_method', '') != 'macro_only'
            and best_sim - c[1] <= 0.15
        ]
        if cup_candidates:
            candidates = cup_candidates

        idx, sim, entry, matched = candidates[0]
        src = entry.get('source', 'unified')
        method = entry.get('density_method', 'unknown')
        source_tag = f"{src}_faiss(sim={sim:.2f}),{method}"
        return entry, matched, sim, source_tag

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def get_density(self, food_name: str) -> float:
        return self._get_density_with_match(food_name)[0]

    def _get_density_with_match(self, food_name: str, top_k: int = 20):
        """Return (density_g_ml, matched_name, source). Used standalone if needed."""
        if self._use_unified:
            density, matched, source = self._lookup_unified(food_name, 'density_g_ml', top_k)
        else:
            density, matched, source = self._lookup_legacy_density(food_name)

        if density is not None:
            return float(density), matched, source

        density = self._gemini_lookup(food_name, 'density_g_ml')
        if density is not None:
            return density, food_name, "gemini_grounding"

        logger.warning(f"No density found for '{food_name}' — all lookups failed")
        return 0.9, "fallback", "fallback_default"

    def get_calories_per_100g(self, food_name: str) -> float:
        return self._get_calories_with_match(food_name)[0]

    def _get_calories_with_match(self, food_name: str, top_k: int = 20):
        """Return (kcal_per_100g, matched_name, source). Used standalone if needed."""
        if self._use_unified:
            kcal, matched, source = self._lookup_unified(food_name, 'calories_per_100g', top_k)
        else:
            kcal, matched, source = self._lookup_legacy_calories(food_name)

        if kcal is not None:
            return float(kcal), matched, source

        kcal = self._gemini_lookup(food_name, 'calories_per_100g')
        if kcal is not None:
            return kcal, food_name, "gemini_grounding"

        logger.warning(f"No calories found for '{food_name}' — all lookups failed")
        return 200.0, "fallback", "fallback_default"

    def get_nutrition(self, food_name: str, volume_ml: float) -> dict:
        density, kcal_per_100g, matched, source = self._resolve_nutrition(food_name)
        mass_g     = volume_ml * density
        total_kcal = (mass_g / 100.0) * kcal_per_100g
        return {
            "food_name":         food_name,
            "volume_ml":         volume_ml,
            "density_g_per_ml":  density,
            "mass_g":            round(mass_g, 1),
            "calories_per_100g": round(kcal_per_100g, 1),
            "total_calories":    round(total_kcal, 1),
        }

    def _resolve_nutrition(self, food_name: str):
        """
        One FAISS search → density + calories from the SAME entry.
        Falls back to Gemini grounding per-field only if the entry is missing a value.
        Returns (density, kcal_per_100g, matched_name, source_tag).
        """
        if self._use_unified:
            entry, matched, sim, source_tag = self._lookup_best_entry(food_name)
        else:
            entry, matched, sim, source_tag = None, None, None, None

        if entry is not None:
            density      = float(entry['density_g_ml'])
            kcal_per_100g = float(entry['calories_per_100g'])
            return density, kcal_per_100g, matched, source_tag

        # Unified entry not found — try Gemini for each field independently
        density = self._gemini_lookup(food_name, 'density_g_ml')
        kcal    = self._gemini_lookup(food_name, 'calories_per_100g')

        density = density if density is not None else 0.9
        kcal    = kcal    if kcal    is not None else 200.0
        src     = "gemini_grounding" if self._gemini_api_key else "fallback_default"
        return density, kcal, food_name, src

    def get_nutrition_for_food(
        self,
        food_name: str,
        volume_ml: float,
        mass_g: Optional[float] = None,
        quantity: int = 1,
    ) -> dict:
        density, kcal_per_100g, matched, source_tag = self._resolve_nutrition(food_name)
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
            "density_matched":   matched,
            "density_source":    source_tag,
            "calorie_matched":   matched,
            "calorie_source":    source_tag,
            "density_grounding_metadata": (
                self.get_grounding_metadata(food_name, 'density_g_ml')
                if source_tag == "gemini_grounding" else None
            ),
            "calorie_grounding_metadata": (
                self.get_grounding_metadata(food_name, 'calories_per_100g')
                if source_tag == "gemini_grounding" else None
            ),
        }
