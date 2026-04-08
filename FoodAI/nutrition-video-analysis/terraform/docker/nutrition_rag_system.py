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
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

# Minimum cosine similarity from FAISS to consider a candidate.
_MIN_FAISS_SIM = 0.45
_MIN_RERANK_SCORE = -5.0

# Per-source minimum similarity — FAO has only 634 entries and can match off-target
# at lower similarities; USDA/CoFID are larger and more tolerant.
_MIN_SIM_BY_SOURCE = {'fao': 0.65, 'usda': 0.50, 'cofid': 0.60}

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
        self._usda_faiss_path   = Path(usda_faiss_path)   if usda_faiss_path   else None
        self._usda_foods_path   = Path(usda_foods_path)   if usda_foods_path   else None
        self._usda_names_path   = Path(usda_names_path)   if usda_names_path   else None

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
        """Load indexes and retrieval/reranking models."""
        logger.info("Loading NutritionRAG...")

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
            logger.info(f"  Unified index:   {len(self._unified_foods)} entries (FAO+USDA+CoFID)")
            self._build_clip_index()
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
    def _is_reliable_density_entry(entry: dict) -> bool:
        density = entry.get('density_g_ml')
        if density is None:
            return False

        source = (entry.get('source') or '').strip().lower()
        method = (entry.get('density_method') or '').strip().lower()
        if source == 'fao':
            return method == 'fao_measured'
        if source == 'usda':
            return method.startswith('cup(')
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
        return any(
            qw in mw or mw in qw
            for qw in q_words for mw in m_words
        )

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
            if has_prepared_marker:
                score += 8.0
            if has_unprepared_marker:
                score -= 18.0

        # Strongly discourage falling back to a generic single-food label like "Rice"
        # when the query is more specific, e.g. "yellow rice".
        if len(query_tokens) >= 2 and normalized_matched in matched_tokens and len(matched_tokens) == 1:
            score -= 16.0

        # Favor richer USDA/CoFID dish descriptions over generic FAO commodity labels
        # when the query itself is a multi-word plated dish.
        if len(query_tokens) >= 2 and entry.get('source') == 'fao' and len(matched_tokens) <= 2:
            score -= 6.0

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
        return score

    def _retrieve_candidates(
        self,
        food_name: str,
        crop_image: Optional[Image.Image],
        top_k: int = 10,
    ) -> list[dict]:
        if not self._use_unified:
            return []

        normalized = self._normalize_food_name(food_name)

        # Sentence-transformer FAISS search (primary recall — same embedder used to build the index)
        st_hits: dict[int, float] = {}
        if self._unified_index is not None and self._embedder is not None:
            variants = list(dict.fromkeys([normalized, food_name.strip().lower()]))
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
            matched = self._unified_names[idx] if idx < len(self._unified_names) else ""
            if not matched:
                continue
            cross_score = float(self._cross_encoder.predict([(food_name, matched)])[0]) if self._cross_encoder else float("-inf")
            lexical_score = self._lexical_override_score(normalized, matched.lower())
            rank_score = self._candidate_rank_score(normalized, matched.lower(), entry, sim)
            candidates.append({
                "idx": idx,
                "entry": entry,
                "matched": matched,
                "clip_sim": sim,
                "cross_score": cross_score,
                "lexical_score": lexical_score,
                "rank_score": rank_score,
            })

        candidates.sort(key=lambda item: (-item["cross_score"], -item["lexical_score"], -item["rank_score"], -item["clip_sim"]))
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

    # ──────────────────────────────────────────────────────────────────────────
    # Unified lookup (density or calories)
    # ──────────────────────────────────────────────────────────────────────────

    _SKIP_WORDS = {'oil', 'fat', 'extract', 'powder', 'concentrate', 'flavoring', 'supplement'}

    # Hardcoded density overrides for foods where the unified DB cup-ratio measurement
    # is known to be wrong.  Checked before FAISS lookup in _resolve_nutrition.
    _DENSITY_HARDCODED: dict = {
        # Fresh tomatoes: ~1.0 g/ml (mostly water), DB cup ratio gives ~0.67
        "tomato":           1.00,
        "tomatoes":         1.00,
        # Falafel: dense fried chickpea balls (~1.07 g/ml), DB cup ratio gives 0.60
        "falafel":          1.07,
        "falafel ball":     1.07,
        "falafel balls":    1.07,
        # Hot / chili sauces: liquid ~1.06 g/ml, DB matches pepper solids giving 0.57
        "hot sauce":        1.06,
        "red hot sauce":    1.06,
        "red chili sauce":  1.06,
        "chili sauce":      1.06,
        "sriracha":         1.06,
        "hot chili sauce":  1.06,
        "chili garlic sauce": 1.06,
        # Cooking oils: pure oil ~0.92 g/ml
        "cooking oil":      0.92,
        "vegetable oil":    0.92,
        "olive oil":        0.91,
        "canola oil":       0.92,
        "sunflower oil":    0.92,
    }

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
        normalized = self._normalize_food_name(food_name)
        retrieved_candidates = self._apply_lexical_override(
            normalized,
            self._retrieve_candidates(food_name, crop_image=crop_image, top_k=top_k),
        )

        candidates = []
        for candidate in retrieved_candidates:
            idx = candidate["idx"]
            sim = candidate["clip_sim"]
            entry = candidate["entry"]
            value = entry.get(field)
            if value is None:
                continue
            matched = candidate["matched"]

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

            # Word-overlap guard against phonetic false positives
            if not self._words_overlap(normalized, matched.lower()):
                logger.info(f"Skip (no word overlap): '{food_name}' vs '{matched}'")
                continue
            # Form-conflict check:
            # Always applied for density (wrong-form density = wrong mass calculation).
            # For kcal: applied as normal EXCEPT when the food name is a multi-word
            # specific dish phrase that appears verbatim in the matched description.
            # This lets USDA dry-packet entries through for named dishes like:
            #   "yellow rice" → "Yellow rice with seasoning, dry packet mix, unprepared" ✓
            # while still blocking generic single-word queries from canned/sauce matches:
            #   "tomato"      → "Tomato products, canned, sauce"  ✗ (blocked)
            #   "diced onions"→ "Onions, frozen, chopped, unprepared" ✗ (phrase not in desc)
            if self._has_conflicting_form(normalized, matched.lower()):
                if field == 'density_g_ml':
                    logger.info(f"Skip (conflicting form for density): '{food_name}' vs '{matched}'")
                    continue
                # kcal: only pass through if query is a multi-word phrase found verbatim
                food_phrase = food_name.strip().lower()
                phrase_in_desc = (
                    len(food_phrase.split()) >= 2
                    and food_phrase in matched.lower()
                )
                if not phrase_in_desc:
                    logger.info(f"Skip (conflicting form for kcal): '{food_name}' vs '{matched}'")
                    continue

            rank_score = candidate["rank_score"]
            candidates.append((candidate["cross_score"], candidate["lexical_score"], rank_score, idx, sim, float(value), matched, entry))

        if not candidates:
            return None, None, None, None

        candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[4]))
        best = candidates[0]

        cross_score, lexical_score, rank_score, idx, sim, value, matched, entry = best
        if cross_score < _MIN_RERANK_SCORE:
            logger.info(f"[{field}] '{food_name}' top rerank score too low ({cross_score:.2f}) - using Gemini fallback path")
            return None, None, None, None
        src = entry.get('source', 'unified')
        source = f"{src}_faiss(sim={sim:.2f})"
        if field == 'density_g_ml':
            source += f",{entry.get('density_method', 'unknown')}"
        logger.info(
            f"[{field}] '{food_name}' → '{matched}' "
            f"({source}, clip={sim:.2f}, cross={cross_score:.2f}, lexical={lexical_score:.2f}, rank={rank_score:.2f}): {value}"
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
                entry   = self._usda_foods[idx]
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
                    f"Most foods are between 0.5 and 1.5 g/ml. "
                    f"Reply with ONLY a single decimal number like 0.85 — no units, no explanation, no other text."
                )
                lo, hi = 0.05, 3.0
            else:  # calories_per_100g
                prompt = (
                    f"How many kilocalories (kcal) are in 100 grams of '{food_name}'? "
                    f"Use standard nutritional data (e.g. USDA). "
                    f"Reply with ONLY a single integer or decimal number — no units, no explanation, no other text."
                )
                lo, hi = 1.0, 900.0

            def _call_new_sdk(with_grounding: bool):
                from google import genai as genai_new
                from google.genai import types
                client = genai_new.Client(api_key=self._gemini_api_key)
                config_kwargs: dict = {"temperature": 0.0}
                if with_grounding and hasattr(types, "Tool"):
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

            text = ""
            grounding_metadata = None
            try:
                response = _call_new_sdk(with_grounding=True)
                text = (response.text or "").strip()
                grounding_metadata = self._extract_grounding_metadata(response)
            except Exception as new_sdk_error:
                logger.warning(
                    f"Gemini grounding [{field}] new SDK failed for '{food_name}': {new_sdk_error}"
                )
                # Retry without grounding tool (some SDK versions differ)
                try:
                    response = _call_new_sdk(with_grounding=False)
                    text = (response.text or "").strip()
                except Exception:
                    pass

            if not text:
                # Last resort: legacy SDK
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=self._gemini_api_key)
                    model = genai.GenerativeModel(
                        self._gemini_model,
                        generation_config={"temperature": 0.0},
                    )
                    response = model.generate_content(prompt)
                    text = (response.text or "").strip()
                except Exception as legacy_err:
                    logger.warning(f"Gemini grounding [{field}] legacy SDK also failed for '{food_name}': {legacy_err}")

            # Try to parse the first number out of the response
            match = re.search(r'(\d+\.?\d*)', text)
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
                else:
                    logger.warning(
                        f"Gemini grounding [{field}] '{food_name}': parsed {val} but outside [{lo}, {hi}] range — text was: {text!r}"
                    )
            else:
                logger.warning(
                    f"Gemini grounding [{field}] '{food_name}': could not parse a number from response: {text!r}"
                )
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
            entry = candidate["entry"]
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

            candidates.append((
                candidate["cross_score"],
                candidate["lexical_score"],
                candidate["rank_score"],
                idx,
                sim,
                entry,
                matched,
            ))

        if not candidates:
            return None, None, None, None

        candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[4]))
        cross_score, lexical_score, rank_score, idx, sim, entry, matched = candidates[0]
        if cross_score < _MIN_RERANK_SCORE:
            logger.info(f"[shared_lookup] '{food_name}' top rerank score too low ({cross_score:.2f}) - not using shared DB entry")
            return None, None, None, None
        src = entry.get('source', 'unified')
        method = entry.get('density_method', 'unknown')
        source_tag = f"{src}_faiss(sim={sim:.2f}),{method}"
        logger.info(
            f"[shared_lookup] '{food_name}' → '{matched}' "
            f"({source_tag}, clip={sim:.2f}, cross={cross_score:.2f}, lexical={lexical_score:.2f}, rank={rank_score:.2f})"
        )
        return entry, matched, sim, source_tag

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def get_density(self, food_name: str, crop_image: Optional[Image.Image] = None) -> float:
        return self._get_density_with_match(food_name, crop_image=crop_image)[0]

    def _get_density_with_match(self, food_name: str, top_k: int = 10, crop_image: Optional[Image.Image] = None):
        """Return (density_g_ml, matched_name, source, entry|None)."""
        if self._use_unified:
            density, matched, source, entry = self._lookup_unified(food_name, 'density_g_ml', top_k, crop_image=crop_image)
        else:
            density, matched, source = self._lookup_legacy_density(food_name)
            entry = None

        if density is not None:
            return float(density), matched, source, entry

        density = self._gemini_lookup(food_name, 'density_g_ml')
        if density is not None:
            return density, food_name, "gemini_grounding", None

        logger.warning(f"No density found for '{food_name}' — all lookups failed")
        return 0.9, "fallback", "fallback_default", None

    def get_calories_per_100g(self, food_name: str, crop_image: Optional[Image.Image] = None) -> float:
        return self._get_calories_with_match(food_name, crop_image=crop_image)[0]

    def _get_calories_with_match(self, food_name: str, top_k: int = 10, crop_image: Optional[Image.Image] = None):
        """Return (kcal_per_100g, matched_name, source). Entry not needed here."""
        if self._use_unified:
            kcal, matched, source, _ = self._lookup_unified(food_name, 'calories_per_100g', top_k, crop_image=crop_image)
        else:
            kcal, matched, source = self._lookup_legacy_calories(food_name)

        if kcal is not None:
            return float(kcal), matched, source

        kcal = self._gemini_lookup(food_name, 'calories_per_100g')
        if kcal is not None:
            return kcal, food_name, "gemini_grounding"

        logger.warning(f"No calories found for '{food_name}' — all lookups failed")
        return 200.0, "fallback", "fallback_default"

    def get_nutrition(self, food_name: str, volume_ml: float, crop_image: Optional[Image.Image] = None) -> dict:
        density, kcal_per_100g, density_matched, density_source, calorie_matched, calorie_source = self._resolve_nutrition(food_name, crop_image=crop_image)
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

    def _resolve_nutrition(self, food_name: str, crop_image: Optional[Image.Image] = None):
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
        # Check hardcoded density overrides FIRST — these win over everything else,
        # including DB cup-ratio measurements that are known to be wrong (e.g. tomato,
        # falafel, hot sauce).
        _food_key = food_name.strip().lower()
        _hardcoded_density: Optional[float] = self._DENSITY_HARDCODED.get(_food_key)
        if _hardcoded_density is not None:
            logger.info(f"[density] '{food_name}' → hardcoded override {_hardcoded_density:.3f} g/ml")

        # ── Step 1: kcal via CLIP-FAISS + cross-encoder ──────────────────────────
        kcal_entry: Optional[dict] = None
        if self._use_unified:
            kcal_per_100g, calorie_matched, calorie_source, kcal_entry = self._lookup_unified(
                food_name, 'calories_per_100g', crop_image=crop_image
            )
        else:
            kcal_per_100g, calorie_matched, calorie_source = self._lookup_legacy_calories(food_name)

        if kcal_per_100g is None:
            kcal_per_100g_gem = self._gemini_lookup(food_name, 'calories_per_100g')
            if kcal_per_100g_gem is not None:
                kcal_per_100g   = kcal_per_100g_gem
                calorie_matched = food_name
                calorie_source  = "gemini_grounding"
            else:
                logger.warning(f"No calories found for '{food_name}' — using default")
                kcal_per_100g   = 200.0
                calorie_matched = "fallback"
                calorie_source  = "fallback_default"

        # ── Step 1a: reuse kcal entry density if reliable AND no hardcoded override ──
        # Skip this shortcut when a hardcoded density exists — the override always wins.
        if _hardcoded_density is None and kcal_entry is not None and self._is_reliable_density_entry(kcal_entry):
            density = float(kcal_entry['density_g_ml'])
            logger.info(
                f"[shared_entry] '{food_name}' → '{calorie_matched}' "
                f"({calorie_source}, reusing entry density={density}, kcal={kcal_per_100g})"
            )
            return density, float(kcal_per_100g), calorie_matched, calorie_source, calorie_matched, calorie_source

        # ── Step 2: density — hardcoded → matched description → raw name ─────────
        density: Optional[float] = _hardcoded_density
        density_matched: Optional[str] = food_name if _hardcoded_density is not None else None
        density_source: Optional[str]  = "hardcoded_override" if _hardcoded_density is not None else None

        density_queries = list(dict.fromkeys(
            q for q in [calorie_matched, food_name] if q and q.strip()
        ))

        for dq in density_queries if density is None else []:
            d, dm, ds, _de = self._get_density_with_match(dq, crop_image=crop_image)
            # Accept only real DB / Gemini hits — not the bare "fallback_default".
            if d is not None and float(d) > 0 and ds != "fallback_default":
                density, density_matched, density_source = d, dm, ds
                logger.info(
                    f"[density] '{food_name}' — found via query '{dq}' → '{dm}' "
                    f"({ds}, {d:.3f} g/ml)"
                )
                break

        if density is None:
            # All FAISS queries returned fallback_default; invoke Gemini / 0.9 default.
            density, density_matched, density_source, _ = self._get_density_with_match(
                food_name, crop_image=crop_image
            )

        return density, float(kcal_per_100g), density_matched, density_source, calorie_matched, calorie_source

    def get_nutrition_for_food(
        self,
        food_name: str,
        volume_ml: float,
        mass_g: Optional[float] = None,
        quantity: int = 1,
        crop_image: Optional[Image.Image] = None,
    ) -> dict:
        density, kcal_per_100g, density_matched, density_source, calorie_matched, calorie_source = self._resolve_nutrition(food_name, crop_image=crop_image)
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
        }
