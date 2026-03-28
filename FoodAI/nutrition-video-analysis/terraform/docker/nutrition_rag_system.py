"""
Nutrition RAG System
Uses pre-built FAISS indexes for fast density (FAO) and calorie (USDA) lookups.
"""
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional

import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Fallback densities (g/ml) for common foods not in FAO database
DENSITY_FALLBACKS = {
    # Grains / starchy
    'rice': 0.85, 'pasta': 0.90, 'noodle': 0.90, 'bread': 0.40,
    # Proteins
    'chicken': 1.05, 'beef': 1.05, 'pork': 1.05, 'fish': 1.05, 'salmon': 1.05,
    # Dairy / fats
    'egg': 1.03, 'cheese': 1.10, 'butter': 0.91, 'oil': 0.92,
    # Vegetables
    'potato': 1.08, 'carrot': 1.05, 'broccoli': 0.37, 'lettuce': 0.20,
    'tomato': 1.00, 'onion': 1.02, 'mushroom': 0.60, 'spinach': 0.30,
    'brussels': 0.55, 'celery': 0.60, 'cucumber': 0.96, 'pepper': 0.60,
    'zucchini': 0.95, 'eggplant': 0.42,
    # Fruits
    'apple': 0.87, 'banana': 0.94, 'orange': 0.88, 'grape': 1.00,
    'olive': 0.90, 'avocado': 1.02, 'berry': 0.87, 'mango': 0.93,
    # Liquids / sauces
    'milk': 1.03, 'yogurt': 1.05, 'cream': 0.99, 'sauce': 1.05,
    'soup': 1.00, 'curry': 1.05, 'stew': 1.05, 'caramel': 1.30,
    'syrup': 1.30, 'honey': 1.40, 'jam': 1.30, 'gravy': 1.05,
    # Baked goods / desserts
    'cake': 0.50, 'cookie': 0.60, 'brownie': 0.75, 'muffin': 0.55,
    'pie': 0.65, 'tart': 0.65, 'crumble': 0.60, 'crisp': 0.55,
    'bar': 0.65, 'pastry': 0.55, 'croissant': 0.35, 'donut': 0.45,
    'waffle': 0.45, 'pancake': 0.55, 'scone': 0.55,
    # Other
    'chocolate': 1.30, 'salad': 0.40, 'dressing': 1.00,
    'hummus': 1.00, 'tofu': 1.05, 'nuts': 0.65,
}
DEFAULT_DENSITY = 0.90  # g/ml — conservative default for mixed foods

# Minimum cosine similarity thresholds.
# FAO index is smaller / more agricultural → needs stricter threshold to avoid wrong matches.
# USDA index is large and food-specific → more lenient threshold is safe.
_MIN_FAO_SIMILARITY = 0.65
_MIN_USDA_SIMILARITY = 0.50

# Calorie fallbacks (kcal/100g) for common foods USDA may not match well at the threshold.
CALORIE_FALLBACKS = {
    'olive': 116, 'kalamata': 116,
    'avocado': 160, 'walnut': 654, 'almond': 579, 'cashew': 553, 'peanut': 567,
    'bacon': 541, 'sausage': 301, 'ham': 145, 'salami': 336,
    'pizza': 266, 'burger': 295, 'hotdog': 290, 'sandwich': 250,
    'pie': 260, 'tart': 300, 'crumble': 190, 'crisp': 190,
    'bar': 380, 'brownie': 415, 'muffin': 370, 'scone': 364,
    'croissant': 406, 'donut': 452, 'waffle': 291, 'pancake': 227,
    'caramel': 380, 'chocolate': 546, 'ice cream': 207, 'gelato': 170,
    'hummus': 177, 'falafel': 333, 'tahini': 595,
    'tofu': 76, 'tempeh': 195, 'edamame': 121,
    'quinoa': 368, 'lentil': 116, 'chickpea': 164, 'bean': 127,
}
DEFAULT_CALORIES = 200.0  # kcal/100g fallback


class NutritionRAG:
    """
    Nutrition lookup using pre-built FAISS indexes.
    - FAO density index:  food name -> density (g/ml)
    - USDA calorie index: food name -> calories_per_100g
    """

    def __init__(
        self,
        fao_faiss_path: Path,
        fao_density_path: Path,
        fao_names_path: Path,
        usda_faiss_path: Path,
        usda_foods_path: Path,
        usda_names_path: Path,
        usda_density_faiss_path: Optional[Path] = None,
        usda_density_path: Optional[Path] = None,
        usda_density_names_path: Optional[Path] = None,
        gemini_api_key: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.fao_faiss_path = Path(fao_faiss_path)
        self.fao_density_path = Path(fao_density_path)
        self.fao_names_path = Path(fao_names_path)
        self.usda_faiss_path = Path(usda_faiss_path)
        self.usda_foods_path = Path(usda_foods_path)
        self.usda_names_path = Path(usda_names_path)
        self.usda_density_faiss_path = Path(usda_density_faiss_path) if usda_density_faiss_path else None
        self.usda_density_path = Path(usda_density_path) if usda_density_path else None
        self.usda_density_names_path = Path(usda_density_names_path) if usda_density_names_path else None
        self._gemini_api_key = gemini_api_key

        self._embedder: Optional[SentenceTransformer] = None
        self._fao_index = None
        self._fao_density: list = []        # [{description, density_g_ml}]
        self._fao_names: list = []          # [str]
        self._usda_index = None
        self._usda_foods: list = []         # [{fdc_id, description, calories_per_100g, ...}]
        self._usda_names: list = []         # [str]
        self._usda_density_index = None
        self._usda_density: list = []       # [{fdc_id, density_g_ml}]
        self._usda_density_names: list = [] # [str]
        self._gemini_density_cache: dict = {}   # food_name -> density (avoid repeat API calls)
        self._embedding_model_name = embedding_model

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self):
        """Load all indexes and data files."""
        print("Loading NutritionRAG (FAO density + USDA calories + USDA density)...")
        self._embedder = SentenceTransformer(self._embedding_model_name)

        # FAO density
        if not self.fao_faiss_path.exists():
            raise FileNotFoundError(f"FAO FAISS index not found: {self.fao_faiss_path}")
        self._fao_index = faiss.read_index(str(self.fao_faiss_path))
        with open(self.fao_density_path) as f:
            self._fao_density = json.load(f)
        with open(self.fao_names_path) as f:
            self._fao_names = json.load(f)
        print(f"  FAO density index:  {len(self._fao_density)} entries")

        # USDA calories
        if not self.usda_faiss_path.exists():
            raise FileNotFoundError(f"USDA FAISS index not found: {self.usda_faiss_path}")
        self._usda_index = faiss.read_index(str(self.usda_faiss_path))
        with open(self.usda_foods_path) as f:
            self._usda_foods = json.load(f)
        with open(self.usda_names_path) as f:
            self._usda_names = json.load(f)
        print(f"  USDA calorie index: {len(self._usda_foods)} entries")

        # USDA density (from volumetric portion measurements — optional)
        if (self.usda_density_faiss_path and self.usda_density_faiss_path.exists()
                and self.usda_density_path and self.usda_density_path.exists()):
            self._usda_density_index = faiss.read_index(str(self.usda_density_faiss_path))
            with open(self.usda_density_path) as f:
                self._usda_density = json.load(f)
            with open(self.usda_density_names_path) as f:
                self._usda_density_names = json.load(f)
            print(f"  USDA density index: {len(self._usda_density)} entries")
        else:
            print("  USDA density index: not available (skipping)")

        print("NutritionRAG ready")

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    # Parenthetical descriptors Gemini often appends that hurt embedding similarity.
    _LABEL_STRIP_RE = None  # compiled lazily

    # Aliases applied before stripping — map Gemini's varied descriptions to canonical
    # food names that embed closer to USDA/FAO entries.
    _LABEL_ALIASES = [
        # "crumb bar" / "crumb" → "crumble" (apple crumb bar → apple crumble)
        (r'\bcrumb\s+bar\b', 'crumble'),
        (r'\bcrumb\b', 'crumble'),
        # apple crisp ≈ apple crumble in USDA
        (r'\bcrisp\b', 'crumble'),
        # fritter → pastry for lookup purposes
        (r'\bfritter\b', 'pastry'),
        # biscuit (UK) → cookie; biscuit (US gravy biscuit) left as-is handled by context
        (r'\bshortbread\b', 'cookie'),
    ]

    # Form/portion words stripped after aliasing — these are descriptors, not food identity
    _FORM_WORDS_RE = r'\b(bar|slice|slices|piece|pieces|wedge|wedges|portion|portions|' \
                     r'serving|servings|ball|balls|patty|patties|log|logs|chunk|chunks)\b'

    @classmethod
    def _normalize_food_name(cls, name: str) -> str:
        """Strip parenthetical descriptors and cooking-state qualifiers for cleaner FAISS lookup.

        Examples:
            "Kalamata Olives (pitted)"  → "kalamata olives"
            "Brussels Sprouts (cooked)" → "brussels sprouts"
            "Chopped Celery, raw"       → "celery"
            "apple crumb bar"           → "apple crumble"
        """
        import re
        name = re.sub(r'\([^)]*\)', '', name)     # remove (...)
        name = re.sub(r',.*', '', name)            # remove everything after first comma
        # apply food-name aliases (crumb → crumble, crisp → crumble, etc.)
        for pattern, replacement in cls._LABEL_ALIASES:
            name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
        # strip common cooking-state/prep words that don't affect nutrition class
        qualifiers = r'\b(chopped|sliced|diced|cooked|raw|fresh|dried|grilled|fried|'  \
                     r'boiled|steamed|baked|roasted|sautéed|sauteed|mashed|whole|'    \
                     r'halved|quartered|shredded|grated|minced|pitted|peeled|'        \
                     r'crumbled|crushed|mixed|assorted)\b'
        name = re.sub(qualifiers, '', name, flags=re.IGNORECASE)
        # strip form/portion words (bar, slice, piece, wedge...)
        name = re.sub(cls._FORM_WORDS_RE, '', name, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', name).strip().lower()

    def _embed(self, text: str) -> np.ndarray:
        vec = self._embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32)

    def _multi_search(self, index, labels: list, top_k: int):
        """Search FAISS with multiple label variants; return merged (sim, idx) list sorted by sim desc."""
        all_hits = {}  # idx -> best sim
        for label in labels:
            if not label.strip():
                continue
            vec = self._embed(label)
            dists, idxs = index.search(vec, top_k)
            for i in range(top_k):
                idx = int(idxs[0][i]); sim = float(dists[0][i])
                if idx not in all_hits or sim > all_hits[idx]:
                    all_hits[idx] = sim
        return sorted(all_hits.items(), key=lambda x: -x[1])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Entries whose descriptions contain these words are unlikely to be the right
    # match for a whole food item — skip them when better candidates exist.
    _DENSITY_SKIP = {'oil', 'fat', 'extract', 'powder', 'concentrate', 'flavoring'}
    _CALORIE_SKIP = {'oil', 'fat', 'extract', 'powder', 'concentrate', 'flavoring', 'supplement'}

    def get_density(self, food_name: str, top_k: int = 5) -> float:
        """Return density in g/ml for the given food name."""
        return self._get_density_with_match(food_name, top_k)[0]

    def _get_density_with_match(self, food_name: str, top_k: int = 5):
        """
        Return (density_g_ml, matched_name, source).

        Lookup order:
          1. FAO FAISS  (agricultural authority, 634 entries, threshold 0.65)
          2. USDA density FAISS  (from cup/volume portions, 2452 entries, threshold 0.60)
          3. Keyword fallback table
          4. Gemini grounding (Google Search) — only when api key available
          5. Hard default (0.90 g/ml)
        """
        normalized = self._normalize_food_name(food_name)
        # Also try the raw lowercased label as a variant (in case normalization over-strips)
        raw_lower = food_name.strip().lower()
        variants = list(dict.fromkeys([normalized, raw_lower]))  # deduplicated, normalized first
        vec = self._embed(normalized) if self._embedder else None

        # ── 1. FAO ──────────────────────────────────────────────────────────
        if self._fao_index is not None and self._embedder is not None:
            try:
                hits = self._multi_search(self._fao_index, variants, top_k)
                food_lower = normalized
                for idx, sim in hits:
                    if sim < _MIN_FAO_SIMILARITY:
                        break
                    if not (0 <= idx < len(self._fao_density)):
                        continue
                    matched = self._fao_names[idx] if idx < len(self._fao_names) else ""
                    if any(skip in matched.lower() for skip in self._DENSITY_SKIP):
                        if not any(skip in food_lower for skip in self._DENSITY_SKIP):
                            continue
                    density = float(self._fao_density[idx].get("density_g_ml", DEFAULT_DENSITY))
                    logger.info(f"FAO density '{food_name}' → '{matched}' (sim={sim:.3f}): {density:.3f} g/ml")
                    return density, matched, f"fao_faiss(sim={sim:.2f})"
            except Exception as e:
                logger.warning(f"FAO FAISS lookup failed for '{food_name}': {e}")

        # ── 2. USDA density (combined cup+macro, all 7756 foods) ─────────────
        if self._usda_index is not None and self._embedder is not None:
            try:
                hits = self._multi_search(self._usda_index, variants, top_k)
                food_lower = normalized
                for idx, sim in hits:
                    if sim < 0.55:
                        break
                    if not (0 <= idx < len(self._usda_foods)):
                        continue
                    entry = self._usda_foods[idx]
                    density = entry.get("density_g_ml")
                    if density is None:
                        continue
                    matched = self._usda_names[idx] if idx < len(self._usda_names) else ""
                    if any(skip in matched.lower() for skip in self._DENSITY_SKIP):
                        if not any(skip in food_lower for skip in self._DENSITY_SKIP):
                            continue
                    method = entry.get("density_method", "usda")
                    # macro_only densities ignore air pockets in baked goods — require a
                    # high-confidence match before trusting them.
                    if method == "macro_only" and sim < 0.70:
                        continue
                    # Plausibility: at least one content word (≥4 chars) from the query
                    # must appear as a substring in the matched name, to guard against
                    # phonetic false positives (e.g. "caramel" ≠ "Carambola").
                    import re as _re
                    q_words = {w for w in food_lower.split() if len(w) >= 4}
                    m_words = set(_re.sub(r'[^\w ]', ' ', matched.lower()).split())
                    if q_words and not any(
                        qw in mw or mw in qw
                        for qw in q_words for mw in m_words if len(mw) >= 4
                    ):
                        logger.info(f"USDA density skip (no word overlap): '{food_name}' vs '{matched}'")
                        continue
                    logger.info(f"USDA density '{food_name}' → '{matched}' (sim={sim:.3f}, {method}): {density:.3f} g/ml")
                    return float(density), matched, f"usda_faiss(sim={sim:.2f},{method})"
            except Exception as e:
                logger.warning(f"USDA density FAISS lookup failed for '{food_name}': {e}")

        # ── 3. Keyword fallback table ─────────────────────────────────────────
        density, kw = self._fallback_density_with_kw(food_name)
        if kw is not None:
            return density, kw, f"fallback_keyword:{kw}"

        # ── 4. Gemini grounding (Google Search) ───────────────────────────────
        gemini_density = self._gemini_density_lookup(food_name)
        if gemini_density is not None:
            return gemini_density, food_name, "gemini_grounding"

        # ── 5. Hard default ───────────────────────────────────────────────────
        logger.info(f"Default density for '{food_name}': {DEFAULT_DENSITY} g/ml")
        return DEFAULT_DENSITY, "default", "fallback_default"

    def _gemini_density_lookup(self, food_name: str) -> Optional[float]:
        """
        Ask Gemini (with Google Search grounding) for food density.
        Returns density in g/ml, or None if unavailable/failed.
        Results are cached to avoid repeat API calls.
        """
        if not self._gemini_api_key:
            return None
        if food_name in self._gemini_density_cache:
            return self._gemini_density_cache[food_name]
        try:
            import google.generativeai as genai
            import re
            genai.configure(api_key=self._gemini_api_key)
            model = genai.GenerativeModel(
                "gemini-pro-latest",
                generation_config={"temperature": 0.0},
                tools="google_search_retrieval",
            )
            prompt = (
                f"What is the density of '{food_name}' in grams per milliliter (g/ml) "
                f"when served as a typical food portion? "
                f"Reply with ONLY a single decimal number (e.g. 0.85). No units, no explanation."
            )
            response = model.generate_content(prompt)
            text = response.text.strip()
            # Extract first float-like number from response
            match = re.search(r'\b(\d+\.?\d*)\b', text)
            if match:
                val = float(match.group(1))
                if 0.05 <= val <= 3.0:
                    logger.info(f"Gemini grounding density '{food_name}': {val:.3f} g/ml")
                    self._gemini_density_cache[food_name] = val
                    return val
        except Exception as e:
            logger.warning(f"Gemini density grounding failed for '{food_name}': {e}")
        return None

    def _fallback_density(self, food_name: str) -> float:
        density, _ = self._fallback_density_with_kw(food_name)
        return density

    def _fallback_density_with_kw(self, food_name: str):
        """Return (density, matched_keyword_or_None)."""
        name_lower = self._normalize_food_name(food_name)
        best_kw, best_density = None, None
        for keyword, density in DENSITY_FALLBACKS.items():
            if keyword in name_lower:
                if best_kw is None or len(keyword) > len(best_kw):
                    best_kw, best_density = keyword, density
        if best_kw is not None:
            logger.info(f"Fallback density '{food_name}' -> keyword '{best_kw}': {best_density} g/ml")
            return best_density, best_kw
        logger.info(f"Default density for '{food_name}': {DEFAULT_DENSITY} g/ml")
        return DEFAULT_DENSITY, None

    def get_calories_per_100g(self, food_name: str, top_k: int = 5) -> float:
        """Return kcal per 100g for the given food name."""
        return self._get_calories_with_match(food_name, top_k)[0]

    def _get_calories_with_match(self, food_name: str, top_k: int = 5):
        """
        Return (kcal_per_100g, matched_name, source) where source is one of:
          'usda_faiss', 'usda_fallback_keyword', 'usda_fallback_default'
        """
        if self._usda_index is None:
            logger.warning(f"USDA index not loaded -- using fallback kcal for '{food_name}'")
            kcal, kw = self._fallback_calories_with_kw(food_name)
            return kcal, kw or "default", "fallback_default"

        normalized = self._normalize_food_name(food_name)
        raw_lower = food_name.strip().lower()
        variants = list(dict.fromkeys([normalized, raw_lower]))
        try:
            hits = self._multi_search(self._usda_index, variants, top_k)
            food_lower = normalized
            for idx, sim in hits:
                if sim < _MIN_USDA_SIMILARITY:
                    break
                if not (0 <= idx < len(self._usda_foods)):
                    continue
                matched = self._usda_names[idx] if idx < len(self._usda_names) else ""
                matched_lower = matched.lower()
                if any(skip in matched_lower for skip in self._CALORIE_SKIP):
                    if not any(skip in food_lower for skip in self._CALORIE_SKIP):
                        continue
                entry = self._usda_foods[idx]
                kcal = float(entry.get("calories_per_100g", 200.0))
                logger.info(f"USDA kcal '{food_name}' -> '{matched}' (sim={sim:.3f}): {kcal:.1f} kcal/100g")
                return kcal, matched, f"usda_faiss(sim={sim:.2f})"
        except Exception as e:
            logger.warning(f"USDA FAISS lookup failed for '{food_name}': {e}")

        kcal, kw = self._fallback_calories_with_kw(food_name)
        src = f"fallback_keyword:{kw}" if kw else "fallback_default"
        return kcal, kw or "default", src

    def _fallback_calories(self, food_name: str) -> float:
        kcal, _ = self._fallback_calories_with_kw(food_name)
        return kcal

    def _fallback_calories_with_kw(self, food_name: str):
        """Return (kcal, matched_keyword_or_None)."""
        name_lower = self._normalize_food_name(food_name)
        best_kw, best_kcal = None, None
        for keyword, kcal in CALORIE_FALLBACKS.items():
            if keyword in name_lower:
                if best_kw is None or len(keyword) > len(best_kw):
                    best_kw, best_kcal = keyword, kcal
        if best_kw is not None:
            logger.info(f"Fallback kcal '{food_name}' -> keyword '{best_kw}': {best_kcal} kcal/100g")
            return float(best_kcal), best_kw
        logger.info(f"Default kcal for '{food_name}': {DEFAULT_CALORIES} kcal/100g")
        return DEFAULT_CALORIES, None

    def get_nutrition(self, food_name: str, volume_ml: float) -> dict:
        """
        Full nutrition calculation:
          mass_g     = volume_ml x density_g_ml
          total_kcal = (mass_g / 100) x calories_per_100g
        """
        density = self.get_density(food_name)
        kcal_per_100g = self.get_calories_per_100g(food_name)
        mass_g = volume_ml * density
        total_kcal = (mass_g / 100.0) * kcal_per_100g

        return {
            "food_name": food_name,
            "volume_ml": volume_ml,
            "density_g_per_ml": density,
            "mass_g": round(mass_g, 1),
            "calories_per_100g": round(kcal_per_100g, 1),
            "total_calories": round(total_kcal, 1),
            "calorie_source": "usda_rag",
            "density_source": "fao_rag",
        }

    def get_nutrition_for_food(
        self,
        food_name: str,
        volume_ml: float,
        mass_g: Optional[float] = None,
        quantity: int = 1,
    ) -> dict:
        """
        If mass_g is already known (e.g. from Gemini), use it directly for
        calorie calc; otherwise derive from volume x density.
        """
        if mass_g is not None and mass_g > 0:
            kcal_per_100g = self.get_calories_per_100g(food_name)
            density = self.get_density(food_name)
            total_kcal = (mass_g / 100.0) * kcal_per_100g
            return {
                "food_name": food_name,
                "quantity": quantity,
                "volume_ml": volume_ml,
                "density_g_per_ml": density,
                "density_source": "fao_rag",
                "density_similarity": 1.0,
                "mass_g": round(float(mass_g), 1),
                "calories_per_100g": round(kcal_per_100g, 1),
                "total_calories": round(total_kcal, 1),
                "calorie_source": "usda_rag",
                "calorie_similarity": 1.0,
                "matched_food": food_name,
            }
        result = self.get_nutrition(food_name, volume_ml)
        result["quantity"] = quantity
        result["density_similarity"] = 1.0
        result["calorie_similarity"] = 1.0
        result["matched_food"] = food_name
        return result
