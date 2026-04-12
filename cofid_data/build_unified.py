"""
Build a single unified FAISS index from FAO + USDA + CoFID.

Each entry in unified_foods.json has:
  description, retrieval_text, calories_per_100g, density_g_ml, density_method, source

retrieval_text = description + food_category (USDA) + known synonyms/aliases
This enriched text is used for FAISS embedding and cross-encoder re-ranking,
enabling cross-regional name matching (e.g. "yorkshire pudding" ↔ "popover",
"aubergine" ↔ "eggplant", "courgette" ↔ "zucchini").

Output:
  unified_data/unified_foods.json
  unified_data/unified_food_names.json
  unified_data/unified_faiss.index
"""

import csv, json, sys
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent.parent  # food-detection/
OUT  = BASE / "unified_data"
OUT.mkdir(exist_ok=True)

# ── Synonym / alias map ───────────────────────────────────────────────────────
# Maps a keyword (must be lowercase, substring-matched) to a list of aliases.
# Each direction is listed explicitly so "popover" → adds "yorkshire pudding"
# AND "yorkshire pudding" → adds "popover".
FOOD_SYNONYMS: dict[str, list[str]] = {
    # Baked goods
    "popover":           ["yorkshire pudding"],
    "yorkshire pudding": ["popover"],

    # Vegetables (UK ↔ US)
    "aubergine":  ["eggplant"],
    "eggplant":   ["aubergine"],
    "courgette":  ["zucchini", "summer squash"],
    "zucchini":   ["courgette"],
    "rocket":     ["arugula"],
    "arugula":    ["rocket"],
    "spring onion": ["scallion", "green onion"],
    "scallion":   ["spring onion", "green onion"],
    "green onion": ["spring onion", "scallion"],
    "beetroot":   ["beet"],
    "beet":       ["beetroot"],
    "mangetout":  ["snow pea", "snap pea"],
    "mange tout": ["snow pea", "snap pea"],
    "snow pea":   ["mangetout", "mange tout"],
    "pak choi":   ["bok choy", "chinese cabbage"],
    "bok choy":   ["pak choi", "chinese cabbage"],
    "swede":      ["rutabaga"],
    "rutabaga":   ["swede"],
    "capsicum":   ["bell pepper", "sweet pepper"],
    "bell pepper":["capsicum", "sweet pepper"],
    "sweet pepper":["capsicum", "bell pepper"],
    "coriander":  ["cilantro"],
    "cilantro":   ["coriander"],

    # Legumes
    "chickpea":   ["garbanzo", "garbanzo bean"],
    "garbanzo":   ["chickpea"],
    "falafel":    ["chickpea patty"],

    # Meat
    "mince":      ["ground beef", "minced meat"],
    "ground beef":["mince", "minced beef"],
    "minced beef":["ground beef", "mince"],
    "prawn":      ["shrimp"],
    "shrimp":     ["prawn"],

    # Dairy
    "double cream":  ["heavy cream", "whipping cream"],
    "heavy cream":   ["double cream", "whipping cream"],
    "whipping cream":["double cream", "heavy cream"],
    "single cream":  ["half and half", "light cream"],
    "half and half": ["single cream", "light cream"],

    # Sweets / confectionery
    "biscuit":    ["cookie"],
    "cookie":     ["biscuit"],
    "sweets":     ["candy"],
    "candy":      ["sweets"],
    "icing sugar":      ["powdered sugar", "confectioners sugar"],
    "powdered sugar":   ["icing sugar", "confectioners sugar"],
    "confectioners sugar": ["icing sugar", "powdered sugar"],
    "caster sugar":     ["superfine sugar", "fine sugar"],
    "superfine sugar":  ["caster sugar"],

    # Starch / flour
    "cornflour":  ["cornstarch"],
    "cornstarch": ["cornflour"],
    "wholemeal":  ["whole wheat", "wholewheat"],
    "whole wheat":["wholemeal"],

    # Snacks
    "crisps":       ["potato chips"],
    "potato chips": ["crisps"],
    "french fries": ["chips", "fries"],
    "fries":        ["french fries", "chips"],

    # Other
    "sultana":           ["golden raisin"],
    "golden raisin":     ["sultana"],
    "treacle":           ["molasses"],
    "molasses":          ["treacle"],
    "desiccated coconut":["shredded coconut"],
    "shredded coconut":  ["desiccated coconut"],
    "paneer":            ["cottage cheese", "indian cheese", "fresh cheese"],
    "palak paneer":      ["saag paneer", "spinach paneer", "spinach with cheese"],
    "saag paneer":       ["palak paneer", "spinach paneer"],
    "naan":              ["flatbread", "indian bread"],
    "chapati":           ["roti", "flatbread", "indian bread"],
    "roti":              ["chapati", "flatbread"],
}


def _build_retrieval_text(description: str, category: str = "") -> str:
    """
    Compose retrieval_text from description, USDA food category, and synonyms.
    The result is embedded for FAISS and used by the cross-encoder for re-ranking.
    """
    desc_lower = description.lower()
    aliases: set[str] = set()
    for keyword, alts in FOOD_SYNONYMS.items():
        if keyword in desc_lower:
            aliases.update(alts)

    parts = [description]
    if category:
        parts.append(category)
    if aliases:
        parts.append("also known as: " + ", ".join(sorted(aliases)))

    return "; ".join(parts)


# ── Load USDA food → category map from raw CSV (optional, graceful fallback) ──
def _load_usda_category_map() -> dict[str, str]:
    """Returns {fdc_id: category_description}. Empty dict if raw files absent."""
    raw = BASE / "usda_data/usda_raw"
    cat_csv  = raw / "food_category.csv"
    food_csv = raw / "food.csv"
    if not cat_csv.exists() or not food_csv.exists():
        print("  (USDA raw CSVs not found — category enrichment skipped)")
        return {}

    cat_map: dict[str, str] = {}
    with open(cat_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat_map[row["id"]] = row["description"]

    fdc_to_cat: dict[str, str] = {}
    with open(food_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat_id = row.get("food_category_id", "")
            if cat_id in cat_map:
                fdc_to_cat[row["fdc_id"]] = cat_map[cat_id]

    print(f"  Loaded {len(fdc_to_cat):,} USDA food→category mappings")
    return fdc_to_cat


# ── Load FAO ──────────────────────────────────────────────────────────────────
def load_fao():
    density_list = json.load(open(BASE / "fao_data/fao_density.json"))
    names_list   = json.load(open(BASE / "fao_data/fao_food_names.json"))
    foods = []
    for entry, name in zip(density_list, names_list):
        foods.append({
            "description":       name,
            "retrieval_text":    _build_retrieval_text(name),
            "calories_per_100g": None,
            "density_g_ml":      entry.get("density_g_ml"),
            "density_method":    "fao_measured",
            "source":            "fao",
        })
    print(f"  FAO:   {len(foods)} entries")
    return foods


# ── Load USDA ─────────────────────────────────────────────────────────────────
def load_usda(fdc_to_cat: dict[str, str]):
    usda_foods = json.load(open(BASE / "usda_data/usda_foods.json"))
    foods = []
    for f in usda_foods:
        desc     = f.get("description", "")
        fdc_id   = str(f.get("fdc_id", ""))
        category = fdc_to_cat.get(fdc_id, "")
        foods.append({
            "id":                f"usda:{fdc_id}",
            "fdc_id":            fdc_id,
            "description":       desc,
            "retrieval_text":    _build_retrieval_text(desc, category),
            "calories_per_100g": f.get("calories_per_100g"),
            "protein_g":         f.get("protein_per_100g"),
            "carbs_g":           f.get("carb_per_100g"),
            "fat_g":             f.get("fat_per_100g"),
            "density_g_ml":      f.get("density_g_ml"),
            "density_method":    f.get("density_method"),
            "density_source":    "usda" if f.get("density_g_ml") else None,
            "cup_g":             f.get("cup_g"),
            "tbsp_g":            f.get("tbsp_g"),
            "tsp_g":             f.get("tsp_g"),
            "fl_oz_g":           f.get("fl_oz_g"),
            "source":            "usda",
        })
    print(f"  USDA:  {len(foods)} entries")
    return foods


# ── Load CoFID ────────────────────────────────────────────────────────────────
def load_cofid():
    cofid_foods = json.load(open(BASE / "cofid_data/cofid_foods.json"))
    foods = []
    for f in cofid_foods:
        desc = f.get("description", "")
        foods.append({
            "id":                f"cofid:{f.get('food_code','')}",
            "description":       desc,
            "retrieval_text":    _build_retrieval_text(desc),
            "calories_per_100g": f.get("calories_per_100g"),
            "protein_g":         f.get("protein_g"),
            "carbs_g":           f.get("carbs_g"),
            "fat_g":             f.get("fat_g"),
            "density_g_ml":      f.get("density_g_ml"),
            "density_method":    f.get("density_method"),
            "density_source":    "cofid" if f.get("density_g_ml") else None,
            "cup_g":             None,
            "tbsp_g":            None,
            "tsp_g":             None,
            "fl_oz_g":           None,
            "source":            "cofid",
        })
    print(f"  CoFID: {len(foods)} entries")
    return foods


# ── Build ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading USDA category map...")
    fdc_to_cat = _load_usda_category_map()

    print("Loading datasets...")
    foods = load_fao() + load_usda(fdc_to_cat) + load_cofid()
    print(f"  Total: {len(foods)} entries")

    # Embed retrieval_text (enriched) — not just description
    retrieval_texts = [f["retrieval_text"] for f in foods]
    names           = [f["description"]    for f in foods]   # kept for CLIP index

    print("Embedding unified index (retrieval_text)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = model.encode(retrieval_texts, batch_size=256, normalize_embeddings=True,
                        show_progress_bar=True)
    vecs = np.array(vecs, dtype=np.float32)

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    print(f"  FAISS: {index.ntotal} vectors, dim={vecs.shape[1]}")

    with open(OUT / "unified_foods.json", "w") as f:
        json.dump(foods, f)
    with open(OUT / "unified_food_names.json", "w") as f:
        json.dump(names, f)
    faiss.write_index(index, str(OUT / "unified_faiss.index"))

    print(f"\nSaved to {OUT}/")
    print(f"  unified_foods.json       ({len(foods)} entries)")
    print(f"  unified_food_names.json")
    print(f"  unified_faiss.index")

    # Spot-check synonym enrichment
    print("\nRetrieval text samples:")
    for food_name in ["Popovers, dry mix, enriched", "Yorkshire pudding", "Eggplant",
                      "Falafel, home-prepared", "Paneer"]:
        fn_lower = food_name.lower()
        for e in foods:
            if fn_lower in e["description"].lower():
                print(f"  [{e['source']}] {e['description']}")
                print(f"    → {e['retrieval_text']}")
                break


if __name__ == "__main__":
    main()
