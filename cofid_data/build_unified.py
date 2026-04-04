"""
Build a single unified FAISS index from FAO + USDA + CoFID.

Each entry in unified_foods.json has:
  description, calories_per_100g, density_g_ml, density_method, source

FAO and USDA may contribute exact density values.
CoFID contributes calorie data, but density stays empty unless an exact
measurement-based CoFID density source is added upstream.

Output:
  unified_data/unified_foods.json
  unified_data/unified_food_names.json
  unified_data/unified_faiss.index
"""

import json, sys
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent.parent  # food-detection/
OUT  = BASE / "unified_data"
OUT.mkdir(exist_ok=True)

# ── Load FAO ──────────────────────────────────────────────────────────────────
def load_fao():
    density_list = json.load(open(BASE / "fao_data/fao_density.json"))
    names_list   = json.load(open(BASE / "fao_data/fao_food_names.json"))
    foods = []
    for entry, name in zip(density_list, names_list):
        foods.append({
            "description":       name,
            "calories_per_100g": None,   # FAO index has no kcal
            "density_g_ml":      entry.get("density_g_ml"),
            "density_method":    "fao_measured",
            "source":            "fao",
        })
    print(f"  FAO:   {len(foods)} entries")
    return foods

# ── Load USDA ─────────────────────────────────────────────────────────────────
def load_usda():
    usda_foods = json.load(open(BASE / "usda_data/usda_foods.json"))
    foods = []
    for f in usda_foods:
        foods.append({
            "description":       f.get("description", ""),
            "calories_per_100g": f.get("calories_per_100g"),
            "density_g_ml":      f.get("density_g_ml"),
            "density_method":    f.get("density_method"),
            "source":            "usda",
        })
    print(f"  USDA:  {len(foods)} entries")
    return foods

# ── Load CoFID ────────────────────────────────────────────────────────────────
def load_cofid():
    cofid_foods = json.load(open(BASE / "cofid_data/cofid_foods.json"))
    foods = []
    for f in cofid_foods:
        foods.append({
            "description":       f.get("description", ""),
            "calories_per_100g": f.get("calories_per_100g"),
            "density_g_ml":      f.get("density_g_ml"),
            "density_method":    f.get("density_method"),
            "source":            "cofid",
        })
    print(f"  CoFID: {len(foods)} entries")
    return foods

# ── Build ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading datasets...")
    foods = load_fao() + load_usda() + load_cofid()
    print(f"  Total: {len(foods)} entries")

    names = [f["description"] for f in foods]

    print("Embedding unified index...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = model.encode(names, batch_size=256, normalize_embeddings=True,
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

if __name__ == "__main__":
    main()
