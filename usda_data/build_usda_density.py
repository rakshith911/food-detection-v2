"""
Build usda_density.json, usda_density_names.json, and usda_density_faiss.index
from the raw USDA FoodData Central CSVs.

Density is computed from volumetric portion measurements:
  density (g/ml) = gram_weight / volume_ml

Priority:
  2 = explicit volume unit (cup, ml, fl oz, etc.)
  1 = modifier text contains "cup" (parsed)
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path

RAW = Path(__file__).parent / 'usda_raw'
OUT = Path(__file__).parent

# ---------- volume unit conversions ----------
UNIT_ML = {
    '1000': 236.588,   # cup
    '1001': 14.7868,   # tablespoon
    '1002': 4.92892,   # teaspoon
    '1003': 1000.0,    # liter
    '1004': 1.0,       # milliliter
    '1005': 16.3871,   # cubic inch
    '1006': 1.0,       # cubic centimeter
    '1007': 3785.41,   # gallon
    '1008': 473.176,   # pint
    '1009': 29.5735,   # fl oz
}

# Qualifiers that make the density unrepresentative of the food as served
SKIP_MODIFIERS = {
    'packed', 'unpacked', 'sifted', 'crumbled', 'crushed', 'broken',
    "bb's", 'pieces', 'chunks', 'slices', 'strips', 'patties',
}

def parse_cup_modifier(modifier, amount):
    mod = modifier.lower().strip()
    if 'cup' not in mod:
        return None
    if any(s in mod for s in SKIP_MODIFIERS):
        return None
    return float(amount) * 236.588

# ---------- load food descriptions ----------
print("Loading food descriptions...")
food_desc = {}
with open(RAW / 'food.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        food_desc[row['fdc_id']] = row.get('description', '').strip()

# ---------- build per-food density list ----------
print("Extracting volumetric portions...")
densities = defaultdict(list)

with open(RAW / 'food_portion.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        fdc_id   = row['fdc_id']
        amount   = float(row['amount'])   if row['amount']      else 1.0
        gw       = float(row['gram_weight']) if row['gram_weight'] else 0.0
        unit_id  = row['measure_unit_id']
        modifier = (row.get('modifier', '') or '') + ' ' + (row.get('portion_description','') or '')

        if gw <= 0 or amount <= 0:
            continue

        vol_ml   = None
        priority = 0

        if unit_id in UNIT_ML:
            vol_ml   = amount * UNIT_ML[unit_id]
            priority = 2
        elif unit_id == '9999':
            vol_ml   = parse_cup_modifier(modifier, amount)
            if vol_ml:
                priority = 1

        if vol_ml and vol_ml > 0:
            density = gw / vol_ml
            if 0.05 <= density <= 3.0:   # sanity bounds
                densities[fdc_id].append((density, priority))

# ---------- collapse to one density per food ----------
print("Collapsing to best density per food...")
results = []
names   = []

for fdc_id, entries in densities.items():
    desc = food_desc.get(fdc_id, '').strip()
    if not desc:
        continue
    max_pri = max(e[1] for e in entries)
    top     = [e[0] for e in entries if e[1] == max_pri]
    avg_d   = round(sum(top) / len(top), 4)
    results.append({'fdc_id': fdc_id, 'density_g_ml': avg_d})
    names.append(desc)

print(f"  → {len(results)} foods with density")

json.dump(results, open(OUT / 'usda_density.json', 'w'), indent=2)
json.dump(names,   open(OUT / 'usda_density_names.json', 'w'), indent=2)
print("Saved usda_density.json and usda_density_names.json")

# ---------- build FAISS index ----------
print("Building FAISS index (this may take ~1 min)...")
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

BATCH = 512
embeddings = []
for i in range(0, len(names), BATCH):
    batch = names[i:i+BATCH]
    vecs  = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    embeddings.append(vecs.astype(np.float32))
    if (i // BATCH) % 5 == 0:
        print(f"  {i}/{len(names)}")

embeddings = np.vstack(embeddings)
dim   = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)   # inner-product on normalised → cosine sim
index.add(embeddings)
faiss.write_index(index, str(OUT / 'usda_density_faiss.index'))
print(f"Saved usda_density_faiss.index  (dim={dim}, {index.ntotal} entries)")
print("Done.")
