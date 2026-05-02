# EXPERIMENT 1 — TRELLIS GLB Volume Estimation

## Goal
Validate the end-to-end volume pipeline:
- Two TRELLIS GLBs per job (GLB_A: food + plate, GLB_B: food only)
- GLB_A scaled using ground truth plate diameter → subtract plate volume → food volume
- GLB_B scaled using food spread (Gemini-estimated) → independent food volume
- Cross-check delta between the two paths
- Feed final food volume to Gemini (constrained distribution) → ingredient volumes → RAG → kcal

---

## Ground Truth Plate Measurements (Physical)

| Measurement | Value |
|---|---|
| Outer diameter | 27.0 cm |
| Rim height | 2.9 cm (midpoint of 2.8–3.0 cm) |
| Inner radius (food surface) | 11.5 cm (outer 13.5 cm − 2 cm rim) |
| Concavity depth | 2.0 cm |
| Mass | 744 g |
| Ceramic density | ~2.4 g/cm³ |
| Material volume | ~310 ml |

**Plate volume formula (enclosed geometric shape):**
```
V_plate = π × 13.5² × 2.9  −  π × 11.5² × 2.0
        = 1,662             −  831
        = 831 ml
```
Hardcoded as `_PLATE_VOLUME_ML = 831.0` in `pipeline.py`.

---

## Pipeline Architecture

```
Image
  │
  ├─ Gemini first pass → labels, plate detection, plate diameter estimate
  │
  ├─ Gemini metric depth → depth map (DISPLAY ONLY — not used for volume)
  │
  ├─ Gemini clean (food only)      → _food_only image   ─┐
  ├─ Gemini clean (food + plate)   → _with_plate image  ─┤
  │                                                       │
  └─ Single TRELLIS call (both images) ──────────────────┘
       │
       ├─ GLB_A (with plate)
       │    trimesh → raw unit volume
       │    scale: plate_diameter_m / max(extents XY) → linear_scale
       │    GLB_A_scaled = raw × scale³ × 1,000,000
       │    food_volume_A = GLB_A_scaled − 831 ml (plate ground truth)
       │
       └─ GLB_B (food only)
            trimesh → raw unit volume
            scale: food_spread_cm (Gemini) / max(extents XY) → linear_scale
            food_volume_B = raw × scale³ × 1,000,000
            cross-check delta = food_volume_A − food_volume_B

  Primary volume = food_volume_A (GLB_A path)
  Cross-check    = food_volume_B (GLB_B path)

  → _gemini_distribute_volumes_constrained(total=food_volume_A)
  → ingredient volume map
  → RAG → kcal per ingredient
```

---

## Run 1 — 2026-05-01 (PARTIAL — GLB_B failed)

**Job ID:** `5b33ba64-ac5b-490d-9dae-b38e3b9ef1cf`
**Dish:** Quinoa salad with greens and shredded cheese

### Ingredients Detected (Gemini)
| Ingredient | Confidence |
|---|---|
| Quinoa | 0.98 |
| Shredded cheese | 0.98 |
| Spinach | 0.95 |
| Radicchio | 0.90 |
| Roasted sweet potato | 0.85 |
| Zucchini | 0.75 |

### Plate Detection (Gemini)
- vessel_type: plate
- diameter_cm: 26.5 cm (ground truth: 27.0 cm — off by 0.5 cm, 1.9% error)
- confidence: 0.95

### GLB_A Trimesh Metrics
| Field | Value |
|---|---|
| is_watertight | **false** |
| volume (true) | null (non-watertight) |
| volume_convex_hull | 0.10066 units³ ← used |
| n_vertices | 722,009 |
| n_faces | 995,940 |
| extents XY | [1.0015, 1.0002] |
| euler_number | 13,985 (should be 0 for clean closed mesh) |

### Volume Chain (GLB_A path)
| Step | Value |
|---|---|
| Plate units (max XY extent) | 1.0015 units |
| Ground truth plate diameter | 0.27 m |
| Linear scale | 0.2696 m/unit |
| Volume scale | 19,595 ml/unit³ |
| GLB_A raw (convex hull) | 0.10066 units³ |
| GLB_A scaled | **1,972 ml** |
| Plate volume (ground truth) | 831 ml |
| **Food volume** | **1,141 ml** |

### Ingredient Distribution (Gemini constrained to 1,141 ml)
| Ingredient | Volume (ml) | Proportion |
|---|---|---|
| Spinach | 342 | 30% |
| Quinoa | 285 | 25% |
| Shredded cheese | 171 | 15% |
| Radicchio | 171 | 15% |
| Roasted sweet potato | 114 | 10% |
| Zucchini | 57 | 5% |

### Summary
| Metric | Value |
|---|---|
| Total food volume | 1,141.5 ml |
| Total mass | 544.5 g |
| Total calories | **653 kcal** |
| GLB_A S3 key | `v2/trellis/outputs/5b33ba64.../image_..._plate.glb` |
| GLB_B | ❌ FAILED |

### GLB_B Failure
**Error:** `IncorrectInstanceState — instance i-0b1f38c1962e885fd cannot be started`

**Root cause:** Two sequential `run_trellis_for_job` calls — GLB_A finished, trellis_gpu.py stopped the instance, GLB_B immediately tried `StartInstances` while the instance was mid-stopping → `IncorrectInstanceState`.

**Fix applied (f8f950d):** Both images now sent in a single `run_trellis_for_job` call (`image_paths=[_with_plate_tmp, _food_only_tmp]`). One GPU start/stop, both GLBs returned in one result dict.

---

## Run 2 — 2026-05-01 (PENDING)

**Job ID:** TBD
**Build:** `food-detection-v2-worker-build:57b885fb`

Changes from Run 1:
- Single TRELLIS call for both images (GLB_B fix)
- GLB_B now scaled independently using food_spread_cm (Gemini-estimated)
- `_gemini_estimate_food_spread_cm` added — Gemini estimates food spread from image using plate diameter as reference
- Cross-check delta between GLB_A and GLB_B paths will be logged

Expected new debug fields:
- `food_spread_cm_gemini` — Gemini-estimated food spread diameter
- `glb_b_metrics` — trimesh metrics for food-only GLB
- `glb_b_scaled_ml` — food volume from GLB_B path
- `glb_a_vs_glb_b_delta_ml` — difference between both paths

---

## Known Issues & Observations

### 1. Convex Hull Instead of True Mesh Volume
GLB_A mesh is **not watertight** (euler_number = 13,985, should be 0). Trimesh falls back to `convex_hull.volume` which is always an overestimate (wraps the shape in a convex envelope, ignoring concavities). For a food+plate scene with irregular shapes this could be 10–30% high.

**Impact on Run 1:** Food volume may be overestimated. The 1,141 ml result should be treated as an upper bound.

**Possible fix:** TRELLIS mesh repair (trimesh.repair, fill_holes) before volume calculation — not yet implemented, pending evaluation.

### 2. Plate Diameter Gemini vs Ground Truth
Gemini estimated plate diameter as 26.5 cm vs ground truth 27.0 cm (−1.9%). Small but worth noting — we use ground truth (27 cm) for scale, not Gemini's estimate.

### 3. GLB_B Scaling Uncertainty
GLB_B uses food spread (Gemini-estimated) as the scale reference. Food spread is less precise than plate diameter (food is irregular, not a known geometric shape). Expect higher variance in GLB_B path vs GLB_A path. The delta between both paths will quantify this.

### 4. No-Plate Path
If no plate is detected, the pipeline falls back to GLB_B + food spread only. No plate subtraction. This path is implemented but untested.

### 5. Volume Scale Units
TRELLIS GLBs appear to be in ~1-unit = ~27cm scale (GLB_A XY extents ≈ 1.0 unit for a 27cm plate). Not standard GLTF meters. Volume scale = 19,595 ml/unit³. This is consistent and derived correctly from the plate calibration.

---

## Code Locations

| Component | File | Function |
|---|---|---|
| GLB_A + GLB_B TRELLIS call | `app/pipeline.py` | `_run_production_image_pipeline` |
| trimesh volume | `app/pipeline.py` | `_estimate_volume_from_glb` |
| Food-only image clean | `app/pipeline.py` | `_gemini_clean_image` |
| Plate+food image clean | `app/pipeline.py` | `_gemini_clean_image_keep_plate` |
| Food spread estimation | `app/pipeline.py` | `_gemini_estimate_food_spread_cm` |
| Constrained distribution | `app/pipeline.py` | `_gemini_distribute_volumes_constrained` |
| Plate volume constant | `app/pipeline.py` | `_PLATE_VOLUME_ML = 831.0` |
| Plate diameter constant | `app/pipeline.py` | `_PLATE_DIAMETER_CM = 27.0` |

## Commits
| Hash | Description |
|---|---|
| `72fbe3c` | TRELLIS dual-GLB architecture (initial stub) |
| `278728c` | Add trimesh to requirements |
| `256b447` | Implement _estimate_volume_from_glb (EXP01 pattern) |
| `1c1db64` | Wire GLB_A scale calibration from ground truth plate diameter |
| `22c463e` | Hardcode plate volume 831 ml |
| `f8f950d` | Fix: single TRELLIS call for both images |
| current | Add GLB_B food spread scaling + _gemini_estimate_food_spread_cm |
