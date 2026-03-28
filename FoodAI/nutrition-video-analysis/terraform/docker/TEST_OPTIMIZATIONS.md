# Testing Gemini API Optimizations Locally

## Quick Test Command

```bash
cd /Users/leo/FoodProject/food-detection/FoodAI/nutrition-video-analysis/terraform/docker

# Activate virtual environment
source venv/bin/activate

# Set Gemini API key (if not already set)
export GEMINI_API_KEY="your-gemini-api-key-here"

# Run test on the image
python3 run_pipeline.py /Users/leo/FoodProject/food-detection/unhealthy-fast-food-delivery-menu-featuring-assorted-burgers-cheeseburgers-nuggets-french-fries-soda-high-calorie-low-356045884.jpg-2.jpg
```

## What to Look For

The optimized pipeline should make **~3 Gemini API calls** for a single image:

1. **VQA format + filter** (1 call) - Combined formatting and non-food filtering
2. **Volume validation + estimation** (1 call) - Combined validation of calculated volumes and estimation of untracked items
3. **Deduplicate + combine** (1 call) - Combined deduplication and item combining

## Expected Output

You should see in the logs:
- `🔵 Gemini Call #1: format_and_filter_with_gemini` (or similar)
- `🔵 Gemini Call #2: batch_validate_and_estimate_volumes_with_gemini`
- `🔵 Gemini Call #3: deduplicate_and_combine_with_gemini`

## Before vs After

**Before optimization:**
- VQA formatting: 1 call
- Filter non-food: 1 call
- Volume validation: ~2 calls (per object)
- Volume estimation: 1 call (batched)
- Deduplicate: 1 call
- Combine: 1 call
- **Total: ~7 calls**

**After optimization:**
- VQA format + filter: 1 call (combined)
- Volume validation + estimation: 1 call (combined)
- Deduplicate + combine: 1 call (combined)
- **Total: ~3 calls**

## Troubleshooting

If you see errors:
1. Make sure models are downloaded (checkpoints, etc.)
2. Make sure GEMINI_API_KEY is set
3. Check that all dependencies are installed: `pip install -r requirements.txt`

## Verify Results

After running, check:
- `results.json` - Should contain all detected items
- Logs should show fewer Gemini API calls
- All detected items should appear in results (even untracked ones with estimated volumes)
