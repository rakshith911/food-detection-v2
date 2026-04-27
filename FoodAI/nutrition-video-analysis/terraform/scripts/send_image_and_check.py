#!/usr/bin/env python3
"""
Send an image to the nutrition pipeline queue via the upload API, then poll status
and print results when complete.

Usage:
  API_BASE_URL=https://your-api.execute-api.us-east-1.amazonaws.com/v1 python send_image_and_check.py [image_path]
  or
  python send_image_and_check.py /path/to/image.jpeg

Requires: network access, API deployed and ECS worker running.
"""
import base64
import json
import os
import sys
import time
from pathlib import Path

try:
    import urllib.request
except ImportError:
    urllib = None

DEFAULT_IMAGE = "/Users/leo/FoodProject/food-detection/gemini_test.jpeg"
DEFAULT_API_BASE = "https://c89txc5qr6.execute-api.us-east-1.amazonaws.com/v1"
POLL_INTERVAL = 10
MAX_POLL_ATTEMPTS = 60  # ~10 minutes


def request(url, data=None, method="GET", headers=None):
    if headers is None:
        headers = {}
    if data is not None and method == "POST":
        data = json.dumps(data).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    api_base = (os.environ.get("API_BASE_URL") or DEFAULT_API_BASE).rstrip("/")

    path = Path(image_path)
    if not path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    with open(path, "rb") as f:
        data_b64 = base64.b64encode(f.read()).decode("utf-8")

    filename = path.name
    content_type = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    if path.suffix.lower() == ".png":
        content_type = "image/png"

    body = {
        "type": "base64",
        "filename": filename,
        "content_type": content_type,
        "video_data": data_b64,
    }

    url = f"{api_base}/api/upload"
    print(f"Uploading {filename} to {api_base} ...")

    try:
        out = request(url, data=body, method="POST")
    except Exception as e:
        print(f"Upload failed: {e}")
        sys.exit(1)

    job_id = out.get("job_id")
    status = out.get("status", "unknown")
    if not job_id:
        print("Response:", json.dumps(out, indent=2))
        sys.exit(1)

    print(f"Job queued: {job_id}")
    print(f"Status: {status}")
    print()

    # Poll status
    print("Polling for completion (status every {}s)...".format(POLL_INTERVAL))
    for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
        time.sleep(POLL_INTERVAL)
        try:
            status_resp = request(f"{api_base}/api/status/{job_id}")
        except Exception as e:
            print(f"  Attempt {attempt}: status request failed: {e}")
            continue
        status = status_resp.get("status", "unknown")
        progress = status_resp.get("progress")
        print(f"  Attempt {attempt}/{MAX_POLL_ATTEMPTS}: status={status}" + (f", progress={progress}%" if progress is not None else ""))

        if status == "completed":
            print("Completed.")
            break
        if status == "failed":
            print("Job failed:", status_resp.get("error", "unknown"))
            sys.exit(1)
    else:
        print("Timeout waiting for completion.")
        print(f"Check status: {api_base}/api/status/{job_id}")
        print(f"Results:     {api_base}/api/results/{job_id}")
        sys.exit(1)

    # Fetch results
    print()
    print("Fetching results...")
    try:
        results = request(f"{api_base}/api/results/{job_id}?detailed=true")
    except Exception as e:
        print(f"Results request failed: {e}")
        sys.exit(1)

    # Print summary: items from DynamoDB 'items', or detected_foods, or detailed_results.nutrition.items
    items = results.get("items") or []
    if not items and results.get("detected_foods"):
        items = [{"food_name": x.get("name"), "total_calories": x.get("calories")} for x in results["detected_foods"]]
    if not items and results.get("detailed_results"):
        items = results["detailed_results"].get("nutrition", {}).get("items") or []
    summary = results.get("nutrition_summary") or {}
    if not summary and results.get("detailed_results"):
        summary = results["detailed_results"].get("nutrition", {}).get("summary") or {}

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    if items:
        print(f"Detected {len(items)} food item(s):")
        for i, item in enumerate(items, 1):
            name = item.get("food_name") or item.get("name") or "Unknown"
            qty = item.get("quantity", 1)
            if qty and int(qty) > 1:
                name = f"{qty} × {name}"
            cal = item.get("total_calories")
            mass = item.get("mass_g")
            print(f"  {i}. {name}")
            if cal is not None:
                print(f"     Calories: {cal:.1f} kcal")
            if mass is not None:
                print(f"     Mass: {mass:.1f} g")
        print()
    if summary:
        print("Meal summary:")
        print(f"  Total calories: {summary.get('total_calories_kcal', 'N/A')} kcal")
        print(f"  Total mass:     {summary.get('total_mass_g', 'N/A')} g")
        print(f"  Food items:     {summary.get('num_food_items', len(items))}")
    print("=" * 60)
    print(f"Full results: {api_base}/api/results/{job_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
