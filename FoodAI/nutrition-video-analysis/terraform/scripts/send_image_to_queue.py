#!/usr/bin/env python3
"""
Send an image to the nutrition pipeline queue via the upload API (base64).
Usage:
  API_BASE_URL=https://your-api.execute-api.us-east-1.amazonaws.com/v1 python send_image_to_queue.py [image_path]
  or
  python send_image_to_queue.py /path/to/image.jpeg
"""
import base64
import json
import os
import sys
from pathlib import Path

DEFAULT_IMAGE = "/Users/leo/FoodProject/food-detection/gemini_test.jpeg"
# Default API (same as test_segmented_images_integration.sh); override with API_BASE_URL env
DEFAULT_API_BASE = "https://c89txc5qr6.execute-api.us-east-1.amazonaws.com/v1"


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
        import urllib.request
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            out = json.loads(resp.read().decode())
    except Exception as e:
        print(f"Request failed: {e}")
        if hasattr(e, "read") and callable(getattr(e, "read", None)):
            try:
                print(e.read().decode())
            except Exception:
                pass
        sys.exit(1)

    job_id = out.get("job_id")
    status = out.get("status", "unknown")
    if not job_id:
        print("Response:", json.dumps(out, indent=2))
        sys.exit(1)

    print(f"Job queued: {job_id}")
    print(f"Status: {status}")
    print(f"Check status: {api_base}/api/status/{job_id}")
    print(f"Results:      {api_base}/api/results/{job_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
