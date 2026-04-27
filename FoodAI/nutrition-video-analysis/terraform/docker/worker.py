#!/usr/bin/env python3
"""
ECS Worker - Polls SQS queue and processes video files for nutrition analysis.
"""

# VERSION CHECK: This ensures we're running the latest code (no mock fallback)
WORKER_VERSION = "v2.0.0-no-mock-2024-12-23"
print(f"🚀 Worker starting - Version: {WORKER_VERSION}")
print(f"🚀 This version has NO mock fallback - errors will fail clearly")

# CRITICAL: Patch transformers.utils BEFORE importing anything else
# This must be the FIRST thing that runs, before any imports
# Florence-2's custom code executes at import time and needs this function
import sys
import importlib.util

# Patch transformers.utils module before it's imported
def patch_transformers_utils():
    """Patch transformers.utils to add missing flash_attn function"""
    # Import transformers.utils (this will import transformers if not already imported)
    import transformers.utils as transformers_utils
    
    if not hasattr(transformers_utils, 'is_flash_attn_greater_or_equal_2_10'):
        def is_flash_attn_greater_or_equal_2_10():
            """Check if flash_attn version >= 2.10. Returns False for CPU-only environments."""
            return False  # Always False for CPU, which is what we want
        transformers_utils.is_flash_attn_greater_or_equal_2_10 = is_flash_attn_greater_or_equal_2_10
        print("✅ Applied monkey patch for is_flash_attn_greater_or_equal_2_10 at worker.py startup")
        return True
    return False

# Apply patch immediately
patch_applied = patch_transformers_utils()

# CRITICAL: Import NumPy BEFORE PyTorch to ensure PyTorch can detect it
# PyTorch's torch.from_numpy() requires NumPy to be imported first
import numpy as np
print(f"✅ NumPy {np.__version__} imported before PyTorch")

import json
import os
import sys
import time
import tempfile
import threading
import traceback
import urllib.request
import urllib.error
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

import boto3
from botocore.config import Config

# AWS clients
# s3v4_client is used for presigned URLs — KMS-encrypted buckets require Signature Version 4
s3 = boto3.client('s3')
s3v4 = boto3.client('s3', config=Config(signature_version='s3v4'))
sqs = boto3.client('sqs')
dynamodb = boto3.resource('dynamodb')

# Environment variables
S3_VIDEOS_BUCKET = os.environ.get('S3_VIDEOS_BUCKET')
S3_RESULTS_BUCKET = os.environ.get('S3_RESULTS_BUCKET')
S3_MODELS_BUCKET = os.environ.get('S3_MODELS_BUCKET')
DYNAMODB_JOBS_TABLE = os.environ.get('DYNAMODB_JOBS_TABLE')
SQS_VIDEO_QUEUE_URL = os.environ.get('SQS_VIDEO_QUEUE_URL')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
DEVICE = os.environ.get('DEVICE', 'cpu')
USE_PRODUCTION_IMAGE_PIPELINE = os.environ.get('USE_PRODUCTION_IMAGE_PIPELINE', 'true')
PRODUCTION_ASSETS_PREFIX = os.environ.get('PRODUCTION_ASSETS_PREFIX', 'production_assets/model_assets')

# Processing settings
MAX_FRAMES = int(os.environ.get('MAX_FRAMES', '60'))
FRAME_SKIP = int(os.environ.get('FRAME_SKIP', '10'))
DETECTION_INTERVAL = int(os.environ.get('DETECTION_INTERVAL', '30'))
MESSAGE_VISIBILITY_TIMEOUT_SECONDS = int(os.environ.get('MESSAGE_VISIBILITY_TIMEOUT_SECONDS', '3600'))
VISIBILITY_HEARTBEAT_INTERVAL_SECONDS = int(os.environ.get('VISIBILITY_HEARTBEAT_INTERVAL_SECONDS', '240'))

print(f"🚀 Worker env: DEVICE={DEVICE} USE_PRODUCTION_IMAGE_PIPELINE={USE_PRODUCTION_IMAGE_PIPELINE}")
print(f"🚀 Worker env: PRODUCTION_ASSETS_PREFIX={PRODUCTION_ASSETS_PREFIX}")


def convert_floats_to_decimal(obj):
    """Recursively convert floats to Decimal for DynamoDB compatibility."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: convert_floats_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats_to_decimal(item) for item in obj]
    return obj


def update_job_status(job_id: str, status: str, **kwargs):
    """Update job status in DynamoDB."""
    table = dynamodb.Table(DYNAMODB_JOBS_TABLE)

    update_expr = 'SET #status = :status, updated_at = :updated_at'
    expr_names = {'#status': 'status'}
    expr_values = {
        ':status': status,
        ':updated_at': datetime.utcnow().isoformat() + 'Z'
    }

    for key, value in kwargs.items():
        update_expr += f', #{key} = :{key}'
        expr_names[f'#{key}'] = key
        expr_values[f':{key}'] = convert_floats_to_decimal(value)

    table.update_item(
        Key={'job_id': job_id},
        UpdateExpression=update_expr,
        ExpressionAttributeNames=expr_names,
        ExpressionAttributeValues=expr_values
    )


def _start_visibility_heartbeat(receipt_handle: str, job_id: str) -> tuple[threading.Event, threading.Thread]:
    """Keep SQS visibility alive while long-running jobs are processing."""
    stop_event = threading.Event()
    interval = max(30, min(VISIBILITY_HEARTBEAT_INTERVAL_SECONDS, max(60, MESSAGE_VISIBILITY_TIMEOUT_SECONDS // 2)))

    def _heartbeat():
        while not stop_event.wait(interval):
            try:
                sqs.change_message_visibility(
                    QueueUrl=SQS_VIDEO_QUEUE_URL,
                    ReceiptHandle=receipt_handle,
                    VisibilityTimeout=MESSAGE_VISIBILITY_TIMEOUT_SECONDS,
                )
                print(
                    f"[SQS] Extended visibility for job {job_id} "
                    f"by {MESSAGE_VISIBILITY_TIMEOUT_SECONDS}s"
                )
            except Exception as hb_err:
                print(f"[SQS] WARNING: Failed to extend visibility for job {job_id}: {hb_err}")

    thread = threading.Thread(
        target=_heartbeat,
        name=f"sqs-visibility-{job_id[:8]}",
        daemon=True,
    )
    thread.start()
    return stop_event, thread


def download_media(s3_bucket: str, s3_key: str, local_path: str):
    """Download media file (video or image) from S3."""
    print(f"Downloading media from s3://{s3_bucket}/{s3_key}")
    s3.download_file(s3_bucket, s3_key, local_path)
    print(f"Downloaded to {local_path}")


def upload_results(job_id: str, results: dict):
    """Upload results to S3."""
    results_key = f'results/{job_id}/results.json'

    s3.put_object(
        Bucket=S3_RESULTS_BUCKET,
        Key=results_key,
        Body=json.dumps(results, indent=2, default=str),
        ContentType='application/json'
    )

    print(f"Results uploaded to s3://{S3_RESULTS_BUCKET}/{results_key}")
    return results_key


def send_expo_push_notification(push_token: str, title: str, body: str, data: Optional[dict] = None):
    """Send push notification via Expo Push API."""
    if not push_token:
        print("[PushNotif] ⚠️ No push token — skipping notification")
        return
    print(f"[PushNotif] Sending push notification to token ending ...{push_token[-12:]}")
    print(f"[PushNotif]   title: {title}")
    print(f"[PushNotif]   body:  {body}")
    payload = {
        "to": push_token,
        "title": title,
        "body": body,
        "sound": "default",
        "data": data or {},
    }
    req = urllib.request.Request(
        "https://exp.host/--/api/v2/push/send",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            response_body = resp.read().decode("utf-8")
            print(f"[PushNotif] ✅ Push sent. Expo response: {response_body[:300]}")
    except urllib.error.HTTPError as e:
        print(f"[PushNotif] ❌ Push send failed (HTTP {e.code}): {e.read().decode('utf-8')[:300]}")
    except Exception as e:
        print(f"[PushNotif] ❌ Push send failed: {e}")


def is_image_file(filename: str) -> bool:
    """Check if file is an image based on extension."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    return Path(filename).suffix.lower() in image_extensions


def is_video_file(filename: str) -> bool:
    """Check if file is a video based on extension."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
    return Path(filename).suffix.lower() in video_extensions


def process_media(media_path: str, job_id: str, user_context: dict = None, pipeline=None) -> dict:
    """
    Process media file (image or video) through the nutrition analysis pipeline.

    Pipeline:
    1. Detect file type (image vs video)
    2. Extract frames (1 for image, multiple for video)
    3. Detect food items with Florence-2
    4. Track objects with SAM2
    5. Estimate depth with Metric3D
    6. Calculate volumes
    7. Look up nutrition data
    8. Return results
    """

    # Update status to processing
    update_job_status(job_id, 'processing', progress=0)

    try:
        # Import processing modules (loaded here to avoid import errors during container startup)
        sys.path.insert(0, '/app')

        from app.pipeline import NutritionVideoPipeline
        from app.models import ModelManager
        from app.config import Settings

        if pipeline is None:
            # Fallback: initialize fresh (slower path, kept for safety)
            print("Initializing configuration...")
            config = Settings()
            config.DEVICE = DEVICE
            config.GEMINI_API_KEY = GEMINI_API_KEY

            update_job_status(job_id, 'processing', progress=5)

            print("Loading AI models...")
            model_manager = ModelManager(config)

            update_job_status(job_id, 'processing', progress=10)

            print("Initializing processing pipeline...")
            pipeline = NutritionVideoPipeline(model_manager, config)

        update_job_status(job_id, 'processing', progress=15)

        # Process media based on type
        media_path_obj = Path(media_path)

        if is_image_file(media_path):
            print(f"Processing image: {media_path}")
            results = pipeline.process_image(media_path_obj, job_id, user_context=user_context)
        elif is_video_file(media_path):
            print(f"Processing video: {media_path}")
            results = pipeline.process_video(media_path_obj, job_id, user_context=user_context)
        else:
            raise ValueError(f"Unsupported file type: {media_path_obj.suffix}")

        update_job_status(job_id, 'processing', progress=95)

        # Transform results to expected format (pipeline returns nutrition.summary, not meal_summary)
        nutrition = results.get('nutrition', {})
        meal_summary = nutrition.get('meal_summary') or nutrition.get('summary', {})

        return {
            'job_id': job_id,
            'media_path': media_path,
            'media_type': results.get('media_type', 'unknown'),
            'detected_items': nutrition.get('items', []),
            'meal_summary': meal_summary,
            'processing_info': {
                'frames_processed': results.get('num_frames_processed', 0),
                'device': DEVICE,
                'mock': False,
                'calibration': results.get('calibration', {})
            },
            'tracking': results.get('tracking', {}),
            'full_results': results
        }

    except ImportError as e:
        print(f"❌ CRITICAL: Pipeline not available: {e}")
        print(f"❌ Worker version: {WORKER_VERSION} - NO MOCK FALLBACK")
        traceback.print_exc()
        raise  # Fail instead of using mock
    except Exception as e:
        print(f"❌ CRITICAL: Real processing failed: {e}")
        print(f"❌ Worker version: {WORKER_VERSION} - NO MOCK FALLBACK")
        traceback.print_exc()
        raise  # Fail instead of using mock


def process_video(video_path: str, job_id: str) -> dict:
    """
    Process video through the nutrition analysis pipeline.

    Pipeline:
    1. Extract frames
    2. Detect food items with Florence-2
    3. Track objects with SAM2
    4. Estimate depth with Metric3D
    5. Calculate volumes
    6. Look up nutrition data
    7. Return results
    """

    # Update status to processing
    update_job_status(job_id, 'processing', progress=0)

    try:
        # Import processing modules (loaded here to avoid import errors during container startup)
        sys.path.insert(0, '/app')

        from app.pipeline import NutritionVideoPipeline
        from app.models import ModelManager
        from app.config import Settings

        # Initialize configuration
        print("Initializing configuration...")
        config = Settings()
        config.DEVICE = DEVICE
        config.GEMINI_API_KEY = GEMINI_API_KEY

        update_job_status(job_id, 'processing', progress=5)

        # Initialize models
        print("Loading AI models...")
        model_manager = ModelManager(config)

        update_job_status(job_id, 'processing', progress=10)

        # Initialize pipeline
        print("Initializing processing pipeline...")
        pipeline = NutritionVideoPipeline(model_manager, config)

        update_job_status(job_id, 'processing', progress=15)

        # Process video
        from pathlib import Path
        print(f"Processing video: {video_path}")
        results = pipeline.process_video(Path(video_path), job_id)

        update_job_status(job_id, 'processing', progress=95)

        # Transform results to expected format
        meal_summary = results.get('nutrition', {}).get('meal_summary', {})

        return {
            'job_id': job_id,
            'video_path': video_path,
            'detected_items': results.get('nutrition', {}).get('items', []),
            'meal_summary': meal_summary,
            'processing_info': {
                'frames_processed': results.get('num_frames_processed', 0),
                'device': DEVICE,
                'mock': False,
                'calibration': results.get('calibration', {})
            },
            'tracking': results.get('tracking', {}),
            'full_results': results
        }

    except ImportError as e:
        print(f"❌ CRITICAL: Pipeline not available: {e}")
        print(f"❌ Worker version: {WORKER_VERSION} - NO MOCK FALLBACK")
        traceback.print_exc()
        raise  # Fail instead of using mock
    except Exception as e:
        print(f"❌ CRITICAL: Real processing failed: {e}")
        print(f"❌ Worker version: {WORKER_VERSION} - NO MOCK FALLBACK")
        traceback.print_exc()
        raise  # Fail instead of using mock


def real_process_video(video_path: str, job_id: str) -> dict:
    """Real video processing using AI pipeline."""
    from pathlib import Path
    from app.config import Settings
    from app.models import ModelManager
    from app.pipeline import NutritionVideoPipeline

    print("Running real AI video processing...")

    # Initialize configuration
    config = Settings()
    config.DEVICE = DEVICE
    config.GEMINI_API_KEY = GEMINI_API_KEY

    # Initialize models
    print("Loading AI models...")
    update_job_status(job_id, 'processing', progress=5)
    model_manager = ModelManager(config)

    # Initialize pipeline
    print("Initializing processing pipeline...")
    update_job_status(job_id, 'processing', progress=10)
    pipeline = NutritionVideoPipeline(model_manager, config)

    # Process video
    print(f"Processing video: {video_path}")
    update_job_status(job_id, 'processing', progress=15)

    results = pipeline.process_video(Path(video_path), job_id)

    update_job_status(job_id, 'processing', progress=95)

    # Transform results to expected format
    meal_summary = results.get('nutrition', {}).get('meal_summary', {})

    return {
        'job_id': job_id,
        'video_path': video_path,
        'detected_items': results.get('nutrition', {}).get('items', []),
        'meal_summary': meal_summary,
        'processing_info': {
            'frames_processed': results.get('num_frames_processed', 0),
            'device': DEVICE,
            'mock': False,
            'calibration': results.get('calibration', {})
        },
        'tracking': results.get('tracking', {}),
        'full_results': results
    }


# Mock processing function removed - code must fail on errors, not silently use mock


def process_message(message: dict, pipeline=None):
    """Process a single SQS message."""
    receipt_handle = message['ReceiptHandle']
    stop_visibility_event = None
    visibility_thread = None
    
    try:
        body = json.loads(message['Body'])
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in message body: {e}")
        print(f"   Message body (first 200 chars): {str(message.get('Body', ''))[:200]}")
        print(f"   Deleting malformed message from queue...")
        # Delete the malformed message so it doesn't keep retrying
        try:
            sqs.delete_message(
                QueueUrl=SQS_VIDEO_QUEUE_URL,
                ReceiptHandle=receipt_handle
            )
            print(f"   SUCCESS: Deleted malformed message")
        except Exception as delete_error:
            print(f"   WARNING: Failed to delete message: {delete_error}")
        return  # Skip this message
    
    # Validate required fields
    if not all(key in body for key in ['job_id', 's3_bucket', 's3_key']):
        print(f"ERROR: Missing required fields in message. Got: {list(body.keys())}")
        print(f"   Expected: ['job_id', 's3_bucket', 's3_key']")
        # Delete the invalid message
        try:
            sqs.delete_message(
                QueueUrl=SQS_VIDEO_QUEUE_URL,
                ReceiptHandle=receipt_handle
            )
            print(f"   SUCCESS: Deleted invalid message")
        except Exception as delete_error:
            print(f"   WARNING: Failed to delete message: {delete_error}")
        return  # Skip this message
    
    job_id = body['job_id']
    s3_bucket = body['s3_bucket']
    s3_key = body['s3_key']
    push_token = body.get('push_token')
    raw_ctx = body.get('user_context')
    if isinstance(raw_ctx, str):
        try:
            user_context = json.loads(raw_ctx)
        except (json.JSONDecodeError, TypeError):
            user_context = {}
    elif isinstance(raw_ctx, dict):
        user_context = raw_ctx
    else:
        user_context = {}
    print(f"[PushNotif] Job {job_id} — push token present in SQS message: {bool(push_token)}")
    print(f"[worker] user_context keys: {list(user_context.keys()) if user_context else 'none'}")

    print(f"\n{'='*60}")
    print(f"Processing job: {job_id}")
    print(f"Media: s3://{s3_bucket}/{s3_key}")
    print(f"{'='*60}\n")

    try:
        # Update status
        update_job_status(job_id, 'processing', progress=0)
        stop_visibility_event, visibility_thread = _start_visibility_heartbeat(receipt_handle, job_id)

        # Create temp directory for processing
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download media file (video or image)
            media_filename = os.path.basename(s3_key)
            media_path = os.path.join(tmpdir, media_filename)
            download_media(s3_bucket, s3_key, media_path)

            # Log media type for debugging (videos must have correct extension for pipeline)
            ext = (os.path.splitext(media_filename)[1] or "").lower()
            is_video = ext in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
            print(f"Media type: {'VIDEO' if is_video else 'IMAGE'} (filename={media_filename}, ext={ext})")

            update_job_status(job_id, 'processing', progress=5)

            # Process media (image or video) — pass user_context so Gemini prompts are enriched
            results = process_media(media_path, job_id, user_context=user_context, pipeline=pipeline)

            # Upload segmentation debug images to S3.
            # Presigned URLs are generated fresh by the Lambda results_handler on each
            # getResults call — ECS task STS credentials cap ExpiresIn at <7 days and
            # expire before the stored URL would be used anyway.
            output_dir = f"/app/data/outputs/production_{job_id}"
            image_assets = {
                "tagged": f"{output_dir}/tagged.png",  # labeled overlay — shown first in app
                "rgb": f"{output_dir}/rgb.png",
            }
            full_results = results.get("full_results", {}) if isinstance(results, dict) else {}
            gemini_depth_assets = (
                full_results.get("production_debug", {}).get("gemini_depth_assets")
                or full_results.get("production_debug", {}).get("depth_outputs", {}).get("gemini_metric_depth")
                or {}
            )
            if gemini_depth_assets.get("full_depth_path"):
                image_assets["gemini_depth_full"] = gemini_depth_assets["full_depth_path"]
            for ingredient_asset in gemini_depth_assets.get("ingredients") or []:
                asset_path = ingredient_asset.get("path")
                slug = ingredient_asset.get("slug") or ingredient_asset.get("name") or "ingredient"
                if asset_path:
                    image_assets[f"gemini_depth_ingredient_{slug}"] = asset_path
            uploaded_keys = []
            for asset_name, local_path in image_assets.items():
                if os.path.exists(local_path):
                    asset_s3_key = f"results/{job_id}/{asset_name}.png"
                    try:
                        s3.upload_file(local_path, S3_RESULTS_BUCKET, asset_s3_key, ExtraArgs={"ContentType": "image/png"})
                        uploaded_keys.append({"name": asset_name, "s3_key": asset_s3_key})
                        print(f"[Images] Uploaded {asset_name} → s3://{S3_RESULTS_BUCKET}/{asset_s3_key}")
                    except Exception as img_err:
                        print(f"[Images] Failed to upload {asset_name}: {img_err}")
            if uploaded_keys:
                results["segmented_images"] = {"asset_keys": uploaded_keys}
                print(f"[Images] {len(uploaded_keys)} segmented images uploaded to S3")

            # Upload results
            results_key = upload_results(job_id, results)

            # Clean up local output directory to free disk space (all assets already in S3)
            import shutil
            for output_dir_candidate in [
                f"/app/data/outputs/production_{job_id}",
                f"/app/data/outputs/{job_id}",
            ]:
                if os.path.exists(output_dir_candidate):
                    shutil.rmtree(output_dir_candidate, ignore_errors=True)

            # Extract food items — video pipeline stores them under full_results.nutrition.items;
            # fall back to detected_items for the legacy image pipeline.
            raw_items = (
                results.get('full_results', {}).get('nutrition', {}).get('items')
                or results.get('detected_items')
                or []
            )
            food_items = [
                {
                    'name': item.get('food_name') or item.get('name'),
                    'calories': item.get('total_calories') or item.get('calories'),
                    'mass_g': item.get('mass_g'),
                }
                for item in raw_items
                if item.get('food_name') or item.get('name')
            ]

            # Update job as completed
            update_job_status(
                job_id,
                'completed',
                progress=100,
                completed_at=datetime.utcnow().isoformat() + 'Z',
                results_s3_key=results_key,
                nutrition_summary=results.get('meal_summary', {}),
                detected_foods=food_items
            )

            # Send completion push notification (works even when app is closed)
            if push_token:
                meal_summary = results.get('meal_summary', {}) or {}
                kcal = meal_summary.get('total_calories_kcal') or 0
                send_expo_push_notification(
                    push_token=push_token,
                    title="UKcal",
                    body="Your analysis is complete",
                    data={"job_id": job_id, "status": "completed"},
                )

            print(f"\n{'='*60}")
            print(f"Job {job_id} completed successfully!")
            print(f"{'='*60}\n")

        # Delete message from queue
        if stop_visibility_event is not None:
            stop_visibility_event.set()
        if visibility_thread is not None:
            visibility_thread.join(timeout=2)
        sqs.delete_message(
            QueueUrl=SQS_VIDEO_QUEUE_URL,
            ReceiptHandle=receipt_handle
        )

    except Exception as e:
        if stop_visibility_event is not None:
            stop_visibility_event.set()
        if visibility_thread is not None:
            visibility_thread.join(timeout=2)
        print(f"CRITICAL ERROR processing job {job_id}: {str(e)}")
        traceback.print_exc()

        # Update job as failed
        update_job_status(
            job_id,
            'failed',
            error=str(e)
        )

        deterministic_errors = (
            "Gemini metric-depth volume estimation did not return a positive total_volume_ml",
        )
        if any(msg in str(e) for msg in deterministic_errors):
            print(f"Deterministic failure for job {job_id}; deleting SQS message to avoid endless retries")
            try:
                sqs.delete_message(
                    QueueUrl=SQS_VIDEO_QUEUE_URL,
                    ReceiptHandle=receipt_handle
                )
            except Exception as delete_error:
                print(f"WARNING: Failed to delete deterministic-failure message: {delete_error}")
            return

        # Re-raise to ensure we don't silently continue
        raise


def poll_queue():
    """Poll SQS queue for messages."""

    print(f"Starting worker...")
    print(f"Queue URL: {SQS_VIDEO_QUEUE_URL}")
    print(f"Videos bucket: {S3_VIDEOS_BUCKET}")
    print(f"Results bucket: {S3_RESULTS_BUCKET}")
    print(f"Device: {DEVICE}")
    print("")

    # Load models once at startup so they are reused across all jobs
    pipeline = None
    try:
        sys.path.insert(0, '/app')
        from app.pipeline import NutritionVideoPipeline
        from app.models import ModelManager
        from app.config import Settings

        print("⏳ Pre-loading AI models at startup (one-time cost)...")
        config = Settings()
        config.DEVICE = DEVICE
        config.GEMINI_API_KEY = GEMINI_API_KEY
        model_manager = ModelManager(config)
        pipeline = NutritionVideoPipeline(model_manager, config)
        print("✅ Models loaded and ready — subsequent jobs will be fast")
    except Exception as e:
        print(f"⚠️ Could not pre-load models: {e} — will load per-job instead")
        pipeline = None

    while True:
        try:
            # Receive messages
            response = sqs.receive_message(
                QueueUrl=SQS_VIDEO_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,  # Long polling
                VisibilityTimeout=MESSAGE_VISIBILITY_TIMEOUT_SECONDS
            )

            messages = response.get('Messages', [])

            if messages:
                for message in messages:
                    process_message(message, pipeline=pipeline)
            else:
                print("No messages in queue, waiting...")

        except KeyboardInterrupt:
            print("\nShutting down worker...")
            break
        except Exception as e:
            print(f"Error polling queue: {str(e)}")
            traceback.print_exc()
            time.sleep(5)  # Wait before retrying


def download_models_from_s3():
    """Download AI model checkpoints from S3 if they don't exist locally."""
    if not S3_MODELS_BUCKET:
        print("S3_MODELS_BUCKET not set, skipping model download")
        return
    
    # Define models to download
    models = [
        # Unified RAG index (USDA + CoFID) plus separate FAO density fallback
        ('rag/unified_faiss.index', '/app/data/rag/unified_faiss.index'),
        ('rag/unified_foods.json', '/app/data/rag/unified_foods.json'),
        ('rag/unified_food_names.json', '/app/data/rag/unified_food_names.json'),
        ('rag/fao_faiss.index', '/app/data/rag/fao_faiss.index'),
        ('rag/fao_foods.json', '/app/data/rag/fao_foods.json'),
        ('rag/fao_food_names.json', '/app/data/rag/fao_food_names.json'),
        ('rag/branded_foods.json', '/app/data/rag/branded_foods.json'),
    ]
    
    print(f"Downloading models from s3://{S3_MODELS_BUCKET}...")
    
    for s3_key, local_path in models:
        # Create directory if needed
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Skip if already exists
        if os.path.exists(local_path):
            print(f"  ✓ {os.path.basename(local_path)} already exists")
            continue
        
        try:
            print(f"  Downloading {s3_key}...")
            s3.download_file(S3_MODELS_BUCKET, s3_key, local_path)
            print(f"  ✓ Downloaded {os.path.basename(local_path)}")
        except Exception as e:
            print(f"  ✗ Failed to download {s3_key}: {e}")
    
    print("Model download complete! Production asset sync skipped (Gemini metric-depth mode).")


if __name__ == '__main__':
    # Validate environment
    required_vars = [
        'S3_VIDEOS_BUCKET',
        'S3_RESULTS_BUCKET',
        'DYNAMODB_JOBS_TABLE',
        'SQS_VIDEO_QUEUE_URL'
    ]

    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        print(f"Error: Missing required environment variables: {missing}")
        sys.exit(1)

    # Download models from S3 before starting
    download_models_from_s3()

    poll_queue()
