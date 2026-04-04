"""
Gemini Direct Processor Lambda
Downloads image or video from S3, sends directly to Gemini API, returns nutrition results.
No Florence-2 / SAM2 / Metric3D — just Gemini.

Required environment variables:
  GEMINI_API_KEY          - Google Gemini API key
  S3_VIDEOS_BUCKET        - Bucket where uploaded media lives
  S3_RESULTS_BUCKET       - Bucket to store result JSON
  DYNAMODB_JOBS_TABLE     - DynamoDB table for job tracking

Recommended Lambda settings:
  Timeout: 300 seconds (5 minutes)
  Memory:  512 MB
"""

import json
import os
import re
import tempfile
import traceback
import urllib.request
import urllib.error
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

import boto3

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

GEMINI_API_KEY       = os.environ.get('GEMINI_API_KEY')
S3_VIDEOS_BUCKET     = os.environ.get('S3_VIDEOS_BUCKET')
S3_RESULTS_BUCKET    = os.environ.get('S3_RESULTS_BUCKET')
DYNAMODB_JOBS_TABLE  = os.environ.get('DYNAMODB_JOBS_TABLE')

# Files <= 20 MB are sent inline; larger files use the Gemini File API
GEMINI_INLINE_LIMIT = 20 * 1024 * 1024

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

GEMINI_MODELS = [
    'gemini-2.5-flash',
    'gemini-2.0-flash-exp',
    'gemini-1.5-flash',
    'gemini-1.5-pro',
]

BASE_NUTRITION_PROMPT = """You are a nutrition expert. Analyze this food image or video and return a detailed nutritional breakdown.

Return ONLY valid JSON with this exact structure (no markdown, no extra text):
{
  "meal_name": "Name of the overall meal or dish",
  "meal_confidence": 0.9,
  "cuisine_type": "Cuisine type if known",
  "cuisine_confidence": 0.8,
  "cooking_method": "Cooking method if known",
  "cooking_method_confidence": 0.75,
  "items": [
    {
      "food_name": "Specific food item name",
      "mass_g": 150,
      "total_calories": 320,
      "confidence": 0.88
    }
  ],
  "nutrition_summary": {
    "total_calories_kcal": 320,
    "total_mass_g": 150,
    "num_food_items": 1,
    "total_food_volume_ml": 150,
    "overall_confidence": 0.86
  }
}

Rules:
- List every distinct food item you can see as a separate entry in "items"
- Use realistic portion weights (e.g. a chicken breast ~150g, a slice of bread ~30g)
- total_calories is the estimated calories for that specific item at its estimated weight
- nutrition_summary totals must match the sum of all items
- If you cannot identify food, return an empty items array and zero totals
- Output ONLY the JSON object, nothing else
"""


def build_nutrition_prompt(user_context: dict) -> str:
    """Build a Gemini prompt incorporating user questionnaire answers."""
    prompt = BASE_NUTRITION_PROMPT

    if not user_context:
        return prompt

    additions = []

    # Hidden / missing ingredients
    hidden = user_context.get('hidden_ingredients', [])
    if hidden:
        hidden_text = ', '.join(
            f"{i['name']} ({i['quantity']})" if i.get('quantity') else i['name']
            for i in hidden if i.get('name')
        )
        if hidden_text:
            additions.append(
                f"The user has suggested that the following ingredients may be present but not visible: {hidden_text}. "
                "Treat these as hypotheses to verify against the dish, not as ground truth. "
                "Only include them if they are genuinely plausible for this meal."
            )

    # Extras / cooking additions
    extras = user_context.get('extras', [])
    if extras:
        extras_text = ', '.join(
            f"{i['name']} ({i['quantity']})" if i.get('quantity') else i['name']
            for i in extras if i.get('name')
        )
        if extras_text:
            additions.append(
                f"The user has suggested these extras or cooking additions: {extras_text}. "
                "Treat these as hypotheses to verify against the dish, not as ground truth. "
                "Only include the incremental extra amount if it is plausible and not already accounted for in the visible base dish."
            )

    # Recipe / menu description
    recipe = user_context.get('recipe_description', '').strip()
    if recipe:
        additions.append(
            f"The user provided this recipe or menu description: \"{recipe}\". "
            "Use this to improve the accuracy of your portion and calorie estimates."
        )

    if additions:
        prompt += "\nIMPORTANT ADDITIONAL CONTEXT PROVIDED BY THE USER:\n"
        for addition in additions:
            prompt += f"- {addition}\n"
        prompt += (
            "\nFactor all of the above context into your analysis, but do not blindly trust it. "
            "Validate questionnaire claims against the meal itself before including them.\n"
        )

    return prompt


# ── helpers ──────────────────────────────────────────────────────────────────

def convert_floats_to_decimal(obj):
    """Recursively convert floats to Decimal for DynamoDB."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: convert_floats_to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_floats_to_decimal(i) for i in obj]
    return obj


def update_job_status(job_id: str, status: str, **kwargs):
    table = dynamodb.Table(DYNAMODB_JOBS_TABLE)
    expr        = 'SET #status = :status, updated_at = :updated_at'
    names       = {'#status': 'status'}
    values      = {
        ':status':     status,
        ':updated_at': datetime.utcnow().isoformat() + 'Z',
    }
    for key, value in kwargs.items():
        expr   += f', #{key} = :{key}'
        names[f'#{key}']  = key
        values[f':{key}'] = convert_floats_to_decimal(value)

    table.update_item(
        Key={'job_id': job_id},
        UpdateExpression=expr,
        ExpressionAttributeNames=names,
        ExpressionAttributeValues=values,
    )


def mime_type_for(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    mapping = {
        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.png': 'image/png',  '.bmp': 'image/bmp',
        '.webp': 'image/webp',
        '.mp4': 'video/mp4',  '.mov': 'video/quicktime',
        '.avi': 'video/x-msvideo', '.mkv': 'video/x-matroska',
        '.webm': 'video/webm',
    }
    return mapping.get(ext, 'application/octet-stream')


def parse_gemini_json(text: str) -> dict:
    """Strip markdown fences and parse JSON from Gemini response."""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return json.loads(text.strip())


def build_gemini_output_entry(stage: str, job_id: str, model_name: str, prompt: str, response_text: str, parsed_output=None, metadata: Optional[dict] = None) -> dict:
    return {
        'index': 1,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'job_id': job_id,
        'stage': stage,
        'model': model_name,
        'prompt': prompt,
        'response_text': response_text,
        'parsed_output': parsed_output,
        'metadata': metadata or {},
    }


def send_expo_push_notification(push_token: str, title: str, body: str, data: Optional[dict] = None):
    """Send push notification via Expo Push API."""
    if not push_token:
        print('[PushNotif] ⚠️ No push token — skipping notification')
        return
    print(f'[PushNotif] Sending push notification to token ending ...{push_token[-12:]}')
    print(f'[PushNotif]   title: {title}')
    print(f'[PushNotif]   body:  {body}')
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
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            response_body = resp.read().decode("utf-8")
            print(f'[PushNotif] ✅ Push sent. Expo response: {response_body[:300]}')
    except urllib.error.HTTPError as e:
        print(f'[PushNotif] ❌ Push send failed (HTTP {e.code}): {e.read().decode("utf-8")[:300]}')
    except Exception as e:
        print(f'[PushNotif] ❌ Push send failed: {e}')


# ── Gemini call ───────────────────────────────────────────────────────────────

def call_gemini(local_path: str, mime: str, prompt: str) -> dict:
    """Send media to Gemini and return model name plus raw response text."""
    try:
        from google import genai as genai_new
        from google.genai import types
    except ImportError:
        raise RuntimeError(
            'google-genai package not installed. '
            'Add google-genai to requirements.txt and redeploy.'
        )

    client = genai_new.Client(api_key=GEMINI_API_KEY)
    size   = os.path.getsize(local_path)

    for model_name in GEMINI_MODELS:
        try:
            if size <= GEMINI_INLINE_LIMIT:
                # Send bytes inline
                with open(local_path, 'rb') as f:
                    file_bytes = f.read()
                parts = [
                    types.Part(inline_data=types.Blob(data=file_bytes, mime_type=mime)),
                    types.Part(text=prompt),
                ]
                response = client.models.generate_content(
                    model=model_name,
                    contents=types.Content(parts=parts),
                )
            else:
                # Upload via File API for larger files
                print(f'[Gemini] File > 20 MB, uploading via File API...')
                myfile = client.files.upload(file=local_path)
                response = client.models.generate_content(
                    model=model_name,
                    contents=[myfile, prompt],
                )

            text = (response.text or '').strip()
            if text:
                print(f'[Gemini] Got response from {model_name} ({len(text)} chars)')
                return {'model_name': model_name, 'response_text': text}

        except Exception as e:
            print(f'[Gemini] Model {model_name} error: {e}')
            continue

    raise RuntimeError('All Gemini models failed to return a response')


# ── normalise Gemini output ───────────────────────────────────────────────────

def normalise_results(data: dict) -> tuple[list, dict, str]:
    """
    Returns (items, nutrition_summary, meal_name) in the format the
    existing results_handler + frontend expect.
    items: [{ food_name, mass_g, total_calories }, ...]
    nutrition_summary: { total_calories_kcal, total_mass_g, num_food_items, total_food_volume_ml }
    """
    raw_items = data.get('items', [])
    meal_name = data.get('meal_name', 'Analyzed Meal')

    items = []
    for item in raw_items:
        items.append({
            'food_name':     item.get('food_name') or item.get('name') or 'Unknown Food',
            'mass_g':        float(item.get('mass_g') or item.get('estimated_quantity_grams') or 0),
            'total_calories': float(item.get('total_calories') or item.get('calories') or 0),
            'confidence':    float(item.get('confidence') or 0),
        })

    ns = data.get('nutrition_summary', {})
    if ns:
        nutrition_summary = {
            'total_calories_kcal':  float(ns.get('total_calories_kcal') or 0),
            'total_mass_g':         float(ns.get('total_mass_g') or 0),
            # Always derive from actual items — Gemini sometimes returns non-zero counts with empty items
            'num_food_items':       len(items),
            'total_food_volume_ml': float(ns.get('total_food_volume_ml') or 0),
            'overall_confidence':   float(ns.get('overall_confidence') or 0),
        }
    else:
        # Build summary from items if Gemini didn't include it
        total_cal  = sum(i['total_calories'] for i in items)
        total_mass = sum(i['mass_g'] for i in items)
        nutrition_summary = {
            'total_calories_kcal':  total_cal,
            'total_mass_g':         total_mass,
            'num_food_items':       len(items),
            'total_food_volume_ml': total_mass,
            'overall_confidence':   float(data.get('meal_confidence') or 0),
        }

    return items, nutrition_summary, meal_name


# ── Lambda entrypoint ─────────────────────────────────────────────────────────

def lambda_handler(event, context):
    job_id     = event.get('job_id')
    s3_bucket  = event.get('s3_bucket')
    s3_key     = event.get('s3_key')
    push_token = event.get('push_token')
    # user_context may arrive as a dict (direct Lambda invoke) or JSON string (via SQS/DynamoDB)
    raw_ctx    = event.get('user_context')
    if isinstance(raw_ctx, str):
        try:
            user_context = json.loads(raw_ctx)
        except (json.JSONDecodeError, TypeError):
            user_context = {}
    elif isinstance(raw_ctx, dict):
        user_context = raw_ctx
    else:
        user_context = {}

    print(f'[gemini_processor] job={job_id}  s3://{s3_bucket}/{s3_key}')
    print(f'[PushNotif] Job {job_id} — push token present in Lambda event: {bool(push_token)}')
    print(f'[gemini_processor] user_context keys: {list(user_context.keys()) if user_context else "none"}')

    try:
        update_job_status(job_id, 'processing', progress=10)

        filename = os.path.basename(s3_key)
        ext      = Path(filename).suffix.lower()

        if ext not in IMAGE_EXTS | VIDEO_EXTS:
            raise ValueError(f'Unsupported file type: {ext}')

        mime = mime_type_for(filename)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, filename)

            # 1. Download from S3
            print(f'[gemini_processor] Downloading from S3...')
            s3.download_file(s3_bucket, s3_key, local_path)
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f'[gemini_processor] Downloaded {file_size_mb:.1f} MB  mime={mime}')
            update_job_status(job_id, 'processing', progress=30)

            # 2. Build prompt (incorporates user questionnaire answers if provided)
            prompt = build_nutrition_prompt(user_context)
            print(f'[gemini_processor] Prompt length: {len(prompt)} chars')

            # 3. Call Gemini
            print(f'[gemini_processor] Calling Gemini...')
            gemini_response = call_gemini(local_path, mime, prompt)
            response_text = gemini_response['response_text']
            update_job_status(job_id, 'processing', progress=70)

            # 4. Parse response
            try:
                data = parse_gemini_json(response_text)
            except json.JSONDecodeError as e:
                raise ValueError(f'Gemini response is not valid JSON: {e}\nRaw: {response_text[:500]}')

            items, nutrition_summary, meal_name = normalise_results(data)

            # 5. Build detected_foods list (lightweight, stored in DynamoDB)
            detected_foods = [
                {'name': i['food_name'], 'calories': i['total_calories'], 'mass_g': i['mass_g']}
                for i in items
            ]

            # 6. Build full result for S3 (same structure the existing results_handler returns)
            full_result = {
                'job_id':            job_id,
                'meal_name':         meal_name,
                'items':             items,           # picked up by getResults() at Location 2
                'detected_items':    items,           # picked up at Location 1 (fallback)
                'nutrition_summary': nutrition_summary,
                'detected_foods':    detected_foods,
                'gemini_outputs':    [
                    build_gemini_output_entry(
                        stage='lambda_direct_nutrition_analysis',
                        job_id=job_id,
                        model_name=gemini_response['model_name'],
                        prompt=prompt,
                        response_text=response_text,
                        parsed_output=data,
                        metadata={'mime_type': mime, 'source_s3_key': s3_key},
                    )
                ],
            }

            # 7. Upload results.json to S3
            results_key = f'results/{job_id}/results.json'
            s3.put_object(
                Bucket=S3_RESULTS_BUCKET,
                Key=results_key,
                Body=json.dumps(full_result, indent=2, default=str),
                ContentType='application/json',
            )
            print(f'[gemini_processor] Uploaded results to s3://{S3_RESULTS_BUCKET}/{results_key}')

            # 8. Mark job completed in DynamoDB
            update_job_status(
                job_id,
                'completed',
                progress=100,
                completed_at=datetime.utcnow().isoformat() + 'Z',
                results_s3_key=results_key,
                nutrition_summary=nutrition_summary,
                detected_foods=detected_foods,
                items=items,
                gemini_outputs=full_result['gemini_outputs'],
            )

            total_kcal = nutrition_summary['total_calories_kcal']
            if push_token:
                send_expo_push_notification(
                    push_token=push_token,
                    title='UKcal Analysis Complete',
                    body=f'Your analysis is ready ({int(total_kcal)} kcal).',
                    data={'job_id': job_id, 'status': 'completed'},
                )
            print(f'[gemini_processor] Done: {len(items)} items, {total_kcal:.0f} kcal')
            return {'job_id': job_id, 'status': 'completed'}

    except Exception as e:
        print(f'[gemini_processor] FAILED job={job_id}: {e}')
        traceback.print_exc()
        update_job_status(job_id, 'failed', error=str(e))
        raise
