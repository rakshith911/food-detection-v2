import json
import os
import boto3
from botocore.config import Config
from decimal import Decimal

# Use Signature Version 4 for KMS-encrypted S3 objects
s3 = boto3.client('s3', config=Config(signature_version='s3v4'))
dynamodb = boto3.resource('dynamodb')

S3_RESULTS_BUCKET = os.environ.get('S3_RESULTS_BUCKET')
DYNAMODB_JOBS_TABLE = os.environ.get('DYNAMODB_JOBS_TABLE')


class DecimalEncoder(json.JSONEncoder):
    """Handle Decimal types from DynamoDB."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)


PRIVATE_RESPONSE_KEYS = {
    'florence_detections',
    'questionnaire_verification',
    'gemini_outputs',
    'pipeline_runtime',
    'production_debug',
    'segmented_images',
    'density_grounding_metadata',
    'calorie_grounding_metadata',
}


def strip_private_response_fields(payload):
    """Remove backend-only validation/debug data from user-facing API responses."""
    if isinstance(payload, dict):
        cleaned = {}
        for key, value in payload.items():
            if key in PRIVATE_RESPONSE_KEYS:
                continue
            cleaned[key] = strip_private_response_fields(value)
        return cleaned
    if isinstance(payload, list):
        return [strip_private_response_fields(item) for item in payload]
    return payload


def lambda_handler(event, context):
    """Get job results from S3 and DynamoDB."""

    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'GET,OPTIONS'
    }

    try:
        # Get job_id from path parameters
        job_id = event.get('pathParameters', {}).get('job_id')

        if not job_id:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'job_id is required'})
            }

        # Get job from DynamoDB
        table = dynamodb.Table(DYNAMODB_JOBS_TABLE)
        response = table.get_item(Key={'job_id': job_id})

        if 'Item' not in response:
            return {
                'statusCode': 404,
                'headers': headers,
                'body': json.dumps({'error': 'Job not found'})
            }

        job = response['Item']

        # Check if job is completed
        if job['status'] != 'completed':
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({
                    'error': 'Job not completed',
                    'status': job['status'],
                    'message': 'Please check /api/status/{job_id} for current status'
                })
            }

        # Get query parameters
        query_params = event.get('queryStringParameters') or {}
        detailed = query_params.get('detailed', 'false').lower() == 'true'

        # Build response with summary from DynamoDB
        result = {
            'job_id': job['job_id'],
            'status': 'completed',
            'created_at': job.get('created_at'),
            'completed_at': job.get('completed_at'),
            'filename': job.get('filename')
        }

        # Add nutrition summary if available in DynamoDB
        if 'nutrition_summary' in job:
            result['nutrition_summary'] = job['nutrition_summary']

        # Add detected foods if available in DynamoDB
        if 'detected_foods' in job:
            result['detected_foods'] = job['detected_foods']

        # Add items list if available in DynamoDB
        if 'items' in job:
            result['items'] = job['items']

        # Generate presigned URL for downloading full results
        results_key = job.get('results_s3_key', f'results/{job_id}/results.json')
        try:
            download_url = s3.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': S3_RESULTS_BUCKET,
                    'Key': results_key
                },
                ExpiresIn=3600  # 1 hour
            )
            result['download_url'] = download_url
        except Exception:
            pass  # Skip if can't generate URL

        # Detailed requests: fetch full results JSON and generate segmented image URLs.
        # Kept behind ?detailed=true so routine status-style calls stay fast.
        if detailed:
            try:
                s3_response = s3.get_object(
                    Bucket=S3_RESULTS_BUCKET,
                    Key=results_key
                )
                detailed_results = json.loads(s3_response['Body'].read().decode('utf-8'))
                result['detailed_results'] = strip_private_response_fields(detailed_results)

                # Generate fresh presigned URLs for segmentation assets.
                # The pipeline uploads:
                #   results/{job_id}/sam3_segmentation.png  (+ rgb.png, zoedepth_colored.png)
                #   segmented_images/{job_id}/segmented_overlay_video.mp4
                # Worker stores asset_keys in results["segmented_images"]; we read them here
                # and generate presigned URLs so the frontend never touches stale ECS credentials.
                presigned_expires = 3600  # 1 hour — same as download_url
                overlay_urls = []

                # Static image overlays — read asset_keys stored by the ECS worker
                asset_keys = (
                    detailed_results.get('segmented_images', {}).get('asset_keys') or []
                )
                for asset in asset_keys:
                    s3_key = asset.get('s3_key')
                    name = asset.get('name', '')
                    if not s3_key:
                        continue
                    try:
                        url = s3.generate_presigned_url(
                            'get_object',
                            Params={'Bucket': S3_RESULTS_BUCKET, 'Key': s3_key},
                            ExpiresIn=presigned_expires,
                        )
                        overlay_urls.append({'name': name, 'url': url})
                    except Exception:
                        pass

                # Video overlay — predictable key written by pipeline._generate_segmented_video_from_sam3
                video_overlay_url = None
                video_s3_key = f'segmented_images/{job_id}/segmented_overlay_video.mp4'
                try:
                    s3.head_object(Bucket=S3_RESULTS_BUCKET, Key=video_s3_key)
                    video_overlay_url = s3.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': S3_RESULTS_BUCKET, 'Key': video_s3_key},
                        ExpiresIn=presigned_expires,
                    )
                except Exception:
                    pass  # video overlay not present (image jobs) or not yet uploaded

                if overlay_urls or video_overlay_url:
                    result['segmented_images'] = {
                        'overlay_urls': overlay_urls,
                        'video_overlay_url': video_overlay_url,
                    }

                # TRELLIS MP4 preview — prefer stored key, fall back to scanning predictable prefix
                trellis_mp4_s3_key = detailed_results.get('trellis_mp4_s3_key')
                if not trellis_mp4_s3_key:
                    # Pipeline may not have saved the key — scan the output folder
                    try:
                        trellis_prefix = f'v2/trellis/outputs/{job_id}/'
                        resp = s3.list_objects_v2(Bucket=S3_RESULTS_BUCKET, Prefix=trellis_prefix)
                        for obj in resp.get('Contents', []):
                            if obj['Key'].endswith('.mp4'):
                                trellis_mp4_s3_key = obj['Key']
                                break
                    except Exception:
                        pass
                if trellis_mp4_s3_key:
                    try:
                        trellis_mp4_url = s3.generate_presigned_url(
                            'get_object',
                            Params={'Bucket': S3_RESULTS_BUCKET, 'Key': trellis_mp4_s3_key},
                            ExpiresIn=presigned_expires,
                        )
                        result['trellis_mp4_url'] = trellis_mp4_url
                    except Exception:
                        pass

            except s3.exceptions.NoSuchKey:
                result['detailed_results'] = None
                result['warning'] = 'Detailed results not found in S3'
            except Exception as e:
                result['detailed_results'] = None
                result['warning'] = f'Error fetching detailed results: {str(e)}'

        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps(result, cls=DecimalEncoder)
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': str(e)})
        }
