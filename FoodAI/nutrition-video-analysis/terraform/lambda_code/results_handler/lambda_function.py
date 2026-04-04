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
                result['detailed_results'] = detailed_results
            except s3.exceptions.NoSuchKey:
                result['detailed_results'] = None
                result['warning'] = 'Detailed results not found in S3'
            except Exception as e:
                result['detailed_results'] = None
                result['warning'] = f'Error fetching detailed results: {str(e)}'

            # Generate fresh presigned URLs for production image pipeline assets.
            # The worker stores images under results/{job_id}/ and records only S3 keys
            # (no presigned URLs) because ECS STS credentials cap ExpiresIn at <7 days.
            # The Lambda generates fresh 1-hour URLs on every detailed request.
            PRODUCTION_ASSET_NAMES = ['sam3_segmentation', 'zoedepth_colored', 'rgb']
            production_overlay_urls = []
            for asset_name in PRODUCTION_ASSET_NAMES:
                key = f'results/{job_id}/{asset_name}.png'
                try:
                    s3.head_object(Bucket=S3_RESULTS_BUCKET, Key=key)
                    presigned_url = s3.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': S3_RESULTS_BUCKET, 'Key': key},
                        ExpiresIn=3600  # 1 hour — Lambda credentials are always fresh
                    )
                    production_overlay_urls.append({'name': asset_name, 'url': presigned_url})
                except Exception:
                    pass  # Asset not yet uploaded or doesn't exist for this job

            if production_overlay_urls:
                result['segmented_images'] = {'overlay_urls': production_overlay_urls}
                print(f'Generated {len(production_overlay_urls)} fresh presigned URLs for production assets')
            else:
                # Fallback: legacy video pipeline segmented images under segmented_images/{job_id}/
                try:
                    segmented_prefix = f'segmented_images/{job_id}/'
                    segmented_objects = s3.list_objects_v2(
                        Bucket=S3_RESULTS_BUCKET,
                        Prefix=segmented_prefix,
                        MaxKeys=100
                    )

                    if 'Contents' in segmented_objects and len(segmented_objects['Contents']) > 0:
                        segmented_images = {
                            'overlay_urls': [],
                            'mask_urls': [],
                            'video_overlay_url': None
                        }

                        for obj in segmented_objects['Contents']:
                            key = obj['Key']
                            try:
                                presigned_url = s3.generate_presigned_url(
                                    'get_object',
                                    Params={'Bucket': S3_RESULTS_BUCKET, 'Key': key},
                                    ExpiresIn=3600
                                )
                                frame_match = None
                                if '/frame_' in key:
                                    parts = key.split('/frame_')
                                    if len(parts) > 1:
                                        frame_match = parts[1].split('/')[0]

                                if key.endswith('.mp4'):
                                    segmented_images['video_overlay_url'] = presigned_url
                                elif 'overlays' in key and 'all_masks.png' in key:
                                    segmented_images['overlay_urls'].append({
                                        'frame': frame_match or '00000',
                                        'url': presigned_url,
                                        'key': key,
                                        'type': 'overlay'
                                    })
                                elif 'masks' in key and key.endswith('.png'):
                                    segmented_images['mask_urls'].append({
                                        'frame': frame_match or '00000',
                                        'url': presigned_url,
                                        'key': key,
                                        'type': 'mask',
                                        'object_id': key.split('_obj_')[1].split('_')[0] if '_obj_' in key else None
                                    })
                            except Exception as e:
                                print(f"Warning: Could not generate URL for {key}: {e}")
                                continue

                        if segmented_images['overlay_urls'] or segmented_images['mask_urls'] or segmented_images['video_overlay_url']:
                            result['segmented_images'] = segmented_images
                            print(f"Added {len(segmented_images['overlay_urls'])} overlays, {len(segmented_images['mask_urls'])} masks")
                except Exception as e:
                    print(f"Warning: Could not generate segmented image URLs: {e}")

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
