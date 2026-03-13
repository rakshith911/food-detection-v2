"""
User Data Handler Lambda
Manages per-user data backup/restore to S3.

Routes:
  PUT    /user-data/{userId}/{dataType}  — Save JSON to S3 at UKcal/{userId}/{dataType}.json
  GET    /user-data/{userId}/{dataType}  — Read JSON from S3 at UKcal/{userId}/{dataType}.json
  DELETE /user-data/{userId}/account    — Wipe ALL user data from S3 + DynamoDB on account deletion

dataType for GET/PUT must be one of: profile, history, settings
"""

import json
import os
import boto3
from botocore.exceptions import ClientError

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

BUCKET = os.environ.get('USER_DATA_BUCKET', 'ukcal-user-uploads')
S3_PREFIX = os.environ.get('S3_PREFIX', 'UKcal')
VIDEOS_BUCKET = os.environ.get('VIDEOS_BUCKET', 'nutrition-video-analysis-dev-videos-dbenpoj2')
RESULTS_BUCKET = os.environ.get('RESULTS_BUCKET', 'nutrition-video-analysis-dev-results-dbenpoj2')
DYNAMO_TABLE = os.environ.get('DYNAMO_TABLE', 'ukcal-business-profiles')
ALLOWED_DATA_TYPES = {'profile', 'history', 'settings'}
MAX_BODY_SIZE = 5 * 1024 * 1024  # 5 MB limit per data type


def _cors_headers():
    return {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,Authorization,X-Amz-Date,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'GET,PUT,DELETE,OPTIONS',
    }


def _response(status_code, body):
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            **_cors_headers(),
        },
        'body': json.dumps(body) if not isinstance(body, str) else body,
    }


def _delete_s3_prefix(bucket, prefix):
    """Delete all objects under a given S3 prefix. Returns count deleted."""
    deleted = 0
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects = page.get('Contents', [])
        if not objects:
            continue
        s3.delete_objects(
            Bucket=bucket,
            Delete={'Objects': [{'Key': obj['Key']} for obj in objects], 'Quiet': True},
        )
        deleted += len(objects)
    return deleted


def _handle_delete_account(user_id, event):
    """Wipe all data for a user across S3 buckets and DynamoDB."""
    body = {}
    try:
        raw = event.get('body') or '{}'
        body = json.loads(raw)
    except Exception:
        pass

    job_ids = body.get('job_ids', [])
    if not isinstance(job_ids, list):
        job_ids = []

    deleted_summary = {}

    # 1. Delete user backup data from ukcal-user-uploads
    user_prefix = f'{S3_PREFIX}/{user_id}/'
    n = _delete_s3_prefix(BUCKET, user_prefix)
    deleted_summary['user_backup'] = n
    print(f'[AccountDelete] Deleted {n} objects from {BUCKET}/{user_prefix}')

    # 2. Delete per-job data for each job_id
    jobs_deleted = 0
    for job_id in job_ids:
        if not job_id or not isinstance(job_id, str):
            continue
        # Uploaded video/image
        n = _delete_s3_prefix(VIDEOS_BUCKET, f'uploads/{job_id}/')
        jobs_deleted += n
        # Analysis results JSON
        n = _delete_s3_prefix(RESULTS_BUCKET, f'results/{job_id}/')
        jobs_deleted += n
        # Segmented images + overlay video
        n = _delete_s3_prefix(RESULTS_BUCKET, f'segmented_images/{job_id}/')
        jobs_deleted += n
    deleted_summary['job_data'] = jobs_deleted
    print(f'[AccountDelete] Deleted {jobs_deleted} job objects across {len(job_ids)} jobs')

    # 3. Delete DynamoDB business profile
    try:
        table = dynamodb.Table(DYNAMO_TABLE)
        table.delete_item(Key={'userId': user_id})
        deleted_summary['dynamo'] = True
        print(f'[AccountDelete] Deleted DynamoDB profile for {user_id}')
    except Exception as e:
        deleted_summary['dynamo'] = False
        print(f'[AccountDelete] DynamoDB delete failed (non-fatal): {e}')

    return _response(200, {
        'message': 'Account data deleted',
        'deleted': deleted_summary,
    })


def lambda_handler(event, context):
    http_method = event.get('httpMethod', '')
    path_params = event.get('pathParameters') or {}

    # Handle CORS preflight
    if http_method == 'OPTIONS':
        return _response(200, {'message': 'OK'})

    user_id = path_params.get('userId', '')
    data_type = path_params.get('dataType', '')

    if not user_id:
        return _response(400, {'error': 'Missing userId'})

    # Account deletion — DELETE /user-data/{userId}/account
    if http_method == 'DELETE' and data_type == 'account':
        return _handle_delete_account(user_id, event)

    if data_type not in ALLOWED_DATA_TYPES:
        return _response(400, {
            'error': f'Invalid dataType: {data_type}. Must be one of: {", ".join(sorted(ALLOWED_DATA_TYPES))}'
        })

    s3_key = f'{S3_PREFIX}/{user_id}/{data_type}.json'

    if http_method == 'PUT':
        return _handle_put(s3_key, event, user_id, data_type)
    elif http_method == 'GET':
        return _handle_get(s3_key, user_id, data_type)
    else:
        return _response(405, {'error': f'Method not allowed: {http_method}'})


def _handle_put(s3_key, event, user_id, data_type):
    """Save JSON data to S3."""
    body = event.get('body', '')
    if not body:
        return _response(400, {'error': 'Empty request body'})

    if len(body) > MAX_BODY_SIZE:
        return _response(413, {'error': f'Body too large. Max size: {MAX_BODY_SIZE} bytes'})

    try:
        json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return _response(400, {'error': 'Invalid JSON body'})

    try:
        s3.put_object(
            Bucket=BUCKET,
            Key=s3_key,
            Body=body,
            ContentType='application/json',
            ServerSideEncryption='aws:kms',
        )
        print(f'[UserData] Saved {data_type} for user {user_id} -> s3://{BUCKET}/{s3_key}')
        return _response(200, {
            'message': f'{data_type} saved successfully',
            'key': s3_key,
        })
    except ClientError as e:
        print(f'[UserData] S3 PutObject error: {e}')
        return _response(500, {'error': 'Failed to save data'})


def _handle_get(s3_key, user_id, data_type):
    """Read JSON data from S3."""
    try:
        response = s3.get_object(Bucket=BUCKET, Key=s3_key)
        body = response['Body'].read().decode('utf-8')
        print(f'[UserData] Retrieved {data_type} for user {user_id} from s3://{BUCKET}/{s3_key}')
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                **_cors_headers(),
            },
            'body': body,
        }
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            print(f'[UserData] No {data_type} found for user {user_id}')
            return _response(404, {'error': f'No {data_type} data found'})
        print(f'[UserData] S3 GetObject error: {e}')
        return _response(500, {'error': 'Failed to retrieve data'})
