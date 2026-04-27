#!/bin/bash
set -e

IMAGE_PATH="/Users/leo/FoodProject/food-detection/unhealthy-fast-food-delivery-menu-featuring-assorted-burgers-cheeseburgers-nuggets-french-fries-soda-high-calorie-low-356045884.jpg-2.jpg"
S3_BUCKET="nutrition-video-analysis-dev-videos-dbenpoj2"
SQS_QUEUE_URL="https://sqs.us-east-1.amazonaws.com/185329004895/food-detection-v2-jobs"
REGION="us-east-1"

JOB_ID=$(python3 -c "import uuid; print(uuid.uuid4())")
IMAGE_FILENAME=$(basename "$IMAGE_PATH")
S3_KEY="uploads/${JOB_ID}/${IMAGE_FILENAME}"

echo "=========================================="
echo "Uploading Image and Triggering Processing"
echo "=========================================="
echo "Job ID: $JOB_ID"
echo ""

echo "Step 1: Uploading image to S3..."
aws s3 cp "$IMAGE_PATH" "s3://${S3_BUCKET}/${S3_KEY}" --region "$REGION"
echo "Uploaded to s3://${S3_BUCKET}/${S3_KEY}"
echo ""

echo "Step 2: Sending SQS message..."
MESSAGE_BODY="{\"job_id\":\"$JOB_ID\",\"s3_bucket\":\"$S3_BUCKET\",\"s3_key\":\"$S3_KEY\"}"

aws sqs send-message \
    --queue-url "$SQS_QUEUE_URL" \
    --message-body "$MESSAGE_BODY" \
    --region "$REGION" \
    --output json

echo ""
echo "=========================================="
echo "Processing Triggered!"
echo "=========================================="
echo "Job ID: $JOB_ID"
echo ""
echo "Monitor logs:"
echo "aws logs tail /ecs/food-detection-v2-worker --follow --region $REGION | grep $JOB_ID"
echo ""
echo "Check results:"
echo "aws s3 ls s3://nutrition-video-analysis-dev-results-dbenpoj2/results/$JOB_ID/ --region $REGION"
echo ""
echo "Check segmented images:"
echo "aws s3 ls s3://nutrition-video-analysis-dev-results-dbenpoj2/segmented_images/$JOB_ID/ --recursive --region $REGION"
