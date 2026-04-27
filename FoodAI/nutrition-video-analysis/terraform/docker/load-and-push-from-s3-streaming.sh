#!/bin/bash

set -e

echo "==========================================="
echo "Loading Docker image from S3 and pushing to ECR (streaming mode)"
echo "==========================================="

# Configuration
S3_BUCKET="nutrition-video-analysis-dev-videos-60ppnqfp"
ECR_REPO="185329004895.dkr.ecr.us-east-1.amazonaws.com/food-detection-v2-worker"
IMAGE_TAG="latest"
REGION="us-east-1"

echo "Step 1: Downloading split parts from S3 and streaming to docker load..."
cd /tmp

# Download and concatenate parts, pipe directly to docker load
echo "Downloading and loading image (this will take 10-15 minutes)..."
(
  for part in aa ab ac ad ae af ag ah; do
    aws s3 cp s3://${S3_BUCKET}/docker-images/video-processor-part-${part} - --region ${REGION}
  done
) | docker load

echo ""
echo "Step 2: Logging into ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_REPO%%/*}

echo ""
echo "Step 3: Pushing to ECR..."
docker push ${ECR_REPO}:${IMAGE_TAG}

echo ""
echo "==========================================="
echo "✓ Docker image successfully pushed to ECR!"
echo "==========================================="
echo "Image: ${ECR_REPO}:${IMAGE_TAG}"
echo ""
echo "Next: Force ECS service update:"
echo "aws ecs update-service --cluster food-detection-v2-cluster --service food-detection-v2-worker --force-new-deployment --region ${REGION}"
