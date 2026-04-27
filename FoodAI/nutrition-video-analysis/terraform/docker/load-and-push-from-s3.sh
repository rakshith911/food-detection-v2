#!/bin/bash

# Script to download split Docker image from S3, reassemble, load into Docker, and push to ECR
# Run this in AWS CloudShell or an EC2 instance

set -e

echo "==========================================="
echo "Loading Docker image from S3 and pushing to ECR"
echo "==========================================="

# Configuration
S3_BUCKET="nutrition-video-analysis-dev-videos-60ppnqfp"
ECR_REPO="185329004895.dkr.ecr.us-east-1.amazonaws.com/food-detection-v2-worker"
IMAGE_TAG="latest"
REGION="us-east-1"

echo "Step 1: Downloading split parts from S3..."
cd /tmp
for part in aa ab ac ad ae af ag ah; do
  echo "Downloading video-processor-part-${part}..."
  aws s3 cp s3://${S3_BUCKET}/docker-images/video-processor-part-${part} /tmp/video-processor-part-${part} --region ${REGION}
done

echo ""
echo "Step 2: Reassembling Docker image..."
cat video-processor-part-* > video-processor.tar
echo "Reassembled image size:"
ls -lh video-processor.tar

echo ""
echo "Step 3: Loading Docker image..."
docker load -i video-processor.tar

echo ""
echo "Step 4: Logging into ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_REPO%%/*}

echo ""
echo "Step 5: Pushing to ECR..."
docker push ${ECR_REPO}:${IMAGE_TAG}

echo ""
echo "Step 6: Cleanup..."
rm -f /tmp/video-processor-part-* /tmp/video-processor.tar

echo ""
echo "==========================================="
echo "✓ Docker image successfully pushed to ECR!"
echo "==========================================="
echo "Image: ${ECR_REPO}:${IMAGE_TAG}"
echo ""
echo "Next steps:"
echo "1. Verify image in ECR:"
echo "   aws ecr describe-images --repository-name food-detection-v2-worker --region ${REGION}"
echo ""
echo "2. Force ECS service update to use new image:"
echo "   aws ecs update-service --cluster food-detection-v2-cluster --service food-detection-v2-worker --force-new-deployment --region ${REGION}"
