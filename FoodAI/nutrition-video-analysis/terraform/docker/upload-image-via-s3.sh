#!/bin/bash

set -e

echo "=========================================="
echo "Uploading Docker image to ECR via S3"
echo "=========================================="

# Configuration
ECR_REPO="185329004895.dkr.ecr.us-east-1.amazonaws.com/food-detection-v2-worker"
IMAGE_TAG="latest"
LOCAL_IMAGE="${ECR_REPO}:${IMAGE_TAG}"
S3_BUCKET="nutrition-video-analysis-dev-videos-60ppnqfp"
S3_KEY="docker-images/video-processor-${IMAGE_TAG}.tar"
REGION="us-east-1"

echo "Step 1: Saving Docker image to tar file..."
docker save ${LOCAL_IMAGE} -o /tmp/video-processor.tar
echo "Image saved to /tmp/video-processor.tar"
ls -lh /tmp/video-processor.tar

echo ""
echo "Step 2: Uploading tar file to S3..."
aws s3 cp /tmp/video-processor.tar s3://${S3_BUCKET}/${S3_KEY} --region ${REGION}
echo "Uploaded to s3://${S3_BUCKET}/${S3_KEY}"

echo ""
echo "Step 3: Downloading on EC2/remote and loading to ECR..."
echo "Run the following commands on an EC2 instance (or Cloud Shell):"
echo ""
echo "# Download from S3"
echo "aws s3 cp s3://${S3_BUCKET}/${S3_KEY} /tmp/video-processor.tar --region ${REGION}"
echo ""
echo "# Load image into Docker"
echo "docker load -i /tmp/video-processor.tar"
echo ""
echo "# Login to ECR"
echo "aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_REPO%%/*}"
echo ""
echo "# Push to ECR"
echo "docker push ${LOCAL_IMAGE}"
echo ""
echo "# Cleanup"
echo "rm /tmp/video-processor.tar"

echo ""
echo "=========================================="
echo "Local steps complete!"
echo "=========================================="
echo "Image saved to S3: s3://${S3_BUCKET}/${S3_KEY}"
echo "Size: $(ls -lh /tmp/video-processor.tar | awk '{print $5}')"
echo ""
echo "Next: Run the commands above on an EC2 instance or AWS CloudShell to complete the push to ECR"
