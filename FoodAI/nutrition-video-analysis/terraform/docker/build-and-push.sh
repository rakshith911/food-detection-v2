#!/bin/bash
# Build and push Docker image to ECR

set -e

# Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-185329004895}"
ECR_REPO="food-detection-v2-worker"
IMAGE_TAG="${IMAGE_TAG:-latest}"

ECR_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
FULL_IMAGE="${ECR_URL}/${ECR_REPO}:${IMAGE_TAG}"

echo "=========================================="
echo "Building Docker image for ECS worker"
echo "=========================================="
echo "ECR URL: ${ECR_URL}"
echo "Image: ${FULL_IMAGE}"
echo ""

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_URL}

# Build image
echo ""
echo "Building Docker image..."
docker build -t ${ECR_REPO}:${IMAGE_TAG} .

# Tag for ECR
echo ""
echo "Tagging image for ECR..."
docker tag ${ECR_REPO}:${IMAGE_TAG} ${FULL_IMAGE}

# Push to ECR
echo ""
echo "Pushing to ECR..."
docker push ${FULL_IMAGE}

echo ""
echo "=========================================="
echo "Successfully pushed: ${FULL_IMAGE}"
echo "=========================================="
echo ""
echo "To update ECS service, run:"
echo "  aws ecs update-service --cluster food-detection-v2-cluster --service food-detection-v2-worker --force-new-deployment"
