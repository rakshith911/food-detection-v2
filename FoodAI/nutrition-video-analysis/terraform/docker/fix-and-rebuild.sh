#!/bin/bash
# Fix and Rebuild Docker Image with CUDA Support
# This script rebuilds the Docker image with the fixed Dockerfile and pushes to ECR

set -e  # Exit on error

echo "==========================================="
echo " Nutrition API - Docker Image Rebuild"
echo "==========================================="
echo ""

# Configuration
REGION="us-east-1"
ECR_REPO_NAME="food-detection-v2-worker"

# Get AWS account ID
echo "[1/7] Getting AWS account ID..."
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}"
echo "Account ID: $ACCOUNT_ID"
echo "ECR Repository: $ECR_REPO"
echo ""

# Verify worker.py exists
echo "[2/7] Verifying required files..."
if [ ! -f "worker.py" ]; then
    echo "ERROR: worker.py not found in current directory"
    echo "Please run this script from terraform/docker directory"
    exit 1
fi

if [ ! -f "Dockerfile" ]; then
    echo "ERROR: Dockerfile not found"
    exit 1
fi

echo "✓ worker.py found"
echo "✓ Dockerfile found"
echo ""

# Login to ECR
echo "[3/7] Logging in to ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_REPO}
echo ""

# Build Docker image
echo "[4/7] Building Docker image..."
echo "This may take 10-15 minutes..."
docker build -t nutrition-api:latest .

if [ $? -ne 0 ]; then
    echo "ERROR: Docker build failed"
    exit 1
fi
echo "✓ Build complete"
echo ""

# Verify worker.py is in the image
echo "[5/7] Verifying worker.py in image..."
docker run --rm nutrition-api:latest ls -la /app/worker.py

if [ $? -ne 0 ]; then
    echo "ERROR: worker.py not found in Docker image"
    exit 1
fi
echo "✓ worker.py verified in image"
echo ""

# Tag image
echo "[6/7] Tagging image..."
docker tag nutrition-api:latest ${ECR_REPO}:latest
docker tag nutrition-api:latest ${ECR_REPO}:$(date +%Y%m%d-%H%M%S)
echo "✓ Tagged as latest and timestamped"
echo ""

# Push to ECR
echo "[7/7] Pushing to ECR..."
echo "Pushing latest tag..."
docker push ${ECR_REPO}:latest

echo "Pushing timestamped tag..."
docker push ${ECR_REPO}:$(date +%Y%m%d-%H%M%S)

if [ $? -ne 0 ]; then
    echo "ERROR: Docker push failed"
    exit 1
fi
echo "✓ Push complete"
echo ""

echo "==========================================="
echo " ✓ Docker Image Rebuild Complete!"
echo "==========================================="
echo ""
echo "Next steps:"
echo "1. Force ECS service to use new image:"
echo "   aws ecs update-service \\"
echo "     --cluster food-detection-v2-cluster \\"
echo "     --service food-detection-v2-worker \\"
echo "     --force-new-deployment \\"
echo "     --region ${REGION}"
echo ""
echo "2. Monitor ECS logs:"
echo "   aws logs tail /aws/ecs/food-detection-v2-worker --follow --region ${REGION}"
echo ""
echo "3. Check for GPU warning (expected on Fargate):"
echo "   If you see 'NVIDIA Driver was not detected', switch to EC2 or AWS Batch"
echo ""
