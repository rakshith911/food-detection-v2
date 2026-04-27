# Simple rebuild script
$ErrorActionPreference = "Stop"

Write-Host "Rebuilding Docker Image..." -ForegroundColor Cyan

# Get config
$REGION = "us-east-1"
$REPO_NAME = "food-detection-v2-worker"
$ACCOUNT_ID = (aws sts get-caller-identity --query Account --output text)
$ECR_REPO = "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME"

Write-Host "ECR Repo: $ECR_REPO"

# Login to ECR
Write-Host "`nLogging in to ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REPO

# Build
Write-Host "`nBuilding Docker image..."
docker build -t nutrition-api:latest .

# Verify worker.py
Write-Host "`nVerifying worker.py in image..."
docker run --rm nutrition-api:latest ls -la /app/worker.py

# Tag
Write-Host "`nTagging image..."
docker tag nutrition-api:latest ${ECR_REPO}:latest

# Push
Write-Host "`nPushing to ECR..."
docker push ${ECR_REPO}:latest

# Update ECS
Write-Host "`nUpdating ECS service..."
aws ecs update-service --cluster food-detection-v2-cluster --service food-detection-v2-worker --force-new-deployment --region $REGION

Write-Host "`nDone! Monitor logs with:" -ForegroundColor Green
Write-Host "aws logs tail /aws/ecs/food-detection-v2-worker --follow --region $REGION"
