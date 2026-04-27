# Build Docker image locally and push to ECR
# This bypasses CodeBuild issues

$ErrorActionPreference = "Stop"

Write-Host "=== Local Docker Build and Push to ECR ===" -ForegroundColor Cyan

# Configuration
$REGION = "us-east-1"
$ACCOUNT_ID = (aws sts get-caller-identity --query Account --output text)
$ECR_REPO = "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/food-detection-v2-worker"
$IMAGE_TAG = Get-Date -Format "yyyyMMdd-HHmmss"
$DOCKER_DIR = "d:\Nutrition5k\food-detection\FoodAI\nutrition-video-analysis\terraform\docker"

Write-Host "`nConfiguration:" -ForegroundColor Yellow
Write-Host "  Region: $REGION"
Write-Host "  Account: $ACCOUNT_ID"
Write-Host "  ECR Repo: $ECR_REPO"
Write-Host "  Image Tag: $IMAGE_TAG"
Write-Host "  Docker Dir: $DOCKER_DIR"

# Step 1: Login to ECR
Write-Host "`n[1/4] Logging into Amazon ECR..." -ForegroundColor Yellow
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REPO
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to login to ECR" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Logged in successfully" -ForegroundColor Green

# Step 2: Build Docker image
Write-Host "`n[2/4] Building Docker image..." -ForegroundColor Yellow
Write-Host "Running: docker build --no-cache -f Dockerfile -t nutrition-api:latest ." -ForegroundColor Gray

Push-Location $DOCKER_DIR
try {
    docker build --no-cache -f Dockerfile -t nutrition-api:latest .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Docker build failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Image built successfully" -ForegroundColor Green
} finally {
    Pop-Location
}

# Step 3: Tag images
Write-Host "`n[3/4] Tagging images..." -ForegroundColor Yellow
docker tag nutrition-api:latest "${ECR_REPO}:latest"
docker tag nutrition-api:latest "${ECR_REPO}:${IMAGE_TAG}"
Write-Host "✓ Images tagged" -ForegroundColor Green

# Step 4: Push to ECR
Write-Host "`n[4/4] Pushing images to ECR..." -ForegroundColor Yellow
docker push "${ECR_REPO}:latest"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to push :latest tag" -ForegroundColor Red
    exit 1
}

docker push "${ECR_REPO}:${IMAGE_TAG}"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to push :$IMAGE_TAG tag" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== SUCCESS ===" -ForegroundColor Green
Write-Host "Image pushed to ECR:" -ForegroundColor Green
Write-Host "  ${ECR_REPO}:latest" -ForegroundColor Cyan
Write-Host "  ${ECR_REPO}:${IMAGE_TAG}" -ForegroundColor Cyan

Write-Host "`nNext step: Force ECS to deploy new image" -ForegroundColor Yellow
Write-Host "Run: aws ecs update-service --cluster food-detection-v2-cluster --service nutrition-video-analysis-dev-service --force-new-deployment --region $REGION" -ForegroundColor Gray
