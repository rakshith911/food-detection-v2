# Rebuild Docker Image and Deploy to ECS
# Run this from: food-detection/FoodAI/nutrition-video-analysis/terraform/docker/

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Rebuilding Nutrition API Docker Image" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$REGION = "us-east-1"
$REPO_NAME = "food-detection-v2-worker"

# Step 1: Get AWS Account ID
Write-Host "[1/8] Getting AWS Account ID..." -ForegroundColor Yellow
$ACCOUNT_ID = (aws sts get-caller-identity --query Account --output text)
$ECR_REPO = "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME"
Write-Host "Account ID: $ACCOUNT_ID" -ForegroundColor Green
Write-Host "ECR Repo: $ECR_REPO" -ForegroundColor Green
Write-Host ""

# Step 2: Verify files
Write-Host "[2/8] Verifying required files..." -ForegroundColor Yellow
if (-not (Test-Path "worker.py")) {
    Write-Host "ERROR: worker.py not found!" -ForegroundColor Red
    Write-Host "Current directory: $(Get-Location)" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path "Dockerfile")) {
    Write-Host "ERROR: Dockerfile not found!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ worker.py found ($(((Get-Item worker.py).Length / 1KB).ToString('0.0')) KB)" -ForegroundColor Green
Write-Host "✓ Dockerfile found" -ForegroundColor Green
Write-Host ""

# Step 3: Login to ECR
Write-Host "[3/8] Logging in to ECR..." -ForegroundColor Yellow
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REPO
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: ECR login failed" -ForegroundColor Red
    exit 1
}
Write-Host "✓ ECR login successful" -ForegroundColor Green
Write-Host ""

# Step 4: Build Docker image
Write-Host "[4/8] Building Docker image..." -ForegroundColor Yellow
Write-Host "This may take 5-10 minutes..." -ForegroundColor Gray
docker build -t nutrition-api:latest .
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker build failed" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Build complete" -ForegroundColor Green
Write-Host ""

# Step 5: Verify worker.py in image
Write-Host "[5/8] Verifying worker.py in Docker image..." -ForegroundColor Yellow
docker run --rm nutrition-api:latest ls -la /app/worker.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: worker.py not found in Docker image!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ worker.py verified in image" -ForegroundColor Green
Write-Host ""

# Step 6: Tag image
Write-Host "[6/8] Tagging image..." -ForegroundColor Yellow
$TIMESTAMP = Get-Date -Format "yyyyMMdd-HHmmss"
docker tag nutrition-api:latest ${ECR_REPO}:latest
docker tag nutrition-api:latest ${ECR_REPO}:$TIMESTAMP
Write-Host "✓ Tagged as 'latest' and '$TIMESTAMP'" -ForegroundColor Green
Write-Host ""

# Step 7: Push to ECR
Write-Host "[7/8] Pushing to ECR..." -ForegroundColor Yellow
Write-Host "Pushing latest tag..." -ForegroundColor Gray
docker push ${ECR_REPO}:latest
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker push failed" -ForegroundColor Red
    exit 1
}
Write-Host "Pushing timestamped tag..." -ForegroundColor Gray
docker push ${ECR_REPO}:$TIMESTAMP
Write-Host "✓ Push complete" -ForegroundColor Green
Write-Host ""

# Step 8: Force ECS deployment
Write-Host "[8/8] Forcing ECS to use new image..." -ForegroundColor Yellow
aws ecs update-service `
    --cluster food-detection-v2-cluster `
    --service food-detection-v2-worker `
    --force-new-deployment `
    --region $REGION | Out-Null

if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: ECS update may have failed, check manually" -ForegroundColor Yellow
} else {
    Write-Host "✓ ECS service update triggered" -ForegroundColor Green
}
Write-Host ""

Write-Host "============================================" -ForegroundColor Green
Write-Host " ✓ Rebuild and Deploy Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Monitor ECS logs:" -ForegroundColor White
Write-Host "   aws logs tail /aws/ecs/food-detection-v2-worker --follow --region $REGION" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Check ECS task status:" -ForegroundColor White
Write-Host "   cd ../terraform" -ForegroundColor Gray
Write-Host "   powershell -ExecutionPolicy Bypass -File check-processing.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Test with your video:" -ForegroundColor White
Write-Host "   curl -X POST 'https://c89txc5qr6.execute-api.us-east-1.amazonaws.com/v1/api/upload' \" -ForegroundColor Gray
Write-Host "     -F 'file=@D:/Nutrition5k/smart_tracked_WhatsApp Video 2025-09-10 at 05.16.56_9874c762.mp4'" -ForegroundColor Gray
Write-Host ""
