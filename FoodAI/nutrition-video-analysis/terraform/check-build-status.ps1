# Set the PATH to include user and machine environment variables
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host "=== Checking Docker Build Status ===" -ForegroundColor Cyan

# Check for completion marker in S3
Write-Host "`nChecking S3 for build completion marker..." -ForegroundColor Yellow
$s3Check = aws s3 ls s3://nutrition-video-analysis-dev-videos-dbenpoj2/docker-images/build-complete.txt --region us-east-1 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "BUILD COMPLETED!" -ForegroundColor Green
    Write-Host "`nDownloading completion marker..." -ForegroundColor Yellow
    aws s3 cp s3://nutrition-video-analysis-dev-videos-dbenpoj2/docker-images/build-complete.txt - --region us-east-1

    Write-Host "`n=== Next Step: Update ECS Service ===" -ForegroundColor Cyan
    Write-Host "Run this command to force ECS to use the new Docker image:" -ForegroundColor Yellow
    Write-Host "aws ecs update-service --cluster food-detection-v2-cluster --service food-detection-v2-worker --force-new-deployment --region us-east-1" -ForegroundColor White
} else {
    Write-Host "Build not yet completed" -ForegroundColor Yellow

    # Check ECR for new image
    Write-Host "`nChecking ECR for recent image pushes..." -ForegroundColor Yellow
    aws ecr describe-images --repository-name food-detection-v2-worker --region us-east-1 --query 'sort_by(imageDetails,& imagePushedAt)[-1].{Pushed:imagePushedAt,Tags:imageTags[0]}' --output table

    Write-Host "`nBuild is still in progress. Wait a few minutes and check again." -ForegroundColor Yellow
}
