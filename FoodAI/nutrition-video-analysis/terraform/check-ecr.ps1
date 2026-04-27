# Set the PATH
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host "Checking ECR for Docker image..." -ForegroundColor Cyan
$images = aws ecr describe-images --repository-name food-detection-v2-worker --region us-east-1 2>&1 | ConvertFrom-Json

if ($images.imageDetails -and $images.imageDetails.Count -gt 0) {
    Write-Host "`n SUCCESS! Docker image found in ECR!" -ForegroundColor Green
    Write-Host "`nImage Details:" -ForegroundColor Yellow
    $images.imageDetails | Format-List

    Write-Host "`nYour infrastructure is now fully deployed and ready!" -ForegroundColor Green
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "1. Test the API: curl https://c89txc5qr6.execute-api.us-east-1.amazonaws.com/v1/health" -ForegroundColor White
    Write-Host "2. Upload a video through the /api/v1/upload endpoint" -ForegroundColor White
    Write-Host "3. The ECS service will automatically scale up to process videos" -ForegroundColor White
} else {
    Write-Host "`nDocker image not yet in ECR. Build is still in progress..." -ForegroundColor Yellow
    Write-Host "The EC2 instance is building the Docker image. This takes 10-20 minutes." -ForegroundColor Yellow
    Write-Host "`nCheck again in a few minutes or monitor the instance logs." -ForegroundColor Cyan
}
