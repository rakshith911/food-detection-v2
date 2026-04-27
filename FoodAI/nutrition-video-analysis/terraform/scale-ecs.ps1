# Set the PATH to include user and machine environment variables
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host "=== Scaling ECS Service ===" -ForegroundColor Cyan
aws ecs update-service --cluster food-detection-v2-cluster --service food-detection-v2-worker --desired-count 1 --region us-east-1

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nECS service scaled to 1 task successfully!" -ForegroundColor Green
    Write-Host "Waiting for task to start..." -ForegroundColor Yellow
    Start-Sleep -Seconds 30

    Write-Host "`nChecking service status:" -ForegroundColor Cyan
    aws ecs describe-services --cluster food-detection-v2-cluster --services food-detection-v2-worker --region us-east-1 --query 'services[0].{runningCount:runningCount,desiredCount:desiredCount,status:status}' --output json
} else {
    Write-Host "Failed to scale ECS service" -ForegroundColor Red
}
