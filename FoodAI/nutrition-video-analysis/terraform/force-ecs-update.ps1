# Set the PATH to include user and machine environment variables
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host "=== Forcing ECS Service Update ===" -ForegroundColor Cyan
Write-Host "This will force ECS to pull and run the latest image from ECR" -ForegroundColor Yellow

aws ecs update-service --cluster food-detection-v2-cluster --service food-detection-v2-worker --force-new-deployment --region us-east-1 --output json

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nECS service update initiated!" -ForegroundColor Green
    Write-Host "Waiting 60 seconds for new task to start..." -ForegroundColor Yellow
    Start-Sleep -Seconds 60

    Write-Host "`n=== Checking ECS Service Status ===" -ForegroundColor Cyan
    aws ecs describe-services --cluster food-detection-v2-cluster --services food-detection-v2-worker --region us-east-1 --query 'services[0].{runningCount:runningCount,desiredCount:desiredCount,deployments:deployments[*].{status:status,taskDefinition:taskDefinition}}' --output json

    Write-Host "`n=== Checking if tasks are healthy ===" -ForegroundColor Cyan
    aws ecs list-tasks --cluster food-detection-v2-cluster --service-name food-detection-v2-worker --region us-east-1 --output json
} else {
    Write-Host "Failed to update ECS service" -ForegroundColor Red
}
