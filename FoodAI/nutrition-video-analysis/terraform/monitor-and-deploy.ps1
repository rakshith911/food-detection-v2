# Set the PATH to include user and machine environment variables
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host "=== Monitoring Docker Build and Auto-Deploy ===" -ForegroundColor Cyan

$maxAttempts = 30  # 30 attempts * 30 seconds = 15 minutes max
$attempt = 0
$buildComplete = $false

while ($attempt -lt $maxAttempts -and -not $buildComplete) {
    $attempt++
    Write-Host "`n[$attempt/$maxAttempts] Checking build status... ($(Get-Date -Format 'HH:mm:ss'))" -ForegroundColor Yellow

    # Check S3 for completion marker
    $s3Check = aws s3 ls s3://nutrition-video-analysis-dev-videos-dbenpoj2/docker-images/build-complete.txt --region us-east-1 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nBUILD COMPLETED!" -ForegroundColor Green
        $buildComplete = $true

        # Get completion time
        aws s3 cp s3://nutrition-video-analysis-dev-videos-dbenpoj2/docker-images/build-complete.txt - --region us-east-1

        Write-Host "`n=== Forcing ECS to Deploy New Image ===" -ForegroundColor Cyan
        aws ecs update-service --cluster food-detection-v2-cluster --service food-detection-v2-worker --force-new-deployment --region us-east-1 --query 'service.{serviceName:serviceName,desiredCount:desiredCount,runningCount:runningCount,deployments:deployments[0].status}' --output json

        if ($LASTEXITCODE -eq 0) {
            Write-Host "`nECS service update initiated successfully!" -ForegroundColor Green
            Write-Host "Waiting 60 seconds for new task to start..." -ForegroundColor Yellow
            Start-Sleep -Seconds 60

            Write-Host "`n=== Checking ECS Service Status ===" -ForegroundColor Cyan
            aws ecs describe-services --cluster food-detection-v2-cluster --services food-detection-v2-worker --region us-east-1 --query 'services[0].{runningCount:runningCount,desiredCount:desiredCount,deployments:deployments[*].{status:status,runningCount:runningCount,desiredCount:desiredCount}}' --output json

            Write-Host "`n=== Checking SQS Queue ===" -ForegroundColor Cyan
            aws sqs get-queue-attributes --queue-url https://sqs.us-east-1.amazonaws.com/185329004895/food-detection-v2-jobs --attribute-names ApproximateNumberOfMessages ApproximateNumberOfMessagesNotVisible --region us-east-1 --output json

            Write-Host "`n=== Checking Job Status ===" -ForegroundColor Cyan
            curl -s https://c89txc5qr6.execute-api.us-east-1.amazonaws.com/v1/api/status/4d0f67dd-5d58-4e80-8986-9df0e99efeb6

            Write-Host "`n`n=== Summary ===" -ForegroundColor Green
            Write-Host "1. Docker image built with correct worker.py" -ForegroundColor White
            Write-Host "2. ECS service forced to deploy new image" -ForegroundColor White
            Write-Host "3. Worker should start processing SQS messages" -ForegroundColor White
            Write-Host "`nMonitor job progress with:" -ForegroundColor Yellow
            Write-Host "curl https://c89txc5qr6.execute-api.us-east-1.amazonaws.com/v1/api/status/4d0f67dd-5d58-4e80-8986-9df0e99efeb6" -ForegroundColor White
        } else {
            Write-Host "`nFailed to update ECS service" -ForegroundColor Red
        }

        break
    }

    if ($attempt % 5 -eq 0) {
        # Every 5th attempt, check ECR for new image
        Write-Host "Checking ECR for new image..." -ForegroundColor Yellow
        aws ecr describe-images --repository-name food-detection-v2-worker --region us-east-1 --query 'sort_by(imageDetails,& imagePushedAt)[-1].{Pushed:imagePushedAt,Digest:imageDigest}' --output json
    }

    if ($attempt -lt $maxAttempts) {
        Write-Host "Waiting 30 seconds before next check..." -ForegroundColor Gray
        Start-Sleep -Seconds 30
    }
}

if (-not $buildComplete) {
    Write-Host "`nBuild did not complete within 15 minutes. Please check EC2 instance logs manually." -ForegroundColor Red
    Write-Host "Instance ID: i-0fdfdfb7ffe180e17" -ForegroundColor Yellow
}
