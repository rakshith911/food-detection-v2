# Set the PATH to include user and machine environment variables
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host "=== Checking ECS Task Status ===" -ForegroundColor Cyan

# Get the task ARN
$taskArn = aws ecs list-tasks --cluster food-detection-v2-cluster --service-name food-detection-v2-worker --region us-east-1 --query 'taskArns[0]' --output text

if ($taskArn) {
    Write-Host "`nTask ARN: $taskArn" -ForegroundColor Yellow

    # Get task details
    Write-Host "`nTask Details:" -ForegroundColor Cyan
    aws ecs describe-tasks --cluster food-detection-v2-cluster --tasks $taskArn --region us-east-1 --query 'tasks[0].{lastStatus:lastStatus,healthStatus:healthStatus,desiredStatus:desiredStatus,containers:containers[0].{name:name,lastStatus:lastStatus,healthStatus:healthStatus,exitCode:exitCode}}' --output json

    # Get CloudWatch logs
    Write-Host "`n=== Recent CloudWatch Logs ===" -ForegroundColor Cyan
    aws logs tail /aws/ecs/nutrition-video-analysis-dev --since 10m --region us-east-1 --format short
} else {
    Write-Host "No tasks found running" -ForegroundColor Red
}

Write-Host "`n=== Checking SQS Queue ===" -ForegroundColor Cyan
aws sqs get-queue-attributes --queue-url https://sqs.us-east-1.amazonaws.com/$(aws sts get-caller-identity --query Account --output text)/nutrition-video-analysis-dev-video-queue --attribute-names ApproximateNumberOfMessages ApproximateNumberOfMessagesNotVisible --region us-east-1 --output json
