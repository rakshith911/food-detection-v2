# Set the PATH to include user and machine environment variables
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host "=== Finding CloudWatch Log Groups ===" -ForegroundColor Cyan
aws logs describe-log-groups --region us-east-1 --query "logGroups[?contains(logGroupName, 'nutrition') || contains(logGroupName, 'video')].logGroupName" --output json

Write-Host "`n=== Finding SQS Queues ===" -ForegroundColor Cyan
aws sqs list-queues --region us-east-1 --queue-name-prefix nutrition --output json

Write-Host "`n=== Getting Account ID ===" -ForegroundColor Cyan
$accountId = aws sts get-caller-identity --query Account --output text
Write-Host "Account ID: $accountId" -ForegroundColor Yellow

Write-Host "`n=== Task Definition ===" -ForegroundColor Cyan
aws ecs describe-task-definition --task-definition food-detection-v2-worker --region us-east-1 --query "taskDefinition.containerDefinitions[0].{name:name,image:image,logConfiguration:logConfiguration,healthCheck:healthCheck}" --output json
