# Set the PATH to include user and machine environment variables
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host "=== Checking ECS Task Logs ===" -ForegroundColor Cyan
aws logs tail /ecs/food-detection-v2-worker --since 30m --region us-east-1 --format short

Write-Host "`n=== Checking SQS Queue Messages ===" -ForegroundColor Cyan
Write-Host "Food Detection V2 Jobs Queue:" -ForegroundColor Yellow
aws sqs get-queue-attributes --queue-url https://sqs.us-east-1.amazonaws.com/185329004895/food-detection-v2-jobs --attribute-names All --region us-east-1 --query "Attributes.{ApproximateNumberOfMessages:ApproximateNumberOfMessages,ApproximateNumberOfMessagesNotVisible:ApproximateNumberOfMessagesNotVisible,ApproximateNumberOfMessagesDelayed:ApproximateNumberOfMessagesDelayed}" --output json

Write-Host "`n=== Checking Job Status ===" -ForegroundColor Cyan
curl -s https://c89txc5qr6.execute-api.us-east-1.amazonaws.com/v1/api/status/4d0f67dd-5d58-4e80-8986-9df0e99efeb6
