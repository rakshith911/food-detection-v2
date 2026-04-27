# Set the PATH to include user and machine environment variables
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host "=== Launching EC2 Instance for Docker Build ===" -ForegroundColor Cyan

# Configuration
$REGION = "us-east-1"
$INSTANCE_TYPE = "t3.2xlarge"  # 8 vCPUs, 32 GB RAM, good for Docker builds
$AMI_ID = "ami-0c02fb55b5e67b84" # Amazon Linux 2023
$KEY_NAME = "nutrition-video-key"

# User data script to run on instance startup
$USER_DATA = @"
#!/bin/bash
set -e

# Log all output
exec > >(tee /var/log/docker-build.log)
exec 2>&1

echo "Starting Docker build process..."

# Install Docker
sudo yum update -y
sudo yum install -y docker git
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Clone repository
cd /home/ec2-user
git clone https://github.com/leolorence12345/food-detection.git
cd food-detection/FoodAI/nutrition-video-analysis/terraform/docker

# Log in to ECR
aws ecr get-login-password --region us-east-1 | sudo docker login --username AWS --password-stdin 185329004895.dkr.ecr.us-east-1.amazonaws.com

# Build Docker image
echo "Building Docker image..."
sudo docker build --platform linux/amd64 -f Dockerfile -t 185329004895.dkr.ecr.us-east-1.amazonaws.com/food-detection-v2-worker:latest .

# Push to ECR
echo "Pushing to ECR..."
sudo docker push 185329004895.dkr.ecr.us-east-1.amazonaws.com/food-detection-v2-worker:latest

echo "Docker build and push completed successfully!"
echo "Build completed at: \$(date)" > /home/ec2-user/build-complete.txt

# Signal completion
aws s3 cp /home/ec2-user/build-complete.txt s3://nutrition-video-analysis-dev-videos-60ppnqfp/docker-images/build-complete.txt --region us-east-1

echo "Done! You can now terminate this instance."
"@

# Base64 encode user data
$USER_DATA_BASE64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($USER_DATA))

Write-Host "`nLaunching EC2 instance..." -ForegroundColor Yellow
Write-Host "Instance Type: $INSTANCE_TYPE" -ForegroundColor White
Write-Host "Region: $REGION" -ForegroundColor White

# Get default VPC and subnet
$vpcId = aws ec2 describe-vpcs --region $REGION --filters "Name=tag:Name,Values=nutrition-video-analysis-dev-vpc" --query 'Vpcs[0].VpcId' --output text
$subnetId = aws ec2 describe-subnets --region $REGION --filters "Name=vpc-id,Values=$vpcId" "Name=map-public-ip-on-launch,Values=true" --query 'Subnets[0].SubnetId' --output text
$sgId = aws ec2 describe-security-groups --region $REGION --filters "Name=group-name,Values=*docker-push-temp*" "Name=vpc-id,Values=$vpcId" --query 'SecurityGroups[0].GroupId' --output text

Write-Host "VPC ID: $vpcId" -ForegroundColor White
Write-Host "Subnet ID: $subnetId" -ForegroundColor White
Write-Host "Security Group ID: $sgId" -ForegroundColor White

# Get IAM instance profile
$instanceProfile = aws iam list-instance-profiles --region $REGION --query 'InstanceProfiles[?contains(InstanceProfileName, `docker-push-temp`)].InstanceProfileName' --output text

if ([string]::IsNullOrEmpty($instanceProfile)) {
    Write-Host "`nError: IAM instance profile not found" -ForegroundColor Red
    Write-Host "Run 'terraform apply' first to create the required resources" -ForegroundColor Yellow
    exit 1
}

Write-Host "IAM Instance Profile: $instanceProfile" -ForegroundColor White

# Launch instance
Write-Host "`nLaunching instance..." -ForegroundColor Yellow
$instanceId = aws ec2 run-instances `
    --image-id $AMI_ID `
    --instance-type $INSTANCE_TYPE `
    --key-name $KEY_NAME `
    --subnet-id $subnetId `
    --security-group-ids $sgId `
    --iam-instance-profile "Name=$instanceProfile" `
    --user-data $USER_DATA_BASE64 `
    --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=100,VolumeType=gp3}" `
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=docker-builder-temp},{Key=Environment,Value=dev},{Key=Purpose,Value=docker-build}]" `
    --region $REGION `
    --query 'Instances[0].InstanceId' `
    --output text

Write-Host "`nInstance launched successfully!" -ForegroundColor Green
Write-Host "Instance ID: $instanceId" -ForegroundColor Yellow

Write-Host "`nWaiting for instance to be running..." -ForegroundColor Yellow
aws ec2 wait instance-running --instance-ids $instanceId --region $REGION

# Get public IP
$publicIp = aws ec2 describe-instances --instance-ids $instanceId --region $REGION --query 'Reservations[0].Instances[0].PublicIpAddress' --output text

Write-Host "`nInstance is running!" -ForegroundColor Green
Write-Host "Public IP: $publicIp" -ForegroundColor Yellow

Write-Host "`n=== Build Progress ===" -ForegroundColor Cyan
Write-Host "The instance is now building the Docker image. This will take 10-15 minutes." -ForegroundColor Yellow
Write-Host "`nTo monitor progress, SSH to the instance:" -ForegroundColor Cyan
Write-Host "ssh -i ~/.ssh/$KEY_NAME.pem ec2-user@$publicIp" -ForegroundColor White
Write-Host "`nThen run:" -ForegroundColor Cyan
Write-Host "tail -f /var/log/docker-build.log" -ForegroundColor White

Write-Host "`nOr check for completion marker:" -ForegroundColor Cyan
Write-Host "aws s3 ls s3://nutrition-video-analysis-dev-videos-60ppnqfp/docker-images/build-complete.txt --region $REGION" -ForegroundColor White

Write-Host "`n=== Next Steps ===" -ForegroundColor Cyan
Write-Host "1. Wait for Docker build to complete (10-15 minutes)" -ForegroundColor White
Write-Host "2. Once complete, force ECS to update: aws ecs update-service --cluster food-detection-v2-cluster --service food-detection-v2-worker --force-new-deployment --region $REGION" -ForegroundColor White
Write-Host "3. Terminate this instance to save costs: aws ec2 terminate-instances --instance-ids $instanceId --region $REGION" -ForegroundColor White
