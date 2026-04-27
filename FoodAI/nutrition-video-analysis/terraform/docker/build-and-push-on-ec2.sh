#!/bin/bash
set -e

# Configuration
REGION="us-east-1"
ECR_REPOSITORY="185329004895.dkr.ecr.us-east-1.amazonaws.com/food-detection-v2-worker"
S3_BUCKET="nutrition-video-analysis-dev-videos-60ppnqfp"

echo "Starting Docker build and push on EC2..."

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo yum update -y
    sudo yum install -y docker
    sudo service docker start
    sudo usermod -a -G docker ec2-user
fi

# Install git if not present
if ! command -v git &> /dev/null; then
    echo "Installing git..."
    sudo yum install -y git
fi

# Clone repository
echo "Cloning repository..."
cd /home/ec2-user
rm -rf food-detection
git clone https://github.com/leolorence12345/food-detection.git
cd food-detection/FoodAI/nutrition-video-analysis/terraform/docker

# Log in to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_REPOSITORY}

# Build AMD64 image
echo "Building AMD64 Docker image (this will take 10-15 minutes)..."
docker build --platform linux/amd64 -f Dockerfile -t ${ECR_REPOSITORY}:latest .

# Push to ECR
echo "Pushing image to ECR..."
docker push ${ECR_REPOSITORY}:latest

echo "Docker image successfully pushed to ECR!"

# Write completion marker to S3
echo "Build completed at $(date)" > /tmp/build-complete.txt
aws s3 cp /tmp/build-complete.txt s3://${S3_BUCKET}/docker-images/build-complete.txt --region ${REGION}

echo "Done!"
