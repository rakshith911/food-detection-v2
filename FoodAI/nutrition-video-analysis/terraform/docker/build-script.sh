#!/bin/bash
set -ex

# Log everything
exec > >(tee /var/log/docker-build.log)
exec 2>&1

cd /home/ec2-user

# Clone repository
echo "=== Cloning repository ==="
git clone https://github.com/leolorence12345/food-detection.git
cd food-detection/FoodAI/nutrition-video-analysis

# ECR login
echo "=== Logging into ECR ==="
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 185329004895.dkr.ecr.us-east-1.amazonaws.com

# Build AMD64 image
echo "=== Building AMD64 image ==="
docker build --platform linux/amd64 -f terraform/docker/Dockerfile -t 185329004895.dkr.ecr.us-east-1.amazonaws.com/food-detection-v2-worker:latest .

# Push to ECR
echo "=== Pushing to ECR ==="
docker push 185329004895.dkr.ecr.us-east-1.amazonaws.com/food-detection-v2-worker:latest

# Write completion marker
echo "Build completed at $(date)" > /tmp/build-complete.txt
aws s3 cp /tmp/build-complete.txt s3://nutrition-video-analysis-dev-videos-60ppnqfp/docker-images/build-complete.txt --region us-east-1

echo "=== DONE ==="
