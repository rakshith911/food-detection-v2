#!/bin/bash
# Rebuild Docker image with S3 model download fix

set -e

echo "=== Downloading fixed worker.py from S3 ==="
aws s3 cp s3://nutrition-video-analysis-dev-models-dbenpoj2/worker-fix/worker.py /tmp/worker.py

echo "=== Setting up build directory ==="
cd /tmp
rm -rf docker-build
mkdir -p docker-build
cd docker-build

echo "=== Cloning source code ==="
git clone --depth 1 https://github.com/PraneethKumarT/FoodAI.git
cd FoodAI/nutrition-video-analysis/terraform/docker

echo "=== Replacing worker.py with fixed version ==="
cp /tmp/worker.py ./worker.py

echo "=== Logging into ECR ==="
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 185329004895.dkr.ecr.us-east-1.amazonaws.com

echo "=== Building Docker image ==="
docker build -t food-detection-v2-worker .

echo "=== Tagging image ==="
docker tag food-detection-v2-worker:latest 185329004895.dkr.ecr.us-east-1.amazonaws.com/food-detection-v2-worker:latest

echo "=== Pushing to ECR ==="
docker push 185329004895.dkr.ecr.us-east-1.amazonaws.com/food-detection-v2-worker:latest

echo "=== Forcing ECS redeployment ==="
aws ecs update-service --cluster food-detection-v2-cluster --service food-detection-v2-worker --force-new-deployment --region us-east-1

echo "=== DONE! ==="
echo "New Docker image pushed. ECS will redeploy in 2-3 minutes."
