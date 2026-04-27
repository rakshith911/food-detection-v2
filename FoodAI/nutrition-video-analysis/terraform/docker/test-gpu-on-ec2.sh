#!/bin/bash
set -e

# Script to test GPU-enabled Docker container on EC2 GPU instance
# Usage: ./test-gpu-on-ec2.sh

echo "=========================================="
echo "GPU Testing Setup for EC2"
echo "=========================================="

# Configuration
REGION="us-east-1"
ECR_REPOSITORY="185329004895.dkr.ecr.us-east-1.amazonaws.com/food-detection-v2-worker"
IMAGE_TAG="gpu-test"

# Step 1: Verify GPU is available
echo ""
echo "Step 1: Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA driver found"
    nvidia-smi
else
    echo "❌ ERROR: nvidia-smi not found. GPU not available!"
    echo "Make sure you're on a GPU instance (g4dn, g5, etc.)"
    exit 1
fi

# Step 2: Install/Start Docker
echo ""
echo "Step 2: Setting up Docker..."
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo yum update -y
    sudo yum install -y docker
    sudo service docker start
    sudo usermod -a -G docker ec2-user
    echo "⚠️  You may need to log out and back in for docker group to take effect"
    echo "   Or run: newgrp docker"
    exit 1
fi

# Install NVIDIA Container Toolkit
echo "Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify Docker can see GPU
echo "Verifying Docker GPU access..."
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi || {
    echo "❌ Docker cannot access GPU. Check NVIDIA Container Toolkit installation."
    exit 1
}

# Step 3: Clone repository (if needed)
echo ""
echo "Step 3: Preparing code..."
if [ ! -d "food-detection" ]; then
    echo "Cloning repository..."
    cd /home/ec2-user
    git clone https://github.com/leolorence12345/food-detection.git
fi

cd food-detection/FoodAI/nutrition-video-analysis/terraform/docker

# Step 4: Build GPU Docker image
echo ""
echo "Step 4: Building GPU-enabled Docker image..."
echo "This will take 15-20 minutes..."
docker build -f Dockerfile.gpu -t ${ECR_REPOSITORY}:${IMAGE_TAG} .

# Step 5: Test GPU in container
echo ""
echo "Step 5: Testing GPU access in container..."
docker run --rm --gpus all ${ECR_REPOSITORY}:${IMAGE_TAG} \
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

if [ $? -eq 0 ]; then
    echo "✅ GPU test passed!"
else
    echo "❌ GPU test failed!"
    exit 1
fi

# Step 6: Push to ECR (optional, for deployment testing)
echo ""
read -p "Push image to ECR? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Logging in to ECR..."
    aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_REPOSITORY}
    
    echo "Pushing GPU image to ECR..."
    docker push ${ECR_REPOSITORY}:${IMAGE_TAG}
    echo "✅ Image pushed as ${ECR_REPOSITORY}:${IMAGE_TAG}"
fi

echo ""
echo "=========================================="
echo "✅ GPU Testing Setup Complete!"
echo "=========================================="
echo ""
echo "To run the worker with GPU:"
echo "  docker run --rm --gpus all \\"
echo "    -e DEVICE=cuda \\"
echo "    -e S3_VIDEOS_BUCKET=your-bucket \\"
echo "    -e S3_RESULTS_BUCKET=your-results-bucket \\"
echo "    -e SQS_VIDEO_QUEUE_URL=your-queue-url \\"
echo "    ${ECR_REPOSITORY}:${IMAGE_TAG}"
echo ""

