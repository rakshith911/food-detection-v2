#!/bin/bash
set -ex

echo "=== Starting model download and upload ==="

# Configuration
REGION="us-east-1"
S3_BUCKET="nutrition-video-analysis-dev-models-dbenpoj2"
VIDEOS_BUCKET="nutrition-video-analysis-dev-videos-dbenpoj2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PRODUCTION_MODEL_ASSETS_DIR="${REPO_ROOT}/PRODUCTION/model_assets"
PRODUCTION_ASSETS_PREFIX="production_assets/model_assets"

# Create temporary directories
mkdir -p /tmp/checkpoints
mkdir -p /tmp/gdino_checkpoints
cd /tmp

# Download SAM2.1 checkpoints
echo "=== Downloading SAM2.1 checkpoints ==="
SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"

curl -L -o checkpoints/sam2.1_hiera_tiny.pt "${SAM2p1_BASE_URL}/sam2.1_hiera_tiny.pt"
curl -L -o checkpoints/sam2.1_hiera_small.pt "${SAM2p1_BASE_URL}/sam2.1_hiera_small.pt"
curl -L -o checkpoints/sam2.1_hiera_base_plus.pt "${SAM2p1_BASE_URL}/sam2.1_hiera_base_plus.pt"
curl -L -o checkpoints/sam2.1_hiera_large.pt "${SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"

# Download Grounding DINO checkpoints
echo "=== Downloading Grounding DINO checkpoints ==="
GDINO_BASE_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download"

curl -L -o gdino_checkpoints/groundingdino_swint_ogc.pth "${GDINO_BASE_URL}/v0.1.0-alpha/groundingdino_swint_ogc.pth"
curl -L -o gdino_checkpoints/groundingdino_swinb_cogcoor.pth "${GDINO_BASE_URL}/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"

# Upload to S3
echo "=== Uploading SAM2 checkpoints to S3 ==="
aws s3 cp checkpoints/ s3://${S3_BUCKET}/checkpoints/ --recursive --region ${REGION}

echo "=== Uploading Grounding DINO checkpoints to S3 ==="
aws s3 cp gdino_checkpoints/ s3://${S3_BUCKET}/gdino_checkpoints/ --recursive --region ${REGION}

if [ -d "${PRODUCTION_MODEL_ASSETS_DIR}" ]; then
    echo "=== Uploading production model assets to S3 ==="
    aws s3 cp "${PRODUCTION_MODEL_ASSETS_DIR}/" "s3://${S3_BUCKET}/${PRODUCTION_ASSETS_PREFIX}/" --recursive --region ${REGION}
else
    echo "=== Skipping production model assets upload: ${PRODUCTION_MODEL_ASSETS_DIR} not found ==="
fi

# Verify uploads
echo "=== Verifying uploads ==="
echo "SAM2 checkpoints:"
aws s3 ls s3://${S3_BUCKET}/checkpoints/ --region ${REGION} --recursive --human-readable

echo "Grounding DINO checkpoints:"
aws s3 ls s3://${S3_BUCKET}/gdino_checkpoints/ --region ${REGION} --recursive --human-readable

echo "Production model assets:"
aws s3 ls s3://${S3_BUCKET}/${PRODUCTION_ASSETS_PREFIX}/ --region ${REGION} --recursive --human-readable

# Write completion marker
echo "Model upload completed at $(date)" > /tmp/upload-complete.txt
aws s3 cp /tmp/upload-complete.txt s3://${VIDEOS_BUCKET}/models-upload-complete.txt --region ${REGION}

echo "=== DONE ==="
