#!/usr/bin/env bash
# Build the Docker image in CodeBuild (from GitHub) and verify it was pushed to ECR.
# Run from repo root or terraform dir. Requires: push to GitHub first, then run this.
set -e

REGION="${AWS_REGION:-us-east-1}"
PROJECT_NAME="${CODEBUILD_PROJECT_NAME:-nutrition-video-analysis-dev-docker-build}"
ECR_REPO_NAME="${ECR_REPO_NAME:-food-detection-v2-worker}"

echo "=============================================="
echo "1. Starting CodeBuild (builds from GitHub, pushes to ECR)"
echo "=============================================="
BUILD_ID=$(aws codebuild start-build --project-name "$PROJECT_NAME" --region "$REGION" --query 'build.id' --output text)
echo "Build ID: $BUILD_ID"
echo ""

echo "2. Waiting for build to complete (this can take 10–20 min)..."
aws codebuild batch-get-builds --ids "$BUILD_ID" --region "$REGION" --query 'builds[0].{id:id,status:buildStatus,currentPhase:currentPhase}' --output table
while true; do
  STATUS=$(aws codebuild batch-get-builds --ids "$BUILD_ID" --region "$REGION" --query 'builds[0].buildStatus' --output text)
  PHASE=$(aws codebuild batch-get-builds --ids "$BUILD_ID" --region "$REGION" --query 'builds[0].currentPhase' --output text)
  echo "  Status: $STATUS  Phase: $PHASE"
  if [[ "$STATUS" == "SUCCEEDED" ]]; then
    echo "  Build succeeded."
    break
  fi
  if [[ "$STATUS" == "FAILED" ]] || [[ "$STATUS" == "FAULT" ]] || [[ "$STATUS" == "STOPPED" ]]; then
    echo "  Build failed. Check logs:"
    echo "  aws logs tail /aws/codebuild/$PROJECT_NAME --follow --region $REGION"
    exit 1
  fi
  sleep 60
done
echo ""

echo "=============================================="
echo "3. Verifying image in ECR"
echo "=============================================="
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO_NAME"
echo "Repository: $ECR_URI"
aws ecr describe-images --repository-name "$ECR_REPO_NAME" --region "$REGION" --query 'imageDetails | sort_by(@, &imagePushedAt) | [-1].{pushed:imagePushedAt,tags:imageTags}' --output table
echo ""
echo "Latest image in ECR is updated. To deploy to ECS, run:"
echo "  aws ecs update-service --cluster food-detection-v2-cluster --service food-detection-v2-worker --force-new-deployment --region $REGION"
echo ""
