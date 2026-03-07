#!/usr/bin/env bash
# After pushing to git, run this to build via CodeBuild and roll out the new image to ECS.
# Usage: from repo root: ./FoodAI/nutrition-video-analysis/terraform/scripts/start-build-and-deploy.sh

set -e
REGION="${AWS_REGION:-us-east-1}"
PROJECT_NAME="nutrition-video-analysis-dev-docker-build"
CLUSTER="nutrition-video-analysis-dev-cluster"
SERVICE="nutrition-video-analysis-dev-video-processor"

echo "Starting CodeBuild project: $PROJECT_NAME (region: $REGION)"
BUILD_ID=$(aws codebuild start-build --project-name "$PROJECT_NAME" --region "$REGION" --query 'build.id' --output text)
echo "Build started: $BUILD_ID"
echo "Watch logs: aws logs tail /aws/codebuild/$PROJECT_NAME --follow --region $REGION"
echo ""
echo "Waiting for build to complete..."
aws codebuild batch-get-builds --ids "$BUILD_ID" --region "$REGION" --query 'builds[0].buildStatus' --output text
while true; do
  STATUS=$(aws codebuild batch-get-builds --ids "$BUILD_ID" --region "$REGION" --query 'builds[0].buildStatus' --output text)
  echo "  Build status: $STATUS"
  if [[ "$STATUS" == "SUCCEEDED" ]]; then
    break
  fi
  if [[ "$STATUS" == "FAILED" ]] || [[ "$STATUS" == "FAULT" ]] || [[ "$STATUS" == "STOPPED" ]] || [[ "$STATUS" == "TIMED_OUT" ]]; then
    echo "Build failed. Check logs: aws logs tail /aws/codebuild/$PROJECT_NAME --follow --region $REGION"
    exit 1
  fi
  sleep 30
done

echo "Build succeeded. Forcing new ECS deployment..."

# Preserve current desired count so a manually-started server stays running after deploy
CURRENT_DESIRED=$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --region "$REGION" --query 'services[0].desiredCount' --output text)

aws ecs update-service \
  --cluster "$CLUSTER" \
  --service "$SERVICE" \
  --force-new-deployment \
  --desired-count "$CURRENT_DESIRED" \
  --region "$REGION" \
  --query 'service.{serviceName:serviceName,desiredCount:desiredCount,runningCount:runningCount}' \
  --output table

echo "Done. New tasks will pull the image from ECR and replace old ones."
