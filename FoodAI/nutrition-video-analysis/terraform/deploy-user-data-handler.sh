#!/bin/bash
# Deploy the user-data-handler Lambda and API Gateway routes
# This script deploys independently of Terraform (for when state is out of sync)

set -e

REGION="us-east-1"
FUNCTION_NAME="food-detection-v2-user-data-handler"
ROLE_NAME="nutri-analysis-dev-lambda-exec"
API_ID="c89txc5qr6"
STAGE="v1"
BUCKET="ukcal-user-uploads"
VIDEOS_BUCKET="nutrition-video-analysis-dev-videos-dbenpoj2"
RESULTS_BUCKET="nutrition-video-analysis-dev-results-dbenpoj2"
DYNAMO_TABLE="ukcal-business-profiles"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "=== Deploying User Data Handler Lambda ==="
echo "Region: $REGION"
echo "Account: $ACCOUNT_ID"
echo "API Gateway: $API_ID"

# Step 1: Package Lambda code
echo ""
echo "1. Packaging Lambda code..."
cd "$(dirname "$0")/lambda_code/user_data_handler"
zip -r /tmp/user_data_handler.zip lambda_function.py
cd - > /dev/null

# Step 2: Find the Lambda execution role ARN
echo "2. Finding Lambda execution role..."
ROLE_ARN=$(aws iam list-roles --query "Roles[?starts_with(RoleName, '${ROLE_NAME}')].Arn | [0]" --output text --region $REGION)

if [ "$ROLE_ARN" = "None" ] || [ -z "$ROLE_ARN" ]; then
  echo "ERROR: Could not find IAM role starting with '${ROLE_NAME}'"
  echo "Available roles:"
  aws iam list-roles --query "Roles[?contains(RoleName, 'lambda')].RoleName" --output text --region $REGION
  exit 1
fi
echo "   Role ARN: $ROLE_ARN"

# Step 3: Add S3 permissions for ukcal-user-uploads bucket to the role
echo "3. Adding S3 permissions for ${BUCKET}..."
POLICY_DOC=$(cat <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::${BUCKET}/*"
    },
    {
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::${BUCKET}"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:DeleteObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::${VIDEOS_BUCKET}",
        "arn:aws:s3:::${VIDEOS_BUCKET}/*",
        "arn:aws:s3:::${RESULTS_BUCKET}",
        "arn:aws:s3:::${RESULTS_BUCKET}/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": "dynamodb:DeleteItem",
      "Resource": "arn:aws:dynamodb:${REGION}:${ACCOUNT_ID}:table/${DYNAMO_TABLE}"
    }
  ]
}
POLICY
)

# Extract the role name from the ARN
ACTUAL_ROLE_NAME=$(echo "$ROLE_ARN" | awk -F/ '{print $NF}')

aws iam put-role-policy \
  --role-name "$ACTUAL_ROLE_NAME" \
  --policy-name "user-data-s3-access" \
  --policy-document "$POLICY_DOC" \
  --region $REGION 2>/dev/null || echo "   (Policy may already exist, continuing...)"
echo "   S3 permissions added"

# Step 4: Create or update the Lambda function
echo "4. Creating/updating Lambda function..."
if aws lambda get-function --function-name $FUNCTION_NAME --region $REGION > /dev/null 2>&1; then
  echo "   Function exists, updating code..."
  aws lambda update-function-code \
    --function-name $FUNCTION_NAME \
    --zip-file fileb:///tmp/user_data_handler.zip \
    --region $REGION > /dev/null

  # Wait for update to complete
  aws lambda wait function-updated --function-name $FUNCTION_NAME --region $REGION 2>/dev/null || true

  aws lambda update-function-configuration \
    --function-name $FUNCTION_NAME \
    --environment "Variables={USER_DATA_BUCKET=${BUCKET},S3_PREFIX=UKcal,VIDEOS_BUCKET=${VIDEOS_BUCKET},RESULTS_BUCKET=${RESULTS_BUCKET},DYNAMO_TABLE=${DYNAMO_TABLE}}" \
    --timeout 60 \
    --region $REGION > /dev/null
else
  echo "   Creating new function..."
  aws lambda create-function \
    --function-name $FUNCTION_NAME \
    --runtime python3.11 \
    --handler lambda_function.lambda_handler \
    --role "$ROLE_ARN" \
    --zip-file fileb:///tmp/user_data_handler.zip \
    --timeout 60 \
    --memory-size 256 \
    --environment "Variables={USER_DATA_BUCKET=${BUCKET},S3_PREFIX=UKcal,VIDEOS_BUCKET=${VIDEOS_BUCKET},RESULTS_BUCKET=${RESULTS_BUCKET},DYNAMO_TABLE=${DYNAMO_TABLE}}" \
    --region $REGION > /dev/null
fi
echo "   Lambda function ready"

LAMBDA_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}"

# Step 5: Create API Gateway resources
echo "5. Setting up API Gateway routes..."

# Get root resource ID
ROOT_ID=$(aws apigateway get-resources --rest-api-id $API_ID --region $REGION \
  --query "items[?path=='/'].id" --output text)

# Check if /user-data resource already exists
USER_DATA_ID=$(aws apigateway get-resources --rest-api-id $API_ID --region $REGION \
  --query "items[?path=='/user-data'].id" --output text)

if [ -z "$USER_DATA_ID" ] || [ "$USER_DATA_ID" = "None" ]; then
  echo "   Creating /user-data resource..."
  USER_DATA_ID=$(aws apigateway create-resource \
    --rest-api-id $API_ID \
    --parent-id $ROOT_ID \
    --path-part "user-data" \
    --region $REGION \
    --query "id" --output text)
fi
echo "   /user-data: $USER_DATA_ID"

# Check if /user-data/{userId} exists
USER_ID_RESOURCE=$(aws apigateway get-resources --rest-api-id $API_ID --region $REGION \
  --query "items[?path=='/user-data/{userId}'].id" --output text)

if [ -z "$USER_ID_RESOURCE" ] || [ "$USER_ID_RESOURCE" = "None" ]; then
  echo "   Creating /user-data/{userId} resource..."
  USER_ID_RESOURCE=$(aws apigateway create-resource \
    --rest-api-id $API_ID \
    --parent-id $USER_DATA_ID \
    --path-part "{userId}" \
    --region $REGION \
    --query "id" --output text)
fi
echo "   /user-data/{userId}: $USER_ID_RESOURCE"

# Check if /user-data/{userId}/{dataType} exists
DATA_TYPE_RESOURCE=$(aws apigateway get-resources --rest-api-id $API_ID --region $REGION \
  --query "items[?path=='/user-data/{userId}/{dataType}'].id" --output text)

if [ -z "$DATA_TYPE_RESOURCE" ] || [ "$DATA_TYPE_RESOURCE" = "None" ]; then
  echo "   Creating /user-data/{userId}/{dataType} resource..."
  DATA_TYPE_RESOURCE=$(aws apigateway create-resource \
    --rest-api-id $API_ID \
    --parent-id $USER_ID_RESOURCE \
    --path-part "{dataType}" \
    --region $REGION \
    --query "id" --output text)
fi
echo "   /user-data/{userId}/{dataType}: $DATA_TYPE_RESOURCE"

# Helper to create method + integration
create_method() {
  local HTTP_METHOD=$1
  echo "   Setting up $HTTP_METHOD method..."

  # Delete existing method if present (to recreate cleanly)
  aws apigateway delete-method \
    --rest-api-id $API_ID \
    --resource-id $DATA_TYPE_RESOURCE \
    --http-method $HTTP_METHOD \
    --region $REGION 2>/dev/null || true

  # Create method
  aws apigateway put-method \
    --rest-api-id $API_ID \
    --resource-id $DATA_TYPE_RESOURCE \
    --http-method $HTTP_METHOD \
    --authorization-type "NONE" \
    --request-parameters "method.request.path.userId=true,method.request.path.dataType=true" \
    --region $REGION > /dev/null

  # Create Lambda integration
  aws apigateway put-integration \
    --rest-api-id $API_ID \
    --resource-id $DATA_TYPE_RESOURCE \
    --http-method $HTTP_METHOD \
    --type AWS_PROXY \
    --integration-http-method POST \
    --uri "arn:aws:apigateway:${REGION}:lambda:path/2015-03-31/functions/${LAMBDA_ARN}/invocations" \
    --region $REGION > /dev/null
}

# Create GET, PUT, DELETE methods with Lambda proxy integration
create_method "GET"
create_method "PUT"
create_method "DELETE"

# Create OPTIONS method for CORS
echo "   Setting up OPTIONS (CORS)..."
aws apigateway delete-method \
  --rest-api-id $API_ID \
  --resource-id $DATA_TYPE_RESOURCE \
  --http-method OPTIONS \
  --region $REGION 2>/dev/null || true

aws apigateway put-method \
  --rest-api-id $API_ID \
  --resource-id $DATA_TYPE_RESOURCE \
  --http-method OPTIONS \
  --authorization-type "NONE" \
  --region $REGION > /dev/null

aws apigateway put-integration \
  --rest-api-id $API_ID \
  --resource-id $DATA_TYPE_RESOURCE \
  --http-method OPTIONS \
  --type MOCK \
  --request-templates '{"application/json": "{\"statusCode\": 200}"}' \
  --region $REGION > /dev/null

aws apigateway put-method-response \
  --rest-api-id $API_ID \
  --resource-id $DATA_TYPE_RESOURCE \
  --http-method OPTIONS \
  --status-code 200 \
  --response-parameters "method.response.header.Access-Control-Allow-Headers=false,method.response.header.Access-Control-Allow-Methods=false,method.response.header.Access-Control-Allow-Origin=false" \
  --response-models '{"application/json": "Empty"}' \
  --region $REGION > /dev/null

aws apigateway put-integration-response \
  --rest-api-id $API_ID \
  --resource-id $DATA_TYPE_RESOURCE \
  --http-method OPTIONS \
  --status-code 200 \
  --response-parameters '{
    "method.response.header.Access-Control-Allow-Headers": "'"'"'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'"'"'",
    "method.response.header.Access-Control-Allow-Methods": "'"'"'GET,PUT,DELETE,OPTIONS'"'"'",
    "method.response.header.Access-Control-Allow-Origin": "'"'"'*'"'"'"
  }' \
  --region $REGION > /dev/null

# Step 6: Add Lambda invoke permission for API Gateway
echo "6. Adding Lambda invoke permission..."
aws lambda add-permission \
  --function-name $FUNCTION_NAME \
  --statement-id "AllowAPIGatewayInvoke" \
  --action "lambda:InvokeFunction" \
  --principal "apigateway.amazonaws.com" \
  --source-arn "arn:aws:apigateway:${REGION}::/restapis/${API_ID}/*/\*/user-data/*" \
  --region $REGION 2>/dev/null || echo "   (Permission may already exist, continuing...)"

# Step 7: Deploy the API
echo "7. Deploying API to stage '${STAGE}'..."
aws apigateway create-deployment \
  --rest-api-id $API_ID \
  --stage-name $STAGE \
  --description "Added /user-data routes for per-user S3 backup" \
  --region $REGION > /dev/null

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Endpoints:"
echo "  GET  https://${API_ID}.execute-api.${REGION}.amazonaws.com/${STAGE}/user-data/{userId}/{dataType}"
echo "  PUT  https://${API_ID}.execute-api.${REGION}.amazonaws.com/${STAGE}/user-data/{userId}/{dataType}"
echo ""
echo "S3 Storage: s3://${BUCKET}/UKcal/{userId}/{profile|history|settings}.json"

# Cleanup
rm -f /tmp/user_data_handler.zip
