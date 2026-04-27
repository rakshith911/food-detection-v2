#!/bin/bash
set -e

# Test script to verify segmented images integration
API_BASE_URL="https://c89txc5qr6.execute-api.us-east-1.amazonaws.com/v1"
IMAGE_PATH="/Users/leo/FoodProject/food-detection/unhealthy-fast-food-delivery-menu-featuring-assorted-burgers-cheeseburgers-nuggets-french-fries-soda-high-calorie-low-356045884.jpg-2.jpg"
REGION="us-east-1"

echo "=========================================="
echo "Testing Segmented Images Integration"
echo "=========================================="
echo ""

# Step 1: Request upload URL
echo "Step 1: Requesting upload URL..."
UPLOAD_RESPONSE=$(curl -s -X POST "${API_BASE_URL}/api/upload" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "presigned",
    "filename": "test_segmentation.jpg",
    "content_type": "image/jpeg"
  }')

JOB_ID=$(echo "$UPLOAD_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['job_id'])" 2>/dev/null || echo "")
UPLOAD_URL=$(echo "$UPLOAD_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['upload_url'])" 2>/dev/null || echo "")

if [ -z "$JOB_ID" ] || [ -z "$UPLOAD_URL" ]; then
  echo "❌ Failed to get upload URL"
  echo "Response: $UPLOAD_RESPONSE"
  exit 1
fi

echo "✅ Got upload URL"
echo "Job ID: $JOB_ID"
echo ""

# Step 2: Upload image
echo "Step 2: Uploading image..."
UPLOAD_RESULT=$(curl -s -w "\n%{http_code}" -X PUT "$UPLOAD_URL" \
  -H "Content-Type: image/jpeg" \
  -H "x-amz-server-side-encryption: aws:kms" \
  --upload-file "$IMAGE_PATH")

HTTP_CODE=$(echo "$UPLOAD_RESULT" | tail -n1)
if [ "$HTTP_CODE" != "200" ]; then
  echo "❌ Upload failed with HTTP $HTTP_CODE"
  exit 1
fi

echo "✅ Image uploaded successfully"
echo ""

# Step 3: Confirm upload
echo "Step 3: Confirming upload to start processing..."
CONFIRM_RESPONSE=$(curl -s -X POST "${API_BASE_URL}/api/upload" \
  -H "Content-Type: application/json" \
  -d "{
    \"type\": \"confirm\",
    \"job_id\": \"$JOB_ID\"
  }")

echo "✅ Processing started"
echo ""

# Step 4: Poll for results
echo "Step 4: Polling for results (this may take a few minutes)..."
MAX_ATTEMPTS=60
ATTEMPT=0
STATUS="processing"

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
  sleep 5
  ATTEMPT=$((ATTEMPT + 1))
  
  STATUS_RESPONSE=$(curl -s "${API_BASE_URL}/api/status/${JOB_ID}")
  STATUS=$(echo "$STATUS_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))" 2>/dev/null || echo "unknown")
  
  echo "  Attempt $ATTEMPT/$MAX_ATTEMPTS: Status = $STATUS"
  
  if [ "$STATUS" = "completed" ]; then
    echo "✅ Processing completed!"
    break
  elif [ "$STATUS" = "failed" ]; then
    echo "❌ Processing failed"
    echo "Response: $STATUS_RESPONSE"
    exit 1
  fi
done

if [ "$STATUS" != "completed" ]; then
  echo "❌ Timeout waiting for completion"
  exit 1
fi

echo ""

# Step 5: Get results and check for segmented_images
echo "Step 5: Fetching results and checking for segmented_images..."
RESULTS_RESPONSE=$(curl -s "${API_BASE_URL}/api/results/${JOB_ID}?detailed=true")

# Check if segmented_images field exists
HAS_SEGMENTED=$(echo "$RESULTS_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print('yes' if 'segmented_images' in data else 'no')" 2>/dev/null || echo "no")

if [ "$HAS_SEGMENTED" = "yes" ]; then
  echo "✅ segmented_images field found in response!"
  
  # Extract segmented images info
  OVERLAY_COUNT=$(echo "$RESULTS_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); seg=data.get('segmented_images', {}); overlays=seg.get('overlay_urls', []); print(len(overlays))" 2>/dev/null || echo "0")
  MASK_COUNT=$(echo "$RESULTS_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); seg=data.get('segmented_images', {}); masks=seg.get('mask_urls', []); print(len(masks))" 2>/dev/null || echo "0")
  
  echo "  - Overlay URLs: $OVERLAY_COUNT"
  echo "  - Mask URLs: $MASK_COUNT"
  
  if [ "$OVERLAY_COUNT" -gt 0 ]; then
    # Get first overlay URL and test if it's accessible
    OVERLAY_URL=$(echo "$RESULTS_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); seg=data.get('segmented_images', {}); overlays=seg.get('overlay_urls', []); print(overlays[0]['url'] if overlays else '')" 2>/dev/null || echo "")
    
    if [ -n "$OVERLAY_URL" ]; then
      echo ""
      echo "Step 6: Testing overlay URL accessibility..."
      HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$OVERLAY_URL" --max-time 10)
      
      if [ "$HTTP_CODE" = "200" ]; then
        echo "✅ Overlay URL is accessible (HTTP $HTTP_CODE)"
        echo "   URL: $OVERLAY_URL"
      else
        echo "⚠️  Overlay URL returned HTTP $HTTP_CODE (may be expired or inaccessible)"
      fi
    fi
  fi
  
  echo ""
  echo "=========================================="
  echo "✅ Integration Test PASSED!"
  echo "=========================================="
  echo ""
  echo "Job ID: $JOB_ID"
  echo "Results API: ${API_BASE_URL}/api/results/${JOB_ID}?detailed=true"
  echo ""
  echo "You can now test in your frontend app!"
  
else
  echo "❌ segmented_images field NOT found in response"
  echo ""
  echo "Response preview:"
  echo "$RESULTS_RESPONSE" | python3 -m json.tool 2>/dev/null | head -50 || echo "$RESULTS_RESPONSE" | head -50
  echo ""
  echo "=========================================="
  echo "❌ Integration Test FAILED"
  echo "=========================================="
  exit 1
fi
