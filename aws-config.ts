// AWS Configuration
// TODO: Replace these values with your actual AWS credentials

export const awsConfig = {
  Auth: {
    // REQUIRED - Amazon Cognito Region
    region: 'us-east-1',
    
    // REQUIRED - Amazon Cognito User Pool ID
    userPoolId: 'us-east-1_gMgIH1mpO',
    
    // REQUIRED - Amazon Cognito Web Client ID (App Client ID)
    userPoolWebClientId: '4gvuc3u96kvior3ijdggoq0fvj',
    
    // REQUIRED - Amazon Cognito Identity Pool ID (for DynamoDB access)
    identityPoolId: 'us-east-1:279bfad4-591c-486a-8d5b-2e77942b2813',
    
    // OPTIONAL - Enforce user authentication prior to accessing AWS resources
    mandatorySignIn: false,
    
    // OPTIONAL - Configuration for cookie storage
    cookieStorage: {
      domain: 'localhost',
      path: '/',
      expires: 365,
      secure: false,
    },
    
    // Note: We use the standard signUp/confirmSignUp flow for email OTP
    // No need to set authenticationFlowType - Amplify handles this automatically
  },
  
  // DynamoDB Configuration
  DynamoDB: {
    // REQUIRED - DynamoDB Table Name for business profiles
    tableName: 'ukcal-business-profiles', // Change this to your table name
    
    // REQUIRED - AWS Region (should match Cognito region)
    region: 'us-east-1',
  },
  
  // S3 Configuration (for future image/video storage)
  S3: {
    // REQUIRED - S3 Bucket Name
    bucketName: 'ukcal-user-uploads', // Change this to your bucket name

    // REQUIRED - AWS Region
    region: 'us-east-1',
  },

  // API Gateway Configuration
  API: {
    // REQUIRED - API Gateway endpoint
    endpoint: 'https://c89txc5qr6.execute-api.us-east-1.amazonaws.com/v1',
  },

  // User Data Backup Configuration (S3)
  // User data is stored at: s3://{S3.bucketName}/{s3Prefix}/{userId}/profile.json|history.json|settings.json
  // The API endpoint is the nutrition analysis API Gateway (same terraform deployment)
  UserData: {
    s3Prefix: 'UKcal', // S3 key prefix for per-user data folders
    apiEndpoint: 'https://c89txc5qr6.execute-api.us-east-1.amazonaws.com/v1', // food-detection-v2 API
  },

  // Delete Account OTP – dedicated service (e.g. separate Cognito or Lambda + API)
  // When set, the app uses this for delete-account verification emails instead of the sign-in Cognito flow.
  // Leave enabled: false until you provide the details (like for sign-in).
  DeleteAccountOTP: {
    enabled: false,
    // Option A: REST API (Lambda + API Gateway) – provide base URL and optional API key
    apiEndpoint: '', // e.g. 'https://xxxx.execute-api.us-east-1.amazonaws.com/prod'
    apiKey: '',     // optional, if your API requires a key header
    // Paths (optional – defaults: /send-delete-otp and /verify-delete-otp)
    sendPath: '/send-delete-otp',
    verifyPath: '/verify-delete-otp',

    // Option B: Separate Cognito User Pool for delete-account only (if you use a second pool)
    // userPoolId: '',
    // userPoolClientId: '',
    // region: 'us-east-1',
  },
};

// Instructions to get these values:
// 1. Go to AWS Console: https://console.aws.amazon.com/cognito/
// 2. Create a new User Pool (or use existing one)
// 3. Configure:
//    - Email as sign-in option
//    - Email OTP for verification
//    - No password required (passwordless OTP)
// 4. After creation, get:
//    - User Pool ID: From "User Pool Overview"
//    - App Client ID: From "App Integration" → "App clients"
//    - Region: From your AWS region selector
