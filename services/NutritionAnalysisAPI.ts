import * as Notifications from 'expo-notifications';
import { Platform } from 'react-native';

// Nutrition Video Analysis API Service
// Integrates with the AWS-deployed nutrition analysis backend
// Override with EXPO_PUBLIC_NUTRITION_API_URL to point at local/different backend (e.g. http://10.0.2.2:8000 for Android emulator)

const API_BASE_URL =
  (typeof process !== 'undefined' && process.env?.EXPO_PUBLIC_NUTRITION_API_URL) ||
  'https://qx3i66fa87.execute-api.us-east-1.amazonaws.com/v1';

export interface NutritionItem {
  food_name: string;
  mass_g: number;
  volume_ml?: number;
  total_calories?: number;
}

export interface SegmentedImage {
  frame: string;
  url: string;
  key: string;
}

export interface SegmentedImages {
  overlay_urls?: SegmentedImage[];
  mask_urls?: SegmentedImage[];
}

export interface NutritionAnalysisResult {
  job_id: string;
  status: 'pending_upload' | 'uploaded' | 'queued' | 'processing' | 'completed' | 'failed';
  created_at?: string;
  updated_at?: string;
  completed_at?: string;
  filename?: string;
  download_url?: string;
  segmented_images?: SegmentedImages;  // New: URLs to segmented images
  nutrition_summary?: {
    total_food_volume_ml: number;
    total_mass_g: number;
    total_calories_kcal: number;
    num_food_items: number;
  };
  items?: NutritionItem[];
  detailed_results?: any;
  error?: string;
}

export interface UploadResponse {
  job_id: string;
  upload_url: string;
  status: string;
  message: string;
}

export class NutritionAnalysisAPI {
  private baseUrl: string;
  private pushToken: string | null = null;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Get Expo push token for server-side completion notifications.
   * Returns null when unavailable (simulator/permissions denied/etc).
   */
  private async getPushToken(): Promise<string | null> {
    try {
      if (this.pushToken) {
        console.log('[PushNotif] Using cached push token:', this.pushToken.slice(-12));
        return this.pushToken;
      }

      console.log('[PushNotif] Checking notification permissions...');
      const { status: existingStatus } = await Notifications.getPermissionsAsync();
      console.log('[PushNotif] Existing permission status:', existingStatus);
      let status = existingStatus;
      if (status !== 'granted') {
        console.log('[PushNotif] Requesting notification permissions...');
        const req = await Notifications.requestPermissionsAsync();
        status = req.status;
        console.log('[PushNotif] Permission request result:', status);
      }
      if (status !== 'granted') {
        console.warn('[PushNotif] Permission not granted — push notifications disabled');
        return null;
      }

      if (Platform.OS === 'android') {
        console.log('[PushNotif] Setting up Android notification channel...');
        await Notifications.setNotificationChannelAsync('default', {
          name: 'default',
          importance: Notifications.AndroidImportance.DEFAULT,
        });
      }

      console.log('[PushNotif] Fetching Expo push token (projectId: 816a41f0)...');
      const tokenResponse = await Notifications.getExpoPushTokenAsync({
        projectId: '816a41f0-67d2-4dd1-b982-c10c51e6dd37',
      });

      this.pushToken = tokenResponse.data || null;
      if (this.pushToken) {
        console.log('[PushNotif] ✅ Push token acquired:', this.pushToken.slice(-12));
      } else {
        console.warn('[PushNotif] ⚠️ getExpoPushTokenAsync returned empty data');
      }
      return this.pushToken;
    } catch (error) {
      console.warn('[PushNotif] ❌ Push token unavailable:', error);
      return null;
    }
  }

  /**
   * Check if the API service is healthy
   */
  async checkHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      const data = await response.json();
      console.log('[Nutrition API] Health check:', data);
      return data.status === 'healthy';
    } catch (error) {
      console.error('[Nutrition API] Health check failed:', error);
      return false;
    }
  }

  /**
   * Request a presigned URL for video/image upload
   * @param filename - The filename for the upload
   * @param contentType - The MIME type (defaults to 'video/mp4', use 'image/jpeg' for images)
   */
  async requestUploadUrl(filename: string, contentType: string = 'video/mp4'): Promise<UploadResponse | null> {
    try {
      const pushToken = await this.getPushToken();
      console.log('[PushNotif] requestUploadUrl — push token present:', !!pushToken);
      console.log('[Nutrition API] Requesting upload URL from:', `${this.baseUrl}/api/upload`);
      console.log('[Nutrition API] Request body:', { type: 'presigned', filename, content_type: contentType, has_push_token: !!pushToken });
      
      const response = await fetch(`${this.baseUrl}/api/upload`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          type: 'presigned',
          filename,
          content_type: contentType,
          push_token: pushToken || undefined,
        }),
      });

      console.log('[Nutrition API] Response status:', response.status, response.statusText);
      
      if (!response.ok) {
        const errorBody = await response.text();
        console.error('[Nutrition API] Error response body:', errorBody);
        throw new Error(`Upload request failed: ${response.status} - ${errorBody || response.statusText}`);
      }

      const data = await response.json();
      console.log('[Nutrition API] Upload URL received:', data.job_id);
      return data;
    } catch (error: any) {
      console.error('[Nutrition API] Failed to request upload URL:', error);
      console.error('[Nutrition API] Error details:', error.message);
      return null;
    }
  }

  /**
   * Upload video to S3 using presigned URL
   */
  async uploadVideo(presignedUrl: string, videoUri: string): Promise<boolean> {
    try {
      console.log('[Nutrition API] Uploading video from:', videoUri);

      // Fetch the video file
      const videoResponse = await fetch(videoUri);
      const videoBlob = await videoResponse.blob();

      // Upload to S3 with required encryption header
      const uploadResponse = await fetch(presignedUrl, {
        method: 'PUT',
        headers: {
          'Content-Type': 'video/mp4',
          'x-amz-server-side-encryption': 'aws:kms',
        },
        body: videoBlob,
      });

      if (!uploadResponse.ok) {
        throw new Error(`Upload failed: ${uploadResponse.statusText}`);
      }

      console.log('[Nutrition API] Video uploaded successfully');
      return true;
    } catch (error) {
      console.error('[Nutrition API] Failed to upload video:', error);
      return false;
    }
  }

  /**
   * Confirm upload and start processing
   */
  async confirmUpload(jobId: string, userContext?: Record<string, any>): Promise<boolean> {
    try {
      const pushToken = await this.getPushToken();
      console.log('[PushNotif] confirmUpload — push token present:', !!pushToken);
      const response = await fetch(`${this.baseUrl}/api/upload`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          type: 'confirm',
          job_id: jobId,
          push_token: pushToken || undefined,
          user_context: userContext && Object.keys(userContext).length > 0 ? userContext : undefined,
        }),
      });

      if (!response.ok) {
        throw new Error(`Upload confirmation failed: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('[Nutrition API] Upload confirmed, processing queued:', data);
      return true;
    } catch (error) {
      console.error('[Nutrition API] Failed to confirm upload:', error);
      return false;
    }
  }

  /**
   * Check the status of a job
   */
  async checkStatus(jobId: string): Promise<NutritionAnalysisResult | null> {
    try {
      const response = await fetch(`${this.baseUrl}/api/status/${jobId}`);

      if (!response.ok) {
        throw new Error(`Status check failed: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('[Nutrition API] Job status:', data.status);
      return data;
    } catch (error) {
      console.error('[Nutrition API] Failed to check status:', error);
      return null;
    }
  }

  /**
   * Get the results of a completed job
   * @param jobId - The job ID
   * @param detailed - Whether to fetch detailed results (default: true to get segmented images)
   */
  async getResults(jobId: string, detailed: boolean = true): Promise<NutritionAnalysisResult | null> {
    try {
      // Request detailed results to get segmented images
      const url = detailed 
        ? `${this.baseUrl}/api/results/${jobId}?detailed=true`
        : `${this.baseUrl}/api/results/${jobId}`;
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`Results fetch failed: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('[Nutrition API] Results received:', data);
      
      // Check if segmented images are already in the response (from Lambda)
      if (data.segmented_images) {
        console.log('[Nutrition API] Segmented images URLs found in response');
      }

      // Fetch detailed results if download URL is available
      if (data.download_url) {
        try {
          console.log('[Nutrition API] Fetching detailed results from S3...');
          const detailsResponse = await fetch(data.download_url, {
            method: 'GET',
            headers: {
              'Accept': 'application/json',
            },
          });
          console.log('[Nutrition API] Details response status:', detailsResponse.status);

          if (detailsResponse.ok) {
            const detailedResults = await detailsResponse.json();
            console.log('[Nutrition API] Detailed results fetched successfully');
            console.log('[Nutrition API] Raw detailed results:', JSON.stringify(detailedResults, null, 2));
            data.detailed_results = detailedResults;

            // Extract items array from detailed results - check multiple possible locations
            let itemsArray: any[] | null = null;
            
            // Location 1: detected_items (top-level)
            if (detailedResults.detected_items && detailedResults.detected_items.length > 0) {
              console.log('[Nutrition API] Found items in detected_items');
              itemsArray = detailedResults.detected_items;
            }
            // Location 2: items (top-level)
            else if (detailedResults.items && detailedResults.items.length > 0) {
              console.log('[Nutrition API] Found items at top level');
              itemsArray = detailedResults.items;
            }
            // Location 3: full_results.nutrition.items
            else if (detailedResults.full_results?.nutrition?.items && detailedResults.full_results.nutrition.items.length > 0) {
              console.log('[Nutrition API] Found items in full_results.nutrition.items');
              itemsArray = detailedResults.full_results.nutrition.items;
            }
            // Location 4: nutrition.items
            else if (detailedResults.nutrition?.items && detailedResults.nutrition.items.length > 0) {
              console.log('[Nutrition API] Found items in nutrition.items');
              itemsArray = detailedResults.nutrition.items;
            }
            
            if (itemsArray && itemsArray.length > 0) {
              console.log('[Nutrition API] Processing', itemsArray.length, 'detected items:');
              data.items = itemsArray.map((item: any) => ({
                food_name: item.food_name || item.name || 'Unknown',
                mass_g: item.mass_g || 0,
                volume_ml: item.volume_ml,
                total_calories: item.total_calories || item.calories || 0,
              }));

              data.items.forEach((item: NutritionItem, index: number) => {
                console.log(`  ${index + 1}. ${item.food_name} - ${Math.round(item.total_calories || 0)} kcal - ${Math.round(item.mass_g)}g`);
              });
            } else {
              console.warn('[Nutrition API] No items array found in detailed results');
            }
            
            // Extract nutrition_summary from detailed results - check multiple locations
            let nutritionSummary: any = null;
            
            // Location 1: full_results.nutrition.summary
            if (detailedResults.full_results?.nutrition?.summary) {
              console.log('[Nutrition API] Found nutrition summary in full_results.nutrition.summary');
              nutritionSummary = detailedResults.full_results.nutrition.summary;
            }
            // Location 2: nutrition.summary
            else if (detailedResults.nutrition?.summary) {
              console.log('[Nutrition API] Found nutrition summary in nutrition.summary');
              nutritionSummary = detailedResults.nutrition.summary;
            }
            // Location 3: meal_summary
            else if (detailedResults.meal_summary && Object.keys(detailedResults.meal_summary).length > 0) {
              console.log('[Nutrition API] Found nutrition summary in meal_summary');
              nutritionSummary = detailedResults.meal_summary;
            }
            
            // Update data.nutrition_summary if we found it
            if (nutritionSummary && Object.keys(nutritionSummary).length > 0) {
              data.nutrition_summary = {
                total_food_volume_ml: nutritionSummary.total_food_volume_ml || 0,
                total_mass_g: nutritionSummary.total_mass_g || 0,
                total_calories_kcal: nutritionSummary.total_calories_kcal || 0,
                num_food_items: nutritionSummary.num_food_items || (itemsArray?.length || 0),
              };
              console.log('[Nutrition API] Extracted nutrition_summary:', data.nutrition_summary);
            } else if (itemsArray && itemsArray.length > 0) {
              // Calculate summary from items if not available
              const totalCalories = itemsArray.reduce((sum: number, item: any) => 
                sum + (item.total_calories || item.calories || 0), 0);
              const totalMass = itemsArray.reduce((sum: number, item: any) => 
                sum + (item.mass_g || 0), 0);
              const totalVolume = itemsArray.reduce((sum: number, item: any) => 
                sum + (item.volume_ml || 0), 0);
              
              data.nutrition_summary = {
                total_food_volume_ml: totalVolume,
                total_mass_g: totalMass,
                total_calories_kcal: totalCalories,
                num_food_items: itemsArray.length,
              };
              console.log('[Nutrition API] Calculated nutrition_summary from items:', data.nutrition_summary);
            }
            
            // Extract segmented images from detailed results if available
            if (detailedResults.segmented_images || detailedResults.tracking?.objects) {
              // Segmented images might be in detailed_results or we need to construct URLs
              console.log('[Nutrition API] Segmented images data found in detailed results');
            }
          } else {
            const errorText = await detailsResponse.text();
            console.error('[Nutrition API] Failed to fetch details. Status:', detailsResponse.status, 'Error:', errorText.substring(0, 200));
            console.log('[Nutrition API] Falling back to detected_foods from main response');

            // Fallback: use detected_foods from the main response if detailed fetch fails
            if ((data as any).detected_foods && Array.isArray((data as any).detected_foods)) {
              data.items = (data as any).detected_foods.map((item: any) => ({
                food_name: item.name || 'Unknown',
                mass_g: item.mass_g || 0,
                volume_ml: undefined,
                total_calories: item.calories || 0,
              }));
              console.log('[Nutrition API] Using fallback detected_foods:', data.items.length, 'items');
            }
          }
        } catch (detailError) {
          console.error('[Nutrition API] Could not fetch detailed results:', detailError);
          console.log('[Nutrition API] Falling back to detected_foods from main response');

          // Fallback: use detected_foods from the main response
          if ((data as any).detected_foods && Array.isArray((data as any).detected_foods)) {
            data.items = (data as any).detected_foods.map((item: any) => ({
              food_name: item.name || 'Unknown',
              mass_g: item.mass_g || 0,
              volume_ml: undefined,
              total_calories: item.calories || 0,
            }));
            console.log('[Nutrition API] Using fallback detected_foods:', data.items.length, 'items');
          }
        }
      } else {
        console.warn('[Nutrition API] No download_url provided in results');
        // Fallback: use detected_foods from the main response
        if ((data as any).detected_foods && Array.isArray((data as any).detected_foods)) {
          data.items = (data as any).detected_foods.map((item: any) => ({
            food_name: item.name || 'Unknown',
            mass_g: item.mass_g || 0,
            volume_ml: undefined,
            total_calories: item.calories || 0,
          }));
          console.log('[Nutrition API] Using detected_foods from response:', data.items.length, 'items');
        }
      }

      return data;
    } catch (error) {
      console.error('[Nutrition API] Failed to get results:', error);
      return null;
    }
  }

  /**
   * Upload image to S3 using presigned URL
   * @param presignedUrl - The presigned S3 URL
   * @param imageUri - The local image URI
   * @param contentType - The MIME type (must match what was used to generate presigned URL)
   */
  async uploadImage(presignedUrl: string, imageUri: string, contentType: string = 'image/jpeg'): Promise<boolean> {
    try {
      console.log('[Nutrition API] Uploading image from:', imageUri);
      console.log('[Nutrition API] Using content type:', contentType);

      // Fetch the image file
      const imageResponse = await fetch(imageUri);
      const imageBlob = await imageResponse.blob();

      console.log('[Nutrition API] Image blob size:', imageBlob.size);

      // Upload to S3 with required encryption header
      // IMPORTANT: Content-Type MUST match exactly what was used to generate the presigned URL
      const uploadResponse = await fetch(presignedUrl, {
        method: 'PUT',
        headers: {
          'Content-Type': contentType,
          'x-amz-server-side-encryption': 'aws:kms',
        },
        body: imageBlob,
      });

      console.log('[Nutrition API] Upload response status:', uploadResponse.status, uploadResponse.statusText);

      if (!uploadResponse.ok) {
        const errorText = await uploadResponse.text();
        console.error('[Nutrition API] Upload error details:', errorText);
        throw new Error(`Upload failed: ${uploadResponse.status} ${uploadResponse.statusText} - ${errorText}`);
      }

      console.log('[Nutrition API] Image uploaded successfully');
      return true;
    } catch (error) {
      console.error('[Nutrition API] Failed to upload image:', error);
      return false;
    }
  }

  /**
   * Complete workflow: Upload video and wait for results
   */
  async analyzeVideo(
    videoUri: string,
    filename: string,
    onProgress?: (status: string) => void,
    onJobCreated?: (jobId: string) => void,
    userContext?: Record<string, any>
  ): Promise<NutritionAnalysisResult | null> {
    try {
      // Step 1: Request upload URL
      onProgress?.('Requesting upload URL...');
      const uploadData = await this.requestUploadUrl(filename);
      if (!uploadData) {
        throw new Error('Failed to get upload URL');
      }

      // Notify caller of job_id immediately so it can be persisted before any further async work
      console.log('[Nutrition API] Job created, notifying caller with job_id:', uploadData.job_id);
      onJobCreated?.(uploadData.job_id);

      // Step 2: Upload video
      onProgress?.('Uploading video...');
      const uploaded = await this.uploadVideo(uploadData.upload_url, videoUri);
      if (!uploaded) {
        throw new Error('Failed to upload video');
      }

      // Step 3: Confirm upload to start processing (pass user context so it reaches Gemini)
      onProgress?.('Starting analysis...');
      const confirmed = await this.confirmUpload(uploadData.job_id, userContext);
      if (!confirmed) {
        throw new Error('Failed to confirm upload');
      }

      // Step 4: Poll for results (always request detailed to get segmented images)
      // 72 attempts × 5s = 360 seconds (6 minutes) before timeout
      onProgress?.('Processing video...');
      return await this.pollForResults(uploadData.job_id, onProgress, 72, 5000, true);
    } catch (error: any) {
      console.warn('[Nutrition API] Video analysis failed:', error?.message);
      if (error?.message === 'Analysis timeout') {
        console.log('[Nutrition API] Video timed out — job is still queued in SQS, re-throwing for caller');
        throw error;
      }
      return null;
    }
  }

  /**
   * Complete workflow: Upload image and wait for results
   */
  async analyzeImage(
    imageUri: string,
    filename: string,
    onProgress?: (status: string) => void,
    onJobCreated?: (jobId: string) => void,
    userContext?: Record<string, any>
  ): Promise<NutritionAnalysisResult | null> {
    try {
      // Step 1: Request upload URL with image content type
      onProgress?.('Requesting upload URL...');
      const contentType = filename.toLowerCase().endsWith('.png') ? 'image/png' : 'image/jpeg';
      const uploadData = await this.requestUploadUrl(filename, contentType);
      if (!uploadData) {
        throw new Error('Failed to get upload URL');
      }

      // Notify caller of job_id immediately so it can be persisted before any further async work
      console.log('[Nutrition API] Job created, notifying caller with job_id:', uploadData.job_id);
      onJobCreated?.(uploadData.job_id);

      // Step 2: Upload image with matching content type
      onProgress?.('Uploading image...');
      const uploaded = await this.uploadImage(uploadData.upload_url, imageUri, contentType);
      if (!uploaded) {
        throw new Error('Failed to upload image');
      }

      // Step 3: Confirm upload to start processing (pass user context so it reaches Gemini)
      onProgress?.('Starting analysis...');
      const confirmed = await this.confirmUpload(uploadData.job_id, userContext);
      if (!confirmed) {
        throw new Error('Failed to confirm upload');
      }

      // Step 4: Poll for results (always request detailed to get segmented images)
      // 72 attempts × 5s = 360 seconds (6 minutes) before timeout
      onProgress?.('Processing image...');
      return await this.pollForResults(uploadData.job_id, onProgress, 72, 5000, true);
    } catch (error: any) {
      console.warn('[Nutrition API] Image analysis failed:', error?.message);
      if (error?.message === 'Analysis timeout') {
        console.log('[Nutrition API] Image timed out — job is still queued in SQS, re-throwing for caller');
        throw error;
      }
      return null;
    }
  }

  /**
   * Poll for job completion
   * @param detailed - Whether to request detailed results (default: true to get segmented images)
   */
  private async pollForResults(
    jobId: string,
    onProgress?: (status: string) => void,
    maxAttempts: number = 60,
    intervalMs: number = 5000,
    detailed: boolean = true
  ): Promise<NutritionAnalysisResult | null> {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      await new Promise(resolve => setTimeout(resolve, intervalMs));

      const status = await this.checkStatus(jobId);
      if (!status) {
        continue;
      }

      if (status.status === 'completed') {
        onProgress?.('Analysis complete!');
        return await this.getResults(jobId, detailed);
      }

      if (status.status === 'failed') {
        throw new Error(status.error || 'Analysis failed');
      }

      onProgress?.(`Processing... (${attempt + 1}/${maxAttempts})`);
    }

    throw new Error('Analysis timeout');
  }
}

// Export singleton instance (uses API_BASE_URL; set EXPO_PUBLIC_NUTRITION_API_URL to override)
export const nutritionAnalysisAPI = new NutritionAnalysisAPI();
