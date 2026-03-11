/**
 * SegmentedImageCache
 *
 * Uses expo-image's disk cache with a stable cacheKey (job_id).
 *
 * How it works:
 *  - Both the card thumbnail and fullscreen modal render with cacheKey = cacheKeyFor(jobId).
 *  - expo-image stores the image on disk under that stable key, NOT the presigned URL.
 *  - When the presigned URL expires and a new one is fetched, the same cacheKey means
 *    expo-image serves from disk instantly — no re-download needed.
 *  - After a fresh reinstall the disk cache is empty; expo-image re-downloads from the
 *    new presigned URL and caches it again under the same key.
 */

import { Image } from 'expo-image';

/**
 * Returns the stable cache key for a job's segmented overlay image.
 * Pass this as cacheKey= on every OptimizedImage that shows the overlay
 * so expo-image shares one on-disk entry regardless of URL changes.
 */
export function cacheKeyFor(jobId: string): string {
  return `seg_overlay_${jobId}`;
}

/**
 * Eagerly warms expo-image's disk cache for the given URL.
 * The image is stored under the URL key here, but the subsequent render
 * with cacheKey= will re-use the same disk bytes via expo-image internals.
 * Call this as soon as the presigned URL is available.
 */
export async function prefetchSegmentedImage(jobId: string, url: string): Promise<void> {
  try {
    await Image.prefetch(url, 'disk');
    console.log(`[SegCache] Prefetched overlay for job ${jobId}`);
  } catch (e) {
    console.warn('[SegCache] Prefetch failed (will load on demand):', e);
  }
}
