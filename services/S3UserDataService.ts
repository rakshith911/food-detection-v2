import type { AnalysisEntry } from '../store/slices/historySlice';

// Same endpoint used by NutritionAnalysisAPI — the actual deployed API Gateway
const API_BASE = 'https://qx3i66fa87.execute-api.us-east-1.amazonaws.com/v1';

// Convert email to a safe S3 key segment
function toUserKey(email: string): string {
  return email.toLowerCase().replace(/[^a-z0-9._-]/g, '_');
}

async function s3Post(type: string, userKey: string, dataType: string, data?: any): Promise<Response> {
  return fetch(`${API_BASE}/api/upload`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      type,
      userKey,
      dataType,
      ...(data !== undefined ? { data } : {}),
    }),
  });
}

export async function saveProfileToS3(email: string, profile: object): Promise<boolean> {
  try {
    const res = await s3Post('user_data_write', toUserKey(email), 'profile', profile);
    return res.ok;
  } catch {
    return false;
  }
}

export async function loadProfileFromS3(email: string): Promise<object | null> {
  try {
    const res = await s3Post('user_data_read', toUserKey(email), 'profile');
    if (res.status === 404 || !res.ok) return null;
    const json = await res.json();
    return json.data ?? null;
  } catch {
    return null;
  }
}

export async function saveHistoryToS3(email: string, history: AnalysisEntry[]): Promise<boolean> {
  try {
    const res = await s3Post('user_data_write', toUserKey(email), 'history', history);
    if (!res.ok) {
      const text = await res.text();
      console.warn('[S3UserData] saveHistoryToS3 failed:', res.status, text);
    }
    return res.ok;
  } catch (e) {
    console.warn('[S3UserData] saveHistoryToS3 exception:', e);
    return false;
  }
}

export async function loadHistoryFromS3(email: string): Promise<AnalysisEntry[] | null> {
  try {
    const res = await s3Post('user_data_read', toUserKey(email), 'history');
    if (res.status === 404 || !res.ok) return null;
    const json = await res.json();
    return Array.isArray(json.data) ? json.data : null;
  } catch {
    return null;
  }
}

// ── Types expected by store slices (upstream interface) ───────────────────────

export interface ProfileBackup {
  userAccount: {
    userId: string;
    email: string;
    createdAt: number;
    hasCompletedProfile: boolean;
  };
  businessProfile: any | null;
  avatar: { id: number } | null;
  profileImage: string | null;
  consentDate?: string;
  updatedAt: string;
}

export interface HistoryBackup {
  entries: AnalysisEntry[];
  updatedAt: string;
}

export interface SettingsBackup {
  hasConsented: boolean | null;
  hasCompletedProfile: boolean | null;
  updatedAt: string;
}

export interface AllUserData {
  profile: ProfileBackup | null;
  history: HistoryBackup | null;
  settings: SettingsBackup | null;
}

// ── Class-based interface used by store slices ────────────────────────────────
// Uses userId (Cognito sub UUID) directly as the S3 key — already safe for S3.
// Lambda only supports 'profile' and 'history'; settings are silently skipped.

class S3UserDataServiceClass {
  backupInBackground(userId: string, dataType: 'profile' | 'history' | 'settings', data: any): void {
    if (dataType === 'settings') return; // Lambda doesn't support settings yet
    s3Post('user_data_write', userId, dataType, data).catch((e) => {
      console.warn(`[S3UserData] Background backup of ${dataType} failed:`, e);
    });
  }

  async restoreAll(userId: string): Promise<AllUserData> {
    try {
      const [profileRes, historyRes] = await Promise.allSettled([
        s3Post('user_data_read', userId, 'profile').then(async (r) => {
          if (!r.ok) return null;
          const j = await r.json();
          return j.data ?? null;
        }),
        s3Post('user_data_read', userId, 'history').then(async (r) => {
          if (!r.ok) return null;
          const j = await r.json();
          return j.data ?? null;
        }),
      ]);
      return {
        profile: profileRes.status === 'fulfilled' ? profileRes.value : null,
        history: historyRes.status === 'fulfilled' ? historyRes.value : null,
        settings: null,
      };
    } catch {
      return { profile: null, history: null, settings: null };
    }
  }

  async deleteAllUserData(userId: string, jobIds: string[]): Promise<void> {
    try {
      await fetch(`${API_BASE}/user-data/${encodeURIComponent(userId)}/account`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_ids: jobIds }),
      });
      console.log('[S3UserData] Account data deleted from S3/DynamoDB');
    } catch (e) {
      console.warn('[S3UserData] deleteAllUserData failed (non-fatal):', e);
    }
  }
}

export const s3UserDataService = new S3UserDataServiceClass();

// ── Image helpers ─────────────────────────────────────────────────────────────

// In-memory cache so we don't re-request presigned GET URLs within the same session
const imageUrlCache = new Map<string, { url: string; expiresAt: number }>();

/**
 * Get a 24-hour presigned GET URL for the original uploaded image.
 * The image was already uploaded to S3 during analysis — we just look it up by job_id.
 * Results are cached in-memory for the session.
 */
export async function getImagePresignedUrl(jobId: string): Promise<string | null> {
  const cached = imageUrlCache.get(jobId);
  if (cached && Date.now() < cached.expiresAt) return cached.url;
  try {
    const res = await fetch(`${API_BASE}/api/upload`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type: 'user_image_get_url', jobId }),
    });
    if (!res.ok) return null;
    const json = await res.json();
    if (json.url) {
      // Cache for 23 hours (URL is valid for 24h)
      imageUrlCache.set(jobId, { url: json.url, expiresAt: Date.now() + 23 * 3600 * 1000 });
      return json.url;
    }
    return null;
  } catch {
    return null;
  }
}
