import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import historyAPI from '../../services/HistoryAPI';
import { s3UserDataService, HistoryBackup } from '../../services/S3UserDataService';

export interface DishContent {
  id: string;
  name: string;
  weight: string;
  calories: string;
}

export interface FeedbackData {
  ratings: {
    foodDishIdentification: number;
    dishContentsIdentification: number;
    massEstimation: number;
    calorieEstimation: number;
    overall: number;
  };
  comment: string;
  timestamp: string;
}

export interface SegmentedImage {
  frame: string;
  url: string;
  key: string;
  type?: 'overlay' | 'mask';
  object_id?: string;
}

export interface SegmentedImages {
  overlay_urls?: SegmentedImage[];
  mask_urls?: SegmentedImage[];
  video_overlay_url?: string | null;
}

export interface AnalysisEntry {
  id: string;
  type: 'image' | 'video';
  timestamp: string;
  imageUri?: string;
  videoUri?: string;
  textDescription?: string;
  analysisResult: string;
  nutritionalInfo: {
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
  };
  mealName?: string;
  dishContents?: DishContent[];
  feedback?: FeedbackData;
  segmented_images?: SegmentedImages;  // Segmented image URLs (may expire; use job_id to refetch)
  job_id?: string;  // Nutrition API job id – used to refetch fresh segmented_images URLs when they expire
  analysisStatus?: 'analyzing' | 'completed' | 'failed';  // Analysis status
  analysisProgress?: number;  // Progress from 0 to 100
}

interface HistoryState {
  history: AnalysisEntry[];
  isLoading: boolean;
  error: string | null;
}

const initialState: HistoryState = {
  history: [],
  isLoading: false,
  error: null,
};

/** Fire-and-forget backup of history entries to S3 */
const backupHistoryToS3 = (history: AnalysisEntry[], getState: () => unknown) => {
  try {
    const state = getState() as { profile: { userAccount: { userId: string } | null } };
    const userId = state.profile?.userAccount?.userId;
    if (!userId) return;

    const backup: HistoryBackup = {
      entries: history,
      updatedAt: new Date().toISOString(),
    };
    s3UserDataService.backupInBackground(userId, 'history', backup);
  } catch (error) {
    console.warn('[History] S3 backup failed silently:', error);
  }
};

// Async thunks for history operations
export const loadHistory = createAsyncThunk(
  'history/loadHistory',
  async (userEmail: string) => {
    const response = await historyAPI.getHistory(userEmail);
    if (response.success && response.data) {
      return response.data;
    } else {
      throw new Error(response.error || 'Failed to load history');
    }
  }
);

export const addAnalysis = createAsyncThunk(
  'history/addAnalysis',
  async (
    { userEmail, analysis }: { userEmail: string; analysis: Omit<AnalysisEntry, 'id' | 'timestamp'> },
    { getState }
  ) => {
    const response = await historyAPI.saveAnalysis(userEmail, analysis);
    if (response.success && response.data) {
      // Backup updated history to S3
      const state = getState() as { history: HistoryState };
      backupHistoryToS3([response.data, ...state.history.history], getState);
      return response.data;
    } else {
      throw new Error(response.error || 'Failed to save analysis');
    }
  }
);

export const deleteAnalysis = createAsyncThunk(
  'history/deleteAnalysis',
  async ({ userEmail, analysisId }: { userEmail: string; analysisId: string }, { getState }) => {
    const response = await historyAPI.deleteAnalysis(userEmail, analysisId);
    if (response.success) {
      // Backup updated history (without the deleted entry) to S3
      const state = getState() as { history: HistoryState };
      const remaining = state.history.history.filter((item) => item.id !== analysisId);
      backupHistoryToS3(remaining, getState);
      return analysisId;
    } else {
      throw new Error(response.error || 'Failed to delete analysis');
    }
  }
);

export const clearHistory = createAsyncThunk(
  'history/clearHistory',
  async (userEmail: string, { getState }) => {
    const response = await historyAPI.clearHistory(userEmail);
    if (response.success) {
      // Backup empty history to S3
      backupHistoryToS3([], getState);
      return true;
    } else {
      throw new Error(response.error || 'Failed to clear history');
    }
  }
);

export const updateAnalysis = createAsyncThunk(
  'history/updateAnalysis',
  async (
    { userEmail, analysisId, updates }: { userEmail: string; analysisId: string; updates: Partial<AnalysisEntry> },
    { getState, rejectWithValue }
  ) => {
    const state = getState() as { history: HistoryState };
    const existing = state.history.history.find(e => e.id === analysisId);

    // Entry was deleted by the user — skip silently, no error
    if (!existing) {
      return rejectWithValue('ENTRY_DELETED');
    }

    // Include type + timestamp so MockHistoryAPI can recreate the entry if AsyncStorage lost it
    const fullUpdates: Partial<AnalysisEntry> = { type: existing.type, timestamp: existing.timestamp, ...updates };

    const response = await historyAPI.updateAnalysis(userEmail, analysisId, fullUpdates);
    if (response.success && response.data) {
      // Backup updated history to S3
      const state = getState() as { history: HistoryState };
      const updatedHistory = state.history.history.map((item) =>
        item.id === analysisId ? response.data! : item
      );
      backupHistoryToS3(updatedHistory, getState);
      return response.data;
    } else {
      throw new Error(response.error || 'Failed to update analysis');
    }
  }
);

const historySlice = createSlice({
  name: 'history',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null;
    },
    clearHistoryLocal: (state) => {
      state.history = [];
    },
    updateAnalysisProgress: (state, action: PayloadAction<{ id: string; progress: number; status?: 'analyzing' | 'completed' | 'failed' }>) => {
      const index = state.history.findIndex(item => item.id === action.payload.id);
      if (index !== -1) {
        state.history[index].analysisProgress = action.payload.progress;
        if (action.payload.status) {
          state.history[index].analysisStatus = action.payload.status;
        }
      }
    },
  },
  extraReducers: (builder) => {
    // Load history
    builder
      .addCase(loadHistory.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(loadHistory.fulfilled, (state, action) => {
        state.isLoading = false;
        // Deep-clone so we never mutate read-only API/cached objects (avoids "Cannot assign to read-only property")
        state.history = JSON.parse(JSON.stringify(action.payload));
        state.error = null;
      })
      .addCase(loadHistory.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.error.message || 'Failed to load history';
      });

    // Add analysis
    builder
      .addCase(addAnalysis.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(addAnalysis.fulfilled, (state, action) => {
        state.isLoading = false;
        state.history.unshift(JSON.parse(JSON.stringify(action.payload)));
        state.error = null;
      })
      .addCase(addAnalysis.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.error.message || 'Failed to save analysis';
      });

    // Delete analysis - don't set isLoading to prevent loader from showing
    builder
      .addCase(deleteAnalysis.pending, (state) => {
        // Don't set isLoading for delete operations to prevent loader from showing
        state.error = null;
      })
      .addCase(deleteAnalysis.fulfilled, (state, action) => {
        state.history = state.history.filter(item => item.id !== action.payload);
        state.error = null;
      })
      .addCase(deleteAnalysis.rejected, (state, action) => {
        state.error = action.error.message || 'Failed to delete analysis';
      });

    // Clear history
    builder
      .addCase(clearHistory.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(clearHistory.fulfilled, (state) => {
        state.isLoading = false;
        state.history = [];
        state.error = null;
      })
      .addCase(clearHistory.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.error.message || 'Failed to clear history';
      });

    // Update analysis
    builder
      .addCase(updateAnalysis.pending, (state) => {
        state.error = null;
      })
      .addCase(updateAnalysis.fulfilled, (state, action) => {
        const index = state.history.findIndex(item => item.id === action.payload.id);
        if (index !== -1) {
          state.history[index] = JSON.parse(JSON.stringify(action.payload));
        }
        state.error = null;
      })
      .addCase(updateAnalysis.rejected, (_state, action) => {
        if (action.payload === 'ENTRY_DELETED') return; // user deleted the card — expected, no log needed
        console.error('[History] updateAnalysis rejected:', action.error.message);
      });
  },
});

export const { clearError, clearHistoryLocal, updateAnalysisProgress } = historySlice.actions;
export default historySlice.reducer;

