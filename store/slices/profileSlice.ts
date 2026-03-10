import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { userService, BusinessProfile, Avatar, UserAccount } from '../../services/UserService';
import { s3UserDataService, ProfileBackup } from '../../services/S3UserDataService';
import { backupSettings } from './appSlice';

export interface UserStats {
  totalAnalyses: number;
  totalCalories: number;
  averageCalories: number;
  lastAnalysisDate?: string;
}

interface ProfileState {
  // User account data
  userAccount: UserAccount | null;
  
  // Business profile data
  businessProfile: BusinessProfile | null;
  
  // Avatar
  avatar: Avatar | null;
  
  // Profile image
  profileImage: string | null;
  
  // User stats
  userStats: UserStats | null;
  
  // Loading states
  isLoading: boolean;
  isSaving: boolean;
  
  // Error state
  error: string | null;
}

const initialState: ProfileState = {
  userAccount: null,
  businessProfile: null,
  avatar: null,
  profileImage: null,
  userStats: null,
  isLoading: false,
  isSaving: false,
  error: null,
};

/** Build a ProfileBackup payload from current state and trigger background S3 backup */
const backupProfileToS3 = async (state: { profile: ProfileState }) => {
  try {
    const { userAccount, businessProfile, avatar, profileImage } = state.profile;
    if (!userAccount?.userId) return;

    const consentDate = await AsyncStorage.getItem('consent_date');

    const backup: ProfileBackup = {
      userAccount: {
        userId: userAccount.userId,
        email: userAccount.email,
        createdAt: userAccount.createdAt,
        hasCompletedProfile: userAccount.hasCompletedProfile,
      },
      businessProfile,
      avatar: avatar ? { id: avatar.id } : null,
      profileImage,
      consentDate: consentDate ?? undefined,
      updatedAt: new Date().toISOString(),
    };

    s3UserDataService.backupInBackground(userAccount.userId, 'profile', backup);
  } catch (error) {
    console.warn('[Profile] S3 backup failed silently:', error);
  }
};

// Async thunks
export const loadProfile = createAsyncThunk(
  'profile/loadProfile',
  async (_, { rejectWithValue }) => {
    try {
      const [account, profile, stats, avatar] = await Promise.all([
        userService.getUserAccount(),
        userService.getBusinessProfile(),
        userService.getUserStats(),
        userService.getAvatar(),
      ]);

      return {
        account,
        profile,
        stats,
        avatar,
        profileImage: profile?.profileImage || null,
      };
    } catch (error) {
      console.error('[Profile] Error loading profile:', error);
      return rejectWithValue(
        error instanceof Error ? error.message : 'Failed to load profile'
      );
    }
  }
);

export const saveBusinessProfile = createAsyncThunk(
  'profile/saveBusinessProfile',
  async (profileData: BusinessProfile, { getState, dispatch, rejectWithValue }) => {
    try {
      const success = await userService.saveBusinessProfile(profileData);
      if (!success) {
        throw new Error('Failed to save profile');
      }
      // Trigger S3 backup in background after local save
      const state = getState() as { profile: ProfileState };
      backupProfileToS3({
        profile: { ...state.profile, businessProfile: profileData },
      });
      // Also backup settings since profile completion status may have changed
      dispatch(backupSettings());
      return profileData;
    } catch (error) {
      console.error('[Profile] Error saving profile:', error);
      return rejectWithValue(
        error instanceof Error ? error.message : 'Failed to save profile'
      );
    }
  }
);

export const updateProfileFields = createAsyncThunk(
  'profile/updateProfileFields',
  async (fields: Partial<BusinessProfile>, { getState, rejectWithValue }) => {
    try {
      const state = getState() as { profile: ProfileState };
      const currentProfile = state.profile.businessProfile;

      if (!currentProfile) {
        throw new Error('No profile found to update');
      }

      const updatedProfile: BusinessProfile = {
        ...currentProfile,
        ...fields,
      };

      const success = await userService.saveBusinessProfile(updatedProfile);
      if (!success) {
        throw new Error('Failed to update profile');
      }

      // Trigger S3 backup in background
      backupProfileToS3({
        profile: { ...state.profile, businessProfile: updatedProfile },
      });

      return updatedProfile;
    } catch (error) {
      console.error('[Profile] Error updating profile:', error);
      return rejectWithValue(
        error instanceof Error ? error.message : 'Failed to update profile'
      );
    }
  }
);

export const setAvatar = createAsyncThunk(
  'profile/setAvatar',
  async (avatar: Avatar | undefined, { getState, rejectWithValue }) => {
    try {
      const success = await userService.setAvatar(avatar);
      if (!success) {
        throw new Error('Failed to set avatar');
      }

      // Trigger S3 backup in background
      const state = getState() as { profile: ProfileState };
      backupProfileToS3({
        profile: { ...state.profile, avatar: avatar || null },
      });

      return avatar || null;
    } catch (error) {
      console.error('[Profile] Error setting avatar:', error);
      return rejectWithValue(
        error instanceof Error ? error.message : 'Failed to set avatar'
      );
    }
  }
);

export const updateProfileImage = createAsyncThunk(
  'profile/updateProfileImage',
  async (imageUri: string | null, { getState, rejectWithValue }) => {
    try {
      const state = getState() as { profile: ProfileState };
      const currentProfile = state.profile.businessProfile;

      if (!currentProfile) {
        throw new Error('No profile found to update');
      }

      const updatedProfile: BusinessProfile = {
        ...currentProfile,
        profileImage: imageUri || undefined,
      };

      const success = await userService.saveBusinessProfile(updatedProfile);
      if (!success) {
        throw new Error('Failed to update profile image');
      }

      // Trigger S3 backup in background
      backupProfileToS3({
        profile: {
          ...state.profile,
          businessProfile: updatedProfile,
          profileImage: imageUri,
        },
      });

      return imageUri;
    } catch (error) {
      console.error('[Profile] Error updating profile image:', error);
      return rejectWithValue(
        error instanceof Error ? error.message : 'Failed to update profile image'
      );
    }
  }
);

export const loadUserStats = createAsyncThunk(
  'profile/loadUserStats',
  async (_, { rejectWithValue }) => {
    try {
      const stats = await userService.getUserStats();
      return stats;
    } catch (error) {
      console.error('[Profile] Error loading user stats:', error);
      return rejectWithValue(
        error instanceof Error ? error.message : 'Failed to load user stats'
      );
    }
  }
);

const profileSlice = createSlice({
  name: 'profile',
  initialState,
  reducers: {
    clearProfile: (state) => {
      state.userAccount = null;
      state.businessProfile = null;
      state.avatar = null;
      state.profileImage = null;
      state.userStats = null;
      state.error = null;
    },
    clearError: (state) => {
      state.error = null;
    },
    // Local updates (optimistic updates)
    setAvatarLocal: (state, action: PayloadAction<Avatar | null>) => {
      state.avatar = action.payload;
    },
    setProfileImageLocal: (state, action: PayloadAction<string | null>) => {
      state.profileImage = action.payload;
      if (state.businessProfile) {
        state.businessProfile.profileImage = action.payload || undefined;
      }
    },
  },
  extraReducers: (builder) => {
    // Load Profile
    builder
      .addCase(loadProfile.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(loadProfile.fulfilled, (state, action) => {
        state.isLoading = false;
        state.userAccount = action.payload.account;
        state.businessProfile = action.payload.profile;
        state.userStats = action.payload.stats;
        state.avatar = action.payload.avatar;
        state.profileImage = action.payload.profileImage;
        state.error = null;
      })
      .addCase(loadProfile.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });

    // Save Business Profile
    builder
      .addCase(saveBusinessProfile.pending, (state) => {
        state.isSaving = true;
        state.error = null;
      })
      .addCase(saveBusinessProfile.fulfilled, (state, action) => {
        state.isSaving = false;
        state.businessProfile = action.payload;
        state.profileImage = action.payload.profileImage || null;
        state.error = null;
      })
      .addCase(saveBusinessProfile.rejected, (state, action) => {
        state.isSaving = false;
        state.error = action.payload as string;
      });

    // Update Profile Fields
    builder
      .addCase(updateProfileFields.pending, (state) => {
        state.isSaving = true;
        state.error = null;
      })
      .addCase(updateProfileFields.fulfilled, (state, action) => {
        state.isSaving = false;
        state.businessProfile = action.payload;
        state.profileImage = action.payload.profileImage || null;
        state.error = null;
      })
      .addCase(updateProfileFields.rejected, (state, action) => {
        state.isSaving = false;
        state.error = action.payload as string;
      });

    // Set Avatar
    builder
      .addCase(setAvatar.pending, (state) => {
        state.isSaving = true;
        state.error = null;
      })
      .addCase(setAvatar.fulfilled, (state, action) => {
        state.isSaving = false;
        state.avatar = action.payload;
        state.error = null;
      })
      .addCase(setAvatar.rejected, (state, action) => {
        state.isSaving = false;
        state.error = action.payload as string;
      });

    // Update Profile Image
    builder
      .addCase(updateProfileImage.pending, (state) => {
        state.isSaving = true;
        state.error = null;
      })
      .addCase(updateProfileImage.fulfilled, (state, action) => {
        state.isSaving = false;
        state.profileImage = action.payload;
        if (state.businessProfile) {
          state.businessProfile.profileImage = action.payload || undefined;
        }
        state.error = null;
      })
      .addCase(updateProfileImage.rejected, (state, action) => {
        state.isSaving = false;
        state.error = action.payload as string;
      });

    // Load User Stats
    builder
      .addCase(loadUserStats.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(loadUserStats.fulfilled, (state, action) => {
        state.isLoading = false;
        state.userStats = action.payload;
        state.error = null;
      })
      .addCase(loadUserStats.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });
  },
});

export const {
  clearProfile,
  clearError,
  setAvatarLocal,
  setProfileImageLocal,
} = profileSlice.actions;

export default profileSlice.reducer;


