import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { cognitoOTPService as mockCognitoService, CognitoOTPService } from '../../services/CognitoAuthService';
import { realCognitoOTPService } from '../../services/RealCognitoAuthService';
import { userService } from '../../services/UserService';
import { s3UserDataService, AllUserData } from '../../services/S3UserDataService';
import { clearProfile } from './profileSlice';
import { clearHistoryLocal } from './historySlice';
import { backupSettings } from './appSlice';
import { loadProfileFromS3, loadHistoryFromS3 } from '../../services/S3UserDataService';

// 🔧 CONFIGURATION: Switch between mock and real AWS Cognito
const USE_REAL_AWS_COGNITO = true; // 🚀 Real AWS Cognito enabled

// Select the appropriate service based on configuration
const cognitoOTPService: CognitoOTPService = (USE_REAL_AWS_COGNITO ? realCognitoOTPService : mockCognitoService) as CognitoOTPService;

export interface User {
  email: string;
  isVerified: boolean;
}

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

const initialState: AuthState = {
  user: null,
  isAuthenticated: false,
  isLoading: true,
  error: null,
};

/**
 * Restore user data from S3 into local AsyncStorage
 * Called after login when the user may have data backed up from a previous install
 */
const restoreUserDataFromS3 = async (userId: string, email: string): Promise<boolean> => {
  try {
    console.log('[Auth] Checking S3 for backed-up user data...');
    const s3Data: AllUserData = await s3UserDataService.restoreAll(userId);

    let hasAnyData = s3Data.profile || s3Data.history || s3Data.settings;

    // Fallback: check email-based S3 path (used by UserService and S3HistoryAPI)
    if (!hasAnyData) {
      console.log('[Auth] No userId-path S3 data, checking email-path fallback...');
      const [emailProfile, emailHistory] = await Promise.all([
        loadProfileFromS3(email),
        loadHistoryFromS3(email),
      ]);
      if (emailProfile || (emailHistory && emailHistory.length > 0)) {
        console.log('[Auth] Found data at email-path S3, restoring...');
        if (emailProfile) {
          await AsyncStorage.setItem('user_profile', JSON.stringify(emailProfile));
          const hasCompleted = (emailProfile as any).hasCompletedProfile === true;
          if (hasCompleted) {
            await AsyncStorage.setItem('business_profile_completed', 'true');
            await AsyncStorage.setItem('user_consent', 'true');
          }
        }
        if (emailHistory && emailHistory.length > 0) {
          const mockData: { [key: string]: any[] } = {};
          mockData[email] = emailHistory;
          await AsyncStorage.setItem('mockHistoryData', JSON.stringify(mockData));
          console.log('[Auth] History restored from email-path S3:', emailHistory.length, 'entries');
        }
        return true;
      }
      console.log('[Auth] No S3 backup found for this user');
      return false;
    }

    console.log('[Auth] S3 backup found, restoring data...');

    // Restore profile data
    if (s3Data.profile) {
      const { userAccount, businessProfile, avatar } = s3Data.profile;
      // Write user account to AsyncStorage (UserService format)
      const restoredAccount = {
        userId: userAccount.userId,
        email: userAccount.email,
        createdAt: userAccount.createdAt,
        hasCompletedProfile: userAccount.hasCompletedProfile,
        profileData: businessProfile || undefined,
        avatar: avatar || undefined,
      };
      await AsyncStorage.setItem('user_account', JSON.stringify(restoredAccount));

      if (businessProfile) {
        await AsyncStorage.setItem('user_profile', JSON.stringify(businessProfile));
      }

      if (userAccount.hasCompletedProfile) {
        await AsyncStorage.setItem('business_profile_completed', 'true');
      }

      if (s3Data.profile.consentDate) {
        await AsyncStorage.setItem('consent_date', s3Data.profile.consentDate);
      }

      console.log('[Auth] Profile data restored from S3');
    }

    // Restore history data
    if (s3Data.history && s3Data.history.entries.length > 0) {
      // Write to the mockHistoryData key used by MockHistoryAPI
      const mockData: { [key: string]: any[] } = {};
      mockData[email] = s3Data.history.entries;
      await AsyncStorage.setItem('mockHistoryData', JSON.stringify(mockData));
      console.log('[Auth] History restored from S3:', s3Data.history.entries.length, 'entries');
    }

    // Restore settings
    if (s3Data.settings) {
      if (s3Data.settings.hasConsented !== null) {
        await AsyncStorage.setItem('user_consent', s3Data.settings.hasConsented ? 'true' : 'false');
      }
      if (s3Data.settings.hasCompletedProfile !== null && s3Data.settings.hasCompletedProfile) {
        await AsyncStorage.setItem('business_profile_completed', 'true');
      }
      console.log('[Auth] Settings restored from S3');
    }

    console.log('[Auth] S3 data restore complete');
    return true;
  } catch (error) {
    console.warn('[Auth] Failed to restore from S3 (non-fatal):', error);
    return false;
  }
};

// Async thunks for auth operations
export const loadUserFromStorage = createAsyncThunk(
  'auth/loadUserFromStorage',
  async () => {
    try {
      const storedUser = await AsyncStorage.getItem('user');
      if (storedUser) {
        return JSON.parse(storedUser) as User;
      }
      return null;
    } catch (error) {
      console.error('Error loading user from storage:', error);
      throw error;
    }
  }
);

export const sendOTP = createAsyncThunk(
  'auth/sendOTP',
  async ({ input, method }: { input: string; method: 'email' | 'phone' }) => {
    try {
      console.log(`[Auth] Sending ${method} OTP using ${USE_REAL_AWS_COGNITO ? 'AWS Cognito' : 'Mock'} service`);
      
      if (method === 'phone') {
        const result = await cognitoOTPService.sendPhoneOTP(input);
        if (!result) {
          throw new Error('Failed to send OTP');
        }
        return { success: true };
      } else {
        const result = await cognitoOTPService.sendEmailOTP(input);
        if (!result) {
          throw new Error('Failed to send OTP');
        }
        return { success: true };
      }
    } catch (error) {
      console.error('Error sending OTP:', error);
      throw error;
    }
  }
);

export const login = createAsyncThunk(
  'auth/login',
  async ({ input, otp, method }: { input: string; otp: string; method: 'email' | 'phone' }, { dispatch }) => {
    try {
      console.log(`[Auth] Verifying ${method} OTP using ${USE_REAL_AWS_COGNITO ? 'AWS Cognito' : 'Mock'} service`);
      
      let verificationResult: { success: boolean; userId?: string };
      
      if (method === 'phone') {
        verificationResult = await cognitoOTPService.verifyPhoneOTP(input, otp);
      } else {
        verificationResult = await cognitoOTPService.verifyEmailOTP(input, otp);
      }

      if (verificationResult.success) {
        // Check if user account already exists for this email
        // We need to check by email since after logout getUserAccount() returns null
        let userAccount = await userService.getUserAccount();

        console.log('[Auth] Checking for existing account:', {
          hasAccount: !!userAccount,
          accountEmail: userAccount?.email,
          loginEmail: input,
        });

        // If no account exists OR the email doesn't match, create a new account
        // This handles both new users and users logging in with a different email
        if (!userAccount || userAccount.email !== input) {
          console.log('[Auth] OTP verified, creating user account...');

          // Clear any old profile completion flags and saved profile data for new users
          await AsyncStorage.removeItem('business_profile_completed');
          await AsyncStorage.removeItem('user_consent');
          await AsyncStorage.removeItem('consent_date');
          await AsyncStorage.removeItem('business_profile_step1'); // Clear any saved Step 1 data
          await AsyncStorage.removeItem('edit_profile_step1'); // Clear any saved edit profile data
          await AsyncStorage.removeItem('user_profile'); // Clear previous user's profile data

          // Pass Cognito user ID if available (for DynamoDB mode)
          userAccount = await userService.createUserAccount(input, verificationResult.userId);

          // Explicitly set flags to false for new users
          await AsyncStorage.setItem('business_profile_completed', 'false');
          await AsyncStorage.setItem('user_consent', 'false');

          // Try to restore data from S3 (handles reinstall scenario)
          // If the user had data backed up from a previous install, restore it
          const restoredFromS3 = await restoreUserDataFromS3(userAccount.userId, input);
          if (restoredFromS3) {
            // Re-read the user account since S3 restore may have updated it
            const restoredAccount = await userService.getUserAccount();
            if (restoredAccount) {
              userAccount = restoredAccount;
            }
            console.log('[Auth] User data restored from S3 after reinstall');
          } else {
            console.log('[Auth] No S3 backup found — proceeding as new user');
          }

          console.log('[Auth] User account created successfully');
        } else {
          console.log('[Auth] OTP verified, using existing user account');
          console.log('[Auth] Existing user ID:', userAccount.userId);

          // For existing users, if they have ANY history, consider profile completed
          // This handles the case where users created history but profile wasn't marked complete
          const historyAPI = require('../../services/HistoryAPI').default;
          let hasHistory = false;
          try {
            const historyResponse = await historyAPI.getHistory(input);
            hasHistory = historyResponse.success && historyResponse.data && historyResponse.data.length > 0;
            console.log('[Auth] Existing user history check:', { hasHistory, count: historyResponse.data?.length || 0 });
          } catch (error) {
            console.log('[Auth] Could not check history:', error);
          }

          // Restore full profile from S3 (covers consent_date and other fields cleared on logout)
          await restoreUserDataFromS3(userAccount.userId, input);
          // Re-read restored account in case S3 had newer data
          const restoredAccount = await userService.getUserAccount();
          if (restoredAccount) userAccount = restoredAccount;

          // Restore profile completion status for existing users
          // Consider profile complete if: explicitly marked complete OR has history
          if (userAccount.hasCompletedProfile === true || hasHistory) {
            // Existing user with completed profile - set both consent and profile flags
            await AsyncStorage.setItem('business_profile_completed', 'true');
            await AsyncStorage.setItem('user_consent', 'true');
            console.log('[Auth] Restored profile completion and consent status for existing user (hasCompletedProfile:', userAccount.hasCompletedProfile, ', hasHistory:', hasHistory, ')');
          } else {
            // Existing user but profile not completed - check consent separately
            const storedConsent = await AsyncStorage.getItem('user_consent');
            await AsyncStorage.setItem('business_profile_completed', 'false');
            await AsyncStorage.setItem('user_consent', storedConsent || 'false');
            console.log('[Auth] Existing user with incomplete profile - restored consent status');
          }
        }
        
        const userData: User = {
          email: input,
          isVerified: true,
        };
        
        // Save to AsyncStorage
        await AsyncStorage.setItem('user', JSON.stringify(userData));

        console.log('[Auth] User logged in successfully');
        console.log('[Auth] User ID:', userAccount.userId);

        // Backup current settings to S3 in background (captures consent + profile completion state)
        dispatch(backupSettings());

        return userData;
      } else {
        throw new Error('Invalid verification code. Please check and try again.');
      }
    } catch (error) {
      console.error('Error during login:', error);
      throw error;
    }
  }
);

export const logout = createAsyncThunk(
  'auth/logout',
  async (_, { dispatch }) => {
    try {
      // Logout from Cognito
      await cognitoOTPService.logout();

      // Wipe ALL local user data — S3 is the source of truth.
      // On next login, restoreUserDataFromS3 restores the user's data from S3.
      await AsyncStorage.multiRemove([
        'user',
        'user_account',
        'user_profile',
        'business_profile_completed',
        'user_consent',
        'consent_date',
        'business_profile_step1',
        'edit_profile_step1',
        'mockHistoryData',
      ]);

      dispatch(clearProfile());
      dispatch(clearHistoryLocal());

      console.log('[Auth] Logout completed — local data cleared, S3 is source of truth');
    } catch (error) {
      console.error('Error during logout:', error);
      throw error;
    }
  }
);

export const sendDeleteAccountOTP = createAsyncThunk(
  'auth/sendDeleteAccountOTP',
  async (email: string) => {
    try {
      console.log(`[Auth] Sending delete account OTP using ${USE_REAL_AWS_COGNITO ? 'AWS Cognito' : 'Mock'} service`);
      const result = await cognitoOTPService.sendDeleteAccountOTP(email);
      if (!result) {
        throw new Error('Failed to send verification code');
      }
      return { success: true };
    } catch (error) {
      console.error('Error sending delete account OTP:', error);
      throw error;
    }
  }
);

export const verifyDeleteAccountOTPAndDelete = createAsyncThunk(
  'auth/verifyDeleteAccountOTPAndDelete',
  async ({ email, otp }: { email: string; otp: string }, { dispatch }) => {
    try {
      console.log(`[Auth] Verifying delete account OTP using ${USE_REAL_AWS_COGNITO ? 'AWS Cognito' : 'Mock'} service`);
      const verified = await cognitoOTPService.verifyDeleteAccountOTP(email, otp);
      if (!verified) {
        throw new Error('Invalid or expired verification code');
      }
      await dispatch(deleteAccount()).unwrap();
      return { success: true };
    } catch (error) {
      console.error('Error verifying delete account OTP:', error);
      throw error;
    }
  }
);

export const deleteAccount = createAsyncThunk(
  'auth/deleteAccount',
  async (_, { dispatch, getState }) => {
    try {
      // Collect all job_ids and userId before wiping local state
      const state = getState() as { history: { history: Array<{ job_id?: string }> }; profile: { userAccount: { userId: string } | null } };
      const jobIds = state.history.history
        .map((e) => e.job_id)
        .filter((id): id is string => !!id);
      const userId = state.profile?.userAccount?.userId;

      // Wipe all S3 + DynamoDB data for this user
      if (userId) {
        await s3UserDataService.deleteAllUserData(userId, jobIds);
      }

      await userService.deleteUserAccount();
      await cognitoOTPService.logout();

      await AsyncStorage.multiRemove([
        'user', 'user_account', 'user_profile',
        'business_profile_completed', 'user_consent', 'consent_date',
        'business_profile_step1', 'edit_profile_step1', 'mockHistoryData',
      ]);

      dispatch(clearProfile());
      dispatch(clearHistoryLocal());

      console.log('[Auth] Account deleted — local data cleared');
      return { success: true };
    } catch (error) {
      console.error('Error during account deletion:', error);
      throw error;
    }
  }
);

export const withdrawParticipation = createAsyncThunk(
  'auth/withdrawParticipation',
  async (_, { dispatch }) => {
    try {
      await userService.withdrawParticipation();
      await cognitoOTPService.logout();

      await AsyncStorage.multiRemove([
        'user', 'user_account', 'user_profile',
        'business_profile_completed', 'user_consent', 'consent_date',
        'business_profile_step1', 'edit_profile_step1', 'mockHistoryData',
      ]);

      dispatch(clearProfile());
      dispatch(clearHistoryLocal());

      console.log('[Auth] Participation withdrawn — local data cleared');
      return { success: true };
    } catch (error) {
      console.error('Error during withdrawal:', error);
      throw error;
    }
  }
);

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    // Load user from storage
    builder
      .addCase(loadUserFromStorage.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(loadUserFromStorage.fulfilled, (state, action) => {
        state.isLoading = false;
        state.user = action.payload;
        state.isAuthenticated = !!action.payload;
      })
      .addCase(loadUserFromStorage.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.error.message || 'Failed to load user';
      });

    // Send OTP
    builder
      .addCase(sendOTP.pending, (state) => {
        state.error = null;
      })
      .addCase(sendOTP.fulfilled, (state) => {
        state.error = null;
      })
      .addCase(sendOTP.rejected, (state, action) => {
        state.error = action.error.message || 'Failed to send OTP';
      });

    // Login
    builder
      .addCase(login.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(login.fulfilled, (state, action) => {
        state.isLoading = false;
        state.user = action.payload;
        state.isAuthenticated = true;
        state.error = null;
      })
      .addCase(login.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.error.message || 'Login failed';
      });

    // Logout
    builder
      .addCase(logout.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(logout.fulfilled, (state) => {
        state.isLoading = false;
        state.user = null;
        state.isAuthenticated = false;
        state.error = null;
      })
      .addCase(logout.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.error.message || 'Logout failed';
      });

    // Send Delete Account OTP
    builder
      .addCase(sendDeleteAccountOTP.rejected, (state, action) => {
        state.error = action.error.message || 'Failed to send verification code';
      });

    // Verify Delete Account OTP and Delete
    builder
      .addCase(verifyDeleteAccountOTPAndDelete.pending, (state) => {
        // Don't set isLoading — App.tsx shows AppLoader when isLoading && isAuthenticated,
        // which would unmount the nav stack and reset to Results/Tutorial on rejection.
        // Screen uses local isVerifying state instead.
        state.error = null;
      })
      .addCase(verifyDeleteAccountOTPAndDelete.fulfilled, (state) => {
        state.user = null;
        state.isAuthenticated = false;
        state.error = null;
      })
      .addCase(verifyDeleteAccountOTPAndDelete.rejected, (state, action) => {
        state.error = action.error.message || 'Invalid or expired verification code';
      });

    // Delete Account
    builder
      .addCase(deleteAccount.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(deleteAccount.fulfilled, (state) => {
        state.isLoading = false;
        state.user = null;
        state.isAuthenticated = false;
        state.error = null;
      })
      .addCase(deleteAccount.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.error.message || 'Account deletion failed';
      });

    // Withdraw Participation
    builder
      .addCase(withdrawParticipation.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(withdrawParticipation.fulfilled, (state) => {
        state.isLoading = false;
        state.user = null;
        state.isAuthenticated = false;
        state.error = null;
      })
      .addCase(withdrawParticipation.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.error.message || 'Withdrawal failed';
      });
  },
});

export const { clearError } = authSlice.actions;
export default authSlice.reducer;

