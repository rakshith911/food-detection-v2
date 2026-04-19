import React, { useState, useEffect, useRef, useCallback } from 'react';
import { StatusBar } from 'expo-status-bar';
import * as Notifications from 'expo-notifications';
import { Alert, Animated, AppState, View, StyleSheet, LogBox } from 'react-native';

LogBox.ignoreLogs([
  '[Nutrition API] Failed to check status',
  'Network request failed',
]);
import { NavigationContainer, CommonActions } from '@react-navigation/native';
import { createStackNavigator, TransitionPresets } from '@react-navigation/stack';
// ActivityIndicator and Text moved to AppLoader component
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { Provider as PaperProvider, DefaultTheme } from 'react-native-paper';
import { Provider as ReduxProvider } from 'react-redux';
import { PersistGate } from 'redux-persist/integration/react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Amplify } from 'aws-amplify';
import { awsConfig } from './aws-config';
import { store, persistor } from './store';
import { useAppDispatch, useAppSelector } from './store/hooks';
import { loadUserFromStorage } from './store/slices/authSlice';
import { setShowSplash, setShowWelcome } from './store/slices/appSlice';
import { loadHistory, updateAnalysis, updateAnalysisProgress } from './store/slices/historySlice';
import { loadProfile } from './store/slices/profileSlice';
import { nutritionAnalysisAPI } from './services/NutritionAnalysisAPI';
import { userService } from './services/UserService';
import { initSentry, setSentryUser, addBreadcrumb } from './utils/sentry';
import {
  buildDishTablesFromItems,
  getBaseDishContents,
  getMealNameFromTables,
  getOverallCaloriesFromTables,
} from './utils/mealTables';
import { ErrorBoundary } from './components/ErrorBoundary';

// Allow the OS to present notifications (sound + banner).
// The notification function itself skips firing when the app is active,
// so this handler only runs for background-delivered notifications.
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldPlaySound: true,
    shouldSetBadge: false,
    shouldShowBanner: true,
    shouldShowList: true,
  }),
});

// Initialize Sentry FIRST, before anything else
// Sentry DSN - Get from environment variable or use the configured DSN
const SENTRY_DSN = process.env.SENTRY_DSN || 'https://df7d79661602353820060afdbd39fa34@o4510355929366528.ingest.us.sentry.io/4510355930152960';
if (SENTRY_DSN) {
  initSentry(SENTRY_DSN);
  console.log('[App] ✅ Sentry initialized with DSN');
} else {
  console.warn('[App] ⚠️ Sentry DSN not provided, error tracking disabled');
}

// Configure Amplify at app startup - MUST be done before any Amplify services are used
try {
  const amplifyConfig = {
    Auth: {
      Cognito: {
        userPoolId: awsConfig.Auth.userPoolId,
        userPoolClientId: awsConfig.Auth.userPoolWebClientId,
      }
    }
  };
  
  Amplify.configure(amplifyConfig);
  console.log('[App] ✅ AWS Amplify configured successfully');
  console.log('[App] Config:', JSON.stringify(amplifyConfig, null, 2));
} catch (error: any) {
  console.error('[App] ❌ Failed to configure Amplify:', error);
  console.error('[App] Error details:', error?.message, error?.stack);
}
import CameraScreen from './screens/CameraScreen';
import ImageTextTab from './components/ImageTextTab';
import VideoTextTab from './components/VideoTextTab';
import LoginScreen from './screens/LoginScreen';
import EmailLoginScreen from './screens/EmailLoginScreen';
import OTPScreen from './screens/OTPScreen';
import ConsentScreen from './screens/ConsentScreen';
import BusinessProfileStep1Screen from './screens/BusinessProfileStep1Screen';
import BusinessProfileStep2Screen from './screens/BusinessProfileStep2Screen';
import HistoryScreen from './screens/HistoryScreen';
import ResultsScreen from './screens/ResultsScreen';
import MealDetailScreen from './screens/MealDetailScreen';
import FeedbackScreen from './screens/FeedbackScreen';
import SplashScreen from './screens/SplashScreen';
import WelcomeScreen from './screens/WelcomeScreen';
import TutorialScreen from './screens/TutorialScreen';
import ProfileScreen from './screens/ProfileScreen';
import EditProfileScreen from './screens/EditProfileScreen';
import EditProfileStep1Screen from './screens/EditProfileStep1Screen';
import EditProfileStep2Screen from './screens/EditProfileStep2Screen';
import ViewConsentScreen from './screens/ViewConsentScreen';
import AddAvatarScreen from './screens/AddAvatarScreen';
import SendFeedbackScreen from './screens/SendFeedbackScreen';
import DeleteAccountScreen from './screens/DeleteAccountScreen';
import WithdrawParticipationScreen from './screens/WithdrawParticipationScreen';
import TermsAndConditionsScreen from './screens/TermsAndConditionsScreen';
import PrivacyPolicyScreen from './screens/PrivacyPolicyScreen';
import AppLoader from './components/AppLoader';

// Remove alert titles globally so popups only show message text
const originalAlert = Alert.alert;
Alert.alert = ((title: any, messageOrButtons?: any, buttonsOrOptions?: any, options?: any) => {
  if (typeof messageOrButtons === 'string' || messageOrButtons === undefined) {
    const message = typeof messageOrButtons === 'string'
      ? messageOrButtons
      : (typeof title === 'string' ? title : '');
    const buttons = typeof messageOrButtons === 'string' ? buttonsOrOptions : messageOrButtons;
    const alertOptions = typeof messageOrButtons === 'string' ? options : buttonsOrOptions;
    return originalAlert('', message, buttons, alertOptions);
  }

  // When second argument is buttons object/array, treat the first argument as message
  return originalAlert('', typeof title === 'string' ? title : '', messageOrButtons, buttonsOrOptions);
}) as typeof Alert.alert;

const Stack = createStackNavigator();

// Smooth transition configuration for all screens
// Uses consistent 450ms duration for smoother, more polished transitions
const smoothTransitionConfig = {
  transitionSpec: {
    open: {
      animation: 'timing',
      config: {
        duration: 450,
        useNativeDriver: true,
      },
    },
    close: {
      animation: 'timing',
      config: {
        duration: 450,
        useNativeDriver: true,
      },
    },
  },
  cardStyleInterpolator: ({ current, next, layouts }: any) => {
    return {
      cardStyle: {
        backgroundColor: '#000000',
        transform: [
          {
            translateX: current.progress.interpolate({
              inputRange: [0, 1],
              outputRange: [layouts.screen.width, 0],
            }),
          },
        ],
        opacity: current.progress, // Fade out outgoing (white) content; NavigationContainer bg is #000000 so no white shows
      },
      overlayStyle: {
        opacity: current.progress.interpolate({
          inputRange: [0, 1],
          outputRange: [0, 0.3],
        }),
      },
    };
  },
  gestureEnabled: true,
  gestureDirection: 'horizontal', // Swipe left to go back (standard)
};

// Shared screen options with smooth transitions
const defaultScreenOptions = {
  headerShown: false,
  ...smoothTransitionConfig,
};

// RootNavigator - conditionally renders different stacks based on app state
// This is inside a single NavigationContainer, so it never remounts
function RootNavigator() {
  const dispatch = useAppDispatch();
  const isAuthenticated = useAppSelector((state) => state.auth.isAuthenticated);
  const isLoading = useAppSelector((state) => state.auth.isLoading);
  const user = useAppSelector((state) => state.auth.user);
  const businessProfile = useAppSelector((state) => state.profile.businessProfile);
  const userAccount = useAppSelector((state) => state.profile.userAccount);
  const isProfileLoading = useAppSelector((state) => state.profile.isLoading);
  const isHistoryLoading = useAppSelector((state) => state.history.isLoading);
  const showWelcome = useAppSelector((state) => state.app.showWelcome);
  const [hasConsented, setHasConsented] = useState<boolean | null>(null);
  const [hasCompletedProfile, setHasCompletedProfile] = useState<boolean | null>(null);
  const [isCheckingConsent, setIsCheckingConsent] = useState(true);
  const previousAuthState = useRef<boolean>(isAuthenticated);
  const hasShownMainApp = useRef(false); // Once true, never show AppLoader for profile loading again

  // Load history and profile when user logs in
  useEffect(() => {
    if (isAuthenticated && user?.email) {
      console.log('[App] 📥 Loading profile and history for:', user.email);
      dispatch(loadProfile());
      dispatch(loadHistory(user.email));
      // Set Sentry user context
      setSentryUser(user);
      addBreadcrumb('User logged in', 'auth', 'info', { email: user.email });
    } else if (!isAuthenticated) {
      // Clear Sentry user context on logout
      setSentryUser(null);
      addBreadcrumb('User logged out', 'auth', 'info');
    }
  }, [isAuthenticated, user?.email, dispatch]);

  // Reset states immediately when user logs out
  useEffect(() => {
    if (!isAuthenticated) {
      setHasConsented(null);
      setHasCompletedProfile(null);
      setIsCheckingConsent(false);

      // Note: Navigation reset is handled by RootNavigator's conditional rendering
      // No need to manually reset navigation here
    }
    previousAuthState.current = isAuthenticated;
  }, [isAuthenticated]);

  // Check if user has consented and completed profile
  useEffect(() => {
    const checkStatus = async () => {
      if (isAuthenticated && !isLoading) {
        setIsCheckingConsent(true);
        try {
          // Check AsyncStorage first for immediate response (set during login)
          const storedConsent = await AsyncStorage.getItem('user_consent');
          const storedProfileCompleted = await AsyncStorage.getItem('business_profile_completed');
          
          // If both are set and true, use them immediately (faster for existing users)
          if (storedConsent === 'true' && storedProfileCompleted === 'true') {
            console.log('[App] Using AsyncStorage values (existing user)');
            setHasConsented(true);
            setHasCompletedProfile(true);
            setIsCheckingConsent(false);
            return;
          }
          
          // Otherwise, check user account (source of truth)
          const userAccount = await userService.getUserAccount();
          
          let consent: string | null = null;
          let profileCompleted: string | null = null;
          
          if (userAccount) {
            // User account exists - use it as source of truth
            if (userAccount.hasCompletedProfile === true) {
              // Existing user with completed profile
              profileCompleted = 'true';
              consent = 'true'; // Assume consent if profile completed
            } else {
              // New user or incomplete profile - must be false
              profileCompleted = 'false';
              // Check AsyncStorage for consent (user might have consented but not completed profile)
              consent = storedConsent || 'false';
            }
            
            // Update AsyncStorage to match user account
            await AsyncStorage.setItem('business_profile_completed', profileCompleted);
            await AsyncStorage.setItem('user_consent', consent);
          } else {
            // No user account - definitely a new user, set both to false
            profileCompleted = 'false';
            consent = 'false';
            await AsyncStorage.setItem('business_profile_completed', 'false');
            await AsyncStorage.setItem('user_consent', 'false');
          }
          
          console.log('[App] Profile status check:', {
            hasUserAccount: !!userAccount,
            hasCompletedProfile: userAccount?.hasCompletedProfile,
            profileCompleted,
            consent,
          });
          
          setHasConsented(consent === 'true');
          setHasCompletedProfile(profileCompleted === 'true');
        } catch (error) {
          console.error('[App] Error checking status:', error);
          setHasConsented(false);
          setHasCompletedProfile(false);
        } finally {
          setIsCheckingConsent(false);
        }
      } else if (!isLoading) {
        setIsCheckingConsent(false);
      }
    };

    checkStatus();
  }, [isAuthenticated, isLoading]);

  // No need to navigate - Results is the initial screen and will handle Tutorial vs Results logic internally

  console.log('[App] Auth state:', { 
    isAuthenticated, 
    isLoading, 
    userEmail: user?.email, 
    hasConsented, 
    hasCompletedProfile 
  });

  // Only show full-screen loader when authenticated (e.g. logout) or checking consent.
  // When unauthenticated, keep the stack mounted so invalid OTP doesn't unmount and reset to login.
  if ((isLoading && isAuthenticated) || isCheckingConsent) {
    return <AppLoader />;
  }

  // Show welcome screen if not authenticated and welcome hasn't been dismissed
  if (!isAuthenticated && showWelcome) {
    console.log('[RootNavigator] Showing welcome flow');
    return (
      <Stack.Navigator screenOptions={defaultScreenOptions} initialRouteName="Welcome">
        <Stack.Screen name="Welcome" component={WelcomeScreen} />
        <Stack.Screen name="EmailLogin" component={EmailLoginScreen} />
        <Stack.Screen name="OTPScreen" component={OTPScreen} />
        <Stack.Screen name="TermsAndConditions" component={TermsAndConditionsScreen} />
        <Stack.Screen name="PrivacyPolicy" component={PrivacyPolicyScreen} />
      </Stack.Navigator>
    );
  }

  if (!isAuthenticated) {
    console.log('[RootNavigator] Showing login flow');
    return (
      <Stack.Navigator screenOptions={defaultScreenOptions}>
          <Stack.Screen name="EmailLogin" component={EmailLoginScreen} />
          <Stack.Screen name="OTPScreen" component={OTPScreen} />
          <Stack.Screen name="TermsAndConditions" component={TermsAndConditionsScreen} />
          <Stack.Screen name="PrivacyPolicy" component={PrivacyPolicyScreen} />
          <Stack.Screen name="Consent">
            {({ navigation, route }) => (
              <ConsentScreen 
                navigation={navigation} 
                onConsent={() => setHasConsented(true)} 
              />
            )}
          </Stack.Screen>
          <Stack.Screen name="BusinessProfileStep1" component={BusinessProfileStep1Screen} />
          <Stack.Screen name="BusinessProfileStep2" component={BusinessProfileStep2Screen} />
          <Stack.Screen name="Tutorial" component={TutorialScreen} />
          <Stack.Screen name="Results" component={ResultsScreen} />
          <Stack.Screen name="Profile" component={ProfileScreen} />
          <Stack.Screen name="EditProfile" component={EditProfileScreen} />
          <Stack.Screen name="AddAvatar" component={AddAvatarScreen} />
          <Stack.Screen name="SendFeedback" component={SendFeedbackScreen} />
          <Stack.Screen name="DeleteAccount" component={DeleteAccountScreen} />
          <Stack.Screen name="WithdrawParticipation" component={WithdrawParticipationScreen} />
          <Stack.Screen name="ViewConsent" component={ViewConsentScreen} />
          <Stack.Screen name="History" component={HistoryScreen} />
          <Stack.Screen name="Camera" component={CameraScreen} />
          <Stack.Screen name="MealDetail" component={MealDetailScreen} />
          <Stack.Screen name="Feedback" component={FeedbackScreen} />
          <Stack.Screen name="ImageText" component={ImageTextTab} />
          <Stack.Screen name="VideoText" component={VideoTextTab} />
        </Stack.Navigator>
    );
  }

  // If authenticated but hasn't consented or completed profile, show onboarding
  if (!hasConsented || !hasCompletedProfile) {
    console.log('[RootNavigator] Showing onboarding flow (consent/profile)');
    return (
      <Stack.Navigator screenOptions={defaultScreenOptions} initialRouteName={!hasConsented ? "Consent" : "BusinessProfileStep1"}>
          <Stack.Screen name="EmailLogin" component={EmailLoginScreen} />
          <Stack.Screen name="OTPScreen" component={OTPScreen} />
          <Stack.Screen name="Consent">
            {({ navigation, route }) => (
              <ConsentScreen 
                navigation={navigation} 
                onConsent={async () => {
                  setHasConsented(true);
                  // Wait a bit for state to update, then navigate
                  setTimeout(() => {
                    navigation.navigate('BusinessProfileStep1');
                  }, 100);
                }} 
              />
            )}
          </Stack.Screen>
          <Stack.Screen name="TermsAndConditions" component={TermsAndConditionsScreen} />
          <Stack.Screen name="PrivacyPolicy" component={PrivacyPolicyScreen} />
          <Stack.Screen name="BusinessProfileStep1" component={BusinessProfileStep1Screen} />
          <Stack.Screen name="BusinessProfileStep2" component={BusinessProfileStep2Screen} />
          <Stack.Screen name="AddAvatar" component={AddAvatarScreen} />
          <Stack.Screen name="Tutorial" component={TutorialScreen} />
          <Stack.Screen name="Results" component={ResultsScreen} />
          <Stack.Screen name="Profile" component={ProfileScreen} />
          <Stack.Screen name="EditProfile" component={EditProfileScreen} />
          <Stack.Screen name="EditProfileStep1">
            {({ navigation, route }) => <EditProfileStep1Screen navigation={navigation} route={route} />}
          </Stack.Screen>
          <Stack.Screen name="EditProfileStep2">
            {({ navigation, route }) => <EditProfileStep2Screen navigation={navigation} route={route} />}
          </Stack.Screen>
          <Stack.Screen name="SendFeedback" component={SendFeedbackScreen} />
          <Stack.Screen name="DeleteAccount" component={DeleteAccountScreen} />
          <Stack.Screen name="WithdrawParticipation" component={WithdrawParticipationScreen} />
          <Stack.Screen name="ViewConsent" component={ViewConsentScreen} />
          <Stack.Screen name="History" component={HistoryScreen} />
          <Stack.Screen name="Camera" component={CameraScreen} />
          <Stack.Screen name="MealDetail" component={MealDetailScreen} />
          <Stack.Screen name="Feedback" component={FeedbackScreen} />
          <Stack.Screen name="ImageText" component={ImageTextTab} />
          <Stack.Screen name="VideoText" component={VideoTextTab} />
        </Stack.Navigator>
    );
  }

  // CRITICAL: Validate profile before showing main app
  // Only enforce validation if profile is still loading
  // Once loading completes, allow app to show (ResultsScreen will handle missing profile)
  const profileBelongsToCurrentUser = userAccount?.email === user?.email;
  const hasValidProfile = businessProfile && businessProfile.businessName && profileBelongsToCurrentUser;

  if ((isProfileLoading || isHistoryLoading) && !hasShownMainApp.current) {
    console.log('[App] ⏳ Profile is loading...', {
      isProfileLoading,
      userEmail: user?.email,
    });
    return <AppLoader />;
  }

  // Mark that the main app has been shown — suppress AppLoader for future profile loads
  hasShownMainApp.current = true;

  if (!hasValidProfile) {
    console.log('[App] ⚠️ Profile not ready but allowing app to show (ResultsScreen will handle)', {
      hasBusinessProfile: !!businessProfile,
      hasBusinessName: !!businessProfile?.businessName,
      profileMatches: profileBelongsToCurrentUser,
      userEmail: user?.email,
      accountEmail: userAccount?.email,
    });
  } else {
    console.log('[App] ✅ Profile ready - Showing main app', {
      businessName: businessProfile.businessName,
      userEmail: user?.email,
      accountEmail: userAccount?.email,
    });
  }
  return (
    <>
      <StatusBar style="dark" />
      <Stack.Navigator screenOptions={defaultScreenOptions} initialRouteName="Results">
        <Stack.Screen name="Results" component={ResultsScreen} />
        <Stack.Screen name="Profile" component={ProfileScreen} />
        <Stack.Screen name="EditProfile" component={EditProfileScreen} />
        <Stack.Screen name="EditProfileStep1" component={EditProfileStep1Screen} />
        <Stack.Screen name="EditProfileStep2" component={EditProfileStep2Screen} />
        <Stack.Screen name="AddAvatar" component={AddAvatarScreen} />
        <Stack.Screen name="SendFeedback" component={SendFeedbackScreen} />
        <Stack.Screen name="DeleteAccount" component={DeleteAccountScreen} />
        <Stack.Screen name="WithdrawParticipation" component={WithdrawParticipationScreen} />
        <Stack.Screen name="ViewConsent" component={ViewConsentScreen} />
        <Stack.Screen name="History" component={HistoryScreen} />
        <Stack.Screen name="Camera" component={CameraScreen} />
        <Stack.Screen name="MealDetail" component={MealDetailScreen} />
        <Stack.Screen name="Feedback" component={FeedbackScreen} />
        <Stack.Screen name="TermsAndConditions" component={TermsAndConditionsScreen} />
        <Stack.Screen name="PrivacyPolicy" component={PrivacyPolicyScreen} />
        <Stack.Screen name="ImageText" component={ImageTextTab} />
        <Stack.Screen name="VideoText" component={VideoTextTab} />
      </Stack.Navigator>
    </>
  );
}

// Inner app component that uses Redux
function AppContent() {
  const dispatch = useAppDispatch();
  const showSplash = useAppSelector((state) => state.app.showSplash);
  const showWelcome = useAppSelector((state) => state.app.showWelcome);
  const isAuthenticated = useAppSelector((state) => state.auth.isAuthenticated);
  const user = useAppSelector((state) => state.auth.user);
  const history = useAppSelector((state) => state.history.history);

  // Single NavigationContainer at the top level - never remounts
  // MUST be called before any conditional returns to maintain hook order
  const navigationRef = useRef<any>(null);

  // Splash fade-out animation
  const splashOpacity = useRef(new Animated.Value(1)).current;
  const [splashMounted, setSplashMounted] = useState(showSplash);

  // Load user from storage on app start
  useEffect(() => {
    console.log('[Redux] Dispatching loadUserFromStorage...');
    dispatch(loadUserFromStorage());
  }, [dispatch]);

  // Log Redux state changes
  const authState = useAppSelector((state) => state.auth);
  useEffect(() => {
    console.log('[Redux] Auth state:', {
      isAuthenticated: authState.isAuthenticated,
      isLoading: authState.isLoading,
      userEmail: authState.user?.email,
    });
  }, [authState.isAuthenticated, authState.isLoading, authState.user]);

  // Log showWelcome changes
  useEffect(() => {
    console.log('[AppContent] showWelcome changed:', showWelcome, 'isAuthenticated:', isAuthenticated);
  }, [showWelcome, isAuthenticated]);

  // Re-check any analyses stuck at 'analyzing' — covers both foreground re-entry
  // and cold start (where AppState is already 'active' when the listener registers).
  const historyRef = useRef(history);
  historyRef.current = history;
  const userRef = useRef(user);
  userRef.current = user;

  const isHistoryLoading = useAppSelector((state) => state.history.isLoading);
  const hasRunColdStartCheck = useRef(false);
  const historyWasLoading = useRef(false);

  const checkPendingJobs = useCallback(async (source: string) => {
    const pendingEntries = historyRef.current.filter(
      (entry) => entry.analysisStatus === 'analyzing' && entry.job_id
    );

    if (pendingEntries.length === 0) return;

    console.log(`[ResumeCheck] ${source} — re-checking ${pendingEntries.length} pending job(s)`);

    for (const entry of pendingEntries) {
      const jobId = entry.job_id!;

      // Re-read the latest entry from the ref — PreviewScreen polling may have already completed it
      const currentEntry = historyRef.current.find((e) => e.id === entry.id);
      if (!currentEntry || currentEntry.analysisStatus !== 'analyzing') {
        console.log(`[ResumeCheck] Skipping job ${jobId} — already resolved by in-app polling`);
        continue;
      }

      console.log(`[ResumeCheck] Checking job ${jobId} for analysis ${entry.id}...`);
      try {
        const status = await nutritionAnalysisAPI.checkStatus(jobId);
        if (!status) {
          console.warn(`[ResumeCheck] Could not reach backend for job ${jobId}`);
          continue;
        }

        console.log(`[ResumeCheck] Job ${jobId} status: ${status.status}`);

        if (status.status === 'completed') {
          // Check once more that PreviewScreen hasn't resolved it while we were awaiting checkStatus
          const stillPending = historyRef.current.find((e) => e.id === entry.id);
          if (!stillPending || stillPending.analysisStatus !== 'analyzing') {
            console.log(`[ResumeCheck] Skipping update — analysis ${entry.id} already resolved`);
            continue;
          }

          const results = await nutritionAnalysisAPI.getResults(jobId, true, true);
          if (!results) continue;

          const items = results.items ?? [];
          const summary = results.nutrition_summary;
          const dishTables = buildDishTablesFromItems(items, (currentEntry as any)?.questionnaireContext);
          const dishContents = items.length > 0
            ? getBaseDishContents(dishTables)
            : [{ id: `${Date.now()}_0`, name: 'No food detected', weight: '', calories: '0' }];
          const totalCalories = getOverallCaloriesFromTables(dishTables) || summary?.total_calories_kcal || 0;
          const mealName = items.length > 0
            ? (results.meal_name || getMealNameFromTables(dishTables, 'Analyzed Meal'))
            : 'No food detected';

          if (userRef.current?.email) {
            await dispatch(updateAnalysis({
              userEmail: userRef.current.email,
              analysisId: entry.id,
              updates: {
                analysisStatus: 'completed',
                analysisProgress: 100,
                dishTables,
                dishContents,
                mealName,
                nutritionalInfo: { calories: totalCalories, protein: 0, carbs: 0, fat: 0 },
                analysisResult: {
                  summary: `Detected ${dishContents.length} food items with ${Math.round(totalCalories)} calories`,
                  nutrition_summary: summary,
                  job_id: jobId,
                } as any,
                job_id: jobId,
                ...(results.segmented_images ? { segmented_images: results.segmented_images } : {}),
                ...(results.trellis_mp4_url ? { videoUri: results.trellis_mp4_url } : {}),
              },
            }));
            console.log(`[ResumeCheck] ✅ Updated analysis ${entry.id} to completed`);
          }
        } else if (status.status === 'failed') {
          dispatch(updateAnalysisProgress({ id: entry.id, progress: 0, status: 'failed' }));
          console.log(`[ResumeCheck] ❌ Job ${jobId} failed`);
        } else {
          console.log(`[ResumeCheck] Job ${jobId} still in progress (${status.status})`);
        }
      } catch (err) {
        console.warn(`[ResumeCheck] Error checking job ${jobId}:`, err);
      }
    }
  }, [dispatch]);

  // Foreground re-entry: AppState fires when the app transitions background → active
  useEffect(() => {
    const subscription = AppState.addEventListener('change', async (nextState) => {
      if (nextState !== 'active') return;
      await checkPendingJobs('App became active');
    });
    return () => subscription.remove();
  }, [checkPendingJobs]);

  // Cold-start / re-open: app is already 'active' when the listener registers,
  // so AppState never fires. Instead, watch for history to finish its first load
  // and run the same check once.
  useEffect(() => {
    if (!isAuthenticated) {
      // Reset on logout so the next login triggers a fresh cold-start check
      hasRunColdStartCheck.current = false;
      historyWasLoading.current = false;
      return;
    }
    if (isHistoryLoading) {
      historyWasLoading.current = true;
      return;
    }
    // Only run after we've seen at least one loading cycle (prevents firing before
    // loadHistory is even dispatched, when isHistoryLoading is still false)
    if (!historyWasLoading.current) return;
    if (hasRunColdStartCheck.current) return;
    hasRunColdStartCheck.current = true;
    checkPendingJobs('Cold start (history loaded)');
  }, [isAuthenticated, isHistoryLoading, checkPendingJobs]);

  const handleSplashFinish = useCallback(() => {
    Animated.timing(splashOpacity, {
      toValue: 0,
      duration: 500,
      useNativeDriver: true,
    }).start(async () => {
      setSplashMounted(false);
      dispatch(setShowSplash(false));

      // Ask for notification permission on first launch only (single system dialog)
      const asked = await AsyncStorage.getItem('notification_permission_asked');
      if (!asked) {
        await AsyncStorage.setItem('notification_permission_asked', 'true');
        await Notifications.requestPermissionsAsync();
      }
    });
  }, [dispatch, splashOpacity]);

  console.log('[AppContent] Rendering app, showWelcome:', showWelcome, 'isAuthenticated:', isAuthenticated);

  return (
    <View style={appStyles.root}>
      <GestureHandlerRootView style={{ flex: 1 }}>
        <PaperProvider theme={{
          ...DefaultTheme,
          colors: {
            ...DefaultTheme.colors,
            background: '#FFFFFF',
            surface: '#FFFFFF',
            text: '#4a4a4a',
            onSurface: '#4a4a4a',
            placeholder: '#4a4a4a',
            backdrop: 'rgba(0, 0, 0, 0.5)',
          },
          dark: false,
        }}>
          <SafeAreaProvider>
            <NavigationContainer ref={navigationRef} theme={{ dark: false, colors: { primary: '#7BA21B', background: '#000000', card: '#FFFFFF', text: '#000000', border: '#E5E7EB', notification: '#7BA21B' } }}>
              <RootNavigator />
            </NavigationContainer>
          </SafeAreaProvider>
        </PaperProvider>
      </GestureHandlerRootView>

      {/* Splash overlay — fades out smoothly over the main app */}
      {splashMounted && (
        <Animated.View style={[StyleSheet.absoluteFillObject, { opacity: splashOpacity }]}>
          <SplashScreen onFinish={handleSplashFinish} />
        </Animated.View>
      )}
    </View>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <ReduxProvider store={store}>
        <PersistGate loading={<AppLoader />} persistor={persistor}>
          <AppContent />
        </PersistGate>
      </ReduxProvider>
    </ErrorBoundary>
  );
}

// Styles moved to AppLoader component
const appStyles = StyleSheet.create({
  root: {
    flex: 1,
  },
});
