import React, { useRef, useState, useEffect, useCallback } from 'react';
import { StyleSheet, Text, View, Image, TouchableOpacity, ScrollView, StatusBar, Alert, LayoutAnimation, Platform, UIManager, Animated, Linking, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { PanGestureHandler, State } from 'react-native-gesture-handler';
import { Video, ResizeMode } from 'expo-av';
import { Camera } from 'react-native-vision-camera';
import { Ionicons } from '@expo/vector-icons';
import Svg, { Path } from 'react-native-svg';
import { useNavigation, useRoute } from '@react-navigation/native';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { deleteAnalysis, loadHistory, clearHistoryLocal } from '../store/slices/historySlice';
import { loadProfile } from '../store/slices/profileSlice';
import type { AnalysisEntry } from '../store/slices/historySlice';
import ScreenLoader from '../components/ScreenLoader';
import ImageWithLoader from '../components/ImageWithLoader';
import { getImagePresignedUrl } from '../services/S3UserDataService';
import AppHeader from '../components/AppHeader';
import BottomButtonContainer from '../components/BottomButtonContainer';
import TutorialScreen from './TutorialScreen';
import CircularProgressBar from '../components/CircularProgressBar';

// Image component that falls back to S3 presigned URL when the local file is unavailable (e.g. after reinstall)
function HistoryCardImage({
  imageUri,
  jobId,
  style,
  onImageLoad,
  onImageError,
}: {
  imageUri: string;
  jobId?: string;
  style: any;
  onImageLoad?: () => void;
  onImageError?: () => void;
}) {
  const [uri, setUri] = useState(imageUri);
  const hasTriedS3 = useRef(false);

  const handleError = useCallback(async () => {
    // Image was already uploaded to S3 during analysis — retrieve it by job_id
    if (jobId && !hasTriedS3.current) {
      hasTriedS3.current = true;
      try {
        const s3Url = await getImagePresignedUrl(jobId);
        if (s3Url) {
          setUri(s3Url);
          return; // Retry with S3 URL; don't propagate error yet
        }
      } catch {}
    }
    onImageError();
  }, [jobId, onImageError]);

  return (
    <ImageWithLoader
      source={{ uri }}
      style={style}
      resizeMode="cover"
      onImageLoad={onImageLoad}
      onImageError={handleError}
    />
  );
}

// SVG Icon Component
const Group2065Icon = ({ width = 28, height = 28 }: { width?: number; height?: number }) => (
  <Svg width={width} height={height} viewBox="0 0 442 430" fill="none">
    <Path d="M333.238 264.191H234.238V165.191C285.898 168.831 329.188 212.791 333.238 264.191ZM300.238 219.191H255.238V229.191H299.238L300.238 219.191Z" fill="#7BA21B"/>
    <Path d="M333.238 283.191C329.278 334.561 286.258 379.341 234.238 382.191V283.191H333.238ZM255.238 307.191V317.191H293.238V307.191H255.238ZM255.238 325.191V335.191H293.238V325.191H255.238Z" fill="#7BA21B"/>
    <Path d="M214.238 283.191V382.191C162.318 379.331 119.038 334.521 115.238 283.191H214.238ZM192.918 306.491C191.858 305.271 188.778 304.151 187.738 302.211L175.738 313.231L163.738 302.211L157.478 309.711L168.228 321.691C162.138 329.371 151.118 332.061 163.748 340.171L175.748 329.151L187.748 340.171L194.008 332.671L183.258 320.691C185.398 317.061 196.788 310.951 192.918 306.491Z" fill="#7BA21B"/>
    <Path d="M214.238 165.191V264.191H115.238C119.408 212.781 162.558 168.811 214.238 165.191ZM179.238 202.191H169.238V219.191H152.238V229.191H169.238V246.191H179.238V229.191H196.238V219.191H180.738L179.238 217.691V202.191Z" fill="#7BA21B"/>
    <Path d="M107.918 171.491C107.458 170.931 98.9284 164.831 98.3584 164.691C95.9184 164.121 87.1484 167.691 83.6284 168.091C68.0784 169.831 55.1584 162.911 43.7084 153.221L0.688373 108.251C-2.06163 99.9113 3.75837 93.3413 12.2684 95.6613C22.8384 98.5413 39.8384 124.881 50.3284 132.601C59.7484 139.531 68.9184 136.331 63.6984 124.231L24.9984 84.9313C20.7384 75.4313 30.8284 66.6913 39.2984 72.6313C51.7484 81.3513 65.1084 102.991 78.2784 110.641C84.0484 113.991 90.1784 111.071 90.1684 104.681C90.1584 98.3413 61.9584 76.1113 56.2084 69.7113C53.5284 66.7313 48.2384 60.6513 48.1984 56.7013C48.1084 48.7313 57.8884 42.9113 64.2784 48.6413C82.2784 70.6713 123.358 93.8513 122.288 125.711C122.138 130.131 119.438 136.231 120.658 140.221C121.388 142.591 132.628 151.851 134.738 151.851C152.648 139.191 172.608 130.131 194.228 125.671C202.548 123.951 210.938 123.431 219.228 121.671C230.058 95.7713 239.458 69.1413 253.068 44.5113C258.008 35.5613 272.568 10.7713 280.288 5.73127C292.248 -2.07873 307.628 8.12128 304.798 22.2413C303.168 30.3513 286.128 48.7513 280.428 56.8613C270.158 71.4813 255.578 94.3913 249.488 110.921C248.568 113.411 244.808 123.151 247.198 124.731C248.008 125.261 262.968 127.751 266.248 128.681C287.448 134.741 303.628 144.731 321.568 157.021C325.868 154.131 338.208 143.751 338.198 138.831C338.178 132.311 335.618 125.741 336.218 117.641C340.388 61.3313 428.688 34.4713 440.808 93.1213C448.598 130.801 412.838 169.561 374.758 166.171C370.278 165.771 365.618 162.981 360.948 164.391C358.918 165.011 345.248 179.421 345.258 181.681C352.518 194.591 360.748 206.981 365.998 220.931C383.958 268.651 375.568 322.371 346.548 363.681C346.548 368.011 380.818 395.561 385.468 404.471C392.018 417.041 383.458 431.891 368.988 428.951C357.658 426.651 329.168 388.481 323.768 388.481C284.248 420.231 234.828 433.361 185.048 419.901C165.698 414.671 148.538 405.161 131.758 394.511C130.288 394.721 129.338 395.821 128.238 396.661C118.718 403.991 107.588 420.731 97.9684 426.381C83.8184 434.701 67.9384 421.081 74.6984 406.111L107.238 371.691C101.218 360.201 92.9084 350.361 87.3884 338.531C63.0084 286.181 70.4984 230.781 102.418 183.841C104.778 180.361 111.748 176.071 107.928 171.471L107.918 171.491ZM350.358 273.861C350.358 204.211 293.898 147.751 224.248 147.751C154.598 147.751 98.1384 204.211 98.1384 273.861C98.1384 343.511 154.598 399.971 224.248 399.971C293.898 399.971 350.358 343.511 350.358 273.861Z" fill="#7BA21B"/>
    <Path d="M216.248 97.1913C208.538 68.2813 188.418 42.8513 161.718 29.2113C158.748 29.0213 160.958 30.3513 161.888 31.0513C176.998 42.4113 187.108 52.4613 197.258 68.6813C201.608 75.6213 211.668 92.7013 209.298 100.621C195.278 103.691 177.598 102.831 163.988 97.9513C132.248 86.5813 135.818 51.5713 122.408 26.5313C116.548 15.5813 108.198 10.2913 99.2383 2.21133C120.828 -1.68867 143.988 -0.418666 164.998 5.93133C198.268 15.9913 230.118 44.5313 222.958 82.4213C222.628 84.1613 219.048 99.2913 216.238 97.1913H216.248Z" fill="#7BA21B"/>
  </Svg>
);

// Enable LayoutAnimation on Android (run once)
if (Platform.OS === 'android' && UIManager.setLayoutAnimationEnabledExperimental) {
  UIManager.setLayoutAnimationEnabledExperimental(true);
}

export default function ResultsScreen({ navigation: navigationProp }: { navigation?: any }) {
  const navigation = useNavigation<any>();
  const route = useRoute<any>();
  const dispatch = useAppDispatch();
  const history = useAppSelector((state) => state.history?.history ?? []);
  const isLoading = useAppSelector((state) => state.history?.isLoading ?? false);
  const error = useAppSelector((state) => state.history?.error ?? null);
  const user = useAppSelector((state) => state.auth?.user);
  const businessProfile = useAppSelector((state) => state.profile?.businessProfile);
  const userAccount = useAppSelector((state) => state.profile?.userAccount);
  const isProfileLoading = useAppSelector((state) => state.profile?.isLoading ?? false);

  // Note: profileBelongsToCurrentUser is calculated later (used in loader checks)
  // App.tsx now handles initial profile validation before showing this screen

  const deletingRef = useRef<Set<string>>(new Set());
  const [playingVideoId, setPlayingVideoId] = useState<string | null>(null);
  // Resolved S3 URLs for videos whose local file:// URI is no longer on device
  const [resolvedVideoUris, setResolvedVideoUris] = useState<Record<string, string>>({});
  const fetchedVideoIds = useRef<Set<string>>(new Set());

  // Two-step notification: set flag when param arrives (clear param immediately so re-submissions retrigger),
  // then fire alert once dashboard has items visible (history.length > 0).
  const [pendingNotification, setPendingNotification] = useState(false);

  useEffect(() => {
    if (!route.params?.showSubmittedNotification) return;
    setPendingNotification(true);
    navigation.setParams({ showSubmittedNotification: undefined });
  }, [route.params?.showSubmittedNotification]);

  useEffect(() => {
    if (!pendingNotification || history.length === 0) return;
    const t = setTimeout(() => {
      setPendingNotification(false);
      Alert.alert('Submitted!', 'We will notify you when the results are ready.', [{ text: 'OK' }]);
    }, 1000);
    return () => clearTimeout(t);
  }, [pendingNotification, history.length]);

  // Proactively fetch S3 URLs for all video items so playback works even if local file is gone
  useEffect(() => {
    history.forEach(histItem => {
      if (histItem.videoUri && histItem.job_id && !fetchedVideoIds.current.has(histItem.id)) {
        fetchedVideoIds.current.add(histItem.id);
        getImagePresignedUrl(histItem.job_id).then(url => {
          if (url) setResolvedVideoUris(prev => ({ ...prev, [histItem.id]: url }));
        });
      }
    });
  }, [history]);
  const swipePositions = useRef<{ [key: string]: Animated.Value }>({});
  const [canShowTutorial, setCanShowTutorial] = useState(false); // Control when TutorialScreen can be shown
  const [isHistoryLoaded, setIsHistoryLoaded] = useState(false); // State to trigger re-renders when history is loaded
  const [isReturningFromTutorial, setIsReturningFromTutorial] = useState(false); // Track when returning from TutorialScreen
  const rightSwipePosition = useRef(new Animated.Value(0)); // For right swipe to TutorialScreen

  // Use navigation prop if provided (for stack navigation), otherwise use hook
  const nav = navigationProp || navigation;

  const hasLoadedHistory = useRef(false); // Track if we've loaded history at least once
  const hasStartedLoading = useRef(false); // Track if loading has started at least once
  const currentUserEmail = useRef<string | null>(null); // Track current user to detect changes
  const lastRenderedEmail = useRef<string | null>(null); // Track last rendered email to prevent flash
  const loadCompleteTime = useRef<number | null>(null); // Track when loading completed

  // Reset tracking when user changes (logout/login)
  useEffect(() => {
    if (user?.email !== currentUserEmail.current) {
      // User changed - reset all tracking
      const previousEmail = currentUserEmail.current;
      currentUserEmail.current = user?.email || null;

      // CRITICAL: Clear old data immediately to prevent showing stale info
      if (history.length > 0 && previousEmail !== null) {
        dispatch(clearHistoryLocal());
      }

      // Note: Profile is now managed by App.tsx - don't clear it here
      // App.tsx calls loadProfile() when user logs in and clearProfile() on logout

      // When user changes, force a complete reset
      hasLoadedHistory.current = false;
      hasStartedLoading.current = false;
      setIsHistoryLoaded(false);
      setCanShowTutorial(false);
      loadCompleteTime.current = null;
    }
  }, [user?.email, dispatch, history.length]);
  
  // Load history ONLY when user email changes (not on every state change)
  useEffect(() => {
    if (user?.email && !hasStartedLoading.current) {
      hasStartedLoading.current = true;
      hasLoadedHistory.current = false;
      dispatch(loadHistory(user.email));
    }
  }, [user?.email, dispatch]);

  // Profile is loaded by App.tsx — no dispatch needed here

  // Track the initial history load only — do NOT reset hasLoadedHistory on every isLoading change
  // (addAnalysis, deleteAnalysis, etc. all set isLoading and must not re-trigger the full-screen loader)
  useEffect(() => {
    if (isLoading && !hasLoadedHistory.current) {
      hasStartedLoading.current = true;
    }
  }, [isLoading]);

  // Track when loading starts and completes
  useEffect(() => {
    // If history has items, mark as loaded
    if (history.length > 0) {
      hasStartedLoading.current = true;
      hasLoadedHistory.current = true;
      setIsHistoryLoaded(true);
      setCanShowTutorial(false);
      return;
    }

    // Handle loading state transitions for empty history
    if (isLoading) {
      hasStartedLoading.current = true;
      hasLoadedHistory.current = false;
      setIsHistoryLoaded(false);
      loadCompleteTime.current = null;
      setCanShowTutorial(false);
    } else if (hasStartedLoading.current && !isLoading && !hasLoadedHistory.current) {
      hasLoadedHistory.current = true;
      setIsHistoryLoaded(true);
      loadCompleteTime.current = Date.now();

      if (history.length === 0) {
        setCanShowTutorial(true);
      } else {
        setCanShowTutorial(false);
      }
    }
  }, [isLoading, history.length, isHistoryLoaded]);

  // If history already has items, we know it's loaded (even if loading state hasn't updated yet)
  // Also, if history is empty but we're not loading and user exists, mark as loaded
  useEffect(() => {
    if (history.length > 0) {
      // CRITICAL: History has items, so it's definitely loaded
      // Mark as loaded immediately and NEVER allow TutorialScreen
      hasLoadedHistory.current = true;
      hasStartedLoading.current = true;
      setIsHistoryLoaded(true); // STATE UPDATE - triggers re-render
      setCanShowTutorial(false); // Explicitly prevent tutorial from showing
    } else if (!isLoading && user?.email && hasStartedLoading.current) {
      // History is empty, not loading, and we've started loading - mark as loaded
      // This handles both: initial load with no history AND all cards deleted
      hasLoadedHistory.current = true;
      setIsHistoryLoaded(true); // STATE UPDATE - triggers re-render
      setCanShowTutorial(true); // Show tutorial when history becomes empty (including after deleting all cards)
    }
  }, [history.length, isLoading, user?.email]);

  // Cleanup: clear playing state if the item is removed from history
  useEffect(() => {
    if (playingVideoId && !history.find(item => item.id === playingVideoId)) {
      setPlayingVideoId(null);
    }
  }, [history, playingVideoId]);

  // Note: Tutorial screen is now shown directly in the render when history is empty
  // No need to navigate to it separately
  // Note: profileBelongsToCurrentUser is calculated at the top of the component (line 50)

  const handleVideoPlay = (itemId: string, videoUri: string, jobId?: string) => {
    if (playingVideoId === itemId) {
      setPlayingVideoId(null);
    } else {
      // If we don't have an S3 URL yet, fetch it first then play
      if (!resolvedVideoUris[itemId] && jobId) {
        getImagePresignedUrl(jobId).then(url => {
          if (url) setResolvedVideoUris(prev => ({ ...prev, [itemId]: url }));
          setPlayingVideoId(itemId);
        });
      } else {
        setPlayingVideoId(itemId);
      }
    }
  };

  // Initialize swipe position for a card
  const getSwipePosition = (itemId: string) => {
    if (!swipePositions.current[itemId]) {
      swipePositions.current[itemId] = new Animated.Value(0);
    }
    return swipePositions.current[itemId];
  };

  // Handle swipe gesture state change
  const handleSwipeStateChange = (item: AnalysisEntry, event: any) => {
    const { state } = event.nativeEvent;
    const translateX = getSwipePosition(item.id);

    if (state === State.END) {
      const threshold = -100; // Swipe threshold to trigger delete
      const currentValue = (translateX as any)._value || 0;

      if (currentValue < threshold) {
        // Swiped enough - show delete confirmation
        if (!deletingRef.current.has(item.id)) {
          deletingRef.current.add(item.id);
          
          Alert.alert(
            '',
            'Are you sure you want to delete this record?',
            [
              {
                text: 'Cancel',
                style: 'cancel',
                onPress: () => {
                  // Animate back to original position
                  Animated.spring(translateX, {
                    toValue: 0,
                    useNativeDriver: true,
                    tension: 50,
                    friction: 7,
                  }).start();
                  deletingRef.current.delete(item.id);
                },
              },
              {
                text: 'Delete',
                style: 'destructive',
                onPress: () => {
                  // Stop video if playing
                  if (playingVideoId === item.id) {
                    setPlayingVideoId(null);
                  }

                  // Configure smooth layout animation for remaining cards
                  LayoutAnimation.configureNext({
                    duration: 200,
                    create: {
                      type: LayoutAnimation.Types.easeOut,
                      property: LayoutAnimation.Properties.opacity,
                    },
                    update: {
                      type: LayoutAnimation.Types.easeOut,
                      springDamping: 0.7,
                    },
                    delete: {
                      type: LayoutAnimation.Types.easeOut,
                      property: LayoutAnimation.Properties.opacity,
                    },
                  });

                  // Delete immediately
                  if (user?.email) {
                    dispatch(deleteAnalysis({ userEmail: user.email, analysisId: item.id }));
                  }
                  deletingRef.current.delete(item.id);
                  // Clean up animation value
                  delete swipePositions.current[item.id];
                },
              },
            ],
            { 
              cancelable: true, 
              onDismiss: () => {
                // Animate back to original position
                Animated.spring(translateX, {
                  toValue: 0,
                  useNativeDriver: true,
                  tension: 50,
                  friction: 7,
                }).start();
                deletingRef.current.delete(item.id);
              }
            }
          );
        }
      } else {
        // Didn't swipe enough - animate back
        Animated.spring(translateX, {
          toValue: 0,
          useNativeDriver: true,
          tension: 50,
          friction: 7,
        }).start();
      }
    }
  };

  const renderCard = (item: AnalysisEntry, index: number) => {
    const isVideo = !!item.videoUri;
    const isPlaying = playingVideoId === item.id;
    const isAnalyzing = item.analysisStatus === 'analyzing';
    const progress = item.analysisProgress || 0;
    const totalCalories = item.nutritionalInfo?.calories ?? 0;
    const hasMealName = !!(item.mealName && item.mealName.trim());
    const isCompletedOrFailed = item.analysisStatus === 'completed' || item.analysisStatus === 'failed';
    // Treat as "pending" when we have no data yet: show "Analyzing..." until we have calories or meal name.
    // This avoids a brief flash of "Unidentified Food" / "0 kcal" when status is already 'completed' but data hasn't arrived yet.
    const hasNoResultYet = totalCalories === 0 && !hasMealName;
    const isPendingOrAnalyzing =
      isAnalyzing ||
      (hasNoResultYet && item.analysisStatus !== 'failed');
    // Only treat as "unidentified" when there is truly no data — no meal name and no calories.
    // A user-cleared entry (calories=0 but mealName set) should stay tappable and show "-".
    const hasZeroCalories = isCompletedOrFailed && totalCalories === 0 && !hasMealName;
    const isNoFoodDetected = item.mealName === 'No food detected';
    const isNonTappable = isPendingOrAnalyzing || hasZeroCalories || isNoFoodDetected;
    const titleText = isPendingOrAnalyzing ? '' : (hasZeroCalories ? 'Unidentified Food' : (item.mealName || ''));
    const subtitleText = isPendingOrAnalyzing
      ? 'Analyzing...'
      : totalCalories === 0
      ? '-'
      : `${totalCalories} Kcal`;
    const translateX = getSwipePosition(item.id);

    return (
      <View style={styles.cardWrapper} key={item.id}>
        <PanGestureHandler
          onGestureEvent={Animated.event(
            [{ nativeEvent: { translationX: translateX } }],
            { 
              useNativeDriver: true,
              listener: (event: any) => {
                // Clamp to only allow left swipe (negative values)
                const { translationX: tx } = event.nativeEvent;
                if (tx > 0) {
                  translateX.setValue(0);
                }
              }
            }
          )}
          onHandlerStateChange={(event) => handleSwipeStateChange(item, event)}
          activeOffsetX={[-10, 10]}  // Activate on both directions, but listener only handles left
          failOffsetY={[-5, 5]}
        >
          <Animated.View
            style={[
              styles.cardContainer,
              {
                transform: [{ translateX }],
              },
            ]}
          >
            <TouchableOpacity
              style={styles.card}
              onPress={() => {
                if (!isNonTappable) {
                  nav.navigate('MealDetail', { item });
                }
              }}
              activeOpacity={isNonTappable ? 1 : 0.9}
              disabled={isNonTappable}
            >
          <View style={styles.mediaWrapper}>
            {isVideo && (resolvedVideoUris[item.id] || item.videoUri?.startsWith('http')) ? (
              <>
                <Video
                  key={resolvedVideoUris[item.id] || item.id}
                  source={{ uri: resolvedVideoUris[item.id] || item.videoUri! }}
                  style={styles.media}
                  resizeMode={ResizeMode.COVER}
                  isLooping={false}
                  isMuted={false}
                  shouldPlay={isPlaying}
                  useNativeControls={false}
                  onPlaybackStatusUpdate={(status) => {
                    if (status.isLoaded && status.didJustFinish) {
                      setPlayingVideoId(null);
                    }
                  }}
                />
                {!isPlaying && (
                  <TouchableOpacity
                    style={styles.playOverlay}
                    onPress={() => handleVideoPlay(item.id, item.videoUri!, item.job_id)}
                    activeOpacity={0.8}
                  >
                    <View style={styles.playCircle}>
                      <Ionicons name="play" size={28} color="#FFFFFF" />
                    </View>
                  </TouchableOpacity>
                )}
                {isPlaying && (
                  <TouchableOpacity
                    style={styles.playOverlay}
                    onPress={() => handleVideoPlay(item.id, item.videoUri!, item.job_id)}
                    activeOpacity={0.8}
                  >
                    <View style={styles.playCircle}>
                      <Ionicons name="pause" size={28} color="#FFFFFF" />
                    </View>
                  </TouchableOpacity>
                )}
              </>
            ) : item.imageUri ? (
              <HistoryCardImage
                imageUri={item.imageUri}
                jobId={item.job_id}
                style={styles.media}
              />
            ) : (
              <View style={[styles.media, styles.videoFallback]} />
            )}
          </View>

          <View style={styles.infoStrip}>
            <View style={{ flex: 1 }}>
              <Text style={styles.cardTitle}>{titleText}</Text>
              <Text style={styles.cardSubtitle}>{subtitleText}</Text>
            </View>
            {isPendingOrAnalyzing ? (
              <CircularProgressBar
                progress={progress}
                size={18}
                strokeWidth={2}
                color="#7BA21B"
                backgroundColor="#E5E7EB"
              />
            ) : (
              <Group2065Icon width={18} height={18} />
            )}
          </View>
        </TouchableOpacity>
        </Animated.View>
      </PanGestureHandler>
      </View>
    );
  };

  // AGGRESSIVE: Always show loader by default, only hide when we're 100% certain
  // Show loader if ANY of these are true:
  // 1. No user yet
  // 2. Haven't started loading
  // 3. Currently loading
  // 4. Haven't confirmed load complete (using STATE not ref for reactivity)
  // 5. History is empty AND TutorialScreen not explicitly allowed (CRITICAL - prevents flash)
  // 6. History has items but images are still loading (for ResultsScreen)
  // 7. User change is in progress (logout/login transition)
  // 8. Profile doesn't belong to current user (email mismatch)
  // 9. Current user email differs from last rendered email (prevents flash during transition)
  //
  // CRITICAL FIX: Once history has items (length > 0), NEVER show TutorialScreen
  // This prevents the flash when logging back in with existing data
  // Using isHistoryLoaded STATE instead of ref for proper re-renders

  // CRITICAL: Calculate profileBelongsToCurrentUser for loader checks
  const profileBelongsToCurrentUser = userAccount?.email === user?.email;

  // CRITICAL: Detect email mismatch between current and last rendered
  const emailChanged = lastRenderedEmail.current !== null &&
                       lastRenderedEmail.current !== user?.email &&
                       profileBelongsToCurrentUser; // Only block if we have a valid profile that matches

  // Show loader only until the initial history load completes for this session.
  // Do NOT block on isLoading here — addAnalysis/deleteAnalysis also set isLoading
  // and must not re-trigger the full-screen loader once the dashboard is already showing.
  const historyNotReadyYet = !hasLoadedHistory.current;

  const shouldShowLoader = !user?.email ||
                           emailChanged ||
                           (history.length === 0 && (historyNotReadyYet || !canShowTutorial));

  // Update last rendered email once we're sure we're showing the correct data
  if (!shouldShowLoader && user?.email && profileBelongsToCurrentUser) {
    lastRenderedEmail.current = user.email;
  }

  // CRITICAL: Show loader FIRST, before any other checks
  // This MUST be the first return statement to prevent TutorialScreen from ever flashing
  if (shouldShowLoader) {
    return (
      <ScreenLoader isLoading={true}>
        <SafeAreaView style={styles.container} edges={['top']}>
          <StatusBar barStyle="dark-content" />
        </SafeAreaView>
      </ScreenLoader>
    );
  }

  // ONLY show TutorialScreen when canShowTutorial is EXPLICITLY true
  // This is the ONLY place TutorialScreen can render
  // All other conditions are checked in shouldShowLoader above
  // Allow showing TutorialScreen even when history has items (for right swipe navigation)
  if (canShowTutorial) {
    return (
      <TutorialScreen 
        onBack={() => {
          // Reset right swipe position animation
          rightSwipePosition.current.setValue(0);
          setIsReturningFromTutorial(true);
          setCanShowTutorial(false);
          // Refresh history when coming back to ensure data is up to date
          if (user?.email) {
            dispatch(loadHistory(user.email)).then(() => {
              // Hide loader after history is loaded
              setTimeout(() => {
                setIsReturningFromTutorial(false);
              }, 300); // Small delay to ensure smooth transition
            });
          } else {
            setIsReturningFromTutorial(false);
          }
        }} 
      />
    );
  }

  // Show green loader when returning from TutorialScreen
  if (isReturningFromTutorial) {
    return <ScreenLoader isLoading={true} />;
  }

  // CRITICAL: Only calculate display values AFTER all loader checks pass
  // This ensures we never render with stale data from previous user
  const userName = (profileBelongsToCurrentUser && businessProfile?.businessName)
    ? businessProfile.businessName
    : user?.email?.split('@')[0] || 'User';
  const displayName = userName.charAt(0).toUpperCase() + userName.slice(1);

  const lastLoginDate = new Date().toLocaleDateString('en-US', {
    month: 'long',
    day: 'numeric',
    year: 'numeric',
  });
  const lastLoginTime = new Date().toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  });

  // Handle right swipe to navigate to TutorialScreen
  const handleRightSwipeStateChange = (event: any) => {
    const { state, translationX } = event.nativeEvent;
    
    if (state === State.END) {
      const threshold = 100; // Swipe threshold to trigger navigation (positive for right swipe)
      const currentValue = translationX || 0;
      
      if (currentValue > threshold) {
        // Swiped right enough - show TutorialScreen
        setCanShowTutorial(true);
      }
      
      // Always reset position after gesture ends
      Animated.spring(rightSwipePosition.current, {
        toValue: 0,
        useNativeDriver: true,
        tension: 50,
        friction: 7,
      }).start();
    }
  };

  const handleCameraPress = async () => {
    const status = Camera.getCameraPermissionStatus();

    if (status === 'granted') {
      nav.navigate('Camera');
      return;
    }

    if (status === 'denied' || status === 'restricted') {
      Alert.alert(
        'UKcal would like to access your camera',
        'UKcal would like to access your camera',
        [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Open Settings', onPress: () => Linking.openSettings() },
        ]
      );
      return;
    }

    // status is 'not-determined' — show the native OS permission popup
    const result = await Camera.requestCameraPermission();
    if (result === 'granted') {
      nav.navigate('Camera');
    }
  };

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <StatusBar barStyle="dark-content" />

      <AppHeader
        displayName={displayName}
        lastLoginDate={lastLoginDate}
        lastLoginTime={lastLoginTime}
        onProfilePress={() => nav.navigate('Profile')}
      />

      {/* Right Swipe Gesture Handler for TutorialScreen */}
      <PanGestureHandler
        onGestureEvent={Animated.event(
          [{ nativeEvent: { translationX: rightSwipePosition.current } }],
          { 
            useNativeDriver: true,
            listener: (event: any) => {
              // Clamp to only allow right swipe (positive values)
              const { translationX: tx } = event.nativeEvent;
              if (tx < 0) {
                rightSwipePosition.current.setValue(0);
              }
            }
          }
        )}
        onHandlerStateChange={handleRightSwipeStateChange}
        activeOffsetX={[-100, 10]}  // First value very high (never activates on left), second activates on right swipe
        failOffsetY={[-5, 5]}
        simultaneousHandlers={[]}  // Don't interfere with card gestures
      >
        <Animated.View
          style={[
            { flex: 1 },
            {
              transform: [{ translateX: rightSwipePosition.current }],
            },
          ]}
        >
          {/* List */}
          <ScrollView
            style={styles.list}
            contentContainerStyle={{ paddingBottom: 100, flexGrow: 1 }}
            showsVerticalScrollIndicator={false}
            decelerationRate="normal"
            bounces={true}
            scrollEventThrottle={16}
            overScrollMode="never"
            nestedScrollEnabled={true}
          >
        {isLoading ? (
          <View style={styles.emptyState}>
            <ActivityIndicator size="large" color="#7BA21B" />
          </View>
        ) : error && history.length === 0 ? (
          <View style={styles.emptyState}>
            <Text style={styles.emptyTitle}>Error loading history</Text>
            <Text style={styles.emptyText}>{error}</Text>
          </View>
        ) : history.length === 0 ? null : (
          history.map(renderCard)
        )}
          </ScrollView>
        </Animated.View>
      </PanGestureHandler>

      {/* Bottom CTA - Fixed at Bottom */}
      <BottomButtonContainer>
        <TouchableOpacity
          activeOpacity={0.9}
          style={styles.captureButton}
          onPress={handleCameraPress}
        >
          <Ionicons name="camera" size={20} color="#FFFFFF" style={{ marginRight: 8 }} />
          <Text style={styles.captureText}>Snap a Dish</Text>
        </TouchableOpacity>
      </BottomButtonContainer>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  header: {
    backgroundColor: '#FFFFFF',
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  avatar: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#E0F2FE',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 10,
  },
  greeting: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1F2937',
  },
  lastLogin: {
    fontSize: 11,
    color: '#6B7280',
    marginTop: 2,
  },
  list: {
    flex: 1,
    paddingHorizontal: 16,
    paddingTop: 12,
  },
  cardWrapper: {
    marginBottom: 14,
  },
  cardContainer: {
    width: '100%',
  },
  card: {
    backgroundColor: '#FFFFFF',
    borderRadius: 14,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#E5E7EB',
  },
  mediaWrapper: {
    width: '100%',
    height: 180,
    backgroundColor: '#000000',
  },
  media: {
    width: '100%',
    height: '100%',
  },
  videoFallback: {
    backgroundColor: '#111827',
  },
  playOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
  },
  playCircle: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  infoStrip: {
    backgroundColor: '#E6F1D3',
    paddingHorizontal: 14,
    paddingVertical: 12,
    flexDirection: 'row',
    alignItems: 'center',
  },
  cardTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#1F2937',
    marginBottom: 2,
  },
  cardSubtitle: {
    fontSize: 12,
    color: '#6B7280',
  },
  captureButton: {
    height: 56, // Fixed height
    width: '100%', // Fixed width
    backgroundColor: '#7BA21B',
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
  },
  captureText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 60,
    paddingHorizontal: 20,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1F2937',
    marginBottom: 8,
  },
  emptyText: {
    fontSize: 14,
    color: '#6B7280',
    textAlign: 'center',
  },
});


