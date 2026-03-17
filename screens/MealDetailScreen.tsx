import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Pressable,
  TextInput,
  ScrollView,
  StatusBar,
  Alert,
  Platform,
  Keyboard,
  TouchableWithoutFeedback,
  Animated,
  Modal,
  Image,
  KeyboardAvoidingView,
  ActivityIndicator,
  Dimensions,
} from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import { PanGestureHandler, State } from 'react-native-gesture-handler';
import { Video, ResizeMode } from 'expo-av';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useRoute, useFocusEffect } from '@react-navigation/native';
import { useAppSelector, useAppDispatch } from '../store/hooks';
import type { AnalysisEntry, DishContent, SegmentedImages } from '../store/slices/historySlice';
import { updateAnalysis } from '../store/slices/historySlice';
import { nutritionAnalysisAPI } from '../services/NutritionAnalysisAPI';
import { getImagePresignedUrl } from '../services/S3UserDataService';
import { prefetchSegmentedImage, cacheKeyFor } from '../services/SegmentedImageCache';
import VectorBackButtonCircle from '../components/VectorBackButtonCircle';
import OptimizedImage from '../components/OptimizedImage';
import AppHeader from '../components/AppHeader';
import BottomButtonContainer from '../components/BottomButtonContainer';


export default function MealDetailScreen() {
  const navigation = useNavigation<any>();
  const route = useRoute();
  const item = (route.params as any)?.item as AnalysisEntry;
  const user = useAppSelector((state) => state.auth.user);
  const businessProfile = useAppSelector((state) => state.profile.businessProfile);
  const dispatch = useAppDispatch();
  const insets = useSafeAreaInsets();

  const [isVideoPlaying, setIsVideoPlaying] = useState(false);
  const overlayVideoRef = useRef<Video>(null);
  const originalVideoRef = useRef<Video>(null);
  const screenSwipePosition = useRef(new Animated.Value(0)); // For screen-level swipe right gesture

  const handleVideoPlay = useCallback(() => {
    setIsVideoPlaying((prev) => !prev);
  }, []);

  // Use business name as display name, fallback to email if business name not available
  // Only use businessName if it exists and is not empty
  const userName = (businessProfile?.businessName && businessProfile.businessName.trim()) 
    ? businessProfile.businessName 
    : (user?.email?.split('@')[0] || 'User');
  const displayName = userName.charAt(0).toUpperCase() + userName.slice(1);
  
  // Get current date for last login (mock)
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

  if (!item) {
    return (
      <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
        <StatusBar barStyle="dark-content" />
        <AppHeader
          displayName={displayName}
          lastLoginDate={lastLoginDate}
          lastLoginTime={lastLoginTime}
          onProfilePress={() => navigation.navigate('Profile')}
        />
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
          <Text>No meal data available</Text>
        </View>
      </SafeAreaView>
    );
  }

  const isVideo = !!item.videoUri;

  const [editingRowId, setEditingRowId] = useState<string | null>(null);
  const [editingMealName, setEditingMealName] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [showSegmentationOverlay, setShowSegmentationOverlay] = useState(false);
  const [overlayImageUrl, setOverlayImageUrl] = useState<string | null>(null);
  const [showFullImageModal, setShowFullImageModal] = useState(false);
  const [fullImageUri, setFullImageUri] = useState<string | null>(null);
  // When segmented overlay URL fails to load (e.g. expired), refetch or show original image
  const [overlayLoadFailed, setOverlayLoadFailed] = useState(false);
  const [refreshedSegmentedImages, setRefreshedSegmentedImages] = useState<SegmentedImages | null>(null);
  const [refreshingOverlay, setRefreshingOverlay] = useState(false);
  const [mediaLoading, setMediaLoading] = useState(true);
  const [videoOverlayError, setVideoOverlayError] = useState(false);
  // Resolved image/video URI — always fetched fresh from S3 (presigned URL)
  const [resolvedImageUri, setResolvedImageUri] = useState<string | undefined>(undefined);
  const [resolvedVideoUri, setResolvedVideoUri] = useState<string | undefined>(undefined);

  useEffect(() => {
    // Always fetch from S3 when job_id is available — presigned URLs are always fresh
    if (item?.job_id) {
      getImagePresignedUrl(item.job_id)
        .then(url => { setResolvedImageUri(url || item?.imageUri); })
        .catch(() => setResolvedImageUri(item?.imageUri));
    } else {
      setResolvedImageUri(item?.imageUri);
    }
  }, [item?.id, item?.job_id]);

  useEffect(() => {
    // Always fetch from S3 when job_id is available — works on any device after reinstall
    if (item?.job_id) {
      getImagePresignedUrl(item.job_id)
        .then(url => { setResolvedVideoUri(url || item?.videoUri); })
        .catch(() => setResolvedVideoUri(item?.videoUri));
    } else {
      setResolvedVideoUri(item?.videoUri);
    }
  }, [item?.id, item?.job_id]);
  
  // Effective overlay: only use freshly fetched URLs — never show stale stored ones
  // (stored overlay_urls are presigned S3 URLs that expire, so always wait for fresh ones)
  const effectiveSegmentedImages = refreshedSegmentedImages;
  
  // Reset overlay state and loader when item changes
  useEffect(() => {
    setOverlayLoadFailed(false);
    setRefreshedSegmentedImages(null);
    setMediaLoading(true);
    setVideoOverlayError(false);
  }, [item?.id]);

  // Show loader again when overlay URL changes (e.g. after refetch)
  useEffect(() => {
    setMediaLoading(true);
  }, [effectiveSegmentedImages?.overlay_urls?.[0]?.url]);

  // Eagerly prefetch the overlay into expo-image's disk cache whenever the URL is available
  useEffect(() => {
    const remoteUrl = effectiveSegmentedImages?.overlay_urls?.[0]?.url;
    if (!item?.job_id || !remoteUrl) return;
    prefetchSegmentedImage(item.job_id, remoteUrl);
  }, [effectiveSegmentedImages?.overlay_urls?.[0]?.url, item?.job_id]);

  // Fetch segmented image URLs on open — always refetch since presigned URLs expire
  useEffect(() => {
    if (!item?.job_id || !user?.email || refreshingOverlay) return;
    let cancelled = false;
    (async () => {
      setRefreshingOverlay(true);
      try {
        const fresh = await nutritionAnalysisAPI.getResults(item.job_id!, true, false);
        if (cancelled) return;
        const hasResult = fresh?.segmented_images?.overlay_urls?.length || fresh?.segmented_images?.video_overlay_url;
        if (hasResult) {
          setOverlayLoadFailed(false);
          setVideoOverlayError(false);
          setRefreshedSegmentedImages(fresh.segmented_images);
          await dispatch(updateAnalysis({
            userEmail: user.email,
            analysisId: item.id,
            updates: { segmented_images: fresh.segmented_images },
          })).unwrap();
        }
      } catch {
        if (!cancelled) setOverlayLoadFailed(true);
      } finally {
        if (!cancelled) setRefreshingOverlay(false);
      }
    })();
    return () => { cancelled = true; };
  }, [item?.id, item?.job_id, user?.email]);

  const handleOverlayLoadError = useCallback(async () => {
    if (item?.job_id && user?.email) {
      setRefreshingOverlay(true);
      try {
        const fresh = await nutritionAnalysisAPI.getResults(item.job_id, true, false);
        if (fresh?.segmented_images?.overlay_urls?.length || fresh?.segmented_images?.video_overlay_url) {
          setOverlayLoadFailed(false);
          setVideoOverlayError(false);
          setRefreshedSegmentedImages(fresh.segmented_images);
          await dispatch(updateAnalysis({
            userEmail: user.email,
            analysisId: item.id,
            updates: { segmented_images: fresh.segmented_images },
          })).unwrap();
        } else {
          setOverlayLoadFailed(true);
        }
      } catch {
        setOverlayLoadFailed(true);
      } finally {
        setRefreshingOverlay(false);
      }
    } else {
      setOverlayLoadFailed(true);
    }
  }, [item?.id, item?.job_id, user?.email, dispatch]);
  
  // Initialize from saved dish contents; empty array when none saved yet — no fake defaults
  const [dishContents, setDishContents] = useState<DishContent[]>(
    item?.dishContents ?? []
  );
  const [mealName, setMealName] = useState(item?.mealName || 'Burger');
  // Total calories = sum of all dish content rows only; never falls back to API value
  const totalCalories = useMemo(() => {
    return dishContents.reduce((acc, row) => {
      const cal = Number(row.calories);
      return acc + (Number.isFinite(cal) ? cal : 0);
    }, 0);
  }, [dishContents]);

  // Track which input is currently focused
  const focusedInputRef = useRef<{ rowId: string; field: string } | null>(null);
  const scrollViewRef = useRef<ScrollView>(null);
  const tableContainerRef = useRef<View>(null);
  const mealNameContainerRef = useRef<View>(null);
  const [isKeyboardVisible, setIsKeyboardVisible] = useState(false);
  const [keyboardHeight, setKeyboardHeight] = useState(0);
  // Refs for input fields to enable "next" button navigation
  const inputRefs = useRef<{ [rowId: string]: { name?: TextInput; weight?: TextInput; calories?: TextInput } }>({});
  // Refs for row containers to enable scrolling to specific rows
  const rowRefs = useRef<{ [rowId: string]: View | null }>({});

  // Format capture date and time from item.timestamp (same approach as login time)
  const captureDate = item?.timestamp 
    ? (() => {
        try {
          // Try parsing the timestamp - handle ISO strings, locale strings, and numbers
          let date: Date;
          if (typeof item.timestamp === 'string') {
            // Try ISO string first (standard format)
            date = new Date(item.timestamp);
            // If that fails, try parsing as number (timestamp in milliseconds)
            if (isNaN(date.getTime())) {
              const numValue = Number(item.timestamp);
              if (!isNaN(numValue) && numValue > 0) {
                date = new Date(numValue);
              }
            }
          } else if (typeof item.timestamp === 'number') {
            date = new Date(item.timestamp);
          } else {
            return null;
          }
          
          if (isNaN(date.getTime())) {
            return null;
          }
          
          return date;
        } catch (error) {
          return null;
        }
      })()
    : null;
  
  const captureDateText = captureDate
    ? captureDate.toLocaleDateString('en-US', {
        month: 'long',
        day: 'numeric',
        year: 'numeric',
      })
    : null;

  const captureTimeText = captureDate
    ? captureDate.toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
        hour12: true,
      })
    : null;

  // Track keyboard visibility
  useEffect(() => {
    const showEvent = Platform.OS === 'ios' ? 'keyboardWillShow' : 'keyboardDidShow';
    const hideEvent = Platform.OS === 'ios' ? 'keyboardWillHide' : 'keyboardDidHide';
    const showSub = Keyboard.addListener(showEvent, (e) => {
      setIsKeyboardVisible(true);
      setKeyboardHeight(e.endCoordinates?.height ?? 0);
    });
    const hideSub = Keyboard.addListener(hideEvent, () => {
      setIsKeyboardVisible(false);
      setKeyboardHeight(0);
    });
    return () => {
      showSub.remove();
      hideSub.remove();
    };
  }, []);

  // Helper function to scroll to an input - ensure it always works
  const scrollToInput = (rowId?: string) => {
    // Single scroll attempt after keyboard has time to appear
    setTimeout(() => {
      let targetRef: View | null = null;
      
      if (rowId && rowRefs.current[rowId]) {
        targetRef = rowRefs.current[rowId];
      } else if (mealNameContainerRef.current) {
        targetRef = mealNameContainerRef.current;
      } else if (tableContainerRef.current) {
        targetRef = tableContainerRef.current;
      }
      
      if (targetRef && scrollViewRef.current) {
        (targetRef as any).measureLayout(
          scrollViewRef.current as any,
          (x: number, y: number) => {
            // Scroll to show input with padding above
            scrollViewRef.current?.scrollTo({
              y: Math.max(0, y - 100),
              animated: true,
            });
          },
          () => {
            // Fallback: scroll to end
            scrollViewRef.current?.scrollToEnd({ animated: true });
          }
        );
      } else if (scrollViewRef.current) {
        // Fallback: scroll to end if measureLayout fails
        scrollViewRef.current.scrollToEnd({ animated: true });
      }
    }, 300);
  };

  // Save changes to Redux store and backend
  const saveChanges = useCallback(async (): Promise<boolean> => {
    if (!item?.id || !user?.email) {
      console.warn('[MealDetail] Cannot save: missing item.id or user.email', { 
        hasItemId: !!item?.id, 
        hasUserEmail: !!user?.email 
      });
      return false;
    }

    try {
      setIsSaving(true);
      // Include full item data in updates to ensure entry can be recreated if missing
      const updates = {
        ...item, // Include all existing item data
        mealName,
        dishContents,
        nutritionalInfo: {
          ...item.nutritionalInfo,
          calories: totalCalories,
        },
      };
      
      console.log('[MealDetail] Saving changes for analysis:', item.id);
      
      await dispatch(updateAnalysis({
        userEmail: user.email,
        analysisId: item.id,
        updates,
      })).unwrap();
      
      console.log('[MealDetail] Changes saved successfully');
      return true;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.error('[MealDetail] Failed to save changes:', errorMessage, error);
      return false;
    } finally {
      setIsSaving(false);
    }
  }, [item, user?.email, mealName, dishContents, totalCalories, dispatch]);

  // Always keep the ref pointing at the latest saveChanges closure
  const saveChangesRef = useRef(saveChanges);
  useEffect(() => {
    saveChangesRef.current = saveChanges;
  }, [saveChanges]);

  // Auto-save whenever dishContents or mealName changes.
  // Skips the initial mount so we don't fire an unnecessary save on screen open.
  // Uses a ref so the effect deps stay stable and only [dishContents, mealName] retrigger it.
  const isFirstRenderRef = useRef(true);
  useEffect(() => {
    if (isFirstRenderRef.current) {
      isFirstRenderRef.current = false;
      return;
    }
    const timer = setTimeout(() => {
      saveChangesRef.current();
    }, 800);
    return () => clearTimeout(timer);
  }, [dishContents, mealName]);

  // Reset scroll position when returning to this screen
  useFocusEffect(
    useCallback(() => {
      scrollViewRef.current?.scrollTo({ y: 0, animated: false });
      return () => {};
    }, [])
  );

  const handleInputFocus = (rowId: string, field: string) => {
    focusedInputRef.current = { rowId, field };
  };

  const handleInputBlur = (rowId: string) => {
    // Exit edit mode only when clicking truly outside (handled by delay)
    // This allows switching between inputs in the same row
    setTimeout(() => {
      if (focusedInputRef.current?.rowId !== rowId) {
        setEditingRowId(null);
        focusedInputRef.current = null;
      }
    }, 300);
  };

  const handleEdit = (rowId: string) => {
    setEditingRowId(rowId);
  };

  const handleDelete = (rowId: string) => {
    Alert.alert(
      'Delete Item',
      'Are you sure you want to delete this item?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => {
            const updated = dishContents.filter(item => item.id !== rowId);
            setDishContents(updated);
            setEditingRowId(null);
          },
        },
      ]
    );
  };

  const handleUpdateRow = (rowId: string, field: 'name' | 'weight' | 'calories', value: string) => {
    setDishContents(prev =>
      prev.map(row => {
        if (row.id !== rowId) return row;
        if (field === 'weight') {
          const updated = { ...row, weight: value };
          const oldWeight = Number(row.weight);
          const oldCal = Number(row.calories);
          // Recalculate calories proportionally when weight changes (e.g. 100g @ 200 kcal → 200g → 400 kcal)
          if (Number.isFinite(oldWeight) && oldWeight > 0 && Number.isFinite(oldCal)) {
            const newWeight = Number(value);
            if (Number.isFinite(newWeight)) {
              updated.calories = String(Math.round((newWeight / oldWeight) * oldCal));
            }
          }
          return updated;
        }
        return { ...row, [field]: value };
      })
    );
  };

  const handleAddContent = () => {
    const newId = Date.now().toString();
    setDishContents(prev => [
      { id: newId, name: '', weight: '', calories: '' },
      ...prev,
    ]);
    setEditingRowId(newId);
  };

  // Handle swipe right to navigate to TutorialScreen
  const handleSwipeRight = () => {
    console.log('[MealDetail] Swipe right detected, navigating to Tutorial');
    try {
      navigation.navigate('Tutorial');
    } catch (error) {
      console.error('[MealDetail] Error navigating to Tutorial:', error);
    }
  };

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <StatusBar barStyle="dark-content" />

      <PanGestureHandler
        onGestureEvent={Animated.event(
          [{ nativeEvent: { translationX: screenSwipePosition.current } }],
          { useNativeDriver: true }
        )}
        onHandlerStateChange={(event) => {
          const { state, translationX } = event.nativeEvent;

          if (state === State.END) {
            // Trigger on right swipe (positive translationX)
            if (translationX > 80) {
              console.log('[MealDetail] Right swipe detected, navigating to Tutorial');
              handleSwipeRight();
            }
            screenSwipePosition.current.setValue(0);
          }

          if (state === State.CANCELLED || state === State.FAILED) {
            screenSwipePosition.current.setValue(0);
          }
        }}
        activeOffsetX={20}
        failOffsetX={-10}
        failOffsetY={[-10, 10]}
      >
        <Animated.View style={{ flex: 1 }}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={{ flex: 1 }}
        keyboardVerticalOffset={0}
      >
        <TouchableWithoutFeedback onPress={() => Keyboard.dismiss()}>
          <View style={{ flex: 1 }}>
            <AppHeader
              displayName={displayName}
              lastLoginDate={lastLoginDate}
              lastLoginTime={lastLoginTime}
              onProfilePress={() => {
                try {
                  navigation.navigate('Profile');
                } catch (error) {
                  console.error('[MealDetail] Error navigating to Profile:', error);
                }
              }}
            />

            <ScrollView 
            ref={scrollViewRef}
            style={styles.scrollView}
            contentContainerStyle={[styles.scrollContent, { paddingBottom: 100 }]}
            showsVerticalScrollIndicator={false}
            keyboardShouldPersistTaps="handled"
            keyboardDismissMode="on-drag"
            decelerationRate="normal"
            bounces={true}
            scrollEventThrottle={16}
            overScrollMode="never"
            nestedScrollEnabled={true}
          >
        {/* Media Preview */}
        <View style={styles.mediaContainer}>
          {(() => {
            const overlayUri = !overlayLoadFailed ? effectiveSegmentedImages?.overlay_urls?.[0]?.url : undefined;
            const videoOverlayUrl = (!overlayLoadFailed && !videoOverlayError)
              ? effectiveSegmentedImages?.video_overlay_url ?? null
              : null;
            const displayUri = overlayUri || resolvedImageUri || null;
            const showImageLoader = !isVideo && !!displayUri;
            // Use S3 URL only — never fall back to stale local file:// path on reinstall / new device
            const originalVideoUri = resolvedVideoUri ?? (item?.job_id ? null : item?.videoUri ?? null);
            const videoSource = isVideo ? (isVideoPlaying ? (videoOverlayUrl || originalVideoUri) : originalVideoUri) : null;
            return (
          <>
          {isVideo && videoSource ? (
            <>
              {/* Mount both videos simultaneously so the segmented one buffers in background.
                  Visibility toggled via opacity to avoid brown-screen loading flash on play. */}
              {videoOverlayUrl && (
                <Video
                  ref={overlayVideoRef}
                  key={videoOverlayUrl}
                  source={{ uri: videoOverlayUrl }}
                  style={[styles.media, { position: 'absolute', top: 0, left: 0, opacity: isVideoPlaying ? 1 : 0 }]}
                  resizeMode={ResizeMode.COVER}
                  isLooping={false}
                  isMuted={false}
                  shouldPlay={isVideoPlaying}
                  useNativeControls={false}
                  onPlaybackStatusUpdate={(status) => {
                    if (status.isLoaded && status.didJustFinish) {
                      setIsVideoPlaying(false);
                      overlayVideoRef.current?.setStatusAsync({ positionMillis: 0, shouldPlay: false });
                    }
                    if (!status.isLoaded && (status as any).error) {
                      setVideoOverlayError(true);
                    }
                  }}
                />
              )}
              <Video
                ref={originalVideoRef}
                key={originalVideoUri ?? 'original'}
                source={{ uri: originalVideoUri ?? '' }}
                style={[styles.media, { opacity: (isVideoPlaying && videoOverlayUrl && !videoOverlayError) ? 0 : 1 }]}
                resizeMode={ResizeMode.COVER}
                isLooping={false}
                isMuted={true}
                shouldPlay={isVideoPlaying && (!videoOverlayUrl || videoOverlayError)}
                useNativeControls={false}
                onPlaybackStatusUpdate={(status) => {
                  if (status.isLoaded && status.didJustFinish) {
                    setIsVideoPlaying(false);
                    originalVideoRef.current?.setStatusAsync({ positionMillis: 0, shouldPlay: false });
                  }
                }}
              />
              {!isVideoPlaying && (
                <TouchableOpacity
                  style={styles.playButtonOverlay}
                  onPress={handleVideoPlay}
                  activeOpacity={0.7}
                >
                  <View style={styles.playButton}>
                    <Ionicons name="play" size={28} color="#FFFFFF" />
                  </View>
                </TouchableOpacity>
              )}
              {isVideoPlaying && (
                <TouchableOpacity
                  style={styles.playButtonOverlay}
                  onPress={handleVideoPlay}
                  activeOpacity={0.7}
                >
                  <View style={styles.playButton}>
                    <Ionicons name="pause" size={28} color="#FFFFFF" />
                  </View>
                </TouchableOpacity>
              )}
            </>
          ) : (
            // Show segmented overlay when available; on load error refetch by job_id or show original image
            displayUri ? (
              <TouchableOpacity
                style={styles.mediaTouchable}
                activeOpacity={1}
                onPress={() => {
                  setFullImageUri(displayUri);
                  setShowFullImageModal(true);
                }}
              >
                <OptimizedImage
                  source={{ uri: displayUri }}
                  style={styles.media}
                  resizeMode="cover"
                  cachePolicy="memory-disk"
                  priority="normal"
                  onImageLoad={() => setMediaLoading(false)}
                  onError={() => { setMediaLoading(false); setOverlayLoadFailed(true); }}
                />
              </TouchableOpacity>
            ) : (
              <View style={[styles.media, styles.placeholder]} />
            )
          )}
          {showImageLoader && mediaLoading && (
            <View style={[StyleSheet.absoluteFill, styles.mediaLoader]} pointerEvents="none">
              <ActivityIndicator size="large" color="#7BA21B" />
            </View>
          )}
          </>
            );
          })()}
          
          {/* Back Button Overlay */}
          <View style={styles.backButtonOverlay} pointerEvents="box-none">
            <View style={styles.backButtonBackground}>
              <VectorBackButtonCircle
                onPress={() => navigation.goBack()}
                size={24}
              />
            </View>
          </View>
        </View>
        
        {/* Segmentation Overlay Modal */}
        <Modal
          visible={showSegmentationOverlay}
          transparent={true}
          animationType="fade"
          onRequestClose={() => setShowSegmentationOverlay(false)}
        >
          <View style={styles.modalContainer}>
            <View style={styles.modalContent}>
              <View style={styles.modalHeader}>
                <Text style={styles.modalTitle}>
                  {overlayImageUrl === resolvedImageUri ? 'Original Image' : 'Segmentation Overlay'}
                </Text>
                <TouchableOpacity
                  onPress={() => setShowSegmentationOverlay(false)}
                  style={styles.closeButton}
                >
                  <Ionicons name="close" size={24} color="#000" />
                </TouchableOpacity>
              </View>
              {overlayImageUrl && (
                <Image
                  source={{ uri: overlayImageUrl }}
                  style={styles.overlayImage}
                  resizeMode="contain"
                />
              )}
            </View>
          </View>
        </Modal>

        {/* Full-screen image modal */}
        <Modal
          visible={showFullImageModal}
          transparent
          animationType="fade"
          onRequestClose={() => setShowFullImageModal(false)}
        >
          <TouchableOpacity
            style={styles.fullImageModalBackdrop}
            activeOpacity={1}
            onPress={() => setShowFullImageModal(false)}
          >
            <View style={styles.fullImageModalContent} pointerEvents="box-none">
              <TouchableOpacity
                style={styles.fullImageCloseButton}
                onPress={() => setShowFullImageModal(false)}
                hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
              >
                <Ionicons name="close" size={28} color="#FFFFFF" />
              </TouchableOpacity>
              {fullImageUri ? (
                <OptimizedImage
                  source={{ uri: fullImageUri }}
                  style={styles.fullImage}
                  resizeMode="contain"
                  cachePolicy="memory-disk"
                  priority="high"
                />
              ) : null}
            </View>
          </TouchableOpacity>
        </Modal>

        {/* Meal Info */}
        <View style={styles.mealInfo}>
        <View style={styles.mealHeader}>
          <View ref={mealNameContainerRef} style={{ flex: 1 }}>
            {editingMealName ? (
              <TextInput
                style={[styles.mealNameInput, styles.inputFocused]}
                value={mealName}
                onChangeText={setMealName}
                onBlur={() => setEditingMealName(false)}
                onFocus={() => scrollToInput()}
                placeholder="Meal name"
                placeholderTextColor="#D1D5DB"
                autoFocus
              />
            ) : (
              <TouchableOpacity onPress={() => setEditingMealName(true)}>
                <Text style={styles.mealName}>{mealName}</Text>
              </TouchableOpacity>
            )}
            <Text style={styles.mealCalories}>{dishContents.length === 0 ? '-' : `${totalCalories} Kcal`}</Text>
          </View>

          <View style={styles.mealActions}>
            {(() => {
              const canAdd = !dishContents.some(r => !r.name.trim() || !r.calories.trim());
              return (
                <TouchableOpacity
                  style={[styles.addButton, !canAdd && styles.addButtonDisabled]}
                  onPress={() => canAdd && handleAddContent()}
                  activeOpacity={canAdd ? 0.8 : 1}
                >
                  <View style={styles.addButtonIcon}>
                    <Text style={styles.addButtonIconText}>+</Text>
                  </View>
                  <Text style={styles.addButtonText}>Add Ingredient</Text>
                </TouchableOpacity>
              );
            })()}
            <View style={styles.captureInfo}>
              <Text style={styles.captureValue}>
                {captureDateText && captureTimeText
                  ? `${captureDateText}, ${captureTimeText}`
                  : 'Unavailable'}
              </Text>
            </View>
          </View>
        </View>
      </View>

        {/* Dish Contents Table */}
        <Pressable 
          ref={tableContainerRef}
          style={styles.tableContainer}
          onPress={() => {
            if (editingRowId) {
              setEditingRowId(null);
              focusedInputRef.current = null;
            }
          }}
        >
          {/* Table Header */}
          <View style={styles.tableHeader}>
            <Text style={[styles.tableHeaderText, { flex: 2 }]}>Dish Contents</Text>
            <Text style={[styles.tableHeaderText, { flex: 1 }]}>Weight</Text>
            <Text style={[styles.tableHeaderText, { flex: 1 }]}>Kcal</Text>
            <Text style={[styles.tableHeaderText, { width: 50 }]}>Action</Text>
          </View>

          {/* Table Rows */}
          {dishContents.map((row) => {
            const isEditing = editingRowId === row.id;
            return (
              <View 
                key={row.id} 
                ref={(ref) => { rowRefs.current[row.id] = ref; }}
                style={styles.tableRow}
              >
                <View style={[styles.tableCell, { flex: 2 }]} pointerEvents="box-none">
                  {isEditing ? (
                    <TextInput
                      ref={(ref) => {
                        if (!inputRefs.current[row.id]) {
                          inputRefs.current[row.id] = {};
                        }
                        inputRefs.current[row.id].name = ref || undefined;
                      }}
                      style={[styles.tableInput, styles.inputFocused]}
                      value={row.name}
                      onChangeText={(value) => handleUpdateRow(row.id, 'name', value)}
                      onFocus={() => {
                        handleInputFocus(row.id, 'name');
                        scrollToInput(row.id);
                      }}
                      placeholder="Ingredient name"
                      placeholderTextColor="#D1D5DB"
                      keyboardType="default"
                      editable={true}
                      blurOnSubmit={false}
                      returnKeyType="next"
                      onSubmitEditing={() => {
                        // Focus the weight input when "next" is pressed
                        setTimeout(() => {
                          inputRefs.current[row.id]?.weight?.focus();
                        }, 100);
                      }}
                      pointerEvents="auto"
                    />
                  ) : (
                    <Text style={styles.tableCellText}>{row.name || '—'}</Text>
                  )}
                </View>
                <View style={[styles.tableCell, { flex: 1 }]} pointerEvents="box-none">
                  {isEditing ? (
                    <TextInput
                      ref={(ref) => {
                        if (!inputRefs.current[row.id]) {
                          inputRefs.current[row.id] = {};
                        }
                        inputRefs.current[row.id].weight = ref || undefined;
                      }}
                      style={[styles.tableInput, styles.inputFocused]}
                      value={row.weight}
                      onChangeText={(value) => handleUpdateRow(row.id, 'weight', value)}
                      onFocus={() => {
                        handleInputFocus(row.id, 'weight');
                        scrollToInput(row.id);
                      }}
                      onSubmitEditing={() => {
                        // Focus the calories input when "next" is pressed
                        setTimeout(() => {
                          inputRefs.current[row.id]?.calories?.focus();
                        }, 100);
                      }}
                      placeholder="Grams"
                      placeholderTextColor="#D1D5DB"
                      keyboardType={Platform.OS === 'ios' ? 'numbers-and-punctuation' : 'numeric'}
                      editable={true}
                      blurOnSubmit={false}
                      returnKeyType="next"
                      pointerEvents="auto"
                    />
                  ) : (
                    <Text style={styles.tableCellText}>{row.weight && row.weight !== '0' ? `${row.weight} g` : '—'}</Text>
                  )}
                </View>
                <View style={[styles.tableCell, { flex: 1 }]} pointerEvents="box-none">
                  {isEditing ? (
                    <TextInput
                      ref={(ref) => {
                        if (!inputRefs.current[row.id]) {
                          inputRefs.current[row.id] = {};
                        }
                        inputRefs.current[row.id].calories = ref || undefined;
                      }}
                      style={[styles.tableInput, styles.inputFocused]}
                      value={row.calories}
                      onChangeText={(value) => handleUpdateRow(row.id, 'calories', value)}
                      onFocus={() => {
                        handleInputFocus(row.id, 'calories');
                        scrollToInput(row.id);
                      }}
                      onSubmitEditing={() => {
                        Keyboard.dismiss();
                        setEditingRowId(null);
                        focusedInputRef.current = null;
                      }}
                      placeholder="KCal"
                      placeholderTextColor="#D1D5DB"
                      keyboardType={Platform.OS === 'ios' ? 'numbers-and-punctuation' : 'numeric'}
                      editable={true}
                      blurOnSubmit={true}
                      returnKeyType="done"
                      pointerEvents="auto"
                    />
                  ) : (
                    <Text style={styles.tableCellText}>{row.calories || '—'}</Text>
                  )}
                </View>
                <View style={[styles.tableCell, { width: 50, alignItems: 'center' }]}>
                  <TouchableOpacity
                    onPress={() => {
                      if (isEditing) {
                        handleDelete(row.id);
                      } else {
                        handleEdit(row.id);
                      }
                    }}
                    style={styles.actionButton}
                  >
                    <Ionicons
                      name={isEditing ? 'trash' : 'pencil'}
                      size={20}
                      color="#7BA21B"
                    />
                  </TouchableOpacity>
                </View>
              </View>
            );
          })}
        </Pressable>
            </ScrollView>
          </View>
        </TouchableWithoutFeedback>
      </KeyboardAvoidingView>

      {/* Next Button - Fixed at Bottom */}
      <BottomButtonContainer>
        {(() => {
          const hasIncompleteRow = dishContents.some(r => !r.name.trim() || !r.calories.trim());
          const canProceed = !hasIncompleteRow;
          return (
        <TouchableOpacity
          style={[styles.nextButton, !canProceed && styles.nextButtonDisabled]}
          onPress={() => {
            if (!canProceed) return;

            setEditingRowId(null);
            Keyboard.dismiss();

            // Navigate immediately with latest data; save in the background
            const updatedItem = {
              ...item,
              mealName,
              dishContents,
              nutritionalInfo: {
                ...item.nutritionalInfo,
                calories: totalCalories,
              },
            };
            (navigation as any).navigate('Feedback', { item: updatedItem });
            saveChanges();
          }}
          disabled={!canProceed}
        >
          <Text style={styles.nextButtonText}>Next</Text>
        </TouchableOpacity>
          );
        })()}
      </BottomButtonContainer>
        </Animated.View>
      </PanGestureHandler>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: 20,
  },
  mediaContainer: {
    width: '100%',
    height: 250,
    backgroundColor: '#F3F4F6',
    position: 'relative',
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
  },
  mediaLoader: {
    backgroundColor: '#F3F4F6',
    justifyContent: 'center',
    alignItems: 'center',
  },
  mediaTouchable: {
    width: '100%',
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
  },
  media: {
    width: '100%',
    height: '100%',
  },
  placeholder: {
    backgroundColor: '#111827',
  },
  backButtonOverlay: {
    position: 'absolute',
    top: 12,
    left: 12,
    zIndex: 10,
  },
  backButtonBackground: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    width: 24,
    height: 24,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  playButtonOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    zIndex: 5,
  },
  playButton: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  mealInfo: {
    paddingHorizontal: 16,
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
  },
  mealHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  mealName: {
    fontSize: 18,
    fontWeight: '700',
    color: '#6B7280',
    marginBottom: 4,
  },
  mealNameInput: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1F2937',
    marginBottom: 4,
    borderWidth: 2,
    borderColor: '#E5E7EB',
    borderRadius: 4,
    paddingHorizontal: 12,
    paddingVertical: 8,
    backgroundColor: '#FFFFFF',
  },
  mealCalories: {
    fontSize: 16,
    fontWeight: '600',
    color: '#6B7280',
  },
  addButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: '#7BA21B',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 24,
  },
  addButtonDisabled: {
    backgroundColor: '#B5D068',
    opacity: 0.6,
  },
  addButtonIcon: {
    width: 17,
    height: 17,
    borderRadius: 9,
    backgroundColor: '#FFFFFF',
    alignItems: 'center',
    justifyContent: 'center',
  },
  addButtonIconText: {
    color: '#7BA21B',
    fontSize: 16,
    fontWeight: '700',
    lineHeight: 16,
    textAlign: 'center',
  },
  addButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
  },
  mealActions: {
    alignItems: 'flex-end',
  },
  captureInfo: {
    alignItems: 'flex-end',
    marginTop: 4,
  },
  captureLabel: {
    fontSize: 10,
    color: '#9CA3AF',
    textTransform: 'uppercase',
  },
  captureValue: {
    fontSize: 11,
    color: '#6B7280',
    fontWeight: '400',
  },
  tableContainer: {
    paddingHorizontal: 16,
    paddingTop: 16,
  },
  tableHeader: {
    flexDirection: 'row',
    paddingVertical: 12,
    borderBottomWidth: 2,
    borderBottomColor: '#E5E7EB',
    marginBottom: 4,
  },
  tableHeaderText: {
    fontSize: 12,
    fontWeight: '700',
    color: '#6B7280',
    textTransform: 'uppercase',
  },
  tableRow: {
    flexDirection: 'row',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
    alignItems: 'center',
  },
  tableCell: {
    paddingHorizontal: 8,
    justifyContent: 'center',
  },
  tableCellText: {
    fontSize: 14,
    color: '#1F2937',
  },
  tableInput: {
    height: 40,
    borderWidth: 1,
    borderColor: '#E5E7EB',
    borderRadius: 4,
    paddingHorizontal: 6,
    fontSize: 12,
    color: '#1F2937',
    backgroundColor: '#FFFFFF',
  },
  inputFocused: {
    borderWidth: 2,
    borderColor: '#7BA21B',
  },
  actionButton: {
    width: 36,
    height: 36,
    justifyContent: 'center',
    alignItems: 'center',
  },
  nextButton: {
    height: 56, // Fixed height
    width: '100%', // Fixed width
    backgroundColor: '#7BA21B',
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
  },
  nextButtonDisabled: {
    backgroundColor: '#9CA3AF',
    opacity: 0.6,
  },
  nextButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
  },
  segmentationButton: {
    position: 'absolute',
    bottom: 16,
    right: 16,
    backgroundColor: 'rgba(123, 162, 27, 0.9)',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 20,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  segmentationButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
  },
  modalContainer: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    width: '95%',
    maxHeight: '90%',
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 16,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1F2937',
  },
  closeButton: {
    padding: 4,
  },
  overlayImage: {
    width: '100%',
    height: 400,
    borderRadius: 8,
  },
  fullImageModalBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.95)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  fullImageModalContent: {
    width: Dimensions.get('window').width,
    height: Dimensions.get('window').height,
    justifyContent: 'center',
    alignItems: 'center',
  },
  fullImageCloseButton: {
    position: 'absolute',
    top: Platform.OS === 'ios' ? 50 : 40,
    right: 20,
    zIndex: 10,
    padding: 8,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    borderRadius: 24,
  },
  fullImageWrapper: {
    width: '100%',
    flex: 1,
  },
  fullImage: {
    width: Dimensions.get('window').width,
    height: Dimensions.get('window').height,
    backgroundColor: 'transparent',
  },
});

