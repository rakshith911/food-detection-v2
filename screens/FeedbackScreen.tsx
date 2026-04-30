import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  TextInput,
  StatusBar,
  Alert,
  ScrollView,
  Platform,
  Image,
  KeyboardAvoidingView,
  ActivityIndicator,
  Modal,
  Animated,
} from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import { Video, ResizeMode } from 'expo-av';
import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import CubeIcon from '../components/CubeIcon';
import { useNavigation, useRoute } from '@react-navigation/native';
import { useAppSelector, useAppDispatch } from '../store/hooks';
import type { AnalysisEntry, SegmentedImages } from '../store/slices/historySlice';
import { updateAnalysis } from '../store/slices/historySlice';
import { nutritionAnalysisAPI } from '../services/NutritionAnalysisAPI';
import { feedbackAPI } from '../services/FeedbackAPI';
import OptimizedImage from '../components/OptimizedImage';
import VectorBackButtonCircle from '../components/VectorBackButtonCircle';
import AppHeader from '../components/AppHeader';
import BottomButtonContainer from '../components/BottomButtonContainer';
import { toSentenceCase } from '../utils/textCase';
import { getImagePresignedUrl } from '../services/S3UserDataService';

const TRELLIS_PREVIEW_LOOP_MS = 4000;

interface StarRatingProps {
  rating: number;
  onRatingChange: (rating: number) => void;
}

const StarRating: React.FC<StarRatingProps> = ({ rating, onRatingChange }) => {
  return (
    <View style={styles.starContainer}>
      {[1, 2, 3, 4, 5].map((star) => (
        <TouchableOpacity
          key={star}
          onPress={() => onRatingChange(star)}
          activeOpacity={0.7}
        >
          <Ionicons
            name={star <= rating ? 'star' : 'star-outline'}
            size={24}
            color={star <= rating ? '#7BA21B' : '#D1D5DB'}
          />
        </TouchableOpacity>
      ))}
    </View>
  );
};

export default function FeedbackScreen() {
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const route = useRoute();
  const user = useAppSelector((state) => state.auth.user);
  const businessProfile = useAppSelector((state) => state.profile.businessProfile);
  const dispatch = useAppDispatch();
  const item = (route.params as any)?.item as AnalysisEntry;

  const [isVideoPlaying, setIsVideoPlaying] = useState(false);
  const trellisVideoRef = useRef<Video>(null);
  const fullScreenVideoRef = useRef<Video>(null);
  const loopFadeAnim = useRef(new Animated.Value(0)).current;
  const loopFadeStarted = useRef(false);
  const fullScreenFadeAnim = useRef(new Animated.Value(0)).current;
  const fullScreenFadeStarted = useRef(false);

  const handleVideoPlay = useCallback(() => {
    setIsVideoPlaying((prev) => !prev);
  }, []);

  const loopTrellisPreviewAtFourSeconds = useCallback((status: any) => {
    if (!status?.isLoaded) return;
    if (status.positionMillis >= TRELLIS_PREVIEW_LOOP_MS - 350 && !loopFadeStarted.current) {
      loopFadeStarted.current = true;
      Animated.timing(loopFadeAnim, { toValue: 1, duration: 350, useNativeDriver: true }).start();
    }
    if (status.positionMillis >= TRELLIS_PREVIEW_LOOP_MS) {
      trellisVideoRef.current?.setStatusAsync({ positionMillis: 0, shouldPlay: true });
      loopFadeStarted.current = false;
      Animated.timing(loopFadeAnim, { toValue: 0, duration: 350, useNativeDriver: true }).start();
    }
  }, [loopFadeAnim]);

  // Initialize state from existing feedback if available
  const [ratings, setRatings] = useState(
    item?.feedback?.ratings || {
      foodDishIdentification: 3,
      dishContentsIdentification: 3,
      massEstimation: 3,
      calorieEstimation: 3,
      overall: 3,
    }
  );
  const [comment, setComment] = useState(item?.feedback?.comment || '');
  const [isSaving, setIsSaving] = useState(false);
  const [isCommentFocused, setIsCommentFocused] = useState(false);
  const commentInputRef = useRef<TextInput>(null);
  const scrollViewRef = useRef<ScrollView>(null);
  const [showFullMediaModal, setShowFullMediaModal] = useState(false);
  const [fullMediaUri, setFullMediaUri] = useState<string | null>(null);
  const [fullMediaType, setFullMediaType] = useState<'image' | 'video'>('image');
  const [overlayLoadFailed, setOverlayLoadFailed] = useState(false);
  const [refreshedSegmentedImages, setRefreshedSegmentedImages] = useState<SegmentedImages | null>(null);
  const [refreshingOverlay, setRefreshingOverlay] = useState(false);
  const [mediaLoading, setMediaLoading] = useState(true);
  const [refreshedTrellisMP4Url, setRefreshedTrellisMP4Url] = useState<string | null>(item?.trellis_mp4_url ?? null);
  const [resolvedVideoUri, setResolvedVideoUri] = useState<string | undefined>(item?.videoUri);
  const [resolvingVideoUri, setResolvingVideoUri] = useState<boolean>(!!item?.job_id && !!item?.videoUri);

  const effectiveSegmentedImages = refreshedSegmentedImages ?? item?.segmented_images;

  const [selectedDepthIngredient, setSelectedDepthIngredient] = useState<string | null>(null);

  const geminiDepthFull = useMemo(() =>
    effectiveSegmentedImages?.overlay_urls?.find((a: any) => a?.name === 'gemini_depth_full')?.url || null,
    [effectiveSegmentedImages?.overlay_urls]
  );

  const taggedOverlayUri = useMemo(() =>
    effectiveSegmentedImages?.overlay_urls?.find((a: any) => a?.name === 'tagged')?.url || null,
    [effectiveSegmentedImages?.overlay_urls]
  );

  const selectedDepthUri = useMemo(() => {
    if (!selectedDepthIngredient) return null;
    if (selectedDepthIngredient === '__full__') return geminiDepthFull;
    if (selectedDepthIngredient === '__tagged__') return taggedOverlayUri;
    return null;
  }, [selectedDepthIngredient, geminiDepthFull, taggedOverlayUri]);

  useEffect(() => {
    setOverlayLoadFailed(false);
    setRefreshedSegmentedImages(null);
    setMediaLoading(true);
    setRefreshedTrellisMP4Url(item?.trellis_mp4_url ?? null);
    setResolvedVideoUri(item?.videoUri);
  }, [item?.id]);

  // Resolve video URI via presigned S3 URL (same as MealDetailScreen)
  useEffect(() => {
    if (!item?.job_id) { setResolvingVideoUri(false); return; }
    setResolvingVideoUri(true);
    getImagePresignedUrl(item.job_id)
      .then(url => { setResolvedVideoUri(url || item?.videoUri); })
      .catch(() => { setResolvedVideoUri(item?.videoUri); })
      .finally(() => setResolvingVideoUri(false));
  }, [item?.id, item?.job_id]);

  useEffect(() => {
    setMediaLoading(true);
  }, [effectiveSegmentedImages?.overlay_urls?.[0]?.url]);

  // Always re-fetch on open to get fresh presigned URLs (they expire in 1 hour)
  useEffect(() => {
    if (!item?.job_id || !user?.email || refreshingOverlay) return;
    let cancelled = false;
    (async () => {
      setRefreshingOverlay(true);
      try {
        const fresh = await nutritionAnalysisAPI.getResults(item.job_id!, true, true);
        if (cancelled) return;
        if (fresh?.segmented_images?.overlay_urls?.length || fresh?.segmented_images?.video_overlay_url) {
          setRefreshedSegmentedImages(fresh.segmented_images);
          setOverlayLoadFailed(false);
        }
        if (fresh?.trellis_mp4_url) {
          setRefreshedTrellisMP4Url(fresh.trellis_mp4_url);
        }
        const updates: Record<string, any> = {};
        if (fresh?.segmented_images) updates.segmented_images = fresh.segmented_images;
        if (fresh?.trellis_mp4_url) updates.trellis_mp4_url = fresh.trellis_mp4_url;
        if (Object.keys(updates).length) {
          await dispatch(updateAnalysis({
            userEmail: user.email,
            analysisId: item.id,
            updates,
          })).unwrap();
        }
      } catch {
        if (!cancelled && !effectiveSegmentedImages?.overlay_urls?.length) setOverlayLoadFailed(true);
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
        const fresh = await nutritionAnalysisAPI.getResults(item.job_id, true, true);
        if (fresh?.segmented_images?.overlay_urls?.length) {
          setRefreshedSegmentedImages(fresh.segmented_images);
          if (fresh?.trellis_mp4_url) setRefreshedTrellisMP4Url(fresh.trellis_mp4_url);
          await dispatch(updateAnalysis({
            userEmail: user.email,
            analysisId: item.id,
            updates: {
              segmented_images: fresh.segmented_images,
              ...(fresh.trellis_mp4_url ? { trellis_mp4_url: fresh.trellis_mp4_url } : {}),
            },
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

  const openFullScreenMedia = useCallback((uri: string | null | undefined, type: 'image' | 'video') => {
    if (!uri) return;
    setFullMediaUri(uri);
    setFullMediaType(type);
    setShowFullMediaModal(true);
  }, []);

  if (!item) {
    return (
      <SafeAreaView style={styles.container} edges={['top']}>
        <StatusBar barStyle="dark-content" />
        <View style={styles.emptyState}>
          <VectorBackButtonCircle size={24} onPress={() => navigation.goBack()} />
          <Text>No meal data available</Text>
        </View>
      </SafeAreaView>
    );
  }

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

  // Format capture date and time from item.timestamp
  const captureDate = item?.timestamp
    ? (() => {
        try {
          let date: Date;
          if (typeof item.timestamp === 'string') {
            date = new Date(item.timestamp);
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


  const handleSave = async () => {
    if (!user?.email || !item?.id) {
      Alert.alert('Error', 'User not authenticated or item not found');
      return;
    }

    setIsSaving(true);
    try {
      const feedback = {
        ratings,
        comment: comment.trim(),
        timestamp: new Date().toISOString(),
      };

      // Save feedback to the analysis entry via Redux
      await dispatch(updateAnalysis({
        userEmail: user.email,
        analysisId: item.id,
        updates: {
          ...item,
          feedback,
        },
      })).unwrap();

      // Also save to AsyncStorage for backward compatibility
      await feedbackAPI.saveFeedback(user.email, {
        analysisId: item.id,
        ...feedback,
      });
      
      console.log('[Feedback] Feedback saved successfully');
      
      Alert.alert(
        'Success',
        'Thank you for your feedback!',
        [
          {
            text: 'OK',
            onPress: () => {
              // Navigate to Results (cards page)
              (navigation as any).navigate('Results');
            },
          },
        ]
      );
    } catch (error) {
      console.error('[Feedback] Error saving feedback:', error);
      Alert.alert('Error', 'An error occurred while saving feedback. Please try again.');
    } finally {
      setIsSaving(false);
    }
  };

  const isVideo = !!item.videoUri;

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <StatusBar barStyle="dark-content" />

      {/* Full-screen media modal */}
      <Modal
        visible={showFullMediaModal}
        transparent
        animationType="fade"
        onRequestClose={() => setShowFullMediaModal(false)}
      >
        <View
          style={styles.fullImageModalBackdrop}
        >
          <View style={styles.fullImageModalContent}>
            <TouchableOpacity
              style={styles.fullImageCloseButton}
              onPress={() => setShowFullMediaModal(false)}
              hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
            >
              <Ionicons name="close" size={28} color="#FFFFFF" />
            </TouchableOpacity>
            {fullMediaUri && fullMediaType === 'video' ? (
              <>
                <Video
                  ref={fullScreenVideoRef}
                  source={{ uri: fullMediaUri }}
                  style={styles.fullImage}
                  resizeMode={ResizeMode.CONTAIN}
                  shouldPlay
                  useNativeControls
                  isLooping={false}
                  onPlaybackStatusUpdate={(status) => {
                    if (!status.isLoaded) return;
                    if (status.positionMillis >= TRELLIS_PREVIEW_LOOP_MS - 350 && !fullScreenFadeStarted.current) {
                      fullScreenFadeStarted.current = true;
                      Animated.timing(fullScreenFadeAnim, { toValue: 1, duration: 350, useNativeDriver: true }).start();
                    }
                    if (status.positionMillis >= TRELLIS_PREVIEW_LOOP_MS) {
                      fullScreenVideoRef.current?.setStatusAsync({ positionMillis: 0, shouldPlay: true });
                      fullScreenFadeStarted.current = false;
                      Animated.timing(fullScreenFadeAnim, { toValue: 0, duration: 350, useNativeDriver: true }).start();
                    }
                  }}
                />
                <Animated.View
                  style={[StyleSheet.absoluteFill, { backgroundColor: '#000000', opacity: fullScreenFadeAnim }]}
                  pointerEvents="none"
                />
              </>
            ) : fullMediaUri ? (
              <Image
                source={{ uri: fullMediaUri }}
                style={styles.fullImage}
                resizeMode="contain"
              />
            ) : null}
          </View>
        </View>
      </Modal>

      {Platform.OS === 'ios' ? (
        <KeyboardAvoidingView
          behavior="padding"
          style={{ flex: 1 }}
          keyboardVerticalOffset={insets.top}
        >
          <AppHeader
            displayName={displayName}
            lastLoginDate={lastLoginDate}

            onProfilePress={() => navigation.navigate('Profile' as never)}
          />
          <ScrollView
            ref={scrollViewRef}
            style={styles.scrollView}
            contentContainerStyle={[styles.scrollContent, { paddingBottom: 5 }]}
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
          <View style={styles.mediaBackdrop} pointerEvents="none" />
          {isVideo && (resolvedVideoUri || resolvingVideoUri) ? (
            <>
              <Video
                source={{ uri: resolvedVideoUri || item.videoUri || '' }}
                style={styles.media}
                resizeMode={ResizeMode.CONTAIN}
                isLooping={false}
                isMuted={false}
                shouldPlay={isVideoPlaying}
                useNativeControls={false}
                onPlaybackStatusUpdate={(status) => {
                  if (status.isLoaded && status.didJustFinish) {
                    setIsVideoPlaying(false);
                  }
                  }}
              />
              <TouchableOpacity
                style={styles.mediaTapTarget}
                onPress={() => openFullScreenMedia(resolvedVideoUri ?? item.videoUri, 'video')}
                activeOpacity={1}
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
            (() => {
              const trellisMP4Url = refreshedTrellisMP4Url;
              const depthUri = !overlayLoadFailed ? selectedDepthUri : null;
              const overlayUri = effectiveSegmentedImages?.overlay_urls?.[0]?.url;
              const displayUri = depthUri || ((!overlayLoadFailed && overlayUri) ? overlayUri : item.imageUri || null);
              const videoThumbnailUri = item.imageUri || overlayUri || null;
              const showImageLoader = !isVideo && (!!depthUri || !trellisMP4Url) && !!displayUri;
              if (depthUri) {
                return (
                  <TouchableOpacity style={styles.mediaTouchable} activeOpacity={1} onPress={() => openFullScreenMedia(depthUri, 'image')}>
                    <OptimizedImage
                      source={{ uri: depthUri }}
                      style={styles.media}
                      resizeMode="contain"
                      cachePolicy="memory-disk"
                      priority="high"
                      onImageLoad={() => setMediaLoading(false)}
                    />
                  </TouchableOpacity>
                );
              }
              if (trellisMP4Url) {
                return (
                  <>
                    <Video
                      ref={trellisVideoRef}
                      key={trellisMP4Url}
                      source={{ uri: trellisMP4Url }}
                      style={[styles.media, { position: 'absolute', top: 0, left: 0, opacity: isVideoPlaying ? 1 : 0 }]}
                      resizeMode={ResizeMode.CONTAIN}
                      isLooping
                      isMuted
                      shouldPlay={isVideoPlaying}
                      useNativeControls={false}
                      progressUpdateIntervalMillis={100}
                      onPlaybackStatusUpdate={loopTrellisPreviewAtFourSeconds}
                    />
                    {videoThumbnailUri && (
                      <OptimizedImage
                        source={{ uri: videoThumbnailUri }}
                        style={[styles.media, StyleSheet.absoluteFillObject, { opacity: isVideoPlaying ? 0 : 1 }]}
                        resizeMode="contain"
                        cachePolicy="memory-disk"
                        priority="normal"
                        onImageLoad={() => setMediaLoading(false)}
                      />
                    )}
                    <TouchableOpacity
                      style={styles.mediaTapTarget}
                      onPress={() => openFullScreenMedia(isVideoPlaying ? trellisMP4Url : videoThumbnailUri, isVideoPlaying ? 'video' : 'image')}
                      activeOpacity={1}
                    />
                    <Animated.View
                      style={[StyleSheet.absoluteFill, { backgroundColor: '#000000', opacity: loopFadeAnim, zIndex: 6 }]}
                      pointerEvents="none"
                    />
                    <TouchableOpacity style={styles.playButtonOverlay} onPress={handleVideoPlay} activeOpacity={0.7}>
                      <View style={styles.playButton}>
                        <Ionicons name={isVideoPlaying ? 'pause' : 'play'} size={28} color="#FFFFFF" />
                      </View>
                    </TouchableOpacity>
                  </>
                );
              }
              if (displayUri) {
                return (
                  <TouchableOpacity activeOpacity={1} onPress={() => openFullScreenMedia(displayUri, 'image')} style={styles.mediaTouchable}>
                    <OptimizedImage
                      source={{ uri: displayUri }}
                      style={styles.media}
                      resizeMode="contain"
                      cachePolicy="memory-disk"
                      priority="normal"
                      onImageLoad={() => setMediaLoading(false)}
                      onError={() => { setMediaLoading(false); setOverlayLoadFailed(true); }}
                    />
                    {showImageLoader && mediaLoading && (
                      <View style={[StyleSheet.absoluteFill, styles.mediaLoader]} pointerEvents="none">
                        <ActivityIndicator size="large" color="#7BA21B" />
                      </View>
                    )}
                  </TouchableOpacity>
                );
              }
              return <View style={[styles.media, styles.placeholder]} />;
            })()
          )}
          {/* Back Button Overlay */}
          <View style={styles.backButtonOverlay}>
            <View style={styles.backButtonBackground}>
              <VectorBackButtonCircle onPress={() => navigation.goBack()} size={24} />
            </View>
          </View>
        </View>

        {/* Meal Info */}
        <View style={styles.mealInfo}>
          {/* Row 1: meal name (wraps) | right: kcal + Write Comments */}
          <View style={styles.mealHeader}>
            <View style={{ flex: 1, marginRight: 12 }}>
              <Text style={styles.mealName}>{toSentenceCase(item.mealName || 'Burger')}</Text>
            </View>
            <View style={styles.mealActions}>
              <Text style={styles.mealCalories}>{item.nutritionalInfo?.calories ?? 0} Kcal</Text>
              <TouchableOpacity
                style={styles.writeCommentButton}
                onPress={() => {
                  scrollViewRef.current?.scrollToEnd({ animated: true });
                  setTimeout(() => {
                    commentInputRef.current?.focus();
                  }, 300);
                }}
                activeOpacity={0.7}
              >
                <Text style={styles.writeCommentButtonText}>Write Comments</Text>
              </TouchableOpacity>
              <View style={styles.captureInfo}>
                <Text style={styles.captureValue}>
                  {captureDateText && captureTimeText
                    ? `${captureDateText}, ${captureTimeText}`
                    : 'Unavailable'}
                </Text>
              </View>
            </View>
          </View>
          {/* Row 2: 3 view buttons with separators */}
          <View style={styles.mediaActionButtons}>
            <TouchableOpacity
              style={[styles.mediaActionButton, isVideoPlaying && styles.mediaActionButtonActive]}
              onPress={() => { setSelectedDepthIngredient(null); handleVideoPlay(); }}
              activeOpacity={0.7}
            >
              <CubeIcon size={18} color={isVideoPlaying ? '#FFFFFF' : '#7BA21B'} />
            </TouchableOpacity>
            <View style={styles.buttonSeparator} />
            <TouchableOpacity
              style={[styles.mediaActionButton, selectedDepthIngredient === '__full__' && styles.mediaActionButtonActive]}
              onPress={() => { setIsVideoPlaying(false); setSelectedDepthIngredient(selectedDepthIngredient === '__full__' ? null : '__full__'); }}
              activeOpacity={0.7}
            >
              <MaterialCommunityIcons name="image-filter-hdr" size={18} color={selectedDepthIngredient === '__full__' ? '#FFFFFF' : '#6B7280'} />
            </TouchableOpacity>
            <View style={styles.buttonSeparator} />
            <TouchableOpacity
              style={[styles.mediaActionButton, selectedDepthIngredient === '__tagged__' && styles.mediaActionButtonActive]}
              onPress={() => { setIsVideoPlaying(false); setSelectedDepthIngredient(selectedDepthIngredient === '__tagged__' ? null : '__tagged__'); }}
              activeOpacity={0.7}
            >
              <MaterialCommunityIcons name="selection-ellipse" size={18} color={selectedDepthIngredient === '__tagged__' ? '#FFFFFF' : '#6B7280'} />
            </TouchableOpacity>
          </View>
        </View>

        {/* Feedback Section */}
        <View style={styles.feedbackSection}>
          <Text style={styles.feedbackTitle}>Your feedback is valuable to us!</Text>

          {/* Food dish identification */}
          <View style={styles.ratingRow}>
            <Text style={styles.ratingLabel}>Food dish identification</Text>
            <StarRating
              rating={ratings.foodDishIdentification}
              onRatingChange={(rating) =>
                setRatings((prev) => ({ ...prev, foodDishIdentification: rating }))
              }
            />
          </View>

          {/* Dish contents identification */}
          <View style={styles.ratingRow}>
            <Text style={styles.ratingLabel}>Dish contents identification</Text>
            <StarRating
              rating={ratings.dishContentsIdentification}
              onRatingChange={(rating) =>
                setRatings((prev) => ({ ...prev, dishContentsIdentification: rating }))
              }
            />
          </View>

          {/* Mass estimation */}
          <View style={styles.ratingRow}>
            <Text style={styles.ratingLabel}>Mass estimation</Text>
            <StarRating
              rating={ratings.massEstimation}
              onRatingChange={(rating) =>
                setRatings((prev) => ({ ...prev, massEstimation: rating }))
              }
            />
          </View>

          {/* Calorie estimation */}
          <View style={styles.ratingRow}>
            <Text style={styles.ratingLabel}>Calorie estimation</Text>
            <StarRating
              rating={ratings.calorieEstimation}
              onRatingChange={(rating) =>
                setRatings((prev) => ({ ...prev, calorieEstimation: rating }))
              }
            />
          </View>

          {/* Overall */}
          <View style={styles.ratingRow}>
            <Text style={styles.ratingLabel}>Overall</Text>
            <StarRating
              rating={ratings.overall}
              onRatingChange={(rating) =>
                setRatings((prev) => ({ ...prev, overall: rating }))
              }
            />
          </View>

          {/* Comment Section */}
          <View style={styles.commentSection}>
            <TextInput
              ref={commentInputRef}
              style={[styles.commentInput, isCommentFocused && styles.commentInputFocused]}
              placeholder="Anything you would like to tell us? (e.g., wrong item, portion too high, etc.)"
              placeholderTextColor="#9CA3AF"
              value={comment}
              onChangeText={setComment}
              multiline
              numberOfLines={4}
              textAlignVertical="top"
              onFocus={() => {
                setIsCommentFocused(true);
                setTimeout(() => {
                  scrollViewRef.current?.scrollToEnd({ animated: true });
                }, 300);
              }}
              onBlur={() => setIsCommentFocused(false)}
            />
          </View>
        </View>
        </ScrollView>

        </KeyboardAvoidingView>
      ) : (
        <View style={{ flex: 1 }}>
          <AppHeader
            displayName={displayName}
            lastLoginDate={lastLoginDate}

            onProfilePress={() => navigation.navigate('Profile' as never)}
          />
          <ScrollView
            ref={scrollViewRef}
            style={styles.scrollView}
            contentContainerStyle={[styles.scrollContent, { paddingBottom: 5 }]}
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
              <View style={styles.mediaBackdrop} pointerEvents="none" />
              {isVideo && (resolvedVideoUri || resolvingVideoUri) ? (
                <>
                  <Video
                    source={{ uri: resolvedVideoUri || item.videoUri || '' }}
                    style={styles.media}
                    resizeMode={ResizeMode.CONTAIN}
                    isLooping={false}
                    isMuted={false}
                    shouldPlay={isVideoPlaying}
                    useNativeControls={false}
                    onPlaybackStatusUpdate={(status) => {
                      if (status.isLoaded && status.didJustFinish) {
                        setIsVideoPlaying(false);
                      }
                      }}
                  />
                  <TouchableOpacity
                    style={styles.mediaTapTarget}
                    onPress={() => openFullScreenMedia(resolvedVideoUri ?? item.videoUri, 'video')}
                    activeOpacity={1}
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
                (() => {
                  const trellisMP4Url = refreshedTrellisMP4Url;
                  const depthUri = !overlayLoadFailed ? selectedDepthUri : null;
                  const overlayUri = effectiveSegmentedImages?.overlay_urls?.[0]?.url;
                  const displayUri = depthUri || ((!overlayLoadFailed && overlayUri) ? overlayUri : item.imageUri || null);
                  const videoThumbnailUri = item.imageUri || overlayUri || null;
                  const showImageLoader = !isVideo && (!!depthUri || !trellisMP4Url) && !!displayUri;
                  if (depthUri) {
                    return (
                      <TouchableOpacity style={styles.mediaTouchable} activeOpacity={1} onPress={() => openFullScreenMedia(depthUri, 'image')}>
                        <OptimizedImage
                          source={{ uri: depthUri }}
                          style={styles.media}
                          resizeMode="contain"
                          cachePolicy="memory-disk"
                          priority="high"
                          onImageLoad={() => setMediaLoading(false)}
                        />
                      </TouchableOpacity>
                    );
                  }
                  if (trellisMP4Url) {
                    return (
                      <>
                        <Video
                          ref={trellisVideoRef}
                          key={trellisMP4Url}
                          source={{ uri: trellisMP4Url }}
                          style={[styles.media, { position: 'absolute', top: 0, left: 0, opacity: isVideoPlaying ? 1 : 0 }]}
                          resizeMode={ResizeMode.CONTAIN}
                          isLooping
                          isMuted
                          shouldPlay={isVideoPlaying}
                          useNativeControls={false}
                          progressUpdateIntervalMillis={100}
                          onPlaybackStatusUpdate={loopTrellisPreviewAtFourSeconds}
                        />
                        {videoThumbnailUri && (
                          <OptimizedImage
                            source={{ uri: videoThumbnailUri }}
                            style={[styles.media, StyleSheet.absoluteFillObject, { opacity: isVideoPlaying ? 0 : 1 }]}
                            resizeMode="contain"
                            cachePolicy="memory-disk"
                            priority="normal"
                            onImageLoad={() => setMediaLoading(false)}
                          />
                        )}
                        <TouchableOpacity
                          style={styles.mediaTapTarget}
                          onPress={() => openFullScreenMedia(isVideoPlaying ? trellisMP4Url : videoThumbnailUri, isVideoPlaying ? 'video' : 'image')}
                          activeOpacity={1}
                        />
                        <Animated.View
                          style={[StyleSheet.absoluteFill, { backgroundColor: '#000000', opacity: loopFadeAnim, zIndex: 6 }]}
                          pointerEvents="none"
                        />
                        <TouchableOpacity style={styles.playButtonOverlay} onPress={handleVideoPlay} activeOpacity={0.7}>
                          <View style={styles.playButton}>
                            <Ionicons name={isVideoPlaying ? 'pause' : 'play'} size={28} color="#FFFFFF" />
                          </View>
                        </TouchableOpacity>
                      </>
                    );
                  }
                  if (displayUri) {
                    return (
                      <TouchableOpacity
                        activeOpacity={1}
                        onPress={() => openFullScreenMedia(displayUri, 'image')}
                        style={styles.mediaTouchable}
                      >
                        <OptimizedImage
                          source={{ uri: displayUri }}
                          style={styles.media}
                          resizeMode="contain"
                          cachePolicy="memory-disk"
                          priority="normal"
                          onImageLoad={() => setMediaLoading(false)}
                          onError={() => { setMediaLoading(false); setOverlayLoadFailed(true); }}
                        />
                        {showImageLoader && mediaLoading && (
                          <View style={[StyleSheet.absoluteFill, styles.mediaLoader]} pointerEvents="none">
                            <ActivityIndicator size="large" color="#7BA21B" />
                          </View>
                        )}
                      </TouchableOpacity>
                    );
                  }
                  return <View style={[styles.media, styles.placeholder]} />;
                })()
              )}
              <View style={styles.backButtonOverlay}>
                <View style={styles.backButtonBackground}>
                  <VectorBackButtonCircle onPress={() => navigation.goBack()} size={24} />
                </View>
              </View>
            </View>
            <View style={styles.mealInfo}>
              {/* Row 1: meal name (wraps) | right: kcal + Write Comments */}
              <View style={styles.mealHeader}>
                <View style={{ flex: 1, marginRight: 12 }}>
                  <Text style={styles.mealName}>{toSentenceCase(item.mealName || 'Burger')}</Text>
                </View>
                <View style={styles.mealActions}>
                  <Text style={styles.mealCalories}>{item.nutritionalInfo?.calories ?? 0} Kcal</Text>
                  <TouchableOpacity
                    style={styles.writeCommentButton}
                    onPress={() => {
                      scrollViewRef.current?.scrollToEnd({ animated: true });
                      setTimeout(() => {
                        commentInputRef.current?.focus();
                      }, 300);
                    }}
                    activeOpacity={0.7}
                  >
                    <Text style={styles.writeCommentButtonText}>Write Comments</Text>
                  </TouchableOpacity>
                  <View style={styles.captureInfo}>
                    <Text style={styles.captureValue}>
                      {captureDateText && captureTimeText
                        ? `${captureDateText}, ${captureTimeText}`
                        : 'Unavailable'}
                    </Text>
                  </View>
                </View>
              </View>
              {/* Row 2: 3 view buttons with separators */}
              <View style={styles.mediaActionButtons}>
                <TouchableOpacity
                  style={[styles.mediaActionButton, isVideoPlaying && styles.mediaActionButtonActive]}
                  onPress={() => { setSelectedDepthIngredient(null); handleVideoPlay(); }}
                  activeOpacity={0.7}
                >
                  <CubeIcon size={18} color={isVideoPlaying ? '#FFFFFF' : '#7BA21B'} />
                </TouchableOpacity>
                <View style={styles.buttonSeparator} />
                <TouchableOpacity
                  style={[styles.mediaActionButton, selectedDepthIngredient === '__full__' && styles.mediaActionButtonActive]}
                  onPress={() => { setIsVideoPlaying(false); setSelectedDepthIngredient(selectedDepthIngredient === '__full__' ? null : '__full__'); }}
                  activeOpacity={0.7}
                >
                  <MaterialCommunityIcons name="image-filter-hdr" size={18} color={selectedDepthIngredient === '__full__' ? '#FFFFFF' : '#6B7280'} />
                </TouchableOpacity>
                <View style={styles.buttonSeparator} />
                <TouchableOpacity
                  style={[styles.mediaActionButton, selectedDepthIngredient === '__tagged__' && styles.mediaActionButtonActive]}
                  onPress={() => { setIsVideoPlaying(false); setSelectedDepthIngredient(selectedDepthIngredient === '__tagged__' ? null : '__tagged__'); }}
                  activeOpacity={0.7}
                >
                  <MaterialCommunityIcons name="selection-ellipse" size={18} color={selectedDepthIngredient === '__tagged__' ? '#FFFFFF' : '#6B7280'} />
                </TouchableOpacity>
              </View>
            </View>
            <View style={styles.feedbackSection}>
              <Text style={styles.feedbackTitle}>Your feedback is valuable to us!</Text>
              <View style={styles.ratingRow}>
                <Text style={styles.ratingLabel}>Food dish identification</Text>
                <StarRating
                  rating={ratings.foodDishIdentification}
                  onRatingChange={(rating) =>
                    setRatings((prev) => ({ ...prev, foodDishIdentification: rating }))
                  }
                />
              </View>
              <View style={styles.ratingRow}>
                <Text style={styles.ratingLabel}>Dish contents identification</Text>
                <StarRating
                  rating={ratings.dishContentsIdentification}
                  onRatingChange={(rating) =>
                    setRatings((prev) => ({ ...prev, dishContentsIdentification: rating }))
                  }
                />
              </View>
              <View style={styles.ratingRow}>
                <Text style={styles.ratingLabel}>Mass estimation</Text>
                <StarRating
                  rating={ratings.massEstimation}
                  onRatingChange={(rating) =>
                    setRatings((prev) => ({ ...prev, massEstimation: rating }))
                  }
                />
              </View>
              <View style={styles.ratingRow}>
                <Text style={styles.ratingLabel}>Calorie estimation</Text>
                <StarRating
                  rating={ratings.calorieEstimation}
                  onRatingChange={(rating) =>
                    setRatings((prev) => ({ ...prev, calorieEstimation: rating }))
                  }
                />
              </View>
              <View style={styles.ratingRow}>
                <Text style={styles.ratingLabel}>Overall</Text>
                <StarRating
                  rating={ratings.overall}
                  onRatingChange={(rating) =>
                    setRatings((prev) => ({ ...prev, overall: rating }))
                  }
                />
              </View>
              <View style={styles.commentSection}>
                <TextInput
                  ref={commentInputRef}
                  style={[styles.commentInput, isCommentFocused && styles.commentInputFocused]}
                  placeholder="Anything you would like to tell us? (e.g., wrong item, portion too high, etc.)"
                  placeholderTextColor="#9CA3AF"
                  value={comment}
                  onChangeText={setComment}
                  multiline
                  numberOfLines={4}
                  textAlignVertical="top"
                  onFocus={() => {
                    setIsCommentFocused(true);
                    setTimeout(() => {
                      scrollViewRef.current?.scrollToEnd({ animated: true });
                    }, 300);
                  }}
                  onBlur={() => setIsCommentFocused(false)}
                />
              </View>
            </View>
          </ScrollView>

        </View>
      )}

      {/* Save Button - Fixed at Bottom, outside KAV so it doesn't float above keyboard */}
      <BottomButtonContainer>
        <TouchableOpacity
          activeOpacity={0.9}
          style={[styles.saveButton, isSaving && styles.saveButtonDisabled]}
          onPress={handleSave}
          disabled={isSaving}
        >
          <Text style={styles.saveButtonText}>{isSaving ? 'Saving...' : 'Save'}</Text>
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
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: 15,
  },
  mediaContainer: {
    width: '100%',
    height: 250,
    backgroundColor: '#111827',
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
    position: 'relative',
  },
  mediaLoader: {
    backgroundColor: 'rgba(17, 24, 39, 0.36)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  media: {
    width: '100%',
    height: '100%',
    backgroundColor: '#111827',
  },
  placeholder: {
    backgroundColor: '#111827',
  },
  mediaBackdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: '#111827',
  },
  mediaTapTarget: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 4,
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    gap: 12,
  },
  backButtonOverlay: {
    position: 'absolute',
    top: 12,
    left: 12,
    zIndex: 10,
  },
  backButtonBackground: {
    backgroundColor: '#FFFFFF',
    width: 22,
    height: 22,
    borderRadius: 11,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  playButtonOverlay: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    width: 64,
    height: 64,
    marginTop: -32,
    marginLeft: -32,
    justifyContent: 'center',
    alignItems: 'center',
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
    padding: 16,
    backgroundColor: '#FFFFFF',
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
  },
  mealHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  mealName: {
    fontSize: 18,
    fontWeight: '700',
    color: '#6B7280',
    marginBottom: 4,
  },
  mealCalories: {
    fontSize: 16,
    fontWeight: '600',
    color: '#6B7280',
  },
  writeCommentButton: {
    backgroundColor: '#7BA21B',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 24,
  },
  writeCommentButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
  },
  mealActions: {
    alignItems: 'flex-end',
    gap: 6,
  },
  mediaActionButtons: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
    gap: 6,
  },
  mediaActionButton: {
    width: 28,
    height: 28,
    borderRadius: 6,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#F3F4F6',
  },
  mediaActionButtonActive: {
    backgroundColor: '#7BA21B',
  },
  buttonSeparator: {
    width: 1,
    height: 14,
    backgroundColor: '#D1D5DB',
  },
  captureInfo: {
    alignItems: 'flex-end',
    marginTop: 4,
  },
  captureValue: {
    fontSize: 16,
    color: '#6B7280',
    fontWeight: '600',
  },
  feedbackSection: {
    padding: 16,
    backgroundColor: '#FFFFFF',
    marginTop: 16,
  },
  feedbackTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1F2937',
    marginBottom: 20,
    textAlign: 'center',
  },
  ratingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  ratingLabel: {
    fontSize: 14,
    color: '#1F2937',
    flex: 1,
    marginRight: 16,
  },
  starContainer: {
    flexDirection: 'row',
    gap: 4,
  },
  commentSection: {
    marginTop: 8,
  },
  commentInput: {
    borderWidth: 1,
    borderColor: '#E5E7EB',
    borderRadius: 8,
    padding: 12,
    fontSize: 14,
    color: '#1F2937',
    backgroundColor: '#FFFFFF',
    minHeight: 100,
    textAlignVertical: 'top',
  },
  commentInputFocused: {
    borderColor: '#7BA21B',
    borderWidth: 2,
  },
  footer: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#FFFFFF',
    borderTopWidth: 1,
    borderTopColor: '#E5E7EB',
  },
  saveButton: {
    height: 56, // Fixed height
    width: '100%', // Fixed width
    backgroundColor: '#7BA21B',
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
  },
  saveButtonDisabled: {
    opacity: 0.6,
  },
  saveButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
  },
  mediaTouchable: {
    width: '100%',
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
  },
  fullImageModalBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.95)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  fullImageModalContent: {
    width: '100%',
    flex: 1,
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
    width: '100%',
    flex: 1,
  },
});
