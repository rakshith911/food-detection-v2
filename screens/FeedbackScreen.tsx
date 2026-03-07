import React, { useState, useRef, useEffect, useCallback } from 'react';
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
} from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import { Video, ResizeMode } from 'expo-av';
import { Ionicons } from '@expo/vector-icons';
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

  const handleVideoPlay = useCallback(() => {
    setIsVideoPlaying((prev) => !prev);
  }, []);

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
  const [showFullImageModal, setShowFullImageModal] = useState(false);
  const [fullImageUri, setFullImageUri] = useState<string | null>(null);
  const [overlayLoadFailed, setOverlayLoadFailed] = useState(false);
  const [refreshedSegmentedImages, setRefreshedSegmentedImages] = useState<SegmentedImages | null>(null);
  const [refreshingOverlay, setRefreshingOverlay] = useState(false);
  const [mediaLoading, setMediaLoading] = useState(true);

  const effectiveSegmentedImages = refreshedSegmentedImages ?? item?.segmented_images;

  useEffect(() => {
    setOverlayLoadFailed(false);
    setRefreshedSegmentedImages(null);
    setMediaLoading(true);
  }, [item?.id]);

  useEffect(() => {
    setMediaLoading(true);
  }, [effectiveSegmentedImages?.overlay_urls?.[0]?.url]);

  // When we have job_id but no overlay URLs, fetch once so segmented images load
  useEffect(() => {
    if (!item?.job_id || !user?.email || effectiveSegmentedImages?.overlay_urls?.length || refreshingOverlay) return;
    let cancelled = false;
    (async () => {
      setRefreshingOverlay(true);
      try {
        const fresh = await nutritionAnalysisAPI.getResults(item.job_id!, true);
        if (cancelled) return;
        if (fresh?.segmented_images?.overlay_urls?.length) {
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
        const fresh = await nutritionAnalysisAPI.getResults(item.job_id, true);
        if (fresh?.segmented_images?.overlay_urls?.length) {
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
  const lastLoginTime = new Date().toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
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
              <TouchableOpacity
                style={styles.fullImageWrapper}
                activeOpacity={1}
                onPress={() => {}}
              >
                <Image
                  source={{ uri: fullImageUri }}
                  style={styles.fullImage}
                  resizeMode="contain"
                />
              </TouchableOpacity>
            ) : null}
          </View>
        </TouchableOpacity>
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
            lastLoginTime={lastLoginTime}
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
          {isVideo && item.videoUri ? (
            <>
              <Video
                source={{ uri: item.videoUri }}
                style={styles.media}
                resizeMode={ResizeMode.COVER}
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
              const overlayUri = effectiveSegmentedImages?.overlay_urls?.[0]?.url;
              const displayUri = (!overlayLoadFailed && overlayUri) ? overlayUri : item.imageUri || null;
              const showImageLoader = !isVideo && !!displayUri;
              if (displayUri) {
                return (
                  <TouchableOpacity activeOpacity={1} onPress={() => { setFullImageUri(displayUri); setShowFullImageModal(true); }} style={styles.mediaTouchable}>
                    <OptimizedImage
                      source={{ uri: displayUri }}
                      style={styles.media}
                      resizeMode="cover"
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
          <View style={styles.mealHeader}>
            <View style={{ flex: 1 }}>
              <Text style={styles.mealName}>{item.mealName || 'Burger'}</Text>
              <Text style={styles.mealCalories}>{item.nutritionalInfo.calories} Kcal</Text>
            </View>

            <View style={styles.mealActions}>
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
            lastLoginTime={lastLoginTime}
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
              {isVideo && item.videoUri ? (
                <>
                  <Video
                    source={{ uri: item.videoUri }}
                    style={styles.media}
                    resizeMode={ResizeMode.COVER}
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
                  const overlayUri = effectiveSegmentedImages?.overlay_urls?.[0]?.url;
                  const displayUri = (!overlayLoadFailed && overlayUri) ? overlayUri : item.imageUri || null;
                  const showImageLoader = !isVideo && !!displayUri;
                  if (displayUri) {
                    return (
                      <TouchableOpacity activeOpacity={1} onPress={() => { setFullImageUri(displayUri); setShowFullImageModal(true); }} style={styles.mediaTouchable}>
                        <OptimizedImage
                          source={{ uri: displayUri }}
                          style={styles.media}
                          resizeMode="cover"
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
              <View style={styles.mealHeader}>
                <View style={{ flex: 1 }}>
                  <Text style={styles.mealName}>{item.mealName || 'Burger'}</Text>
                  <Text style={styles.mealCalories}>{item.nutritionalInfo.calories} Kcal</Text>
                </View>
                <View style={styles.mealActions}>
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
    backgroundColor: '#F3F4F6',
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
    position: 'relative',
  },
  mediaLoader: {
    backgroundColor: '#F3F4F6',
    justifyContent: 'center',
    alignItems: 'center',
  },
  media: {
    width: '100%',
    height: '100%',
  },
  placeholder: {
    backgroundColor: '#D1D5DB',
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
    padding: 16,
    backgroundColor: '#FFFFFF',
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
  mealCalories: {
    fontSize: 16,
    fontWeight: '600',
    color: '#6B7280',
  },
  writeCommentButton: {
    backgroundColor: '#7BA21B',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 6,
  },
  writeCommentButtonText: {
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
  captureValue: {
    fontSize: 11,
    color: '#6B7280',
    fontWeight: '400',
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
    borderRadius: 28,
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

