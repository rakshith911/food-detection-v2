import React, { useState, useRef } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  TextInput,
  StatusBar,
  useWindowDimensions,
  Dimensions,
  Platform,
  ScrollView,
  Keyboard,
  TouchableWithoutFeedback,
  KeyboardAvoidingView,
  Alert,
} from 'react-native';
import { useEvent, useEventListener } from 'expo';
import { useVideoPlayer, VideoView } from 'expo-video';
import * as Notifications from 'expo-notifications';
import { Ionicons } from '@expo/vector-icons';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import type { QuestionnaireContext } from '../store/slices/historySlice';
import { addAnalysis, updateAnalysis, updateAnalysisProgress } from '../store/slices/historySlice';
import { nutritionAnalysisAPI } from '../services/NutritionAnalysisAPI';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import OptimizedImage from '../components/OptimizedImage';
import VectorBackButtonCircle from '../components/VectorBackButtonCircle';
import AppHeader from '../components/AppHeader';
import BottomButtonContainer from '../components/BottomButtonContainer';
import {
  buildDishTablesFromItems,
  getBaseDishContents,
  getMealNameFromTables,
  getOverallCaloriesFromTables,
} from '../utils/mealTables';
import { toSentenceCase } from '../utils/textCase';

async function scheduleAnalysisCompleteNotification() {
  try {
    const { status } = await Notifications.getPermissionsAsync();
    if (status !== 'granted') {
      const { status: newStatus } = await Notifications.requestPermissionsAsync();
      if (newStatus !== 'granted') return;
    }
    if (Platform.OS === 'android') {
      await Notifications.setNotificationChannelAsync('default', {
        name: 'Analysis',
        importance: Notifications.AndroidImportance.DEFAULT,
      });
    }
    const body = 'Your analysis is ready';
    await Notifications.scheduleNotificationAsync({
      content: {
        title: 'UKcal',
        body,
      },
      trigger: null,
    });
  } catch (e) {
    console.warn('[Preview] Notification failed:', e);
  }
}

interface IngredientRow {
  id: string;
  name: string;
  quantity: string;
}

interface PreviewScreenProps {
  imageUri?: string;
  videoUri?: string;
  onBack: () => void;
  onAnalyze?: () => void;
}

/** Renders video preview using expo-video (only mounted when uri is set so useVideoPlayer runs with a valid source). */
function PreviewVideo({ uri, style }: { uri: string; style: object }) {
  const player = useVideoPlayer(uri, (p) => {
    p.loop = false;
  });
  const playingPayload = useEvent(player, 'playingChange', { isPlaying: player.playing });
  const isPlaying = playingPayload?.isPlaying ?? false;

  useEventListener(player, 'playToEnd', () => {
    player.pause();
    player.replay();
  });

  return (
    <>
      <VideoView
        player={player}
        style={style}
        contentFit="cover"
        nativeControls={false}
      />
      <TouchableOpacity
        style={styles.playButton}
        onPress={() => {
          if (isPlaying) {
            player.pause();
          } else {
            player.play();
          }
        }}
        activeOpacity={0.8}
      >
        <View style={styles.playIconCircle}>
          <Ionicons name={isPlaying ? 'pause' : 'play'} size={40} color="#1F2937" />
        </View>
      </TouchableOpacity>
    </>
  );
}

export default function PreviewScreen({ imageUri, videoUri, onBack, onAnalyze }: PreviewScreenProps) {
  const insets = useSafeAreaInsets();
  const { width, height } = useWindowDimensions();
  const [step, setStep] = useState<1 | 2 | 3 | 4>(1);
  const [hiddenIngrYes, setHiddenIngrYes] = useState(false);
  const [hiddenIngredients, setHiddenIngredients] = useState<IngredientRow[]>([
    { id: '1', name: '', quantity: '' },
  ]);
  const [extrasYes, setExtrasYes] = useState(false);
  const [extras, setExtras] = useState<IngredientRow[]>([
    { id: '1', name: '', quantity: '' },
  ]);
  const [editingIngredientId, setEditingIngredientId] = useState<string | null>(null);
  const [textInput, setTextInput] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isTextInputFocused, setIsTextInputFocused] = useState(false);
  const [currentAnalysisId, setCurrentAnalysisId] = useState<string | null>(null);
  const dispatch = useAppDispatch();
  const user = useAppSelector((state) => state.auth.user);
  const businessProfile = useAppSelector((state) => state.profile.businessProfile);
  const navigation = useNavigation();
  const textInputRef = useRef<TextInput>(null);
  const scrollViewRef = useRef<ScrollView>(null);

  const userName = (businessProfile?.businessName && businessProfile.businessName.trim())
    ? businessProfile.businessName
    : (user?.email?.split('@')[0] || 'User');
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

  const handleBack = () => {
    if (step === 1) {
      onBack();
    } else {
      setStep((s) => (s - 1) as 1 | 2 | 3 | 4);
    }
  };

  const handleNext = () => {
    setStep((s) => (s + 1) as 1 | 2 | 3 | 4);
  };

  const addIngredientRow = (type: 'hidden' | 'extras') => {
    const newRow: IngredientRow = { id: Date.now().toString(), name: '', quantity: '' };
    if (type === 'hidden') {
      setHiddenIngredients((prev) => [...prev, newRow]);
    } else {
      setExtras((prev) => [...prev, newRow]);
    }
  };

  const removeIngredientRow = (type: 'hidden' | 'extras', id: string) => {
    if (type === 'hidden') {
      setHiddenIngredients((prev) => prev.filter((r) => r.id !== id));
    } else {
      setExtras((prev) => prev.filter((r) => r.id !== id));
    }
  };

  const handleIngredientAction = (type: 'hidden' | 'extras', id: string) => {
    if (editingIngredientId === id) {
      // Trash mode — delete directly, no popup
      removeIngredientRow(type, id);
      setEditingIngredientId(null);
    } else {
      // Pencil mode — enter edit mode (show trash icon)
      setEditingIngredientId(id);
    }
  };

  const updateIngredientRow = (
    type: 'hidden' | 'extras',
    id: string,
    field: 'name' | 'quantity',
    value: string
  ) => {
    const nextValue = field === 'name' ? toSentenceCase(value) : value;
    if (type === 'hidden') {
      setHiddenIngredients((prev) =>
        prev.map((r) => (r.id === id ? { ...r, [field]: nextValue } : r))
      );
    } else {
      setExtras((prev) =>
        prev.map((r) => (r.id === id ? { ...r, [field]: nextValue } : r))
      );
    }
  };

  const handleSubmit = async () => {
    if (isSubmitting) return;

    setIsSubmitting(true);

    let analysisId: string | null = null;

    try {
      const analysisType: 'image' | 'video' = imageUri ? 'image' : 'video';
      let analysisResult;
      let result;
      let mealName = 'Analyzed Meal';

      const userContext: QuestionnaireContext = {};

      if (hiddenIngrYes) {
        const filledHidden = hiddenIngredients.filter((r) => r.name.trim());
        if (filledHidden.length > 0) {
          userContext.hidden_ingredients = filledHidden.map((r) => ({
            name: r.name.trim(),
            quantity: r.quantity.trim(),
          }));
        }
      }

      if (extrasYes) {
        const filledExtras = extras.filter((r) => r.name.trim());
        if (filledExtras.length > 0) {
          userContext.extras = filledExtras.map((r) => ({
            name: r.name.trim(),
            quantity: r.quantity.trim(),
          }));
        }
      }

      if (textInput.trim()) {
        userContext.recipe_description = textInput.trim();
      }

      // Generate a thumbnail from the first frame of the video so the history
      // card and MealDetailScreen have an image to show while the video loads.
      let videoThumbnailUri: string | undefined;
      if (videoUri) {
        try {
          // Dynamic require so the app doesn't crash when the native module
          // isn't available (e.g. running in Expo Go without a dev client build)
          // eslint-disable-next-line @typescript-eslint/no-var-requires
          const VideoThumbnails = require('expo-video-thumbnails');
          const { uri } = await VideoThumbnails.getThumbnailAsync(videoUri, { time: 0 });
          videoThumbnailUri = uri;
        } catch {
          // Non-fatal — thumbnail is a nice-to-have, not required
        }
      }

      if (user?.email) {
        const tempAnalysis = {
          type: analysisType,
          imageUri: imageUri || videoThumbnailUri || undefined,
          videoUri: videoUri || undefined,
          textDescription: textInput.trim() || undefined,
          analysisResult: JSON.stringify({ summary: 'Analysis in progress...' }),
          nutritionalInfo: {
            calories: 0,
            protein: 0,
            carbs: 0,
            fat: 0,
          },
          questionnaireContext: userContext,
          analysisStatus: 'analyzing' as const,
          analysisProgress: 0,
        };

        const result_action = await dispatch(addAnalysis({
          userEmail: user.email,
          analysis: tempAnalysis,
        }));

        if (addAnalysis.fulfilled.match(result_action)) {
          analysisId = result_action.payload.id;
          setCurrentAnalysisId(analysisId);

          // Navigate immediately — Results screen will show the notification on mount
          if (onAnalyze) {
            onAnalyze();
          }
        }
      }

      if (videoUri || imageUri) {
        const mediaType = videoUri ? 'video' : 'image';
        console.log(`[PreviewScreen] Starting real ${mediaType} analysis...`);

        console.log('[PreviewScreen] User context:', JSON.stringify(userContext));

        const filename = videoUri
          ? `video_${Date.now()}.mp4`
          : `image_${Date.now()}.jpg`;

        const updateProgress = (status: string) => {
          if (!analysisId) return;
          if (status.includes('Preparing')) {
            dispatch(updateAnalysisProgress({ id: analysisId, progress: 10, status: 'analyzing' }));
          } else if (status.includes('Uploading')) {
            dispatch(updateAnalysisProgress({ id: analysisId, progress: 30, status: 'analyzing' }));
          } else if (status.includes('Starting')) {
            dispatch(updateAnalysisProgress({ id: analysisId, progress: 40, status: 'analyzing' }));
          } else if (status.includes('Processing')) {
            const match = status.match(/\((\d+)\/(\d+)\)/);
            if (match) {
              const current = parseInt(match[1], 10);
              const total = parseInt(match[2], 10);
              const processingProgress = 50 + Math.floor((current / total) * 40);
              dispatch(updateAnalysisProgress({ id: analysisId, progress: processingProgress, status: 'analyzing' }));
            } else {
              dispatch(updateAnalysisProgress({ id: analysisId, progress: 60, status: 'analyzing' }));
            }
          } else if (status.includes('complete') || status.includes('Complete')) {
            dispatch(updateAnalysisProgress({ id: analysisId, progress: 100, status: 'completed' }));
          }
        };

        const onJobCreated = (jobId: string) => {
          if (!analysisId || !user?.email) return;
          console.log('[PreviewScreen] Job created — persisting job_id immediately:', jobId);
          dispatch(updateAnalysis({
            userEmail: user.email,
            analysisId,
            updates: { job_id: jobId },
          }));
        };

        const apiResult = videoUri
          ? await nutritionAnalysisAPI.analyzeVideo(
              videoUri,
              filename,
              (status) => {
                console.log('[PreviewScreen] Progress:', status);
                updateProgress(status);
              },
              onJobCreated,
              userContext
            )
          : await nutritionAnalysisAPI.analyzeImage(
              imageUri!,
              filename,
              (status) => {
                console.log('[PreviewScreen] Progress:', status);
                updateProgress(status);
              },
              onJobCreated,
              userContext
            );

        if (!apiResult || !apiResult.nutrition_summary) {
          throw new Error(`${mediaType} analysis failed or returned no results`);
        } else {
          console.log('[PreviewScreen] API Result nutrition_summary:', apiResult.nutrition_summary);
          console.log('[PreviewScreen] API Result items:', apiResult.items);

          let dishTables;
          let dishContents;

          if (apiResult.items && apiResult.items.length > 0) {
            console.log('[PreviewScreen] Using extracted items:', apiResult.items.length, 'items');
            dishTables = buildDishTablesFromItems(apiResult.items, userContext);
            console.log('[PreviewScreen] dishTables built');
            dishContents = getBaseDishContents(dishTables);
            console.log('[PreviewScreen] dishContents:', dishContents.length, 'rows');
            mealName = toSentenceCase(apiResult.meal_name || getMealNameFromTables(dishTables, 'Analyzed Meal'));
            console.log('[PreviewScreen] mealName resolved:', mealName);
          } else if (apiResult.detailed_results?.items && apiResult.detailed_results.items.length > 0) {
            console.log('[PreviewScreen] Using detailed_results.items');
            dishTables = buildDishTablesFromItems(apiResult.detailed_results.items, userContext);
            dishContents = getBaseDishContents(dishTables);
            mealName = toSentenceCase(apiResult.meal_name || getMealNameFromTables(dishTables, 'Analyzed Meal'));
          } else {
            console.log('[PreviewScreen] No food items found in response');
            dishTables = buildDishTablesFromItems([], userContext);
            dishContents = [{ id: `${Date.now()}_0`, name: toSentenceCase('No food detected'), weight: '', calories: '0' }];
            mealName = toSentenceCase('No food detected');
          }

          console.log('[PreviewScreen] Computing totals...');
          const totalCaloriesFromItems = getOverallCaloriesFromTables(dishTables)
            || apiResult.nutrition_summary?.total_calories_kcal
            || 0;
          console.log('[PreviewScreen] totalCalories:', totalCaloriesFromItems);

          analysisResult = {
            totalCalories: totalCaloriesFromItems,
            totalProtein: 0,
            totalCarbs: 0,
            totalFat: 0,
            dishTables,
            dishContents,
            mealName,
          };

          console.log('[PreviewScreen] analysisResult created, building result...');
          const numItems = apiResult.items?.length || apiResult.nutrition_summary?.num_food_items || dishContents.length;

          result = {
            summary: `Detected ${numItems} food items with ${Math.round(totalCaloriesFromItems)} calories`,
            nutrition_summary: apiResult.nutrition_summary || {
              total_calories_kcal: totalCaloriesFromItems,
              num_food_items: numItems,
              total_mass_g: 0,
              total_food_volume_ml: 0,
            },
            detailed_results: apiResult.detailed_results,
            segmented_images: apiResult.segmented_images,
            job_id: apiResult.job_id,
          };
          console.log('[PreviewScreen] result built, about to dispatch...');
        }
      } else {
        throw new Error('No image or video URI provided');
      }

      if (user?.email && analysisId) {
        const result_action = await dispatch(updateAnalysis({
          userEmail: user.email,
          analysisId: analysisId,
          updates: {
            analysisResult: JSON.parse(JSON.stringify(result)),
            dishTables: analysisResult.dishTables,
            dishContents: analysisResult.dishContents,
            mealName: mealName,
            questionnaireContext: userContext,
            nutritionalInfo: {
              calories: Number(analysisResult.totalCalories) || 0,
              protein: Number(analysisResult.totalProtein) || 0,
              carbs: Number(analysisResult.totalCarbs) || 0,
              fat: Number(analysisResult.totalFat) || 0,
            },
            segmented_images: typeof result === 'object' && result?.segmented_images ? result.segmented_images : undefined,
            job_id: typeof result === 'object' && 'job_id' in result ? (result as any).job_id : undefined,
            analysisStatus: 'completed',
            analysisProgress: 100,
          },
        }));

        if (updateAnalysis.rejected.match(result_action)) {
          console.error('Error updating analysis:', result_action.error);
        }
      }

      try {
        const savedStreak = await AsyncStorage.getItem('streakDays');
        const currentStreak = savedStreak ? parseInt(savedStreak, 10) : 0;
        await AsyncStorage.setItem('streakDays', (currentStreak + 1).toString());
      } catch (error) {
        console.error('Error saving streak:', error);
      }
    } catch (error: any) {
      console.warn('[PreviewScreen] Analysis error:', error?.message);

      if (error?.message === 'Analysis timeout') {
        console.log('[PreviewScreen] Timeout — job remains queued in SQS, analysisId:', analysisId);
        // Job is queued — push notification will fire when ready, no alert needed
      } else if (
        error?.message?.includes('analysis failed') ||
        error?.message?.includes('too many') ||
        error?.message?.includes('rate limit') ||
        error?.message?.includes('503') ||
        error?.message?.includes('429')
      ) {
        console.log('[PreviewScreen] Capacity/rate-limit error:', error?.message);
        if (analysisId && user?.email) {
          dispatch(updateAnalysisProgress({ id: analysisId, progress: 0, status: 'failed' }));
        }
        Alert.alert(
          'High Demand',
          'We are receiving too many requests. We will notify you when its ready.',
          [{ text: 'OK' }]
        );
      } else {
        console.log('[PreviewScreen] Unexpected error — marking failed. analysisId:', analysisId, 'error:', error?.message);
        if (analysisId && user?.email) {
          dispatch(updateAnalysisProgress({ id: analysisId, progress: 0, status: 'failed' }));
        }
        Alert.alert(
          'Analysis Failed',
          'Something went wrong processing your image. Please try again.',
          [{ text: 'OK' }]
        );
      }
    } finally {
      setIsSubmitting(false);
      setCurrentAnalysisId(null);
    }
  };

  const renderRadioGroup = (value: boolean, onChange: (v: boolean) => void) => (
    <View style={styles.radioContainer}>
      <TouchableOpacity style={styles.radioOption} onPress={() => onChange(true)}>
        <View style={[styles.radioOuter, value && styles.radioOuterSelected]}>
          {value && <View style={styles.radioInner} />}
        </View>
        <Text style={styles.radioLabel}>Yes</Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.radioOption} onPress={() => onChange(false)}>
        <View style={[styles.radioOuter, !value && styles.radioOuterSelected]}>
          {!value && <View style={styles.radioInner} />}
        </View>
        <Text style={styles.radioLabel}>No</Text>
      </TouchableOpacity>
    </View>
  );

  const renderIngredientTable = (
    type: 'hidden' | 'extras',
    rows: IngredientRow[],
    headerLabel: string
  ) => {
    const lastRow = rows[rows.length - 1];
    const canAddMore = lastRow ? (lastRow.name.trim().length > 0 && lastRow.quantity.trim().length > 0) : true;
    return (
    <View style={styles.ingredientSection}>
      <View style={styles.tableDivider} />
      <View style={styles.tableHeader}>
        <Text style={[styles.tableHeaderText, { flex: 5, paddingLeft: 10 }]}>{headerLabel}</Text>
        <View style={{ width: 8 }} />
        <Text style={[styles.tableHeaderText, { flex: 4, paddingLeft: 10 }]}>Quantity</Text>
        <Text style={[styles.tableHeaderText, { flex: 2, textAlign: 'center' }]}>Action</Text>
      </View>
      {rows.map((row) => {
        const isRowEditing = editingIngredientId === row.id;
        return (
        <View key={row.id} style={styles.tableRow}>
          <TextInput
            style={[styles.tableInput, { flex: 5 }]}
            placeholder="Ingredient Name"
            value={row.name}
            onChangeText={(v) => updateIngredientRow(type, row.id, 'name', v)}
            placeholderTextColor="#999999"
          />
          <View style={{ width: 8 }} />
          <TextInput
            style={[styles.tableInput, { flex: 4 }]}
            placeholder="Weight or volume"
            value={row.quantity}
            onChangeText={(v) => updateIngredientRow(type, row.id, 'quantity', v)}
            placeholderTextColor="#999999"
          />
          <TouchableOpacity
            style={{ flex: 2, alignItems: 'center', justifyContent: 'center' }}
            onPress={() => handleIngredientAction(type, row.id)}
            activeOpacity={0.7}
          >
            <Ionicons name={isRowEditing ? 'trash' : 'pencil'} size={18} color="#7BA21B" />
          </TouchableOpacity>
        </View>
        );
      })}
      <View style={styles.addIngredientRow}>
        <TouchableOpacity
          style={[styles.addIngredientButton, !canAddMore && styles.addIngredientButtonDisabled]}
          onPress={() => canAddMore && addIngredientRow(type)}
          activeOpacity={canAddMore ? 0.8 : 1}
        >
          <View style={styles.addIngredientIcon}>
            <Text style={styles.addIngredientIconText}>+</Text>
          </View>
          <Text style={styles.addIngredientText}>Add Ingredient</Text>
        </TouchableOpacity>
      </View>
    </View>
    );
  };

  const renderStep1 = () => (
    <>
      <ScrollView
        ref={scrollViewRef}
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        bounces={true}
      >
        {/* Image/Video Preview */}
        <View style={styles.previewContainer}>
          {imageUri ? (
            <OptimizedImage
              source={{ uri: imageUri }}
              style={styles.previewMedia}
              resizeMode="cover"
              cachePolicy="memory-disk"
              priority="high"
            />
          ) : videoUri ? (
            <PreviewVideo uri={videoUri} style={styles.previewMedia} />
          ) : null}
          <View style={styles.darkOverlay} />
          <View style={styles.backButtonContainer}>
            <View style={styles.backButtonBackground}>
              <VectorBackButtonCircle onPress={onBack} size={24} />
            </View>
          </View>
        </View>

        {/* Instruction Text */}
        <View style={styles.instructionContainer}>
          <Text style={styles.instructionTitle}>Please check the snap.</Text>
          <View style={styles.bulletList}>
            <View style={styles.bulletItem}>
              <Text style={styles.bulletDot}>{'\u2022'}</Text>
              <Text style={styles.bulletText}>
                For better accuracy, place a blank business card (or similar-sized reference card) beside the plate for scale.
              </Text>
            </View>
            <View style={styles.bulletItem}>
              <Text style={styles.bulletDot}>{'\u2022'}</Text>
              <Text style={styles.bulletText}>
                Use good lighting and keep the whole dish in view.
              </Text>
            </View>
            <View style={styles.bulletItem}>
              <Text style={styles.bulletDot}>{'\u2022'}</Text>
              <Text style={styles.bulletText}>
                If it looks good, tap Next. To retake, tap the back arrow.
              </Text>
            </View>
          </View>
        </View>
      </ScrollView>

      <BottomButtonContainer paddingHorizontal={10}>
        <TouchableOpacity
          style={styles.primaryButton}
          onPress={handleNext}
          activeOpacity={0.8}
        >
          <Text style={styles.primaryButtonText}>Next</Text>
        </TouchableOpacity>
      </BottomButtonContainer>
    </>
  );

  const renderStep2 = () => (
    <>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={[styles.scrollContent, { paddingBottom: 30 }]}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      >
        <View style={styles.stepBackButton}>
          <VectorBackButtonCircle onPress={handleBack} size={32} />
        </View>

        <Text style={styles.questionText}>
          Are any ingredients hidden or missing from the photo? For example: chicken under pasta, cheese inside, sauce mixed in.
        </Text>

        {renderRadioGroup(hiddenIngrYes, setHiddenIngrYes)}

        {hiddenIngrYes && renderIngredientTable('hidden', hiddenIngredients, 'Ingredients')}
      </ScrollView>

      <BottomButtonContainer paddingHorizontal={10}>
        {(() => {
          const canProceed = !hiddenIngrYes || (
            hiddenIngredients.length > 0 &&
            hiddenIngredients.every(r => r.name.trim().length > 0 && r.quantity.trim().length > 0)
          );
          return (
            <TouchableOpacity
              style={[styles.primaryButton, !canProceed && styles.primaryButtonDisabled]}
              onPress={canProceed ? handleNext : undefined}
              activeOpacity={canProceed ? 0.8 : 1}
            >
              <Text style={styles.primaryButtonText}>Next</Text>
            </TouchableOpacity>
          );
        })()}
      </BottomButtonContainer>
    </>
  );

  const renderStep3 = () => (
    <>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={[styles.scrollContent, { paddingBottom: 30 }]}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      >
        <View style={styles.stepBackButton}>
          <VectorBackButtonCircle onPress={handleBack} size={32} />
        </View>

        <Text style={styles.questionText}>
          Any extras or cooking styles that add calories? For example: extra cheese/oil/butter, double meat, fried, creamy sauce.
        </Text>

        {renderRadioGroup(extrasYes, setExtrasYes)}

        {extrasYes && renderIngredientTable('extras', extras, 'Extra Ingredients')}
      </ScrollView>

      <BottomButtonContainer paddingHorizontal={10}>
        {(() => {
          const canProceed = !extrasYes || (
            extras.length > 0 &&
            extras.every(r => r.name.trim().length > 0 && r.quantity.trim().length > 0)
          );
          return (
            <TouchableOpacity
              style={[styles.primaryButton, !canProceed && styles.primaryButtonDisabled]}
              onPress={canProceed ? handleNext : undefined}
              activeOpacity={canProceed ? 0.8 : 1}
            >
              <Text style={styles.primaryButtonText}>Next</Text>
            </TouchableOpacity>
          );
        })()}
      </BottomButtonContainer>
    </>
  );

  const renderStep4 = () => (
    <>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={{ flex: 1 }}
        keyboardVerticalOffset={Platform.OS === 'ios' ? insets.top : 0}
      >
        <TouchableWithoutFeedback onPress={() => Keyboard.dismiss()}>
          <ScrollView
            ref={scrollViewRef}
            style={styles.scrollView}
            contentContainerStyle={[styles.scrollContent, { paddingBottom: 30 }]}
            showsVerticalScrollIndicator={false}
            keyboardShouldPersistTaps="handled"
            keyboardDismissMode="on-drag"
          >
            <View style={styles.stepBackButton}>
              <VectorBackButtonCircle onPress={handleBack} size={32} />
            </View>

            <Text style={styles.questionText}>
              If you know the recipe or menu description, add it here (Optional).
            </Text>

            <View style={styles.descriptionInputContainer}>
              <TextInput
                ref={textInputRef}
                style={[styles.descriptionInput, isTextInputFocused && styles.descriptionInputFocused]}
                placeholder="For example, Spaghetti with chicken, creamy sauce, 2 tablespoon of oil, parmesan on top."
                value={textInput}
                onChangeText={setTextInput}
                multiline
                textAlignVertical="top"
                placeholderTextColor="#999999"
                onFocus={() => {
                  setIsTextInputFocused(true);
                  setTimeout(() => {
                    scrollViewRef.current?.scrollToEnd({ animated: true });
                  }, 300);
                }}
                onBlur={() => setIsTextInputFocused(false)}
              />
            </View>
          </ScrollView>
        </TouchableWithoutFeedback>
      </KeyboardAvoidingView>

      <BottomButtonContainer paddingHorizontal={10}>
        <TouchableOpacity
          style={[styles.primaryButton, isSubmitting && styles.primaryButtonDisabled]}
          onPress={handleSubmit}
          disabled={isSubmitting}
          activeOpacity={0.8}
        >
          <Text style={styles.primaryButtonText}>
            {isSubmitting ? 'Analyzing...' : 'Submit'}
          </Text>
        </TouchableOpacity>
      </BottomButtonContainer>
    </>
  );

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <StatusBar barStyle="light-content" />
      <AppHeader
        displayName={displayName}
        lastLoginDate={lastLoginDate}
        lastLoginTime={lastLoginTime}
        onProfilePress={() => navigation.navigate('Profile' as never)}
      />
      {step === 1 && renderStep1()}
      {step === 2 && renderStep2()}
      {step === 3 && renderStep3()}
      {step === 4 && renderStep4()}
    </SafeAreaView>
  );
}

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
  },

  // ── Step 1 ──
  previewContainer: {
    width: '100%',
    height: SCREEN_HEIGHT * 0.45,
    position: 'relative',
    backgroundColor: '#000000',
    overflow: 'hidden',
  },
  previewMedia: {
    width: '100%',
    height: '100%',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  darkOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.4)',
  },
  backButtonContainer: {
    position: 'absolute',
    top: 16,
    left: 16,
    zIndex: 10,
  },
  backButtonBackground: {
    backgroundColor: '#FFFFFF',
    width: 24,
    height: 24,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  playButton: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: [{ translateX: -40 }, { translateY: -40 }],
    zIndex: 10,
  },
  playIconCircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  instructionContainer: {
    paddingHorizontal: 20,
    paddingTop: 24,
    paddingBottom: 16,
  },
  instructionTitle: {
    fontSize: 15,
    color: '#1F2937',
    marginBottom: 10,
  },
  bulletList: {
    gap: 8,
  },
  bulletItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  bulletDot: {
    fontSize: 15,
    color: '#1F2937',
    marginRight: 8,
    lineHeight: 22,
  },
  bulletText: {
    flex: 1,
    fontSize: 15,
    color: '#1F2937',
    lineHeight: 22,
  },

  // ── Steps 2–4 shared ──
  stepBackButton: {
    paddingHorizontal: 16,
    paddingTop: 16,
    paddingBottom: 8,
    alignSelf: 'flex-start',
  },
  questionText: {
    fontSize: 16,
    color: '#1F2937',
    lineHeight: 24,
    paddingHorizontal: 20,
    paddingTop: 12,
    paddingBottom: 20,
  },
  radioContainer: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    gap: 32,
    marginBottom: 8,
  },
  radioOption: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  radioOuter: {
    width: 22,
    height: 22,
    borderRadius: 11,
    borderWidth: 2,
    borderColor: '#CCCCCC',
    justifyContent: 'center',
    alignItems: 'center',
  },
  radioOuterSelected: {
    borderColor: '#7BA21B',
  },
  radioInner: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: '#7BA21B',
  },
  radioLabel: {
    fontSize: 16,
    color: '#1F2937',
  },

  // ── Ingredient table ──
  ingredientSection: {
    marginTop: 8,
  },
  tableDivider: {
    height: 1,
    backgroundColor: '#E5E7EB',
    marginBottom: 16,
  },
  tableHeader: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    marginBottom: 12,
  },
  tableHeaderText: {
    fontSize: 14,
    fontWeight: '700',
    color: '#1F2937',
  },
  tableRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    marginBottom: 10,
  },
  tableInput: {
    borderWidth: 1,
    borderColor: '#D1D5DB',
    borderRadius: 6,
    paddingHorizontal: 10,
    paddingVertical: 8,
    fontSize: 13,
    color: '#1F2937',
    backgroundColor: '#FFFFFF',
  },
  addIngredientRow: {
    paddingHorizontal: 20,
    paddingTop: 12,
    alignItems: 'flex-end',
  },
  addIngredientButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#7BA21B',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 24,
    gap: 6,
  },
  addIngredientButtonDisabled: {
    backgroundColor: '#B5D068',
    opacity: 0.6,
  },
  addIngredientIcon: {
    width: 17,
    height: 17,
    borderRadius: 9,
    backgroundColor: '#FFFFFF',
    alignItems: 'center',
    justifyContent: 'center',
  },
  addIngredientIconText: {
    color: '#7BA21B',
    fontSize: 16,
    fontWeight: '700',
    lineHeight: 16,
    textAlign: 'center',
  },
  addIngredientText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
  },

  // ── Step 4 ──
  descriptionInputContainer: {
    marginHorizontal: 20,
    marginTop: 8,
  },
  descriptionInput: {
    borderWidth: 1,
    borderColor: '#E5E7EB',
    borderRadius: 12,
    padding: 16,
    fontSize: 14,
    color: '#1F2937',
    minHeight: 180,
    backgroundColor: '#FFFFFF',
    textAlignVertical: 'top',
  },
  descriptionInputFocused: {
    borderColor: '#7BA21B',
    borderWidth: 2,
  },

  // ── Shared button ──
  primaryButton: {
    height: 56,
    width: '100%',
    backgroundColor: '#7BA21B',
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#7BA21B',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  primaryButtonDisabled: {
    opacity: 0.7,
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
  },
});
