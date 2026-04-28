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
import type { AnalysisEntry, DishContent, DishTableKey, DishTableSection, SegmentedImages } from '../store/slices/historySlice';
import { updateAnalysis } from '../store/slices/historySlice';
import { nutritionAnalysisAPI } from '../services/NutritionAnalysisAPI';
import { getImagePresignedUrl } from '../services/S3UserDataService';
import { prefetchSegmentedImage, cacheKeyFor } from '../services/SegmentedImageCache';
import VectorBackButtonCircle from '../components/VectorBackButtonCircle';
import OptimizedImage from '../components/OptimizedImage';
import AppHeader from '../components/AppHeader';
import BottomButtonContainer from '../components/BottomButtonContainer';
import {
  getBaseDishContents,
  getMealNameFromTables,
  getOverallCaloriesFromTables,
  getTableTotals,
  hydrateDishTables,
} from '../utils/mealTables';
import { toDisplayFoodLabel, toSentenceCase } from '../utils/textCase';


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
  
  if (!item) {
    return (
      <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
        <StatusBar barStyle="dark-content" />
        <AppHeader
          displayName={displayName}
          onProfilePress={() => navigation.navigate('Profile')}
        />
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
          <Text>No meal data available</Text>
        </View>
      </SafeAreaView>
    );
  }

  const isVideo = !!item.videoUri;

  const normalizeDishTablesForSave = useCallback(
    (tables: DishTableSection[]) =>
      tables.map((table) => ({
        ...table,
        rows: table.rows.map((row) => ({
          ...row,
          name: toDisplayFoodLabel(row.name),
        })),
      })),
    []
  );

  const [editingRowId, setEditingRowId] = useState<string | null>(null);
  const [editingMealName, setEditingMealName] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [showSegmentationOverlay, setShowSegmentationOverlay] = useState(false);
  const [overlayImageUrl, setOverlayImageUrl] = useState<string | null>(null);
  const [showFullImageModal, setShowFullImageModal] = useState(false);
  const [fullImageUri, setFullImageUri] = useState<string | null>(null);
  const [selectedDepthIngredient, setSelectedDepthIngredient] = useState<string | null>(null);
  // When segmented overlay URL fails to load (e.g. expired), refetch or show original image
  const [overlayLoadFailed, setOverlayLoadFailed] = useState(false);
  const [refreshedSegmentedImages, setRefreshedSegmentedImages] = useState<SegmentedImages | null>(null);
  const [refreshingOverlay, setRefreshingOverlay] = useState(false);
  const [mediaLoading, setMediaLoading] = useState(true);
  const [videoOverlayError, setVideoOverlayError] = useState(false);
  // Resolved image/video URI — always fetched fresh from S3 (presigned URL).
  // Initialised to the locally-stored URI so something renders immediately while
  // the presigned URL fetch is in-flight; S3 URL overrides once resolved.
  const [resolvedImageUri, setResolvedImageUri] = useState<string | undefined>(item?.imageUri);
  const [resolvedVideoUri, setResolvedVideoUri] = useState<string | undefined>(item?.videoUri);
  // True while we are waiting for the presigned URL — used to show a loading overlay
  const [resolvingVideoUri, setResolvingVideoUri] = useState<boolean>(!!item?.job_id && !!item?.videoUri);

  useEffect(() => {
    // For video analyses, item.imageUri is the saved first-frame thumbnail and should
    // remain the pre-play preview instead of being replaced by a fetched media URL.
    if (isVideo) {
      setResolvedImageUri(item?.imageUri);
      return;
    }

    // Always fetch from S3 for images when job_id is available — presigned URLs are fresh
    if (item?.job_id) {
      getImagePresignedUrl(item.job_id)
        .then(url => { setResolvedImageUri(url || item?.imageUri); })
        .catch(() => setResolvedImageUri(item?.imageUri));
    } else {
      setResolvedImageUri(item?.imageUri);
    }
  }, [isVideo, item?.id, item?.job_id, item?.imageUri]);

  useEffect(() => {
    // Always fetch from S3 when job_id is available — works on any device after reinstall.
    // Local videoUri is used as the initial value so the player is never blank while fetching.
    if (item?.job_id) {
      setResolvingVideoUri(true);
      getImagePresignedUrl(item.job_id)
        .then(url => {
          setResolvedVideoUri(url || item?.videoUri);
          setResolvingVideoUri(false);
        })
        .catch(() => {
          setResolvedVideoUri(item?.videoUri);
          setResolvingVideoUri(false);
        });
    } else {
      setResolvedVideoUri(item?.videoUri);
      setResolvingVideoUri(false);
    }
  }, [item?.id, item?.job_id]);
  
  // Use freshly fetched URLs when available, fall back to stored ones.
  // Stored URLs have 30-day expiry so they remain valid for normal usage patterns.
  const effectiveSegmentedImages = refreshedSegmentedImages ?? item?.segmented_images ?? null;

  const normalizeAssetName = useCallback((value?: string | null) => (
    (value || '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '')
  ), []);

  const geminiDepthAssets = useMemo(() => {
    const overlays = effectiveSegmentedImages?.overlay_urls || [];
    const full = overlays.find((asset: any) => asset?.name === 'gemini_depth_full')?.url || null;
    const ingredients: Record<string, string> = {};
    overlays.forEach((asset: any) => {
      const name = asset?.name || '';
      if (!name.startsWith('gemini_depth_ingredient_') || !asset?.url) return;
      const slug = name.replace('gemini_depth_ingredient_', '');
      ingredients[slug] = asset.url;
      const displaySlug = normalizeAssetName(toDisplayFoodLabel(slug.replace(/_/g, ' ')));
      if (displaySlug && displaySlug !== slug) {
        ingredients[displaySlug] = asset.url;
      }
    });
    return { full, ingredients };
  }, [effectiveSegmentedImages?.overlay_urls, normalizeAssetName]);

  const taggedOverlayUri = useMemo(() =>
    effectiveSegmentedImages?.overlay_urls?.find((a: any) => a?.name === 'tagged')?.url || null,
    [effectiveSegmentedImages?.overlay_urls]
  );

  const selectedDepthUri = useMemo(() => {
    if (!selectedDepthIngredient) return null;
    if (selectedDepthIngredient === '__full__') return geminiDepthAssets.full;
    if (selectedDepthIngredient === '__tagged__') return taggedOverlayUri;
    return geminiDepthAssets.ingredients[normalizeAssetName(selectedDepthIngredient)] || null;
  }, [geminiDepthAssets, normalizeAssetName, selectedDepthIngredient, taggedOverlayUri]);

  // Reset overlay state and loader when item changes
  useEffect(() => {
    setOverlayLoadFailed(false);
    setRefreshedSegmentedImages(null);
    setMediaLoading(true);
    setVideoOverlayError(false);
    setSelectedDepthIngredient(null);
    // Re-seed resolved URIs from the new item immediately so there's no blank frame
    setResolvedImageUri(item?.imageUri);
    setResolvedVideoUri(item?.videoUri);
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
        // fetchDetailedJson=true so we download results.json and get fresh presigned image URLs
        const fresh = await nutritionAnalysisAPI.getResults(item.job_id!, true, true);
        if (cancelled) return;
        const hasResult = fresh?.segmented_images?.overlay_urls?.length || fresh?.segmented_images?.video_overlay_url;
        if (hasResult || fresh?.trellis_mp4_url) {
          setOverlayLoadFailed(false);
          setVideoOverlayError(false);
          setRefreshedSegmentedImages(fresh.segmented_images || null);
          await dispatch(updateAnalysis({
            userEmail: user.email,
            analysisId: item.id,
            updates: {
              segmented_images: fresh.segmented_images,
              ...(fresh.trellis_mp4_url ? { trellis_mp4_url: fresh.trellis_mp4_url } : {}),
            },
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
          setRefreshedSegmentedImages(fresh.segmented_images || null);
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
  
  const [dishTables, setDishTables] = useState<DishTableSection[]>(
    normalizeDishTablesForSave(hydrateDishTables(item?.dishTables, item?.dishContents))
  );
  const [mealName, setMealName] = useState(toSentenceCase(item?.mealName || 'Burger'));
  const dishContents = useMemo(() => getBaseDishContents(dishTables), [dishTables]);
  const totalCalories = useMemo(() => getOverallCaloriesFromTables(dishTables), [dishTables]);

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
      const normalizedDishTables = normalizeDishTablesForSave(dishTables);
      const normalizedMealName = toSentenceCase(mealName);
      // Include full item data in updates to ensure entry can be recreated if missing
      const updates = {
        ...item, // Include all existing item data
        mealName: normalizedMealName,
        dishTables: normalizedDishTables,
        dishContents: getBaseDishContents(normalizedDishTables),
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
  }, [item, user?.email, mealName, dishTables, totalCalories, dispatch, normalizeDishTablesForSave]);

  // Always keep the ref pointing at the latest saveChanges closure
  const saveChangesRef = useRef(saveChanges);
  useEffect(() => {
    saveChangesRef.current = saveChanges;
  }, [saveChanges]);

  // Auto-save whenever table contents or mealName changes.
  // Skips the initial mount so we don't fire an unnecessary save on screen open.
  // Uses a ref so the effect deps stay stable and only [dishTables, mealName] retrigger it.
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
  }, [dishTables, mealName]);

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

  const updateTableRows = (tableKey: DishTableKey, updater: (rows: DishContent[]) => DishContent[]) => {
    setDishTables((prev) =>
      prev.map((table) =>
        table.key === tableKey ? { ...table, rows: updater(table.rows) } : table
      )
    );
  };

  const handleDelete = (tableKey: DishTableKey, rowId: string) => {
    Alert.alert(
      'Delete Item',
      'Are you sure you want to delete this item?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => {
            updateTableRows(tableKey, (rows) => rows.filter((row) => row.id !== rowId));
            setEditingRowId(null);
          },
        },
      ]
    );
  };

  const handleUpdateRow = (
    tableKey: DishTableKey,
    rowId: string,
    field: 'name' | 'weight' | 'calories',
    value: string
  ) => {
    updateTableRows(tableKey, (rows) =>
      rows.map((row) => {
        if (row.id !== rowId) return row;
        const nextValue = field === 'name' ? toSentenceCase(value) : value;
        if (field === 'weight') {
          const updated = { ...row, weight: nextValue };
          const oldWeight = Number(row.weight);
          const oldCal = Number(row.calories);
          // Recalculate calories proportionally when weight changes (e.g. 100g @ 200 kcal → 200g → 400 kcal)
          if (Number.isFinite(oldWeight) && oldWeight > 0 && Number.isFinite(oldCal)) {
            const newWeight = Number(nextValue);
            if (Number.isFinite(newWeight)) {
              updated.calories = String(Math.round((newWeight / oldWeight) * oldCal));
            }
          }
          return updated;
        }
        return { ...row, [field]: nextValue };
      })
    );
  };

  const handleAddContent = () => {
    const newId = Date.now().toString();
    updateTableRows('base', (rows) => [
      { id: newId, name: '', weight: '', calories: '' },
      ...rows,
    ]);
    setEditingRowId(newId);
  };

  const hasIncompleteRow = useMemo(
    () =>
      dishTables.some((table) =>
        table.rows.some((row) => !row.name.trim() || !row.calories.trim())
      ),
    [dishTables]
  );

  const canAddBaseIngredient = useMemo(
    () => !dishContents.some((row) => !row.name.trim() || !row.calories.trim()),
    [dishContents]
  );

  const renderDishTable = (table: DishTableSection) => {
    const totals = getTableTotals(table.rows);
    const firstColumnHeader =
      table.key === 'highCalorie'
        ? 'High Calorie Content'
        : table.key === 'hiddenContent'
          ? 'Hidden Calorie Content'
          : 'Dish Contents';

    return (
      <Pressable
        key={table.key}
        ref={table.key === 'base' ? tableContainerRef : undefined}
        style={styles.tableContainer}
        onPress={() => {
          if (editingRowId) {
            setEditingRowId(null);
            focusedInputRef.current = null;
          }
        }}
      >
        <View style={styles.tableHeader}>
          <Text style={[styles.tableHeaderText, { flex: 2 }]}>{firstColumnHeader}</Text>
          <Text style={[styles.tableHeaderText, { flex: 1 }]}>Weight</Text>
          <Text style={[styles.tableHeaderText, { flex: 1 }]}>Kcal</Text>
          <Text style={[styles.tableHeaderText, { width: 50 }]}>Action</Text>
        </View>

        {table.rows.map((row) => {
          const isEditing = editingRowId === row.id;
          const slug = normalizeAssetName(row.name);
          const hasDepth = !!geminiDepthAssets.ingredients[slug];
          const isSelectedDepthRow = selectedDepthIngredient != null && selectedDepthIngredient !== '__full__' && normalizeAssetName(selectedDepthIngredient) === slug;
          const RowContainer: any = isEditing ? View : TouchableOpacity;
          return (
            <RowContainer
              key={row.id}
              ref={(ref: any) => { rowRefs.current[row.id] = ref; }}
              style={[styles.tableRow, isSelectedDepthRow && styles.tableRowSelected]}
              activeOpacity={hasDepth ? 0.85 : 1}
              onPress={() => {
                if (!row.name.trim() || !hasDepth) return;
                if (isSelectedDepthRow) {
                  setSelectedDepthIngredient(null);
                } else {
                  setIsVideoPlaying(false);
                  setSelectedDepthIngredient(row.name);
                }
              }}
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
                    onChangeText={(value) => handleUpdateRow(table.key, row.id, 'name', value)}
                    onFocus={() => {
                      handleInputFocus(row.id, 'name');
                      scrollToInput(row.id);
                    }}
                    placeholder="Ingredient name"
                    placeholderTextColor="#D1D5DB"
                    keyboardType="default"
                    editable
                    blurOnSubmit={false}
                    returnKeyType="next"
                    onSubmitEditing={() => {
                      setTimeout(() => {
                        inputRefs.current[row.id]?.weight?.focus();
                      }, 100);
                    }}
                  />
                ) : (
                  <Text style={styles.tableCellText}>{toSentenceCase(row.name) || '—'}</Text>
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
                    onChangeText={(value) => handleUpdateRow(table.key, row.id, 'weight', value)}
                    onFocus={() => {
                      handleInputFocus(row.id, 'weight');
                      scrollToInput(row.id);
                    }}
                    onSubmitEditing={() => {
                      setTimeout(() => {
                        inputRefs.current[row.id]?.calories?.focus();
                      }, 100);
                    }}
                    placeholder="Grams"
                    placeholderTextColor="#D1D5DB"
                    keyboardType={Platform.OS === 'ios' ? 'numbers-and-punctuation' : 'numeric'}
                    editable
                    blurOnSubmit={false}
                    returnKeyType="next"
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
                    onChangeText={(value) => handleUpdateRow(table.key, row.id, 'calories', value)}
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
                    editable
                    blurOnSubmit
                    returnKeyType="done"
                  />
                ) : (
                  <Text style={styles.tableCellText}>{row.calories || '—'}</Text>
                )}
              </View>
              <View style={[styles.tableCell, { width: 50, alignItems: 'center' }]}>
                <TouchableOpacity
                  onPress={() => {
                    if (isEditing) {
                      handleDelete(table.key, row.id);
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
            </RowContainer>
          );
        })}

        {table.rows.length > 0 ? (
          <View style={[styles.tableRow, styles.tableTotalRow]}>
            <View style={[styles.tableCell, { flex: 2 }]}>
              <Text style={styles.tableTotalText}>Total</Text>
            </View>
            <View style={[styles.tableCell, { flex: 1 }]}>
              <Text style={styles.tableTotalText}>
                {totals.totalWeight > 0 ? `${Math.round(totals.totalWeight)} g` : '—'}
              </Text>
            </View>
            <View style={[styles.tableCell, { flex: 1 }]}>
              <Text style={styles.tableTotalText}>{Math.round(totals.totalCalories)}</Text>
            </View>
            <View style={[styles.tableCell, { width: 50 }]} />
          </View>
        ) : null}
      </Pressable>
    );
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
            // TRELLIS MP4: for image jobs, use trellis_mp4_url as the "overlay video" (same UI as video pipeline)
            const trellisMP4Url = item.trellis_mp4_url ?? null;
            const depthUri = !overlayLoadFailed ? selectedDepthUri : null;
            const displayUri = depthUri || overlayUri || resolvedImageUri || null;
            const videoThumbnailUri = resolvedImageUri || item.imageUri || overlayUri || null;
            const showImageLoader = !isVideo && (!!depthUri || !trellisMP4Url) && !!displayUri;
            // Use resolvedVideoUri directly — it is pre-initialised to item?.videoUri so
            // the player always has a URI while the presigned URL fetch is in-flight.
            const originalVideoUri = resolvedVideoUri ?? null;
            const videoSource = isVideo ? (isVideoPlaying ? (videoOverlayUrl || originalVideoUri) : originalVideoUri) : null;
            return (
          <>
          {isVideo && (videoSource || resolvingVideoUri) ? (
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
              {originalVideoUri ? (
                <Video
                  ref={originalVideoRef}
                  key={originalVideoUri}
                  source={{ uri: originalVideoUri }}
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
                    if (!status.isLoaded && (status as any).error) {
                      // Video URI failed to load — clear so thumbnail is shown instead
                      setResolvedVideoUri(undefined);
                    }
                  }}
                />
              ) : (
                // No URI yet (presigned fetch still in progress) — show placeholder
                <View style={[styles.media, styles.placeholder]} />
              )}
              {/* Thumbnail overlay: shown when not playing to mask the Video component's
                  black initialization frame. Fades out once the user presses play. */}
              {!isVideoPlaying && videoThumbnailUri && (
                <OptimizedImage
                  source={{ uri: videoThumbnailUri }}
                  style={[styles.media, StyleSheet.absoluteFillObject]}
                  resizeMode="cover"
                  cachePolicy="memory-disk"
                  priority="normal"
                />
              )}
              {!isVideoPlaying && (
                <TouchableOpacity
                  style={styles.playButtonOverlay}
                  onPress={handleVideoPlay}
                  activeOpacity={0.7}
                >
                  <View style={styles.playButton}>
                    <Ionicons name={resolvingVideoUri ? 'hourglass-outline' : 'play'} size={28} color="#FFFFFF" />
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
          ) : depthUri ? (
            // Ingredient depth map selected — tap to deselect and return to normal view
            <TouchableOpacity
              style={styles.mediaTouchable}
              activeOpacity={1}
              onPress={() => { setSelectedDepthIngredient(null); }}
            >
              <OptimizedImage
                source={{ uri: depthUri }}
                style={styles.media}
                resizeMode="cover"
                cachePolicy="memory-disk"
                priority="high"
                onImageLoad={() => setMediaLoading(false)}
              />
            </TouchableOpacity>
          ) : trellisMP4Url ? (
            // Image job with TRELLIS MP4 — same play/pause UI as the video pipeline
            <>
              <Video
                ref={overlayVideoRef}
                key={trellisMP4Url}
                source={{ uri: trellisMP4Url }}
                style={[styles.media, { position: 'absolute', top: 0, left: 0, opacity: isVideoPlaying ? 1 : 0 }]}
                resizeMode={ResizeMode.COVER}
                isLooping
                isMuted
                shouldPlay={isVideoPlaying}
                useNativeControls={false}
                onPlaybackStatusUpdate={(status) => {
                  if (!status.isLoaded && (status as any).error) {
                    setVideoOverlayError(true);
                  }
                }}
              />
              {videoThumbnailUri && (
                <OptimizedImage
                  source={{ uri: videoThumbnailUri }}
                  style={[styles.media, StyleSheet.absoluteFillObject, { opacity: isVideoPlaying ? 0 : 1 }]}
                  resizeMode="cover"
                  cachePolicy="memory-disk"
                  priority="normal"
                  onImageLoad={() => setMediaLoading(false)}
                />
              )}
              <TouchableOpacity
                style={styles.playButtonOverlay}
                onPress={handleVideoPlay}
                activeOpacity={0.7}
              >
                <View style={styles.playButton}>
                  <Ionicons name={isVideoPlaying ? 'pause' : 'play'} size={28} color="#FFFFFF" />
                </View>
              </TouchableOpacity>
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
          {/* Row 1: meal name */}
          <View ref={mealNameContainerRef}>
            {editingMealName ? (
              <TextInput
                style={[styles.mealNameInput, styles.inputFocused]}
                value={mealName}
                onChangeText={(value) => setMealName(toSentenceCase(value))}
                onBlur={() => setEditingMealName(false)}
                onFocus={() => scrollToInput()}
                placeholder="Meal name"
                placeholderTextColor="#D1D5DB"
                autoFocus
              />
            ) : (
              <TouchableOpacity onPress={() => {
                if (selectedDepthIngredient) {
                  setSelectedDepthIngredient(null);
                } else {
                  setEditingMealName(true);
                }
              }}>
                <Text style={styles.mealName}>{toSentenceCase(mealName)}</Text>
              </TouchableOpacity>
            )}
          </View>

          {/* Row 2: [3 view buttons] [kcal flex] [Add Ingredient] */}
          <View style={styles.mealActionsRow}>
            <View style={styles.mediaActionButtons}>
              {/* 3D TRELLIS button */}
              <TouchableOpacity
                style={[styles.mediaActionButton, isVideoPlaying && styles.mediaActionButtonActive]}
                onPress={() => { setSelectedDepthIngredient(null); handleVideoPlay(); }}
                activeOpacity={0.7}
              >
                <Ionicons name="cube-outline" size={18} color={isVideoPlaying ? '#FFFFFF' : '#6B7280'} />
              </TouchableOpacity>
              {/* Metric depth map button */}
              <TouchableOpacity
                style={[styles.mediaActionButton, selectedDepthIngredient === '__full__' && styles.mediaActionButtonActive]}
                onPress={() => {
                  setIsVideoPlaying(false);
                  setSelectedDepthIngredient(selectedDepthIngredient === '__full__' ? null : '__full__');
                }}
                activeOpacity={0.7}
              >
                <Ionicons name="analytics-outline" size={18} color={selectedDepthIngredient === '__full__' ? '#FFFFFF' : '#6B7280'} />
              </TouchableOpacity>
              {/* Tagging overlay button */}
              <TouchableOpacity
                style={[styles.mediaActionButton, selectedDepthIngredient === '__tagged__' && styles.mediaActionButtonActive]}
                onPress={() => {
                  setIsVideoPlaying(false);
                  setSelectedDepthIngredient(selectedDepthIngredient === '__tagged__' ? null : '__tagged__');
                }}
                activeOpacity={0.7}
              >
                <Ionicons name="scan-outline" size={18} color={selectedDepthIngredient === '__tagged__' ? '#FFFFFF' : '#6B7280'} />
              </TouchableOpacity>
            </View>

            <Text style={styles.mealCalories}>
              {dishTables.every((table) => table.rows.length === 0) ? '-' : `${totalCalories} Kcal`}
            </Text>

            <TouchableOpacity
              style={[styles.addButton, !canAddBaseIngredient && styles.addButtonDisabled]}
              onPress={() => canAddBaseIngredient && handleAddContent()}
              activeOpacity={canAddBaseIngredient ? 0.8 : 1}
            >
              <View style={styles.addButtonIcon}>
                <Text style={styles.addButtonIconText}>+</Text>
              </View>
              <Text style={styles.addButtonText}>Add Ingredient</Text>
            </TouchableOpacity>
          </View>
        </View>

        {dishTables
          .filter((table) => table.key === 'base' || table.rows.length > 0)
          .map((table) => renderDishTable(table))}
            </ScrollView>
          </View>
        </TouchableWithoutFeedback>
      </KeyboardAvoidingView>

      {/* Next Button - Fixed at Bottom */}
      <BottomButtonContainer>
        <TouchableOpacity
          style={[styles.nextButton, hasIncompleteRow && styles.nextButtonDisabled]}
          onPress={() => {
            if (hasIncompleteRow) return;

            setEditingRowId(null);
            Keyboard.dismiss();

            // Navigate immediately with latest data; save in the background
            const updatedItem = {
              ...item,
              mealName,
              dishTables,
              dishContents,
              nutritionalInfo: {
                ...item.nutritionalInfo,
                calories: totalCalories,
              },
            };
            (navigation as any).navigate('Feedback', { item: updatedItem });
            saveChanges();
          }}
          disabled={hasIncompleteRow}
        >
          <Text style={styles.nextButtonText}>Next</Text>
        </TouchableOpacity>
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
    flex: 1,
    fontSize: 16,
    fontWeight: '600',
    color: '#6B7280',
    textAlign: 'center',
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
  mealActionsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 10,
    gap: 8,
  },
  mediaActionButtons: {
    flexDirection: 'row',
    gap: 6,
  },
  mediaActionButton: {
    width: 40,
    height: 40,
    borderRadius: 8,
    backgroundColor: '#F3F4F6',
    borderWidth: 1,
    borderColor: '#E5E7EB',
    alignItems: 'center',
    justifyContent: 'center',
  },
  mediaActionButtonActive: {
    backgroundColor: '#7BA21B',
    borderColor: '#7BA21B',
  },
  tableContainer: {
    paddingHorizontal: 16,
    paddingTop: 16,
  },
  tableTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1F2937',
    marginBottom: 8,
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
  tableRowSelected: {
    backgroundColor: '#F5F9EA',
  },
  tableCell: {
    paddingHorizontal: 8,
    justifyContent: 'center',
  },
  tableTotalRow: {
    backgroundColor: '#F9FAFB',
    borderBottomColor: '#E5E7EB',
  },
  tableTotalText: {
    fontSize: 14,
    fontWeight: '700',
    color: '#1F2937',
  },
  tableSummary: {
    marginTop: 10,
    fontSize: 13,
    fontWeight: '600',
    color: '#4B5563',
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
