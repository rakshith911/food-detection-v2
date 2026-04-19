import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ScrollView,
  Image,
  StatusBar,
  useWindowDimensions,
  Alert,
  Linking,
} from 'react-native';
import { Camera, useCameraDevice, useCameraPermission, useMicrophonePermission } from 'react-native-vision-camera';
import { Video, ResizeMode } from 'expo-av';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { addAnalysis } from '../store/slices/historySlice';
import { mockFoodDetectionService } from '../services/MockFoodDetection';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import { BlurView } from 'expo-blur';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import PreviewScreen from './PreviewScreen';
import { captureException } from '../utils/sentry';
import { safeGoBack } from '../utils/navigationHelpers';

export default function CameraScreen() {
  const navigation = useNavigation();
  const { width, height } = useWindowDimensions();
  const insets = useSafeAreaInsets();

  // Vision Camera hooks
  const device = useCameraDevice('back');
  const { hasPermission: hasCameraPermission, requestPermission: requestCameraPermission } = useCameraPermission();
  const { hasPermission: hasMicPermission, requestPermission: requestMicPermission } = useMicrophonePermission();

  const [flashEnabled, setFlashEnabled] = useState(false);
  const [streakDays, setStreakDays] = useState(1);
  const [activeTab, setActiveTab] = useState<'photo' | 'video'>('photo');
  const [lastMediaMode, setLastMediaMode] = useState<'photo' | 'video'>('photo');
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const cameraRef = useRef<Camera>(null);
  const recordingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const recordingAutoStopRef = useRef<NodeJS.Timeout | null>(null);
  const recordingStartTimeRef = useRef<number>(0);
  const isRecordingRef = useRef<boolean>(false);
  const MAX_RECORDING_SECONDS = 5;
  const isTakingPhotoRef = useRef(false);
  const dispatch = useAppDispatch();
  const user = useAppSelector((state) => state.auth.user);

  useEffect(() => {
    // Load streak from storage
    loadStreak();
  }, []);

  // Debug: Log flash state changes
  useEffect(() => {
    console.log('[Camera] Flash state changed to:', flashEnabled);
    console.log('[Camera] Active tab:', activeTab);
    console.log('[Camera] Torch enabled:', flashEnabled);
  }, [flashEnabled, activeTab]);

  // Hide tab bar when CameraScreen is focused or when PreviewScreen is shown
  useFocusEffect(
    React.useCallback(() => {
      // Hide tab bar and activate camera
      const parent = navigation.getParent();
      if (parent) {
        parent.setOptions({
          tabBarStyle: { display: 'none' },
        });
      }

      // Delay camera activation to avoid race conditions
      const timer = setTimeout(() => {
        setIsCameraActive(true);
      }, 300);

      // Cleanup: show tab bar and deactivate camera when leaving
      return () => {
        clearTimeout(timer);
        setIsCameraActive(false);

        const parent = navigation.getParent();
        if (parent) {
          parent.setOptions({
            tabBarStyle: {
              backgroundColor: '#FFFFFF',
              borderTopWidth: 0,
              elevation: 8,
              shadowColor: '#000',
              shadowOffset: { width: 0, height: -2 },
              shadowOpacity: 0.1,
              shadowRadius: 4,
              paddingBottom: 8,
              paddingTop: 8,
              height: 70,
            },
          });
        }
      };
    }, [navigation])
  );

  // Also hide tab bar when preview is shown
  useEffect(() => {
    if (selectedImage || selectedVideo) {
      const parent = navigation.getParent();
      if (parent) {
        parent.setOptions({
          tabBarStyle: { display: 'none' },
        });
      }
    }
  }, [selectedImage, selectedVideo, navigation]);

  const loadStreak = async () => {
    try {
      const savedStreak = await AsyncStorage.getItem('streakDays');
      if (savedStreak) {
        setStreakDays(parseInt(savedStreak, 10));
      }
      
    } catch (error) {
      console.error('Error loading streak:', error);
    }
  };

  const takePhoto = async () => {
    if (!cameraRef.current || isTakingPhotoRef.current) {
      return;
    }

    try {
      isTakingPhotoRef.current = true;
      const photo = await cameraRef.current.takePhoto({
        flash: flashEnabled ? 'on' : 'off',
      });
      if (photo && photo.path) {
        setSelectedImage('file://' + photo.path);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error('Camera error:', error);
      captureException(error instanceof Error ? error : new Error(String(error)), {
        context: 'Camera - Take Photo',
      });
    } finally {
      isTakingPhotoRef.current = false;
    }
  };

  const pickImageFromGallery = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        allowsEditing: false,
        quality: 1,
      });

      if (!result.canceled && result.assets[0]) {
        setSelectedImage(result.assets[0].uri);
      }
    } catch (error) {
      console.error('Gallery error:', error);
      captureException(error instanceof Error ? error : new Error(String(error)), {
        context: 'Camera - Pick Image from Gallery',
      });
    }
  };

  const pickVideoFromGallery = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['videos'],
        allowsEditing: true,
        videoMaxDuration: MAX_RECORDING_SECONDS,
        quality: 1,
      });

      if (!result.canceled && result.assets[0]) {
        const asset = result.assets[0];
        const durationSec = asset.duration ? asset.duration / 1000 : 0;
        if (durationSec > MAX_RECORDING_SECONDS + 0.5) {
          Alert.alert('Video Too Long', `Please select a video that is ${MAX_RECORDING_SECONDS} seconds or shorter.`);
          return;
        }
        setSelectedVideo(asset.uri);
      }
    } catch (error) {
      console.error('Gallery error:', error);
      captureException(error instanceof Error ? error : new Error(String(error)), {
        context: 'Camera - Pick Video from Gallery',
      });
    }
  };

  const pickFromGallery = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images', 'videos'],
        allowsEditing: false,
        videoMaxDuration: MAX_RECORDING_SECONDS,
        quality: 1,
      });

      if (!result.canceled && result.assets[0]) {
        const asset = result.assets[0];
        if (asset.type === 'video') {
          const durationSec = asset.duration ? asset.duration / 1000 : 0;
          if (durationSec > MAX_RECORDING_SECONDS + 0.5) {
            Alert.alert('Video Too Long', `Please select a video that is ${MAX_RECORDING_SECONDS} seconds or shorter.`);
            return;
          }
          setSelectedVideo(asset.uri);
        } else {
          setSelectedImage(asset.uri);
        }
      }
    } catch (error) {
      console.error('Gallery error:', error);
      captureException(error instanceof Error ? error : new Error(String(error)), {
        context: 'Camera - Pick from Gallery',
      });
    }
  };

  const startRecording = async () => {
    if (!cameraRef.current || isRecording) {
      return;
    }

    // Check if microphone permission is granted
    if (!hasMicPermission) {
      console.log('[Camera] Requesting microphone permission for video recording');
      const granted = await requestMicPermission();
      if (!granted) {
        Alert.alert(
          'Microphone Permission Required',
          'Video recording requires microphone access. Please grant permission in your device settings.',
          [{ text: 'OK' }]
        );
        return;
      }
    }

    try {
      setIsRecording(true);
      isRecordingRef.current = true;
      setRecordingTime(0);
      recordingStartTimeRef.current = Date.now();

      // Start timer
      recordingIntervalRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

      // Auto-stop after 5 seconds
      recordingAutoStopRef.current = setTimeout(async () => {
        console.log('[Camera] Auto-stopping recording at 5s limit');
        try {
          if (cameraRef.current && isRecordingRef.current) {
            await cameraRef.current.stopRecording();
          }
        } catch (e) {
          console.warn('[Camera] Auto-stop error (non-fatal):', e);
        }
      }, MAX_RECORDING_SECONDS * 1000);

      console.log('[Camera] Starting video recording...');

      // Start recording with vision camera
      cameraRef.current.startRecording({
        flash: flashEnabled ? 'on' : 'off',
        onRecordingFinished: (video) => {
          console.log('[Camera] Recording completed successfully', video.path);
          setSelectedVideo('file://' + video.path);
          isRecordingRef.current = false;
          setIsRecording(false);
          setRecordingTime(0);
          if (recordingIntervalRef.current) {
            clearInterval(recordingIntervalRef.current);
            recordingIntervalRef.current = null;
          }
          if (recordingAutoStopRef.current) {
            clearTimeout(recordingAutoStopRef.current);
            recordingAutoStopRef.current = null;
          }
        },
        onRecordingError: (error) => {
          console.error('[Camera] Recording error:', error);
          Alert.alert(
            'Recording Error',
            'Failed to record video. Please try again.',
            [{ text: 'OK' }]
          );
          captureException(new Error(String(error)), {
            context: 'Camera - Video Recording',
          });
          isRecordingRef.current = false;
          setIsRecording(false);
          setRecordingTime(0);
          if (recordingIntervalRef.current) {
            clearInterval(recordingIntervalRef.current);
            recordingIntervalRef.current = null;
          }
          if (recordingAutoStopRef.current) {
            clearTimeout(recordingAutoStopRef.current);
            recordingAutoStopRef.current = null;
          }
        },
      });
    } catch (error) {
      console.error('[Camera] Error starting recording:', error);
      isRecordingRef.current = false;
      setIsRecording(false);
      setRecordingTime(0);
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
        recordingIntervalRef.current = null;
      }
      if (recordingAutoStopRef.current) {
        clearTimeout(recordingAutoStopRef.current);
        recordingAutoStopRef.current = null;
      }
    }
  };

  const stopRecording = async () => {
    if (!cameraRef.current || !isRecordingRef.current) {
      console.log('[Camera] Cannot stop - camera ref or recording state invalid');
      return;
    }
    if (recordingAutoStopRef.current) {
      clearTimeout(recordingAutoStopRef.current);
      recordingAutoStopRef.current = null;
    }
    console.log('[Camera] Stopping recording...');
    try {
      await cameraRef.current.stopRecording();
    } catch (e) {
      console.warn('[Camera] stopRecording error (non-fatal):', e);
    }
  };

  const handleClose = () => {
    // If recording, DON'T stop it - just let it complete in background
    if (isRecording) {
      console.log('[Camera] Close pressed while recording - letting recording complete');
      Alert.alert(
        'Recording in Progress',
        'Please stop recording before closing.',
        [{ text: 'OK' }]
      );
      return;
    }
    
    // Navigate back
    safeGoBack(navigation as any, 'Results');
  };

  const handlePreviewBack = () => {
    setSelectedImage(null);
    setSelectedVideo(null);
  };

  // Must be called from a user gesture (tap) so iOS shows the system Allow/Don't Allow dialog
  const handleRequestCameraPermission = async () => {
    const status = Camera.getCameraPermissionStatus();
    if (status === 'denied' || status === 'restricted') {
      Linking.openSettings();
      return;
    }
    const cameraGranted = await requestCameraPermission();
    if (cameraGranted) {
      await requestMicPermission();
    }
  };

  if (hasCameraPermission === null) {
    // Camera permissions are still loading
    return (
      <View style={styles.permissionScreenContainer}>
        <Text style={styles.permissionMessage}>Loading camera...</Text>
      </View>
    );
  }

  // Check if camera device is available (won't be on simulator)
  const isCameraAvailable = device != null;

  // Auto-request permission or show alert if denied
  useEffect(() => {
    if (hasCameraPermission === false) {
      const status = Camera.getCameraPermissionStatus();
      if (status === 'not-determined') {
        // Show native OS popup directly
        handleRequestCameraPermission();
      } else {
        // Already denied — show alert and go back
        Alert.alert(
          'UKcal would like to access your camera',
          'UKcal would like to access your camera',
          [
            {
              text: 'Cancel',
              style: 'cancel',
              onPress: () => safeGoBack(navigation as any, 'Results'),
            },
            {
              text: 'Open Settings',
              onPress: () => {
                Linking.openSettings();
                safeGoBack(navigation as any, 'Results');
              },
            },
          ],
          { cancelable: false }
        );
      }
    }
  }, [hasCameraPermission]);

  if (!hasCameraPermission) {
    // Show minimal loading state while native popup is visible
    return (
      <View style={styles.permissionScreenContainer}>
        <StatusBar barStyle="dark-content" />
      </View>
    );
  }

  // Show preview screen if image or video is selected
  if (selectedImage || selectedVideo) {
    return (
      <PreviewScreen
        imageUri={selectedImage || undefined}
        videoUri={selectedVideo || undefined}
        onBack={handlePreviewBack}
        onAnalyze={() => {
          // After submit, show the new results feed screen
          // @ts-ignore - stack param list not strictly typed here
          (navigation as any).navigate('Results', { showSubmittedNotification: true });
        }}
      />
    );
  }

  return (
    <View style={[styles.container, { marginBottom: 0, paddingBottom: -insets.bottom }]}>
      <StatusBar barStyle="light-content" />
      
      {/* Main Content - Photo, Video, or Text */}
      <View style={[
        styles.cameraContainer,
        (activeTab === 'photo' || activeTab === 'video') && styles.cameraContainerFull
      ]}>
        {(activeTab === 'photo' || activeTab === 'video') && (
          <>
            {isCameraAvailable ? (
              <Camera
                ref={cameraRef}
                style={styles.camera}
                device={device}
                isActive={isCameraActive && !selectedImage && !selectedVideo}
                photo={true}
                video={true}
                audio={true}
                torch={flashEnabled ? 'on' : 'off'}
              />
            ) : (
              <View style={styles.noCameraContainer}>
                <Ionicons name="camera-outline" size={64} color="#666" />
                <Text style={styles.noCameraText}>Camera not available</Text>
                <Text style={styles.noCameraSubtext}>Use Gallery to select media</Text>
                <TouchableOpacity style={styles.galleryButton} onPress={pickFromGallery}>
                  <Ionicons name="images-outline" size={24} color="#FFFFFF" />
                  <Text style={styles.galleryButtonText}>Open Gallery</Text>
                </TouchableOpacity>
              </View>
            )}
            {/* Top Overlays - Outside CameraView */}
            <View style={[styles.topOverlay, { top: insets.top + 16 }]}>
              {isCameraAvailable && (
                <TouchableOpacity
                  style={styles.flashButton}
                  onPress={() => {
                    const newFlashState = !flashEnabled;
                    console.log('[Camera] Toggling torch:', newFlashState);
                    setFlashEnabled(newFlashState);
                  }}
                >
                  <Ionicons
                    name={flashEnabled ? 'flash' : 'flash-off'}
                    size={24}
                    color="#FFFFFF"
                  />
                </TouchableOpacity>
              )}
              {!isCameraAvailable && <View />}

              <View style={styles.topRightControls}>
                <TouchableOpacity
                  style={styles.closeButton}
                  onPress={handleClose}
                >
                  <Ionicons name="close" size={24} color="#FFFFFF" />
                </TouchableOpacity>
              </View>
            </View>

            {activeTab === 'video' && isRecording && (
              <View style={styles.recordingIndicator}>
                <View style={styles.recordingDot} />
                <Text style={styles.recordingText}>Recording: {recordingTime}s</Text>
              </View>
            )}

            {/* Camera Frame Indicators - only when camera is available */}
            {isCameraAvailable && (
              <View style={styles.frameIndicators}>
                <View style={[styles.cornerIndicator, styles.topLeft]} />
                <View style={[styles.cornerIndicator, styles.topRight]} />
                <View style={[styles.cornerIndicator, styles.bottomLeft]} />
                <View style={[styles.cornerIndicator, styles.bottomRight]} />
              </View>
            )}
          </>
        )}

        {/* No TEXT content anymore */}
      </View>

      {/* Bottom Navigation Bar */}
      {
        <BlurView intensity={40} tint="dark" style={[
          styles.blurBar,
          { 
            position: 'absolute', 
            bottom: 0, 
            left: 0, 
            right: 0, 
            zIndex: 100,
            paddingBottom: Math.max(insets.bottom, 24),
          }
        ]} pointerEvents="box-none">
          <View style={styles.blurBarInner}>
          {/* Tab Navigation */}
          <View style={[styles.tabNav]}>
            <TouchableOpacity
              style={styles.tab}
              onPress={() => { setActiveTab('photo'); setLastMediaMode('photo'); }}
            >
              <Text
                style={[
                  styles.tabText,
                  activeTab === 'photo' && styles.tabTextActive,
                ]}
              >
                PHOTO
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.tab}
              onPress={() => { setActiveTab('video'); setLastMediaMode('video'); }}
            >
              <Text
                style={[
                  styles.tabText,
                  activeTab === 'video' && styles.tabTextActive,
                ]}
              >
                VIDEO
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.tab}
              onPress={pickFromGallery}
            >
              <Text
                style={[
                  styles.tabText,
                ]}
              >
                GALLERY
              </Text>
            </TouchableOpacity>
          </View>

           {/* Shutter/Record Button - only show when camera is available */}
           {activeTab === 'photo' && isCameraAvailable && (
             <TouchableOpacity style={styles.shutterButton} onPress={takePhoto}>
               <View style={styles.shutterInner} />
             </TouchableOpacity>
           )}

           {activeTab === 'video' && isCameraAvailable && (
            <TouchableOpacity
              style={[styles.shutterButton, isRecording && styles.recordingButton]}
              onPress={isRecording ? stopRecording : startRecording}
            >
              <View style={[styles.shutterInner, isRecording && styles.recordingInner]} />
            </TouchableOpacity>
          )}
          
           {/* Show gallery prompt when camera not available */}
           {!isCameraAvailable && (activeTab === 'photo' || activeTab === 'video') && (
             <TouchableOpacity style={styles.shutterButton} onPress={pickFromGallery}>
               <Ionicons name="images" size={32} color="#34C759" />
             </TouchableOpacity>
           )}
          </View>
        </BlurView>
      }
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
    overflow: 'hidden',
    marginBottom: 0,
    paddingBottom: 0,
  },
  headerBar: {
    backgroundColor: '#34C759',
    paddingTop: StatusBar.currentHeight || 44,
    paddingBottom: 12,
    paddingHorizontal: 16,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  headerText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  premiumButton: {
    backgroundColor: '#34C759',
    borderWidth: 1,
    borderColor: '#FFFFFF',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 8,
  },
  premiumButtonText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  cameraContainer: {
    flex: 1,
    position: 'relative',
  },
  cameraContainerFull: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    overflow: 'hidden',
    marginBottom: 0,
    paddingBottom: 0,
  },
  camera: {
    flex: 1,
    width: '100%',
    height: '100%',
    marginBottom: 0,
    paddingBottom: 0,
  },
  topOverlay: {
    position: 'absolute',
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    paddingHorizontal: 16,
    zIndex: 10,
  },
  flashButton: {
    padding: 8,
  },
  avatarContainer: {
    alignItems: 'center',
  },
  avatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#FFFFFF',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 4,
  },
  avatarVersion: {
    color: '#FFFFFF',
    fontSize: 10,
    fontWeight: '600',
  },
  topRightControls: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  streakButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 8,
  },
  streakText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '700',
  },
  closeButton: {
    padding: 8,
  },
  carouselContainer: {
    position: 'absolute',
    top: 100,
    left: 0,
    right: 0,
    zIndex: 10,
  },
  carouselContent: {
    paddingHorizontal: 16,
    gap: 12,
  },
  sampleFoodItem: {
    width: 80,
    height: 80,
    borderRadius: 40,
    overflow: 'hidden',
    marginRight: 12,
    borderWidth: 2,
    borderColor: '#FFFFFF',
  },
  sampleFoodImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  instructionText: {
    position: 'absolute',
    top: 200,
    left: 0,
    right: 0,
    textAlign: 'center',
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '500',
    paddingHorizontal: 32,
    zIndex: 10,
    textShadowColor: 'rgba(0, 0, 0, 0.75)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 3,
  },
  frameIndicators: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
  },
  cornerIndicator: {
    position: 'absolute',
    width: 50,
    height: 50,
    borderColor: '#34C759',
    borderWidth: 4,
  },
  topLeft: {
    top: '30%',
    left: '15%',
    borderRightWidth: 0,
    borderBottomWidth: 0,
  },
  topRight: {
    top: '30%',
    right: '15%',
    borderLeftWidth: 0,
    borderBottomWidth: 0,
  },
  bottomLeft: {
    bottom: '40%',
    left: '15%',
    borderRightWidth: 0,
    borderTopWidth: 0,
  },
  bottomRight: {
    bottom: '40%',
    right: '15%',
    borderLeftWidth: 0,
    borderTopWidth: 0,
  },
  previewContainer: {
    flex: 1,
  },
  previewImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  backButton: {
    position: 'absolute',
    top: 16,
    left: 16,
    padding: 8,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    borderRadius: 20,
    zIndex: 10,
  },
  bottomNavBar: {
    backgroundColor: '#5D4037',
    paddingBottom: 24,
    paddingTop: 16,
    borderTopLeftRadius: 0,
    borderTopRightRadius: 0,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: -2,
    },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 8,
  },
  bottomNavBarTransparent: {
    backgroundColor: 'transparent',
    shadowOpacity: 0,
    elevation: 0,
    borderTopWidth: 0,
    borderBottomWidth: 0,
  },
  blurBar: {
    backgroundColor: 'transparent',
    paddingTop: 16,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    overflow: 'hidden',
    marginBottom: 0,
  },
  blurBarInner: {
    paddingTop: 16,
    width: '100%',
    backgroundColor: 'transparent',
  },
  tabNav: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingHorizontal: 16,
    marginBottom: 16,
    backgroundColor: 'transparent',
  },
  tab: {
    flex: 1,
    alignItems: 'center',
    backgroundColor: 'transparent',
  },
  tabText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
    letterSpacing: 0.5,
    textShadowColor: 'rgba(0, 0, 0, 1)',
    textShadowOffset: { width: 0, height: 2 },
    textShadowRadius: 4,
  },
  tabTextActive: {
    color: '#34C759',
    fontWeight: '700',
    textShadowColor: 'rgba(0, 0, 0, 1)',
    textShadowOffset: { width: 0, height: 2 },
    textShadowRadius: 4,
  },
  shutterButton: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderWidth: 5,
    borderColor: '#34C759',
    justifyContent: 'center',
    alignItems: 'center',
    alignSelf: 'center',
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.5,
    shadowRadius: 8,
    elevation: 8,
  },
  shutterInner: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#1C1C1E',
  },
  recordingButton: {
    borderColor: '#EF4444',
  },
  recordingInner: {
    backgroundColor: '#EF4444',
  },
  recordingIndicator: {
    position: 'absolute',
    top: 100,
    left: 16,
    backgroundColor: 'rgba(239, 68, 68, 0.9)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    flexDirection: 'row',
    alignItems: 'center',
    zIndex: 20,
  },
  recordingDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#FFFFFF',
    marginRight: 8,
  },
  recordingText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '700',
  },
  textContainer: {
    flex: 1,
    backgroundColor: '#1C1C1E',
  },
  textContent: {
    padding: 20,
  },
  textHeader: {
    marginBottom: 24,
  },
  textTitle: {
    fontSize: 28,
    fontWeight: '700',
    color: '#FFFFFF',
    marginBottom: 8,
  },
  textSubtitle: {
    fontSize: 16,
    color: '#999',
    lineHeight: 22,
  },
  textInput: {
    backgroundColor: '#2C2C2E',
    borderWidth: 2,
    borderColor: '#3C3C3E',
    borderRadius: 16,
    padding: 16,
    fontSize: 16,
    color: '#FFFFFF',
    minHeight: 200,
    textAlignVertical: 'top',
    marginBottom: 20,
  },
  analyzeTextButton: {
    backgroundColor: '#34C759',
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 20,
    shadowColor: '#34C759',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  analyzeButtonText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '700',
  },
  resultBox: {
    backgroundColor: '#2C2C2E',
    padding: 20,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#34C759',
  },
  resultText: {
    fontSize: 16,
    lineHeight: 24,
    color: '#FFFFFF',
  },
  bottomIcons: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingHorizontal: 16,
  },
  bottomIcon: {
    padding: 8,
  },
  message: {
    textAlign: 'center',
    paddingBottom: 16,
    fontSize: 18,
    color: '#FFFFFF',
    fontWeight: '500',
  },
  noCameraContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#1C1C1E',
    padding: 20,
  },
  noCameraText: {
    color: '#FFFFFF',
    fontSize: 20,
    fontWeight: '600',
    marginTop: 16,
    marginBottom: 8,
  },
  noCameraSubtext: {
    color: '#999',
    fontSize: 14,
    marginBottom: 24,
  },
  galleryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#34C759',
    paddingVertical: 14,
    paddingHorizontal: 24,
    borderRadius: 12,
    gap: 8,
  },
  galleryButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  button: {
    backgroundColor: '#34C759',
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderRadius: 12,
    alignItems: 'center',
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '600',
  },
  permissionScreenContainer: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 24,
  },
  permissionMessage: {
    textAlign: 'center',
    fontSize: 18,
    color: '#1C1C1E',
    fontWeight: '500',
    marginBottom: 24,
  },
  permissionButton: {
    backgroundColor: '#34C759',
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderRadius: 12,
    alignItems: 'center',
  },
  permissionButtonText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '600',
  },
  permissionHint: {
    textAlign: 'center',
    fontSize: 13,
    color: '#666',
    marginTop: 20,
    paddingHorizontal: 16,
  },
});
