import React, { useRef } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Image,
  Dimensions,
  ScrollView,
  Animated,
  Alert,
  Linking,
} from 'react-native';
import { Camera } from 'react-native-vision-camera';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { PanGestureHandler, State } from 'react-native-gesture-handler';
import { useAppSelector } from '../store/hooks';
import ScreenLoader from '../components/ScreenLoader';
import { useImageLoadTracker } from '../hooks/useImageLoadTracker';
import ImageWithLoader from '../components/ImageWithLoader';
import AppHeader from '../components/AppHeader';
import BottomButtonContainer from '../components/BottomButtonContainer';

const { width } = Dimensions.get('window');

interface TutorialScreenProps {
  onBack?: () => void;
}

export default function TutorialScreen({ onBack }: TutorialScreenProps = {}) {
  const navigation = useNavigation<any>();
  const user = useAppSelector((state) => state.auth.user);
  const businessProfile = useAppSelector((state) => state.profile.businessProfile);
  const swipePosition = useRef(new Animated.Value(0)); // For swipe left gesture to go back

  // Track tutorial images loading - wait for actual images to load
  const { isLoading: isImageLoading, handleImageLoad } = useImageLoadTracker({
    imageCount: 2, // We have 2 tutorial images
    minLoadTime: 400,
  });

  // Use business name as username, fallback to email if business name not available
  const userName = businessProfile?.businessName || user?.email?.split('@')[0] || 'User';
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

  const handleSnapDish = async () => {
    const status = Camera.getCameraPermissionStatus();

    if (status === 'granted') {
      navigation.navigate('Camera' as never);
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

    // 'not-determined' — show the native iOS permission popup (appears over this screen)
    const result = await Camera.requestCameraPermission();
    if (result === 'granted') {
      navigation.navigate('Camera' as never);
    } else {
      Alert.alert(
        'UKcal would like to access your camera',
        'UKcal would like to access your camera',
        [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Open Settings', onPress: () => Linking.openSettings() },
        ]
      );
    }
  };

  // Handle swipe left gesture to go back
  const handleSwipeStateChange = (event: any) => {
    const { state, translationX } = event.nativeEvent;
    
    if (state === State.END) {
      const threshold = -100; // Swipe threshold to trigger going back (negative for left swipe)
      const currentValue = translationX || 0;
      
      if (currentValue < threshold && onBack) {
        // Swiped left enough - go back to ResultsScreen
        onBack();
      }
      
      // Always reset position after gesture ends
      Animated.spring(swipePosition.current, {
        toValue: 0,
        useNativeDriver: true,
        tension: 50,
        friction: 7,
      }).start();
    }
  };

  return (
    <ScreenLoader isLoading={isImageLoading}>
      <PanGestureHandler
        onGestureEvent={Animated.event(
          [{ nativeEvent: { translationX: swipePosition.current } }],
          { 
            useNativeDriver: true,
            listener: (event: any) => {
              // Clamp to only allow left swipe (negative values)
              const { translationX: tx } = event.nativeEvent;
              if (tx > 0) {
                swipePosition.current.setValue(0);
              }
            }
          }
        )}
        onHandlerStateChange={handleSwipeStateChange}
        activeOffsetX={-10}
        failOffsetY={[-5, 5]}
      >
        <Animated.View style={{ flex: 1 }}>
          <SafeAreaView style={styles.container} edges={['top']}>
        <ScrollView 
          style={styles.scrollView}
          contentContainerStyle={[styles.contentContainer, { paddingBottom: 100 }]}
          showsVerticalScrollIndicator={false}
          decelerationRate="normal"
          bounces={true}
          scrollEventThrottle={16}
          overScrollMode="never"
          nestedScrollEnabled={true}
        >
          <AppHeader
            displayName={displayName}
            lastLoginDate={lastLoginDate}
            lastLoginTime={lastLoginTime}
            onProfilePress={() => {
              try {
                navigation.navigate('Profile');
              } catch (error) {
                console.error('[Tutorial] Error navigating to Profile:', error);
              }
            }}
          />

          {/* Title */}
          <Text style={styles.mainTitle}>
            From food plate to calories in two easy steps Snap and See!
          </Text>

          {/* Steps Container */}
          <View style={styles.stepsWrapper}>
            {/* Step 1: Snap a Dish */}
            <View style={[styles.stepContainer, styles.stepContainerStep1]}>
              <Text style={styles.stepTitle}>Step 1: Snap a Dish</Text>
              
              <View style={styles.pointsContainer}>
                <View style={styles.bulletPointFullWidth}>
                  <View style={styles.bulletDot} />
                  <Text style={styles.instructionTextFullWidth}>
                    Place the meal on a flat, well-lit surface.
                  </Text>
                </View>
              </View>
              
              <View style={styles.stepContent}>
                <View style={styles.instructionsContainer}>
                  <View style={styles.bulletPoint}>
                    <View style={styles.bulletDot} />
                    <Text style={styles.instructionText}>
                      Take a photo or short 5-second video from above so the whole meal is visible.
                    </Text>
                  </View>
                  
                  <View style={styles.bulletPoint}>
                    <View style={styles.bulletDot} />
                    <Text style={styles.instructionText}>
                      Provide supplemental text (optional), preview, and submit.
                    </Text>
                  </View>
                </View>

                <View style={[styles.illustrationContainer, styles.illustrationContainerStep1]}>
                  <ImageWithLoader
                    source={require('../icons/image.png')} 
                    style={styles.illustrationImage}
                    onImageLoad={handleImageLoad}
                  />
                </View>
              </View>
            </View>

            {/* Step 2: See AI on plate */}
            <View style={[styles.stepContainer, styles.stepContainerStep2]}>
              <Text style={styles.stepTitle}>Step 2: See AI on plate</Text>
              
              <View style={styles.stepContent}>
                <View style={styles.instructionsContainer}>
                  <View style={styles.bulletPoint}>
                    <View style={styles.bulletDot} />
                    <Text style={styles.instructionText}>
                      Review the AI-generated results
                    </Text>
                  </View>
                  
                  <View style={styles.bulletPoint}>
                    <View style={styles.bulletDot} />
                    <Text style={styles.instructionText}>
                      Edit the details if needed.
                    </Text>
                  </View>
                  
                  <View style={styles.bulletPoint}>
                    <View style={styles.bulletDot} />
                    <Text style={styles.instructionText}>
                      Provide quick feedback (optional).
                    </Text>
                  </View>
                  
                  <View style={styles.bulletPoint}>
                    <View style={styles.bulletDot} />
                    <Text style={styles.instructionText}>
                      Save the results
                    </Text>
                  </View>
                </View>

                <View style={[styles.illustrationContainer, styles.illustrationContainerStep2]}>
                  <ImageWithLoader
                    source={require('../icons/image_2.png')} 
                    style={styles.illustrationImage}
                    onImageLoad={handleImageLoad}
                  />
                </View>
              </View>
            </View>
          </View>

        </ScrollView>

        {/* Bottom Action Button - Fixed at Bottom */}
        <BottomButtonContainer>
          <TouchableOpacity 
            style={styles.snapButton}
            onPress={handleSnapDish}
            activeOpacity={0.8}
          >
            <Ionicons name="camera" size={22} color="#FFFFFF" style={styles.cameraIcon} />
            <Text style={styles.snapButtonText}>Snap a Dish</Text>
          </TouchableOpacity>
        </BottomButtonContainer>
      </SafeAreaView>
        </Animated.View>
      </PanGestureHandler>
    </ScreenLoader>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  stepContainerStep1: {
    padding: 15,
  },
  stepContainerStep2: {
    minHeight: 195,
    padding: 15,
    overflow: 'hidden',
  },
  scrollView: {
    flex: 1,
  },
  contentContainer: {
    paddingBottom: 10,
    flexGrow: 1,
  },
  mainTitle: {
    fontSize: 16,
    fontFamily: 'Roboto',
    fontWeight: '700',
    color: '#434343',
    textAlign: 'center',
    marginHorizontal: 20,
    marginTop: 12,
    marginBottom: 12,
    lineHeight: 21,
  },
  stepsWrapper: {
    flexDirection: 'column',
    marginHorizontal: 20,
    gap: 20,
    marginBottom: 16,
  },
  stepContainer: {
    backgroundColor: '#EDF5DE',
    borderRadius: 12,
    padding: 16,
    minHeight: 200,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
    overflow: 'hidden',
  },
  stepTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1F2937',
    marginBottom: 12,
  },
  stepContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: 8,
  },
  instructionsContainer: {
    flex: 1,
    minWidth: 0,
    width: '100%',
  },
  pointsContainer: {
    width: '100%',
    marginBottom: -8,
    marginLeft: -16,
    marginRight: -16,
    paddingVertical: 12,
    paddingHorizontal: 16,
    backgroundColor: '#EDF5DE',
    borderRadius: 0,
    alignSelf: 'stretch',
  },
  bulletPoint: {
    flexDirection: 'row',
    marginBottom: 8,
    alignItems: 'flex-start',
  },
  bulletPointFullWidth: {
    flexDirection: 'row',
    marginBottom: 8,
    alignItems: 'center',
    width: '100%',
  },
  bulletDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#7BA21B',
    marginRight: 8,
    marginTop: 5,
    flexShrink: 0,
  },
  instructionText: {
    flex: 1,
    fontSize: 13,
    fontFamily: 'Roboto',
    fontWeight: '400',
    color: '#6B7280',
    lineHeight: 18,
  },
  instructionTextFullWidth: {
    fontSize: 13,
    fontFamily: 'Roboto',
    fontWeight: '400',
    color: '#6B7280',
    lineHeight: 18,
    flex: 1,
    flexShrink: 1,
  },
  illustrationContainer: {
    alignItems: 'flex-end',
    justifyContent: 'flex-end',
    width: 120,
    height: 150,
    flexShrink: 0,
    overflow: 'hidden',
    marginRight: -22,
    marginBottom: -16,
  },
  illustrationContainerStep1: {
    alignSelf: 'flex-end',
  },
  illustrationContainerStep2: {
    marginRight: -22,
    marginBottom: -32,
  },
  illustrationImage: {
    width: '100%',
    height: '100%',
    borderRadius: 0,
    backgroundColor: 'transparent',
  },
  bottomButtonContainer: {
    backgroundColor: '#FFFFFF',
    paddingBottom: 20,
    paddingTop: 16,
    paddingHorizontal: 20,
    marginTop: 20,
    borderTopWidth: 1,
    borderTopColor: '#E5E7EB',
  },
  snapButton: {
    height: 56, // Fixed height
    width: '100%', // Fixed width
    backgroundColor: '#7BA21B',
    borderRadius: 8,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#7BA21B',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  cameraIcon: {
    marginRight: 8,
  },
  snapButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
  },
});

