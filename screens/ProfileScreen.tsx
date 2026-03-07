import React, { useState } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ScrollView,
  StatusBar,
  Platform,
  Alert,
  Image,
  Linking,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { logout, withdrawParticipation } from '../store/slices/authSlice';
import { updateProfileImage, setAvatar as setAvatarAction } from '../store/slices/profileSlice';
import { useNavigation } from '@react-navigation/native';
import VectorBackButton from '../components/VectorBackButton';
import { testSentryError, testSentryMessage } from '../utils/testSentry';
import { captureException } from '../utils/sentry';
import OptimizedImage from '../components/OptimizedImage';
import BottomButtonContainer from '../components/BottomButtonContainer';
import { safeGoBack } from '../utils/navigationHelpers';

export default function ProfileScreen() {
  const dispatch = useAppDispatch();
  const user = useAppSelector((state) => state.auth.user);
  const profileState = useAppSelector((state) => state.profile);
  const navigation = useNavigation();
  
  // Get avatar and profileImage from Redux
  const avatar = profileState.avatar;
  const profileImage = profileState.profileImage;
  const [isPickerLoading, setIsPickerLoading] = useState(false);

  const selectProfileImage = async () => {
    if (isPickerLoading) return;

    setIsPickerLoading(true);
    try {
      const { granted } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!granted) {
        Alert.alert(
          'Permission Required',
          'UKcal would like to access your photos.',
          [
            { text: 'Cancel', style: 'cancel' },
            { text: 'Open Settings', onPress: () => Linking.openSettings() },
          ]
        );
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.8,
      });

      if (!result.canceled) {
        const imageUri = result.assets[0].uri;
        await dispatch(updateProfileImage(imageUri));
        await dispatch(setAvatarAction(undefined));
      }
    } catch (error) {
      console.error('Error picking image:', error);
      Alert.alert(
        'Permission Required',
        'UKcal would like to access your photos.',
        [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Open Settings', onPress: () => Linking.openSettings() },
        ]
      );
    } finally {
      setIsPickerLoading(false);
    }
  };

  const handleAddAvatar = () => {
    (navigation.navigate as any)('AddAvatar', { fromProfile: true });
  };

  const handleLogout = async () => {
    Alert.alert(
      'Logout',
      'Are you sure you want to logout?',
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Logout',
          style: 'destructive',
          onPress: async () => {
            try {
              // Clear consent and profile completion flags
              // Note: User account is preserved for future login
              await AsyncStorage.removeItem('user_consent');
              await AsyncStorage.removeItem('business_profile_completed');
              await AsyncStorage.removeItem('consent_date');
              
              // Call logout which will clear auth state but preserve user account
              await dispatch(logout());
              console.log('[Profile] User logged out successfully');
            } catch (error) {
              console.error('[Profile] Error during logout:', error);
              captureException(error instanceof Error ? error : new Error(String(error)), {
                context: 'Profile - Logout',
              });
            }
          },
        },
      ]
    );
  };

  const handleEditProfile = () => {
    try {
      console.log('[Profile] Navigating to EditProfileStep1');
      console.log('[Profile] Navigation object:', navigation);
      console.log('[Profile] Navigation type:', typeof navigation);
      console.log('[Profile] Has navigate method:', typeof navigation?.navigate === 'function');
      
      if (navigation && typeof navigation.navigate === 'function') {
        const result = navigation.navigate('EditProfileStep1' as never);
        console.log('[Profile] Navigation result:', result);
      } else {
        console.error('[Profile] Navigation object is invalid:', navigation);
        Alert.alert('Error', 'Navigation is not available');
      }
    } catch (error) {
      console.error('[Profile] Error navigating to EditProfileStep1:', error);
      console.error('[Profile] Error stack:', error instanceof Error ? error.stack : 'No stack');
      Alert.alert('Navigation Error', `Unable to open edit profile: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const handleViewConsent = () => {
    navigation.navigate('ViewConsent' as never);
  };

  const handleContactUs = () => {
    Alert.alert('Contact Us', 'Email: prabodh@nyu.edu\nPhone: +1-3477650774');
  };

  const handleSendFeedback = () => {
    navigation.navigate('SendFeedback' as never);
  };

  const handleWithdrawParticipation = () => {
    Alert.alert(
      'Delete Account',
      'To withdraw participation, please delete your account',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete Account',
          style: 'destructive',
          onPress: () => {
            navigation.navigate('DeleteAccount' as never);
          },
        },
      ],
      { cancelable: true }
    );
  };

  const handleDeleteAccount = () => {
    Alert.alert(
      'Delete Account',
      'Are you sure you want to delete your account?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Continue',
          style: 'destructive',
          onPress: () => {
            navigation.navigate('DeleteAccount' as never);
          },
        },
      ]
    );
  };

        const menuItems = [
          {
            title: 'Edit Profile',
            onPress: handleEditProfile,
          },
          {
            title: 'View Consent',
            onPress: handleViewConsent,
          },
    {
      title: 'Contact Us',
      onPress: handleContactUs,
    },
    {
      title: 'Send Feedback',
      onPress: handleSendFeedback,
    },
    {
      title: 'Withdraw Participation',
      onPress: handleWithdrawParticipation,
    },
    {
      title: 'Delete Account',
      onPress: handleDeleteAccount,
      isLast: true,
    },
    /*
    {
      title: '🔍 View Debug Info (Terminal)',
      onPress: handleViewDebugInfo,
    },
    {
      title: '🧪 Test Sentry (Send Test Error)',
      onPress: () => {
        Alert.alert(
          'Test Sentry',
          'This will send a test error to Sentry. Check your Sentry dashboard to verify it\'s working.',
          [
            { text: 'Cancel', style: 'cancel' },
            {
              text: 'Send Test Error',
              onPress: () => {
                testSentryError();
                Alert.alert('Test Sent', 'Check your Sentry dashboard at https://sentry.io to see the test error!');
              },
            },
            {
              text: 'Send Test Message',
              onPress: () => {
                testSentryMessage();
                Alert.alert('Test Sent', 'Check your Sentry dashboard at https://sentry.io to see the test message!');
              },
            },
          ]
        );
      },
    },
    */
  ];

  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#FFFFFF" />
      
      {/* Header */}
      <View style={styles.header}>
        <VectorBackButton onPress={() => safeGoBack(navigation as any, 'Results')} />
        <Text style={styles.headerTitle}>Profile</Text>
        <View style={{ width: 40 }} />
      </View>

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={[styles.scrollContent, { paddingBottom: 100 }]}
        showsVerticalScrollIndicator={false}
        decelerationRate="normal"
        bounces={true}
        scrollEventThrottle={16}
        overScrollMode="never"
        nestedScrollEnabled={true}
      >
              {/* Profile Avatar / Image */}
              <View style={styles.avatarContainer}>
                <TouchableOpacity
                  onPress={handleAddAvatar}
                  activeOpacity={0.7}
                  style={styles.avatarTouchable}
                >
                  {avatar ? (
                    <Image 
                      source={avatar.imgSrc} 
                      style={styles.avatarImage}
                      resizeMode="cover"
                    />
                  ) : profileImage ? (
                    <OptimizedImage 
                      source={{ uri: profileImage }} 
                      style={styles.avatarImage}
                      resizeMode="cover"
                      cachePolicy="memory-disk"
                      priority="high"
                    />
                  ) : (
                    <View style={styles.avatarPlaceholder}>
                      <Ionicons name="person" size={64} color="#9CA3AF" />
                    </View>
                  )}
                </TouchableOpacity>
                <View style={styles.imageButtonsContainer}>
                  <TouchableOpacity style={[styles.selectImageButton, isPickerLoading && { opacity: 0.6 }]} onPress={selectProfileImage} disabled={isPickerLoading}>
                    <Text style={styles.selectImageText}>{isPickerLoading ? 'Opening...' : 'Upload Photo'}</Text>
                  </TouchableOpacity>
                </View>
              </View>

        {/* Menu Items */}
        <View style={styles.menuContainer}>
          {menuItems.map((item, index) => (
            <TouchableOpacity
              key={index}
              style={[
                styles.menuItem,
                item.isLast && styles.menuItemLast,
              ]}
              onPress={item.onPress}
            >
              <Text style={styles.menuItemText}>{item.title}</Text>
            </TouchableOpacity>
          ))}
        </View>

      </ScrollView>

      {/* Logout Button - Fixed at Bottom */}
      <BottomButtonContainer>
        <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
          <Text style={styles.logoutButtonText}>Log Out</Text>
        </TouchableOpacity>
      </BottomButtonContainer>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    paddingTop: Platform.OS === 'web' ? 0 : 40,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
    backgroundColor: '#FFFFFF',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#000000',
    flex: 1,
    marginLeft: 12,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: 20,
    paddingTop: 30,
  },
  avatarContainer: {
    alignItems: 'center',
    marginBottom: 40,
  },
  avatarTouchable: {
    marginBottom: 4,
  },
  avatarPlaceholder: {
    width: 140,
    height: 140,
    borderRadius: 70,
    backgroundColor: '#E5E7EB',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#7BA21B',
  },
  avatarImage: {
    width: 140,
    height: 140,
    borderRadius: 70,
    borderWidth: 2,
    borderColor: '#7BA21B',
  },
  imageButtonsContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    marginTop: 4,
  },
  selectImageButton: {
    backgroundColor: '#7BA21B',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 6,
    alignSelf: 'center',
  },
  selectImageText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '600',
    textAlign: 'center',
  },
  buttonDivider: {
    width: 1,
    height: 20,
    backgroundColor: '#E5E7EB',
  },
  selectAvatarButton: {
    backgroundColor: '#FFFFFF',
    borderWidth: 2,
    borderColor: '#7BA21B',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 6,
    flex: 1,
  },
  selectAvatarText: {
    color: '#7BA21B',
    fontSize: 14,
    fontWeight: '600',
    textAlign: 'center',
  },
  menuContainer: {
    marginBottom: 20,
  },
  menuItem: {
    paddingVertical: 16,
    paddingHorizontal: 4,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
    marginBottom: 10,
  },
  menuItemLast: {
    borderBottomWidth: 0,
    marginBottom: 0,
  },
  menuItemText: {
    fontSize: 16,
    color: '#1F2937',
    fontWeight: '400',
  },
  logoutButton: {
    height: 56, // Fixed height
    width: '100%', // Fixed width
    backgroundColor: '#7BA21B',
    borderRadius: 28,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
  },
  logoutButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
  },
});

