import React, { useState } from 'react';
import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  TouchableOpacity,
  StatusBar,
  Alert,
  Platform,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useNavigation } from '@react-navigation/native';
import VectorBackButton from '../components/VectorBackButton';
import { safeGoBack } from '../utils/navigationHelpers';
import BottomButtonContainer from '../components/BottomButtonContainer';

interface ConsentScreenProps {
  navigation?: any;
  onConsent?: () => void;
}

export default function ConsentScreen({ navigation: navigationProp, onConsent }: ConsentScreenProps) {
  // Use navigation hook as primary, fallback to prop if provided
  const navigationHook = useNavigation();
  const navigation = navigationProp || navigationHook;
  const [isLoading, setIsLoading] = useState(false);
  const [hasScrolledToBottom, setHasScrolledToBottom] = useState(false);
  
  // Check if we can go back
  const canGoBack = navigation && navigation.canGoBack ? navigation.canGoBack() : false;

  const handleScroll = (event: any) => {
    const { layoutMeasurement, contentOffset, contentSize } = event.nativeEvent;
    
    // Check if user has scrolled to bottom (with 50px threshold)
    const isAtBottom = layoutMeasurement.height + contentOffset.y >= contentSize.height - 50;
    
    if (isAtBottom && !hasScrolledToBottom) {
      setHasScrolledToBottom(true);
    }
  };

  const handleConsent = async () => {
    if (!hasScrolledToBottom) {
      Alert.alert('Please Read All Information', 'You must scroll to the bottom and read all information before consenting.');
      return;
    }
    
    setIsLoading(true);
    try {
      await AsyncStorage.setItem('user_consent', 'true');
      await AsyncStorage.setItem('consent_date', new Date().toISOString());
      
      console.log('[Consent] User accepted terms');
      
      if (onConsent) {
        onConsent();
      }
      
      if (navigation) {
        navigation.navigate('BusinessProfileStep1');
      }
    } catch (error) {
      console.error('[Consent] Error saving consent:', error);
      Alert.alert('Error', 'Failed to save consent. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#FFFFFF" />
      
      {/* Header */}
      <View style={styles.header}>
        <VectorBackButton 
          onPress={() => {
            try {
              if (navigation) {
                // Always navigate to EmailLogin when back button is pressed
                navigation.navigate('EmailLogin' as never);
              }
            } catch (error) {
              console.error('[Consent] Navigation error:', error);
              // If navigation fails, try to go back as fallback
              try {
                if (navigation && navigation.goBack) {
                  navigation.goBack();
                }
              } catch (fallbackError) {
                console.error('[Consent] Fallback navigation error:', fallbackError);
              }
            }
          }} 
        />
        <Text style={styles.headerTitle}>Consent</Text>
        <View style={{ width: 40 }} />
      </View>

      {/* ScrollView with content */}
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        onScroll={handleScroll}
        scrollEventThrottle={16}
        decelerationRate="normal"
        bounces={true}
        overScrollMode="never"
        nestedScrollEnabled={true}
      >
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Study Title and Investigators</Text>
          <Text style={styles.sectionText}>AI-Based Calorie Estimation App for Small and Medium Food Businesses (IRB-FY2025-XXXX)</Text>
          <Text style={styles.sectionText}>Dr. Prabodh Panindre, Dr. Sunil Kumar</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Invitation to Be a Part of a Research Study</Text>
          <Text style={styles.sectionText}>You are invited to participate in a research study. This form has information to help you decide whether or not you wish to participate - please review it carefully. Your participation is voluntary. Please ask any questions you have about the study or about this form before deciding to participate.</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Purpose of the Study</Text>
          <Text style={styles.sectionText}>The purpose of this study is to assess the user interface and experience of an AI-based mobile app for estimating calories in food items sold by small and medium food businesses (SMEs).</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Eligibility to Participate</Text>
          <Text style={styles.sectionText}>You are eligible to participate in this study if you are a food business owner, operator, or staff member (e.g., chef or manager) working in a small or medium-sized enterprise (SME).</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Description of Study Procedures</Text>
          <Text style={styles.sectionText}>Participation in this project will take 30 days. If you agree to participate, you will be asked to download the app, and do the following for a 30-day period:</Text>
          <Text style={styles.sectionText}>1. Download and use the mobile app.</Text>
          <Text style={styles.sectionText}>2. Capture photos or short videos of your menu dishes using the app.</Text>
          <Text style={styles.sectionText}>3. Review AI results, make corrections if needed, and provide quick feedback.</Text>
          <Text style={styles.sectionText}>4. After a 30-day period, the app will request your feedback related to the mobile app's user interface and experience, and the overall project (activity time: 10 minutes).</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Risks or Discomforts</Text>
          <Text style={styles.sectionText}>The smartphone app used as a part of this study will use some of your phone's battery, and it may use cellular data if it's used while not on WIFI. There are no known risks associated with your participation in this project beyond those of everyday life.</Text>
          <Text style={styles.sectionText}>Please tell the researchers if you believe you have been harmed by your participation in the study.</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Benefits</Text>
          <Text style={styles.sectionText}>In this project, you will contribute to the development of an AI-based calorie estimation tool for food businesses.</Text>
          <Text style={styles.sectionText}>You are not expected to directly benefit from participation in the study.</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Voluntary Participation</Text>
          <Text style={styles.sectionText}>Participating in this study is completely voluntary. You may choose not to take part in the study or to stop participating at any time, for any reason, without penalty or negative consequences. You may refuse to participate or withdraw at any time without penalty by clicking on the button located at the top left corner of the app screen or by uninstalling the app. Non-participation or withdrawal will not affect your relationship with NYU or any other entity.</Text>
          <Text style={styles.sectionText}>If you withdraw or are withdrawn from the study early, then we will not keep information about you that has already been collected.</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Privacy & Data Confidentiality</Text>
          <Text style={styles.sectionText}>No information that could be used to identify you will be recorded or linked with the research data collected as a part of this study.</Text>
          <Text style={styles.sectionText}>The data obtained will be fed directly to the AI models in the app without manual intervention. Survey responses will not be linked to the ID or any other personally identifiable information, and will be used for aggregate analysis and program evaluation. After submitting the survey, you may uninstall the app from your smartphone and continue to use your device the way you were using it before participating in the project.</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Future Use of Data</Text>
          <Text style={styles.sectionText}>Information about you will be used by the research team for the research described in this consent form, and the aggregate analysis may be used for future research or shared with other researchers.</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Access to Your Study Information</Text>
          <Text style={styles.sectionText}>We will give you access to the information that is collected about you in this study. Participants can access the information collected about themselves through the app.</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Contact Information</Text>
          <Text style={styles.sectionText}>You are encouraged to ask questions at any time during this study. For information about the study, contact Prabodh Panindre at 6469973860, ppp231@nyu.edu.</Text>
          <Text style={styles.sectionText}>If you have questions about your rights as a research participant or if you believe you have been harmed from the research, please contact the NYU Human Research Protection Program at (212)998-4808 or ask.humansubjects@nyu.edu.</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Agreement to Participate</Text>
          <Text style={styles.sectionText}>By clicking the button provided below, you are agreeing to participate in this study. Make sure you understand what the study involves before you click the button. If you have any questions about the study after you agree to participate, you can contact the research team using the information provided above. If you do not want to participate in the project, you may uninstall the app and quit. You may access this consent agreement by clicking on the button located at the top left corner of the app screen. Thank you for your consideration.</Text>
        </View>

        {/* Extra padding at bottom to ensure last section is visible above button */}
        <View style={{ height: 80 }} />
      </ScrollView>

      {/* Button Container - Fixed at bottom */}
      <BottomButtonContainer>
        <TouchableOpacity
          style={[
            styles.consentButton,
            (!hasScrolledToBottom || isLoading) && styles.consentButtonDisabled
          ]}
          onPress={handleConsent}
          disabled={!hasScrolledToBottom || isLoading}
        >
          <Text style={styles.consentButtonText}>
            {isLoading ? 'Processing...' : hasScrolledToBottom ? 'I Consent' : 'Scroll to bottom to continue'}
          </Text>
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
    borderBottomWidth: 2,
    borderBottomColor: '#E5E7EB',
    backgroundColor: '#FFFFFF',
    zIndex: 10,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: '#000000',
    flex: 1,
    marginLeft: 12,
  },
  scrollView: {
    flex: 1,
    marginBottom: 0,
  },
  scrollContent: {
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 100,
  },
  mainTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#000000',
    marginBottom: 8,
    textAlign: 'center',
  },
  section: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#000000',
    marginBottom: 8,
  },
  sectionText: {
    fontSize: 14,
    lineHeight: 22,
    color: '#000000',
    textAlign: 'justify',
    fontWeight: '400',
  },
  buttonContainer: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    paddingHorizontal: 20,
    paddingVertical: 16,
    paddingBottom: Platform.OS === 'web' ? 20 : 30,
    backgroundColor: '#FFFFFF',
    borderTopWidth: 2,
    borderTopColor: '#E5E7EB',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -4 },
    shadowOpacity: 0.15,
    shadowRadius: 8,
    elevation: 10,
    zIndex: 100,
  },
  consentButton: {
    height: 56, // Fixed height
    width: '100%', // Fixed width
    backgroundColor: '#7BA21B',
    borderRadius: 28,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#7BA21B',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 4,
  },
  consentButtonDisabled: {
    backgroundColor: '#9CA3AF',
    shadowColor: '#000',
    shadowOpacity: 0.1,
  },
  consentButtonText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
});
