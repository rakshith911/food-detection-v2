import React, { useState, useRef, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  TextInput,
  ActivityIndicator,
  Alert,
  StatusBar,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  Keyboard,
} from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import VectorBackButton from '../components/VectorBackButton';
import BottomButtonContainer from '../components/BottomButtonContainer';

interface FeedbackQuestion {
  id: number;
  question: string;
  rating: number | null;
}

const feedbackQuestions: Omit<FeedbackQuestion, 'rating'>[] = [
  {
    id: 1,
    question: 'The purpose of the App and my role in the study were clearly explained.',
  },
  {
    id: 2,
    question: 'The app was easy to install, set up, and use on my smartphone.',
  },
  {
    id: 3,
    question: 'Capturing photos or videos of dishes was simple and well-guided.',
  },
  {
    id: 4,
    question: 'The app correctly identified most of the dishes I uploaded.',
  },
  {
    id: 5,
    question: 'The calorie estimates provided by the app were accurate and useful.',
  },
  {
    id: 6,
    question: 'Editing or providing feedback on the results was straightforward.',
  },
  {
    id: 7,
    question: 'The overall app design and navigation were clear and user-friendly.',
  },
  {
    id: 8,
    question: 'I felt confident that my images and data were stored properly.',
  },
  {
    id: 9,
    question: 'The app could be valuable for small food businesses to understand or share calorie information.',
  },
  {
    id: 10,
    question: 'Overall, I am satisfied with my experience using the app.',
  },
];

export default function SendFeedbackScreen() {
  const insets = useSafeAreaInsets();
  const navigation = useNavigation<any>();
  const [questions, setQuestions] = useState<FeedbackQuestion[]>(
    feedbackQuestions.map(q => ({ ...q, rating: null }))
  );
  const [additionalComments, setAdditionalComments] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [isKeyboardVisible, setIsKeyboardVisible] = useState(false);
  const [keyboardHeight, setKeyboardHeight] = useState(0);
  const [isCommentFocused, setIsCommentFocused] = useState(false);
  const scrollViewRef = useRef<ScrollView>(null);
  const commentsContainerRef = useRef<View>(null);

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

  const totalQuestions = questions.length;
  const currentQuestion = questions[currentQuestionIndex];
  const isLastQuestion = currentQuestionIndex === totalQuestions - 1;
  const isRatingSelected = currentQuestion?.rating !== null;

  const handleRatingChange = (questionId: number, rating: number) => {
    setQuestions(prev =>
      prev.map(q => (q.id === questionId ? { ...q, rating } : q))
    );
  };

  const handleNextQuestion = () => {
    // On first screen (instructions), just move to first question
    if (currentQuestionIndex === 0) {
      setCurrentQuestionIndex(1);
      return;
    }
    
    // On question screens, require rating
    if (!isRatingSelected) {
      Alert.alert('Required', 'Please select a rating before continuing.');
      return;
    }
    setCurrentQuestionIndex(prev => Math.min(prev + 1, totalQuestions - 1));
  };

  const sendFeedback = async () => {
    setIsLoading(true);
    try {
      const feedbackData = {
        questions: questions.map(q => ({
          questionId: q.id,
          question: q.question,
          rating: q.rating,
        })),
        additionalComments: additionalComments.trim() || undefined,
        submittedAt: new Date().toISOString(),
      };

      // TODO: Implement feedback API call
      // await userService.sendFeedback(feedbackData);
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      console.log('[SendFeedback] Feedback data:', JSON.stringify(feedbackData, null, 2));
      
      Alert.alert('', 'We received your feedback. Thank you!');
      navigation.goBack();
    } catch (error) {
      console.error('[SendFeedback] Error sending feedback:', error);
      Alert.alert('Error', 'Failed to send feedback. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const RatingScale = ({ questionId, currentRating }: { questionId: number; currentRating: number | null }) => {
    return (
      <View style={styles.ratingContainer}>
        <View style={styles.ratingScale}>
          {[1, 2, 3, 4, 5].map((rating) => (
            <View key={rating} style={styles.ratingItem}>
              <TouchableOpacity
                style={[
                  styles.ratingButton,
                  currentRating === rating && styles.ratingButtonSelected,
                ]}
                onPress={() => handleRatingChange(questionId, rating)}
                activeOpacity={0.7}
              />
              <Text style={styles.ratingNumber}>{rating}</Text>
            </View>
          ))}
        </View>
      </View>
    );
  };

  const scrollToComments = () => {
    setTimeout(() => {
      if (commentsContainerRef.current && scrollViewRef.current) {
        commentsContainerRef.current.measureLayout(
          scrollViewRef.current as any,
          (x, y) => {
            scrollViewRef.current?.scrollTo({
              y: Math.max(0, y - 100),
              animated: true,
            });
          },
          () => {
            scrollViewRef.current?.scrollToEnd({ animated: true });
          }
        );
      } else if (scrollViewRef.current) {
        scrollViewRef.current.scrollToEnd({ animated: true });
      }
    }, 300);
  };

  const isIOS = Platform.OS === 'ios';

  const scrollContent = (
    <ScrollView
      ref={scrollViewRef}
      style={styles.scrollView}
      contentContainerStyle={[
        styles.scrollContent,
        { paddingBottom: isKeyboardVisible ? 200 : 100 },
      ]}
      showsVerticalScrollIndicator={false}
      keyboardShouldPersistTaps="handled"
      keyboardDismissMode="on-drag"
      decelerationRate="normal"
      bounces={true}
      scrollEventThrottle={16}
      overScrollMode="never"
      nestedScrollEnabled={true}
    >
      <View style={styles.contentContainer}>
        {currentQuestionIndex === 0 ? (
          <Text style={styles.instructionsText}>
            We would like you to rate the app's user interface, experience, helpfulness, and overall project using the rating scale provided below. Circle a number for each item that best reflects how you feel. That is, circle 1 if you strongly disagree, 5 if you strongly agree with the statement, or any number in between.
          </Text>
        ) : (
          <View style={styles.questionContainer}>
            <Text style={styles.questionText}>
              {currentQuestion?.question}
            </Text>
            {currentQuestion && (
              <RatingScale
                questionId={currentQuestion.id}
                currentRating={currentQuestion.rating}
              />
            )}
          </View>
        )}

        {isLastQuestion && (
          <View ref={commentsContainerRef} style={styles.commentsContainer}>
            <Text style={styles.commentsLabel}>
              Give us your feedback
            </Text>
            <TextInput
              style={[styles.commentsInput, isCommentFocused && styles.commentsInputFocused]}
              onChangeText={setAdditionalComments}
              value={additionalComments}
              multiline={true}
              numberOfLines={10}
              placeholder="Enter your comments here..."
              placeholderTextColor="#9CA3AF"
              textAlignVertical="top"
              onFocus={() => { setIsCommentFocused(true); scrollToComments(); }}
              onBlur={() => setIsCommentFocused(false)}
            />
          </View>
        )}
      </View>
    </ScrollView>
  );

  const bottomButton = (
    <BottomButtonContainer>
      <TouchableOpacity
        style={[
          styles.nextButton,
          ((currentQuestionIndex > 0 && !isRatingSelected) || isLoading) && styles.nextButtonDisabled,
        ]}
        onPress={isLastQuestion ? sendFeedback : handleNextQuestion}
        disabled={(currentQuestionIndex > 0 && !isRatingSelected) || isLoading}
        activeOpacity={0.9}
      >
        {isLastQuestion && isLoading ? (
          <ActivityIndicator color="#FFFFFF" />
        ) : (
          <Text style={styles.nextButtonText}>
            {isLastQuestion ? 'Send Feedback' : 'Next'}
          </Text>
        )}
      </TouchableOpacity>
    </BottomButtonContainer>
  );

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <StatusBar barStyle="dark-content" backgroundColor="#FFFFFF" />

      <KeyboardAvoidingView
        behavior={isIOS ? 'padding' : 'padding'}
        style={{ flex: 1 }}
        keyboardVerticalOffset={insets.top}
      >
        <View style={styles.header}>
          <VectorBackButton onPress={() => navigation.goBack()} />
          <Text style={styles.headerTitle}>Send Feedback</Text>
          <View style={{ width: 40 }} />
        </View>

        {scrollContent}
        {bottomButton}
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
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
    flexGrow: 1,
  },
  contentContainer: {
    paddingHorizontal: 20,
    paddingTop: 20,
  },
  instructionsText: {
    fontSize: 17,
    lineHeight: 24,
    color: '#1F2937',
    marginBottom: 24,
    fontWeight: '400',
    textAlign: 'justify',
  },
  questionContainer: {
    marginBottom: 20,
  },
  questionText: {
    fontSize: 18,
    lineHeight: 24,
    color: '#1F2937',
    marginBottom: 20,
    fontWeight: '400',
    textAlign: 'left',
  },
  ratingContainer: {
    marginTop: 8,
  },
  ratingScale: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    gap: 12,
  },
  ratingItem: {
    alignItems: 'center',
    gap: 4,
  },
  ratingButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    borderWidth: 4,
    borderColor: '#7BA21B', // green
    borderStyle: 'solid',
    backgroundColor: '#FFFFFF',
    padding: 12, // Creates the double border effect
  },
  ratingButtonSelected: {
    backgroundColor: '#7BA21B', // green when selected
  },
  ratingNumber: {
    fontSize: 14,
    fontWeight: '500',
    color: '#1F2937',
    marginTop: 4,
  },
  commentsContainer: {
    marginTop: 8,
    marginBottom: 20,
  },
  commentsLabel: {
    fontSize: 18,
    lineHeight: 24,
    color: '#1F2937',
    marginBottom: 12,
    fontWeight: '500',
  },
  commentsInput: {
    borderWidth: 1,
    borderColor: '#E5E7EB',
    borderRadius: 8,
    padding: 16,
    fontSize: 16,
    color: '#1F2937',
    height: 200,
    backgroundColor: '#FFFFFF',
    textAlignVertical: 'top',
  },
  commentsInputFocused: {
    borderColor: '#7BA21B',
    borderWidth: 2,
  },
  nextButton: {
    height: 56,
    width: '100%',
    borderRadius: 28,
    backgroundColor: '#7BA21B',
    alignItems: 'center',
    justifyContent: 'center',
  },
  nextButtonDisabled: {
    backgroundColor: '#B7D17F',
  },
  nextButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
  },
});
