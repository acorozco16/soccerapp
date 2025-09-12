import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Alert,
  Animated,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import drillService from '../services/drills';

// Real Madrid Color Palette
const Colors = {
  gold: '#FCBF00',         // Real Madrid Gold
  blue: '#004996',         // Real Madrid Blue
  white: '#FFFFFF',        // White
  red: '#E62644',          // Real Madrid Red
  lightGray: '#F8F9FA',    // Light Gray
  darkGray: '#6C757D',     // Dark Gray
};

export default function AnalysisProgressScreen({ route, navigation }) {
  const { analysisId, drillType, drillName } = route.params;
  const [status, setStatus] = useState('processing');
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('Uploading video...');
  const [errorMessage, setErrorMessage] = useState('');
  const progressAnim = useState(new Animated.Value(0))[0];
  const ballBounceAnim = useState(new Animated.Value(0))[0];
  const ballRotateAnim = useState(new Animated.Value(0))[0];

  useEffect(() => {
    startProgressTracking();
    startBallAnimation();
    return () => {
      // Cleanup any intervals
    };
  }, []);

  const startBallAnimation = () => {
    // Bouncing animation
    Animated.loop(
      Animated.sequence([
        Animated.timing(ballBounceAnim, {
          toValue: -15,
          duration: 600,
          useNativeDriver: true,
        }),
        Animated.timing(ballBounceAnim, {
          toValue: 0,
          duration: 400,
          useNativeDriver: true,
        }),
      ])
    ).start();

    // Rotation animation
    Animated.loop(
      Animated.timing(ballRotateAnim, {
        toValue: 1,
        duration: 2000,
        useNativeDriver: true,
      })
    ).start();
  };

  const startProgressTracking = async () => {
    let attempts = 0;
    const maxAttempts = 60; // 5 minutes max (5 second intervals)
    
    const checkProgress = async () => {
      try {
        const result = await drillService.getAnalysisStatus(analysisId);
        
        if (result.success) {
          const statusData = result.status;
          updateProgress(statusData);
          
          if (statusData.status === 'completed') {
            // Analysis complete, get results
            const resultsResponse = await drillService.getAnalysisResults(analysisId);
            if (resultsResponse.success) {
              navigation.replace('AnalysisResults', {
                results: resultsResponse.results,
                drillName: drillName,
                drillType: drillType
              });
            } else {
              setErrorMessage('Failed to retrieve analysis results');
              setStatus('error');
            }
            return;
          } else if (statusData.status === 'failed') {
            setErrorMessage(statusData.error || 'Analysis failed');
            setStatus('error');
            return;
          }
        } else {
          console.error('Failed to check status:', result.error);
        }
        
        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(checkProgress, 5000); // Check every 5 seconds
        } else {
          setErrorMessage('Analysis timed out. Please try again.');
          setStatus('error');
        }
      } catch (error) {
        console.error('Progress tracking error:', error);
        setErrorMessage('Connection error. Please check your internet.');
        setStatus('error');
      }
    };
    
    checkProgress();
  };

  const getDrillMessages = (drillType) => {
    const drillMessages = {
      'juggling': [
        'Processing video...',
        'Detecting touches...',
        'Counting total touches...'
      ],
      'v_cuts': [
        'Processing video...',
        'Detecting cuts...',
        'Counting repetitions...'
      ],
      'sole_rolls': [
        'Processing video...',
        'Detecting rolls...',
        'Counting repetitions...'
      ],
      'inside_outside': [
        'Processing video...',
        'Detecting foot touches...',
        'Counting alternations...'
      ],
      'croquetas': [
        'Processing video...',
        'Detecting movements...',
        'Counting repetitions...'
      ],
      'bell_touches': [
        'Processing video...',
        'Detecting touches...',
        'Counting total touches...'
      ],
      'triangles': [
        'Processing video...',
        'Detecting movements...',
        'Counting triangle completions...'
      ],
      'outside_foot_push': [
        'Processing video...',
        'Detecting pushes...',
        'Counting repetitions...'
      ]
    };
    return drillMessages[drillType] || drillMessages['juggling'];
  };

  const updateProgress = (statusData) => {
    let progressPercent = 0;
    let stepText = 'Processing...';
    
    // Get drill-specific messages
    const messages = getDrillMessages(drillType);
    
    switch (statusData.status) {
      case 'uploaded':
        progressPercent = 20;
        stepText = messages[0]; // 'Processing video...'
        break;
      case 'analyzing':
        progressPercent = 50;
        stepText = messages[1]; // 'Detecting touches...' etc
        break;
      case 'generating_feedback':
        progressPercent = 80;
        stepText = messages[2]; // 'Counting total touches...' etc
        break;
      case 'completed':
        progressPercent = 100;
        stepText = 'Analysis complete!';
        break;
      default:
        progressPercent = 10;
        stepText = messages[0];
    }
    
    setProgress(progressPercent);
    setCurrentStep(stepText);
    
    // Animate progress bar
    Animated.timing(progressAnim, {
      toValue: progressPercent / 100,
      duration: 500,
      useNativeDriver: false,
    }).start();
  };

  const handleRetry = () => {
    setStatus('processing');
    setProgress(0);
    setErrorMessage('');
    setCurrentStep('Retrying analysis...');
    startProgressTracking();
  };

  const handleCancel = () => {
    Alert.alert(
      'Cancel Analysis',
      'Are you sure you want to cancel? You will lose this recording.',
      [
        { text: 'Continue Analysis', style: 'cancel' },
        { 
          text: 'Cancel', 
          style: 'destructive',
          onPress: () => navigation.navigate('DrillSelection')
        }
      ]
    );
  };

  if (status === 'error') {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.errorContainer}>
          <MaterialIcons name="error-outline" size={64} color={Colors.red} />
          <Text style={styles.errorTitle}>Analysis Failed</Text>
          <Text style={styles.errorText}>{errorMessage}</Text>
          
          <View style={styles.errorActions}>
            <TouchableOpacity 
              style={[styles.button, styles.retryButton]}
              onPress={handleRetry}
            >
              <Text style={styles.retryButtonText}>Try Again</Text>
            </TouchableOpacity>
            
            <TouchableOpacity 
              style={[styles.button, styles.cancelButton]}
              onPress={handleCancel}
            >
              <Text style={styles.cancelButtonText}>Back to Drills</Text>
            </TouchableOpacity>
          </View>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity 
          style={styles.cancelIcon}
          onPress={handleCancel}
        >
          <MaterialIcons name="close" size={24} color={Colors.darkGray} />
        </TouchableOpacity>
        <Text style={styles.title}>Analyzing Performance</Text>
        <View style={styles.placeholder} />
      </View>

      <View style={styles.content}>
        <View style={styles.drillInfo}>
          <Text style={styles.drillName}>{drillName}</Text>
          <Text style={styles.drillSubtitle}>Video Analysis in Progress</Text>
        </View>

        <View style={styles.progressContainer}>
          <View style={styles.progressRing}>
            <ActivityIndicator size="large" color={Colors.blue} />
          </View>
          
          <Text style={styles.progressPercentage}>{progress}%</Text>
          
          {/* Animated Soccer Ball */}
          <Animated.View 
            style={[
              styles.ballContainer,
              {
                transform: [
                  {
                    translateY: ballBounceAnim
                  },
                  {
                    rotate: ballRotateAnim.interpolate({
                      inputRange: [0, 1],
                      outputRange: ['0deg', '360deg']
                    })
                  }
                ]
              }
            ]}
          >
            <MaterialIcons name="sports-soccer" size={40} color={Colors.gold} />
          </Animated.View>
          
          <Text style={styles.analysisText}>Counting touches...</Text>
          
          <View style={styles.progressBarContainer}>
            <View style={styles.progressBarBackground}>
              <Animated.View 
                style={[
                  styles.progressBarFill,
                  {
                    width: progressAnim.interpolate({
                      inputRange: [0, 1],
                      outputRange: ['0%', '100%']
                    })
                  }
                ]}
              />
            </View>
          </View>
          
          <Text style={styles.stepText}>{currentStep}</Text>
        </View>

      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.lightGray,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 20,
    backgroundColor: Colors.white,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  cancelIcon: {
    padding: 5,
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    color: Colors.blue,
  },
  placeholder: {
    width: 34,
  },
  content: {
    flex: 1,
    padding: 20,
  },
  drillInfo: {
    alignItems: 'center',
    marginBottom: 40,
  },
  drillName: {
    fontSize: 24,
    fontWeight: 'bold',
    color: Colors.blue,
    textAlign: 'center',
  },
  drillSubtitle: {
    fontSize: 16,
    color: Colors.darkGray,
    marginTop: 5,
  },
  progressContainer: {
    alignItems: 'center',
    marginBottom: 40,
  },
  progressRing: {
    marginBottom: 20,
  },
  progressPercentage: {
    fontSize: 32,
    fontWeight: 'bold',
    color: Colors.blue,
    marginBottom: 20,
  },
  progressBarContainer: {
    width: '100%',
    marginBottom: 20,
  },
  progressBarBackground: {
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    backgroundColor: Colors.blue,
    borderRadius: 4,
  },
  stepText: {
    fontSize: 16,
    color: Colors.darkGray,
    textAlign: 'center',
  },
  ballContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 15,
    height: 60,
  },
  analysisText: {
    fontSize: 14,
    color: Colors.blue,
    fontWeight: '500',
    marginBottom: 20,
    textAlign: 'center',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  errorTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: Colors.blue,
    marginTop: 20,
    marginBottom: 10,
  },
  errorText: {
    fontSize: 16,
    color: Colors.darkGray,
    textAlign: 'center',
    marginBottom: 30,
    lineHeight: 22,
  },
  errorActions: {
    width: '100%',
  },
  button: {
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 15,
  },
  retryButton: {
    backgroundColor: Colors.blue,
  },
  retryButtonText: {
    color: Colors.white,
    fontSize: 16,
    fontWeight: '600',
  },
  cancelButton: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#ddd',
  },
  cancelButtonText: {
    color: Colors.darkGray,
    fontSize: 16,
    fontWeight: '500',
  },
});