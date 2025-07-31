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
  primary: '#663399',      // Royal Purple
  gold: '#FFD700',         // Gold
  navy: '#001F3F',         // Navy Blue
  white: '#FFFFFF',        // White
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

  useEffect(() => {
    startProgressTracking();
    return () => {
      // Cleanup any intervals
    };
  }, []);

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

  const updateProgress = (statusData) => {
    let progressPercent = 0;
    let stepText = 'Processing...';
    
    switch (statusData.status) {
      case 'uploaded':
        progressPercent = 20;
        stepText = 'Video uploaded, starting analysis...';
        break;
      case 'analyzing':
        progressPercent = 50;
        stepText = 'Analyzing your technique...';
        break;
      case 'generating_feedback':
        progressPercent = 80;
        stepText = 'Generating feedback and scores...';
        break;
      case 'completed':
        progressPercent = 100;
        stepText = 'Analysis complete!';
        break;
      default:
        progressPercent = 10;
        stepText = 'Processing video...';
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
          <MaterialIcons name="error-outline" size={64} color={Colors.primary} />
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
            <ActivityIndicator size="large" color={Colors.primary} />
          </View>
          
          <Text style={styles.progressPercentage}>{progress}%</Text>
          
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

        <View style={styles.infoContainer}>
          <View style={styles.infoCard}>
            <MaterialIcons name="analytics" size={24} color={Colors.primary} />
            <Text style={styles.infoTitle}>AI Analysis</Text>
            <Text style={styles.infoText}>
              Our system is analyzing your technique, timing, and form
            </Text>
          </View>
          
          <View style={styles.infoCard}>
            <MaterialIcons name="timer" size={24} color={Colors.primary} />
            <Text style={styles.infoTitle}>Processing Time</Text>
            <Text style={styles.infoText}>
              Analysis typically takes 1-3 minutes depending on video length
            </Text>
          </View>
          
          <View style={styles.infoCard}>
            <MaterialIcons name="star" size={24} color={Colors.primary} />
            <Text style={styles.infoTitle}>Detailed Feedback</Text>
            <Text style={styles.infoText}>
              You'll receive scores, tips, and areas for improvement
            </Text>
          </View>
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
    color: Colors.navy,
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
    color: Colors.navy,
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
    color: Colors.primary,
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
    backgroundColor: Colors.primary,
    borderRadius: 4,
  },
  stepText: {
    fontSize: 16,
    color: Colors.darkGray,
    textAlign: 'center',
  },
  infoContainer: {
    flex: 1,
  },
  infoCard: {
    backgroundColor: Colors.white,
    padding: 20,
    borderRadius: 12,
    marginBottom: 15,
    flexDirection: 'row',
    alignItems: 'flex-start',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: Colors.navy,
    marginLeft: 15,
    marginBottom: 5,
    flex: 1,
  },
  infoText: {
    fontSize: 14,
    color: Colors.darkGray,
    marginLeft: 15,
    flex: 1,
    lineHeight: 20,
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
    color: Colors.navy,
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
    backgroundColor: Colors.primary,
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