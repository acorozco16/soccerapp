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

export default function VideoProcessingScreen({ route, navigation }) {
  const { drillType, drillName, videoFile } = route.params;
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState('uploading'); // uploading, processing, finalizing
  const [stageText, setStageText] = useState('Uploading video...');
  const [errorMessage, setErrorMessage] = useState('');
  const [analysisId, setAnalysisId] = useState(null);
  
  const progressAnim = useState(new Animated.Value(0))[0];
  const ballBounceAnim = useState(new Animated.Value(0))[0];
  const ballRotateAnim = useState(new Animated.Value(0))[0];

  useEffect(() => {
    startVideoProcessing();
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

  const startVideoProcessing = async () => {
    try {
      // Stage 1: Upload (0-30%)
      setStage('uploading');
      setStageText('Uploading video...');
      updateProgress(10);

      const uploadResult = await drillService.startDrillAnalysisWithRetry(
        drillType,
        videoFile,
        (uploadProgress) => {
          // Map upload progress to 0-30% of total progress
          const totalProgress = Math.round(uploadProgress * 0.3);
          updateProgress(totalProgress);
        }
      );

      if (!uploadResult.success) {
        throw new Error(uploadResult.error);
      }

      setAnalysisId(uploadResult.analysisId);

      // Stage 2: Processing (30-90%)
      setStage('processing');
      setStageText(getDrillProcessingMessage(drillType));
      updateProgress(40);

      // Start polling for analysis completion
      await pollAnalysisStatus(uploadResult.analysisId);

    } catch (error) {
      console.error('Video processing error:', error);
      setErrorMessage(error.message || 'Processing failed. Please try again.');
    }
  };

  const getDrillProcessingMessage = (drillType) => {
    const messages = {
      'juggling': 'Detecting ball touches...',
      'v_cuts': 'Analyzing movement patterns...',
      'sole_rolls': 'Counting roll movements...',
      'inside_outside': 'Tracking foot touches...',
      'croquetas': 'Analyzing technique...',
      'bell_touches': 'Detecting touches...',
      'triangles': 'Counting triangle completions...',
      'outside_foot_push': 'Analyzing push movements...'
    };
    return messages[drillType] || 'Processing video...';
  };

  const pollAnalysisStatus = async (analysisId) => {
    let attempts = 0;
    const maxAttempts = 60; // 5 minutes max
    
    const checkStatus = async () => {
      try {
        const result = await drillService.getAnalysisStatus(analysisId);
        
        if (result.success) {
          const statusData = result.status;
          
          if (statusData.status === 'completed') {
            // Stage 3: Finalizing (90-100%)
            setStage('finalizing');
            setStageText('Finalizing results...');
            updateProgress(95);

            // Get results and navigate
            const resultsResponse = await drillService.getAnalysisResults(analysisId);
            if (resultsResponse.success) {
              updateProgress(100);
              
              // Small delay to show 100% completion
              setTimeout(() => {
                navigation.replace('AnalysisResults', {
                  results: resultsResponse.results,
                  drillName: drillName,
                  drillType: drillType
                });
              }, 1000);
            } else {
              throw new Error('Failed to retrieve analysis results');
            }
            return;
          } else if (statusData.status === 'failed') {
            throw new Error(statusData.error || 'Analysis failed');
          } else {
            // Still processing - update progress (30-90%)
            const processingProgress = Math.min(90, 30 + (attempts * 2));
            updateProgress(processingProgress);
          }
        }
        
        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(checkStatus, 5000); // Check every 5 seconds
        } else {
          throw new Error('Analysis timed out. Please try again.');
        }
      } catch (error) {
        console.error('Status polling error:', error);
        setErrorMessage(error.message || 'Processing failed. Please try again.');
      }
    };
    
    checkStatus();
  };

  const updateProgress = (progressPercent) => {
    setProgress(progressPercent);
    
    // Animate progress bar
    Animated.timing(progressAnim, {
      toValue: progressPercent / 100,
      duration: 500,
      useNativeDriver: false,
    }).start();
  };

  const handleCancel = () => {
    Alert.alert(
      'Cancel Processing',
      'Are you sure you want to cancel? You will lose this recording.',
      [
        { text: 'Continue Processing', style: 'cancel' },
        { 
          text: 'Cancel', 
          style: 'destructive',
          onPress: () => navigation.navigate('DrillSelection')
        }
      ]
    );
  };

  const handleRetry = () => {
    setProgress(0);
    setErrorMessage('');
    setStage('uploading');
    setStageText('Uploading video...');
    startVideoProcessing();
  };

  if (errorMessage) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.errorContainer}>
          <MaterialIcons name="error-outline" size={64} color={Colors.red} />
          <Text style={styles.errorTitle}>Processing Failed</Text>
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
        <Text style={styles.title}>Processing Video</Text>
        <View style={styles.placeholder} />
      </View>

      <View style={styles.content}>
        <View style={styles.drillInfo}>
          <Text style={styles.drillName}>{drillName}</Text>
          <Text style={styles.drillSubtitle}>Analyzing Your Performance</Text>
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
          
          {/* Stage Indicators */}
          <View style={styles.stageIndicators}>
            <View style={[styles.stageIndicator, stage === 'uploading' && styles.activeStage]}>
              <Text style={[styles.stageText, stage === 'uploading' && styles.activeStageText]}>Upload</Text>
            </View>
            <View style={styles.stageDivider} />
            <View style={[styles.stageIndicator, stage === 'processing' && styles.activeStage]}>
              <Text style={[styles.stageText, stage === 'processing' && styles.activeStageText]}>Analyze</Text>
            </View>
            <View style={styles.stageDivider} />
            <View style={[styles.stageIndicator, stage === 'finalizing' && styles.activeStage]}>
              <Text style={[styles.stageText, stage === 'finalizing' && styles.activeStageText]}>Finish</Text>
            </View>
          </View>
          
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
          
          <Text style={styles.stepText}>{stageText}</Text>
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
  ballContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
    height: 60,
  },
  stageIndicators: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  stageIndicator: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 15,
    backgroundColor: Colors.white,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  activeStage: {
    backgroundColor: Colors.blue,
    borderColor: Colors.blue,
  },
  stageText: {
    fontSize: 12,
    color: Colors.darkGray,
    fontWeight: '500',
  },
  activeStageText: {
    color: Colors.white,
  },
  stageDivider: {
    width: 20,
    height: 1,
    backgroundColor: '#e0e0e0',
    marginHorizontal: 5,
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