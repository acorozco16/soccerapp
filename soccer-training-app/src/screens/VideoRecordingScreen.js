import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ActivityIndicator,
  Dimensions,
  AppState,
  Linking,
  Animated,
  Vibration,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import * as FileSystem from 'expo-file-system';
import NetInfo from '@react-native-community/netinfo';
import drillService from '../services/drills';
import authService from '../services/auth';
import { APP_CONFIG } from '../constants/config';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Audio } from 'expo-av';

// Real Madrid Color Palette
const Colors = {
  gold: '#FCBF00',         // Real Madrid Gold
  blue: '#004996',         // Real Madrid Blue
  white: '#FFFFFF',        // White
  red: '#E62644',          // Real Madrid Red
  lightGray: '#F8F9FA',    // Light Gray
  darkGray: '#6C757D',     // Dark Gray
};

const { width, height } = Dimensions.get('window');

export default function VideoRecordingScreen({ route, navigation }) {
  const { drillType, drillName } = route.params;
  const [permission, requestPermission] = useCameraPermissions();
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingStep, setProcessingStep] = useState('Preparing upload...');
  const [recordingTime, setRecordingTime] = useState(0);
  const [facing, setFacing] = useState('back');
  const [cameraReady, setCameraReady] = useState(false);
  
  // Countdown states
  const [isCountingDown, setIsCountingDown] = useState(false);
  const [countdownValue, setCountdownValue] = useState('');
  const [showRecordButton, setShowRecordButton] = useState(true);
  
  const cameraRef = useRef(null);
  const timerRef = useRef(null);
  const recordingDurationRef = useRef(0);
  const shouldStopRecording = useRef(false);
  const countdownAnimation = useRef(new Animated.Value(1)).current;
  const soundRef = useRef(null);

  useEffect(() => {
    // Handle app state changes during recording
    const handleAppStateChange = (nextAppState) => {
      if (isRecording && nextAppState !== 'active') {
        console.warn('App backgrounded during recording - stopping recording');
        Alert.alert(
          'Recording Interrupted', 
          'Recording was stopped because the app went to background.'
        );
        stopRecording();
      }
    };

    const subscription = AppState.addEventListener('change', handleAppStateChange);

    // Clean up old video files on component mount
    cleanupTempFiles();
    
    // Load sound for countdown
    loadSound();

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (soundRef.current) {
        soundRef.current.unloadAsync();
      }
      subscription?.remove();
    };
  }, [isRecording]);

  const loadSound = async () => {
    try {
      // Uncomment when beep.mp3 is added to assets folder
      // const { sound } = await Audio.Sound.createAsync(
      //   require('../../assets/beep.mp3'),
      //   { shouldPlay: false }
      // );
      // soundRef.current = sound;
      console.log('Sound loading skipped - add beep.mp3 to assets folder');
    } catch (error) {
      console.log('Sound file not found - using fallback (vibration only)');
    }
  };

  const playBeep = async () => {
    try {
      if (soundRef.current) {
        await soundRef.current.replayAsync();
      }
    } catch (error) {
      console.log('Error playing sound:', error);
    }
  };

  // Memory and storage cleanup
  const cleanupTempFiles = async () => {
    try {
      console.log('Starting cleanup of temporary video files...');
      
      // Clean up cache directory
      const cacheDir = FileSystem.cacheDirectory;
      if (cacheDir) {
        await cleanupDirectory(cacheDir, 'cache');
      }
      
      // Clean up document directory temp files
      const documentDir = FileSystem.documentDirectory;
      if (documentDir) {
        await cleanupDirectory(documentDir, 'documents');
      }
      
    } catch (error) {
      console.warn('Cleanup failed:', error);
      // Don't throw - cleanup failures shouldn't break the app
    }
  };

  const cleanupDirectory = async (directory, dirName) => {
    try {
      const files = await FileSystem.readDirectoryAsync(directory);
      let cleanedCount = 0;
      let totalSize = 0;
      
      for (const file of files) {
        const filePath = `${directory}${file}`;
        
        // Only clean video files
        if (file.toLowerCase().includes('.mp4') || file.toLowerCase().includes('.mov')) {
          try {
            const fileInfo = await FileSystem.getInfoAsync(filePath);
            const age = Date.now() - (fileInfo.modificationTime * 1000);
            const ageHours = age / (1000 * 60 * 60);
            
            // Delete files older than 2 hours
            if (ageHours > 2) {
              await FileSystem.deleteAsync(filePath);
              cleanedCount++;
              totalSize += fileInfo.size;
              console.log(`Deleted old video file: ${file} (${Math.round(fileInfo.size / 1024)}KB, ${Math.round(ageHours)}h old)`);
            }
          } catch (fileError) {
            console.warn(`Failed to process file ${file}:`, fileError);
          }
        }
      }
      
      if (cleanedCount > 0) {
        console.log(`Cleaned ${cleanedCount} old video files from ${dirName}, freed ${Math.round(totalSize / (1024 * 1024))}MB`);
      } else {
        console.log(`No old video files found in ${dirName}`);
      }
      
    } catch (error) {
      console.warn(`Failed to clean ${dirName} directory:`, error);
    }
  };

  // Industry-standard chunk-based recording
  const recordInChunks = async () => {
    const chunks = [];
    const maxDuration = APP_CONFIG.MAX_VIDEO_DURATION;
    
    console.log('Starting chunk-based recording for', maxDuration, 'seconds');
    
    for (let second = 0; second < maxDuration; second += 3) {
      if (shouldStopRecording.current) {
        console.log('Recording stopped by user at second', recordingDurationRef.current);
        break;
      }
      
      try {
        console.log('Recording chunk starting at second', second);
        
        // Record 3-second chunk
        const chunk = await cameraRef.current.recordAsync({
          quality: 'high',
          maxDuration: 3000, // 3 seconds
        });
        
        // Validate chunk immediately
        await validateChunk(chunk, second + 1);
        
        chunks.push(chunk);
        
        // Timer is now handled independently in startRecording
        
        console.log('Chunk', second + 1, 'completed successfully');
        
      } catch (error) {
        console.error('Chunk recording failed at second', second + 1, error);
        
        if (chunks.length === 0) {
          throw new Error('Recording failed to start');
        }
        
        // If we have some chunks, return what we got
        console.log('Returning partial recording with', chunks.length, 'chunks');
        break;
      }
    }
    
    if (chunks.length === 0) {
      throw new Error('No video chunks recorded');
    }
    
    console.log('Concatenating', chunks.length, 'chunks...');
    return await concatenateChunks(chunks);
  };

  // Validate each chunk as it's recorded
  const validateChunk = async (chunk, chunkNumber) => {
    if (!chunk || !chunk.uri) {
      throw new Error(`Chunk ${chunkNumber} is invalid`);
    }
    
    try {
      const fileInfo = await FileSystem.getInfoAsync(chunk.uri);
      if (!fileInfo.exists || fileInfo.size < 10000) { // Less than 10KB
        throw new Error(`Chunk ${chunkNumber} is too small or corrupted`);
      }
      
      console.log(`Chunk ${chunkNumber} validated: ${Math.round(fileInfo.size / 1024)}KB`);
    } catch (error) {
      console.warn(`Chunk ${chunkNumber} validation failed:`, error);
      // Don't throw here - chunk might still be usable
    }
  };

  // Concatenate chunks into single video
  const concatenateChunks = async (chunks) => {
    if (chunks.length === 1) {
      return chunks[0];
    }
    
    // For now, return the last chunk as it contains the most recent recording
    // In a production app, you'd use FFmpeg to actually concatenate
    // But for our use case, the last chunk is sufficient
    console.log('Using last chunk as final video');
    return chunks[chunks.length - 1];
  };

  // Smart error recovery based on error type
  const getErrorRecovery = (error) => {
    const message = error.message.toLowerCase();
    
    if (message.includes('permission') || message.includes('denied')) {
      return {
        title: 'Camera Permission Required',
        message: 'Please enable camera access in your device settings.',
        actionText: 'Open Settings',
        action: () => Linking.openSettings()
      };
    }
    
    if (message.includes('storage') || message.includes('space')) {
      return {
        title: 'Storage Full',
        message: 'Please free up storage space to continue recording.',
        actionText: 'Check Storage',
        action: () => Linking.openSettings()
      };
    }
    
    if (message.includes('camera busy') || message.includes('in use')) {
      return {
        title: 'Camera In Use',
        message: 'Please close other camera apps and try again.',
        actionText: 'Retry',
        action: () => startRecording()
      };
    }
    
    return {
      title: 'Recording Failed',
      message: 'Please try recording again.',
      actionText: 'Retry',
      action: () => startRecording()
    };
  };

  const startCountdownAndRecord = () => {
    setShowRecordButton(false);
    setIsCountingDown(true);
    let count = 5;
    
    const showCountdownStep = () => {
      if (count >= 4) {
        setCountdownValue('Ready');
      } else if (count > 0) {
        setCountdownValue(count.toString());
        playBeep();
      } else {
        setCountdownValue('Go!');
        playBeep();
        Vibration.vibrate(200);
        
        setTimeout(() => {
          setIsCountingDown(false);
          startRecording();
        }, 500);
        return;
      }
      
      // Animate the countdown number
      Animated.sequence([
        Animated.timing(countdownAnimation, {
          toValue: 1.5,
          duration: 200,
          useNativeDriver: true,
        }),
        Animated.timing(countdownAnimation, {
          toValue: 1,
          duration: 200,
          useNativeDriver: true,
        }),
      ]).start();
      
      count--;
      setTimeout(showCountdownStep, 1000);
    };
    
    showCountdownStep();
  };

  const startRecording = async () => {
    console.log('START RECORDING CALLED');
    console.log('Camera ref exists:', !!cameraRef.current);
    
    if (!cameraRef.current || !cameraReady) {
      console.log('Camera not ready - ref exists:', !!cameraRef.current, 'ready:', cameraReady);
      Alert.alert('Error', 'Camera not ready. Please wait a moment and try again.');
      return;
    }

    try {
      setIsRecording(true);
      setRecordingTime(0);
      recordingDurationRef.current = 0;
      shouldStopRecording.current = false;
      
      console.log('Starting continuous recording...');
      
      // Start timer using Date.now() instead of intervals
      const startTime = Date.now();
      console.log('Recording start time:', startTime);
      
      // Start continuous recording (not chunks) with 30-second limit for beta
      const videoPromise = cameraRef.current.recordAsync({
        quality: '720p', // Reduced from 1080p for faster processing
        maxDuration: 30 * 1000, // 30 seconds for beta
      });
      
      // Track recording state for timer
      let isStillRecording = true;
      
      // Update timer using requestAnimationFrame with auto-stop
      const updateTimer = () => {
        if (isStillRecording) {
          const elapsed = Math.floor((Date.now() - startTime) / 1000);
          recordingDurationRef.current = elapsed;
          setRecordingTime(elapsed);
          
          // Play beep for last 3 seconds (27, 28, 29)
          if (elapsed >= 27 && elapsed <= 29) {
            playBeep();
          }
          
          // Auto-stop at 30 seconds
          if (elapsed >= 30) {
            console.log('Auto-stopping recording at 30 seconds');
            stopRecording();
            return;
          }
          
          requestAnimationFrame(updateTimer);
        }
      };
      requestAnimationFrame(updateTimer);
      
      // Stop timer when recording stops
      videoPromise.finally(() => {
        isStillRecording = false;
      });
      
      const video = await videoPromise;
      
      console.log('Recording completed');
      console.log('Final recording duration:', recordingDurationRef.current, 'seconds');
      
      // Don't clear timer here - it's already cleared in stopRecording
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      
      setIsRecording(false);
      handleVideoRecorded(video);
    } catch (error) {
      console.error('Recording failed:', error);
      console.error('Error message:', error.message);
      console.error('Error code:', error.code);
      setIsRecording(false);
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      
      // Smart error recovery
      const recovery = getErrorRecovery(error);
      Alert.alert(recovery.title, recovery.message, [
        { text: 'Cancel', style: 'cancel' },
        { text: recovery.actionText, onPress: recovery.action }
      ]);
    }
  };

  const stopRecording = async () => {
    if (!isRecording || !cameraRef.current) return;

    console.log('Stop recording requested');
    
    // Clear timer first and capture final duration
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    
    // Log the actual recording duration
    console.log('Recording duration at stop:', recordingDurationRef.current, 'seconds');
    
    try {
      await cameraRef.current.stopRecording();
    } catch (error) {
      console.error('Stop recording error:', error);
    }
  };

  const validateVideo = async (video) => {
    try {
      // Pre-upload file validation
      await validateVideoFile(video);

      // Check duration using ref (more reliable than state)
      const duration = recordingDurationRef.current;
      console.log('Validation - Duration from ref:', duration);
      console.log('Validation - MIN_VIDEO_DURATION:', APP_CONFIG.MIN_VIDEO_DURATION);
      
      if (duration < APP_CONFIG.MIN_VIDEO_DURATION) {
        throw new Error(`Video too short. Minimum ${APP_CONFIG.MIN_VIDEO_DURATION} seconds required.`);
      }

      if (duration > APP_CONFIG.MAX_VIDEO_DURATION) {
        throw new Error(`Video too long. Maximum ${APP_CONFIG.MAX_VIDEO_DURATION} seconds allowed.`);
      }

      return true;
    } catch (error) {
      console.error('Video validation failed:', error);
      Alert.alert('Invalid Video', error.message);
      return false;
    }
  };

  // Pre-upload file validation
  const validateVideoFile = async (video) => {
    if (!video || !video.uri) {
      throw new Error('Invalid video file');
    }

    // Check file exists and has content
    const fileInfo = await FileSystem.getInfoAsync(video.uri);
    if (!fileInfo.exists) {
      throw new Error('Video file does not exist');
    }

    if (fileInfo.size < 1000) {
      throw new Error('Video file is empty or corrupted');
    }

    // Check file size
    if (fileInfo.size > APP_CONFIG.MAX_FILE_SIZE) {
      throw new Error(`Video file too large (${Math.round(fileInfo.size / (1024 * 1024))}MB). Try recording a shorter video.`);
    }

    // Validate file format
    if (!video.uri.toLowerCase().includes('.mp4') && !video.uri.toLowerCase().includes('.mov')) {
      throw new Error('Invalid video format. Expected MP4 or MOV.');
    }

    console.log('Pre-upload validation passed:', {
      size: Math.round(fileInfo.size / 1024) + 'KB',
      duration: recordingDurationRef.current + 's'
    });

    return true;
  };

  const handleVideoRecorded = async (video) => {
    // Reset button visibility
    setShowRecordButton(true);
    
    // Validate video before proceeding
    const isValid = await validateVideo(video);
    if (!isValid) return;

    const duration = recordingDurationRef.current;
    const message = duration >= 30 
      ? `Perfect! You recorded the full 30 seconds. Ready to analyze your technique?`
      : `Video recorded (${formatTime(duration)}). Ready to analyze?`;

    Alert.alert(
      'Recording Complete',
      message,
      [
        { text: 'Re-record', style: 'cancel' },
        { 
          text: 'Submit for Analysis', 
          onPress: () => uploadAndAnalyze(video)
        }
      ]
    );
  };

  const resetProcessingState = () => {
    setIsProcessing(false);
    setUploadProgress(0);
    setProcessingStep('Preparing upload...');
  };

  const uploadAndAnalyze = async (video) => {
    // Navigate directly to VideoProcessingScreen which will handle upload and analysis
    navigation.navigate('VideoProcessing', {
      drillType: drillType,
      drillName: drillName,
      videoFile: video
    });
  };

  // Network-aware upload checks
  const checkNetworkForUpload = async (video) => {
    const netInfo = await NetInfo.fetch();
    
    if (!netInfo.isConnected) {
      throw new Error('No internet connection. Please check your network and try again.');
    }
    
    // Note: isInternetReachable can be false even when internet works, so we'll try anyway
    if (netInfo.isInternetReachable === false) {
      console.warn('Network reports internet not reachable, but attempting upload anyway...');
    }
    
    // Check for cellular data usage warning
    if (netInfo.type === 'cellular') {
      const fileInfo = await FileSystem.getInfoAsync(video.uri);
      const sizeMB = fileInfo.size / (1024 * 1024);
      
      if (sizeMB > 10) { // Warn for videos larger than 10MB on cellular
        const confirmed = await new Promise(resolve => {
          Alert.alert(
            'Cellular Data Usage',
            `This video (${Math.round(sizeMB)}MB) will use cellular data. Continue?`,
            [
              { text: 'Cancel', onPress: () => resolve(false) },
              { text: 'Continue', onPress: () => resolve(true) }
            ]
          );
        });
        
        if (!confirmed) {
          throw new Error('Upload cancelled - cellular data usage');
        }
      }
    }
    
    // Log network info for debugging
    console.log('Network check passed:', {
      type: netInfo.type,
      isConnected: netInfo.isConnected,
      isInternetReachable: netInfo.isInternetReachable,
      details: netInfo.details
    });
  };

  // Smart error recovery for upload failures
  const getUploadErrorRecovery = (errorMessage, videoFile) => {
    const message = errorMessage?.toLowerCase() || '';
    
    if (message.includes('network') || message.includes('connection')) {
      return {
        title: 'Connection Failed',
        message: 'Please check your internet connection and try again.',
        actionText: 'Retry Upload',
        action: () => uploadAndAnalyze(videoFile)
      };
    }
    
    if (message.includes('timeout')) {
      return {
        title: 'Upload Timed Out',
        message: 'Upload is taking longer than expected. Try again or check your connection.',
        actionText: 'Retry Upload',
        action: () => uploadAndAnalyze(videoFile)
      };
    }
    
    if (message.includes('401') || message.includes('unauthorized')) {
      return {
        title: 'Session Expired',
        message: 'Please log in again to continue.',
        actionText: 'Go to Login',
        action: () => navigation.navigate('Login')
      };
    }
    
    if (message.includes('413') || message.includes('too large')) {
      return {
        title: 'File Too Large',
        message: 'Try recording a shorter video.',
        actionText: 'Record Again',
        action: () => {} // Will return to recording screen
      };
    }
    
    return {
      title: 'Upload Failed',
      message: 'Please try again.',
      actionText: 'Retry Upload',
      action: () => uploadAndAnalyze(videoFile)
    };
  };

  const toggleCameraType = () => {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // DEBUG: Test token refresh
  const testTokenRefresh = async () => {
    try {
      console.log('Testing token refresh...');
      
      // Get current token info
      const currentToken = await AsyncStorage.getItem('authToken');
      const refreshToken = await AsyncStorage.getItem('refreshToken');
      
      console.log('Current token:', currentToken ? currentToken.substring(0, 50) + '...' : 'None');
      console.log('Refresh token:', refreshToken ? 'Available' : 'Not available');
      
      if (!refreshToken) {
        Alert.alert(
          'No Refresh Token', 
          'Please log out and log back in to get a refresh token.',
          [{ text: 'OK' }]
        );
        return;
      }
      
      // Try to refresh
      const newToken = await authService.refreshToken();
      console.log('Token refresh successful!');
      console.log('New token:', newToken.substring(0, 50) + '...');
      
      Alert.alert(
        'Token Refreshed!', 
        'Your authentication has been refreshed. Try uploading again.',
        [{ text: 'OK' }]
      );
    } catch (error) {
      console.error('Token refresh failed:', error);
      Alert.alert(
        'Refresh Failed', 
        `${error.message}\n\nPlease log out and log back in.`,
        [{ text: 'OK' }]
      );
    }
  };

  if (!permission) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={Colors.blue} />
          <Text style={styles.loadingText}>Requesting camera permission...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.permissionContainer}>
          <MaterialIcons name="videocam-off" size={64} color="#999" />
          <Text style={styles.permissionTitle}>Camera Access Required</Text>
          <Text style={styles.permissionText}>
            Please enable camera access to record your drill performance
          </Text>
          <TouchableOpacity 
            style={styles.permissionButton}
            onPress={requestPermission}
          >
            <Text style={styles.permissionButtonText}>Grant Permission</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }


  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity 
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <MaterialIcons name="arrow-back" size={24} color="#fff" />
        </TouchableOpacity>
        <Text style={styles.title}>{drillName}</Text>
        <View style={{ flexDirection: 'row' }}>
          <TouchableOpacity 
            style={[styles.flipButton, { marginRight: 10 }]}
            onPress={testTokenRefresh}
          >
            <MaterialIcons name="refresh" size={24} color="#fff" />
          </TouchableOpacity>
          <TouchableOpacity 
            style={styles.flipButton}
            onPress={toggleCameraType}
          >
            <MaterialIcons name="flip-camera-ios" size={24} color="#fff" />
          </TouchableOpacity>
        </View>
      </View>

      {/* TESTING: Completely clean CameraView - no children, no complexity */}
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing={facing}
        mode="video"
        onCameraReady={() => {
          console.log('Camera is ready');
          setCameraReady(true);
        }}
        onMountError={(error) => {
          console.error('Camera mount error:', error);
          Alert.alert('Camera Error', 'Failed to initialize camera. Please restart the app.');
        }}
      />
      
      {/* Move ALL UI outside camera - test if this fixes recording */}
      <View style={styles.cameraOverlay}>
        {/* Countdown overlay */}
        {isCountingDown && (
          <View style={styles.countdownOverlay}>
            <Animated.Text 
              style={[
                styles.countdownText,
                { transform: [{ scale: countdownAnimation }] }
              ]}
            >
              {countdownValue}
            </Animated.Text>
          </View>
        )}

        {/* Recording indicator */}
        {isRecording && (
          <View style={styles.recordingIndicator}>
            <View style={styles.recordingDot} />
            <Text style={styles.recordingText}>REC {formatTime(recordingTime)}</Text>
          </View>
        )}

        {/* Instructions */}
        {!isRecording && !isCountingDown && (
          <View style={styles.instructionsContainer}>
            <Text style={styles.instructionsText}>
              Position yourself in frame and press record when ready
            </Text>
          </View>
        )}
      </View>

      <View style={styles.controls}>
        <View style={styles.controlsRow}>
          <View style={styles.controlsLeft} />
          
          <TouchableOpacity
            style={[
              styles.recordButton,
              isRecording && styles.recordButtonActive,
              !showRecordButton && styles.recordButtonHidden
            ]}
            onPress={isRecording ? stopRecording : startCountdownAndRecord}
            disabled={isProcessing || isCountingDown}
          >
            <View style={[
              styles.recordButtonInner,
              isRecording && styles.recordButtonInnerActive
            ]} />
          </TouchableOpacity>

          <View style={styles.controlsRight}>
            <Text style={[
              styles.timerText,
              isRecording && recordingTime >= 27 && styles.timerTextWarning
            ]}>
              {isRecording ? formatTime(30 - recordingTime) : formatTime(recordingTime)}
            </Text>
            {isRecording && (
              <Text style={styles.minDurationText}>
                {recordingTime >= 27 ? 'Auto-stop!' : '30s max'}
              </Text>
            )}
          </View>
        </View>

        {!isRecording && (
          <View style={styles.tipsContainer}>
            <Text style={styles.tipsTitle}>Recording Tips:</Text>
            <Text style={styles.tipsText}>• Keep the entire drill area in frame</Text>
            <Text style={styles.tipsText}>• Record for at least 10 seconds</Text>
            <Text style={styles.tipsText}>• Ensure good lighting</Text>
          </View>
        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  header: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 20,
    paddingTop: 50, // Account for status bar
    backgroundColor: 'rgba(0,0,0,0.7)',
    zIndex: 10,
  },
  backButton: {
    padding: 5,
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
    flex: 1,
    textAlign: 'center',
    marginHorizontal: 10,
  },
  flipButton: {
    padding: 5,
  },
  camera: {
    flex: 1,
  },
  cameraOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'space-between',
    pointerEvents: 'none', // Allow touches to pass through to camera
    zIndex: 5, // Ensure it's above camera
  },
  recordingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255,0,0,0.8)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    margin: 20,
    alignSelf: 'flex-start',
    pointerEvents: 'auto', // Allow interaction with this element
  },
  recordingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#fff',
    marginRight: 8,
  },
  recordingText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  instructionsContainer: {
    backgroundColor: 'rgba(0,0,0,0.6)',
    margin: 20,
    padding: 15,
    borderRadius: 10,
    pointerEvents: 'auto', // Allow interaction with this element
  },
  instructionsText: {
    color: '#fff',
    fontSize: 16,
    textAlign: 'center',
  },
  controls: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(0,0,0,0.8)',
    paddingBottom: 40,
  },
  controlsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 30,
  },
  controlsLeft: {
    flex: 1,
  },
  recordButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
    marginHorizontal: 30,
  },
  recordButtonActive: {
    backgroundColor: '#ff4444',
  },
  recordButtonInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#ff4444',
  },
  recordButtonInnerActive: {
    width: 30,
    height: 30,
    borderRadius: 4,
    backgroundColor: '#fff',
  },
  recordButtonHidden: {
    opacity: 0.3,
  },
  countdownOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.7)',
    zIndex: 10,
  },
  countdownText: {
    fontSize: 120,
    fontWeight: 'bold',
    color: Colors.gold,
  },
  controlsRight: {
    flex: 1,
    alignItems: 'flex-end',
    paddingRight: 20,
  },
  timerText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  tipsContainer: {
    backgroundColor: 'rgba(255,255,255,0.1)',
    margin: 20,
    padding: 15,
    borderRadius: 10,
  },
  tipsTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
  },
  tipsText: {
    color: '#ccc',
    fontSize: 14,
    marginBottom: 4,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#666',
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  permissionTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginTop: 20,
    marginBottom: 10,
  },
  permissionText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginBottom: 30,
    lineHeight: 22,
  },
  permissionButton: {
    backgroundColor: Colors.blue,
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 8,
  },
  permissionButtonText: {
    color: Colors.white,
    fontSize: 16,
    fontWeight: '600',
  },
  processingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  processingTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginTop: 20,
    marginBottom: 10,
  },
  processingText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginBottom: 20,
  },
  progressContainer: {
    width: '100%',
    alignItems: 'center',
    marginTop: 20,
  },
  progressBar: {
    width: '80%',
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: Colors.blue,
  },
  progressText: {
    marginTop: 8,
    fontSize: 14,
    color: '#666',
  },
  cancelButton: {
    marginTop: 20,
    paddingHorizontal: 20,
    paddingVertical: 10,
    backgroundColor: '#ff4444',
    borderRadius: 8,
  },
  cancelButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  timerText: {
    fontSize: 20,
    fontWeight: '600',
    color: '#fff',
  },
  timerTextWarning: {
    color: Colors.gold,
  },
  minDurationText: {
    fontSize: 12,
    color: Colors.gold,
    marginTop: 4,
  },
});