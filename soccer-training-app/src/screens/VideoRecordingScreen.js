import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ActivityIndicator,
  Dimensions,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
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

const { width, height } = Dimensions.get('window');

export default function VideoRecordingScreen({ route, navigation }) {
  const { drillType, drillName } = route.params;
  const [permission, requestPermission] = useCameraPermissions();
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [facing, setFacing] = useState('back');
  const cameraRef = useRef(null);
  const timerRef = useRef(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);


  const startRecording = async () => {
    if (!cameraRef.current) return;

    try {
      setIsRecording(true);
      setRecordingTime(0);
      
      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime(prevTime => prevTime + 1);
      }, 1000);

      const video = await cameraRef.current.recordAsync({
        maxDuration: 60, // Max 60 seconds
      });

      if (timerRef.current) {
        clearInterval(timerRef.current);
      }

      setIsRecording(false);
      handleVideoRecorded(video);
    } catch (error) {
      console.error('Recording failed:', error);
      setIsRecording(false);
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      Alert.alert('Error', 'Failed to record video. Please try again.');
    }
  };

  const stopRecording = async () => {
    if (!cameraRef.current || !isRecording) return;

    try {
      await cameraRef.current.stopRecording();
    } catch (error) {
      console.error('Stop recording failed:', error);
    }
  };

  const handleVideoRecorded = (video) => {
    Alert.alert(
      'Recording Complete',
      `Video recorded (${formatTime(recordingTime)}). Ready to analyze?`,
      [
        { text: 'Re-record', style: 'cancel' },
        { 
          text: 'Analyze', 
          onPress: () => uploadAndAnalyze(video)
        }
      ]
    );
  };

  const uploadAndAnalyze = async (video) => {
    setIsProcessing(true);
    
    try {
      const result = await drillService.startDrillAnalysis(drillType, video);
      
      if (result.success) {
        // Navigate to analysis progress screen
        navigation.navigate('AnalysisProgress', {
          analysisId: result.analysisId,
          drillType: result.drillType,
          drillName: drillName
        });
      } else {
        Alert.alert('Upload Failed', result.error);
      }
    } catch (error) {
      console.error('Upload failed:', error);
      Alert.alert('Error', 'Failed to upload video for analysis');
    } finally {
      setIsProcessing(false);
    }
  };

  const toggleCameraType = () => {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (!permission) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={Colors.primary} />
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

  if (isProcessing) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.processingContainer}>
          <ActivityIndicator size="large" color={Colors.primary} />
          <Text style={styles.processingTitle}>Processing Video</Text>
          <Text style={styles.processingText}>
            Uploading and starting analysis...
          </Text>
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
        <TouchableOpacity 
          style={styles.flipButton}
          onPress={toggleCameraType}
        >
          <MaterialIcons name="flip-camera-ios" size={24} color="#fff" />
        </TouchableOpacity>
      </View>

      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing={facing}
      >
        <View style={styles.cameraOverlay}>
          {/* Recording indicator */}
          {isRecording && (
            <View style={styles.recordingIndicator}>
              <View style={styles.recordingDot} />
              <Text style={styles.recordingText}>REC {formatTime(recordingTime)}</Text>
            </View>
          )}

          {/* Instructions */}
          <View style={styles.instructionsContainer}>
            <Text style={styles.instructionsText}>
              Position yourself in frame and press record when ready
            </Text>
          </View>
        </View>
      </CameraView>

      <View style={styles.controls}>
        <View style={styles.controlsRow}>
          <View style={styles.controlsLeft} />
          
          <TouchableOpacity
            style={[
              styles.recordButton,
              isRecording && styles.recordButtonActive
            ]}
            onPress={isRecording ? stopRecording : startRecording}
            disabled={isProcessing}
          >
            <View style={[
              styles.recordButtonInner,
              isRecording && styles.recordButtonInnerActive
            ]} />
          </TouchableOpacity>

          <View style={styles.controlsRight}>
            <Text style={styles.timerText}>{formatTime(recordingTime)}</Text>
          </View>
        </View>

        <View style={styles.tipsContainer}>
          <Text style={styles.tipsTitle}>Recording Tips:</Text>
          <Text style={styles.tipsText}>• Keep the entire drill area in frame</Text>
          <Text style={styles.tipsText}>• Record for at least 10 seconds</Text>
          <Text style={styles.tipsText}>• Ensure good lighting</Text>
        </View>
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
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 20,
    backgroundColor: 'rgba(0,0,0,0.7)',
    zIndex: 1,
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
    flex: 1,
    justifyContent: 'space-between',
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
  },
  instructionsText: {
    color: '#fff',
    fontSize: 16,
    textAlign: 'center',
  },
  controls: {
    backgroundColor: '#000',
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
    backgroundColor: Colors.primary,
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
  },
});