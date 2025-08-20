import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
  Vibration,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import { Audio } from 'expo-av';

// App Color Palette
const Colors = {
  primary: '#004996',      // Blue
  gold: '#FCBF00',         // Gold  
  white: '#FFFFFF',        // White
  accent: '#E62644',       // Red
  lightGray: '#F8F9FA',    // Light Gray
  darkGray: '#6C757D',     // Dark Gray
  navy: '#004996',         // Navy (same as primary)
};

export default function TimerScreen({ route, navigation }) {
  const { drillType, drillName, duration: selectedDuration } = route.params;
  
  // States
  const [isCountingDown, setIsCountingDown] = useState(false);
  const [countdownValue, setCountdownValue] = useState('');
  const [isTimerRunning, setIsTimerRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [timeRemaining, setTimeRemaining] = useState(selectedDuration);
  
  // Refs
  const countdownAnimation = useRef(new Animated.Value(1)).current;
  const timerInterval = useRef(null);
  const soundRef = useRef(null);

  useEffect(() => {
    // Load sound
    loadSound();
    
    // Start countdown immediately
    startCountdown();
    
    return () => {
      if (timerInterval.current) {
        clearInterval(timerInterval.current);
      }
      if (soundRef.current) {
        soundRef.current.unloadAsync();
      }
    };
  }, []);

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

  const startCountdown = () => {
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
          startTimer();
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

  const startTimer = () => {
    setIsTimerRunning(true);
    setIsPaused(false);
    
    timerInterval.current = setInterval(() => {
      setTimeRemaining((prev) => {
        if (prev <= 1) {
          clearInterval(timerInterval.current);
          onTimerComplete();
          return 0;
        }
        
        // Play beep for last 3 seconds
        if (prev <= 4 && prev > 1) {
          playBeep();
        }
        
        return prev - 1;
      });
    }, 1000);
  };

  const pauseTimer = () => {
    if (timerInterval.current) {
      clearInterval(timerInterval.current);
    }
    setIsPaused(true);
  };

  const resumeTimer = () => {
    setIsPaused(false);
    startTimer();
  };

  const onTimerComplete = () => {
    setIsTimerRunning(false);
    Vibration.vibrate([0, 200, 100, 200]);
    
    // Navigate to manual log screen after a brief delay to avoid render issues
    setTimeout(() => {
      navigation.replace('ManualLog', {
        drillType,
        drillName,
        duration: selectedDuration / 60, // Convert to minutes
      });
    }, 100);
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleCancel = () => {
    if (timerInterval.current) {
      clearInterval(timerInterval.current);
    }
    navigation.goBack();
  };

  if (isCountingDown) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.countdownContainer}>
          <Animated.Text 
            style={[
              styles.countdownText,
              { transform: [{ scale: countdownAnimation }] }
            ]}
          >
            {countdownValue}
          </Animated.Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={handleCancel} style={styles.cancelButton}>
          <MaterialIcons name="close" size={24} color={Colors.white} />
        </TouchableOpacity>
        <Text style={styles.drillName}>{drillName}</Text>
        <View style={styles.headerRight} />
      </View>

      <View style={styles.timerContainer}>
        <Text style={styles.timerText}>{formatTime(timeRemaining)}</Text>
        
        <View style={styles.progressContainer}>
          <View 
            style={[
              styles.progressBar,
              { 
                width: `${((selectedDuration - timeRemaining) / selectedDuration) * 100}%` 
              }
            ]}
          />
        </View>

        {isTimerRunning && (
          <View style={styles.controlsContainer}>
            {isPaused ? (
              <TouchableOpacity style={styles.controlButton} onPress={resumeTimer}>
                <MaterialIcons name="play-arrow" size={48} color={Colors.white} />
                <Text style={styles.controlText}>Resume</Text>
              </TouchableOpacity>
            ) : (
              <TouchableOpacity style={styles.controlButton} onPress={pauseTimer}>
                <MaterialIcons name="pause" size={48} color={Colors.white} />
                <Text style={styles.controlText}>Pause</Text>
              </TouchableOpacity>
            )}
          </View>
        )}
      </View>

      <View style={styles.motivationContainer}>
        <Text style={styles.motivationText}>Keep going! Focus on your touches</Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.navy,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 20,
  },
  cancelButton: {
    padding: 5,
  },
  drillName: {
    fontSize: 20,
    fontWeight: '600',
    color: Colors.white,
    flex: 1,
    textAlign: 'center',
  },
  headerRight: {
    width: 34,
  },
  countdownContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  countdownText: {
    fontSize: 120,
    fontWeight: 'bold',
    color: Colors.gold,
  },
  timerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  timerText: {
    fontSize: 120,
    fontWeight: '200',
    color: Colors.white,
    marginBottom: 40,
  },
  progressContainer: {
    width: '100%',
    height: 8,
    backgroundColor: Colors.white + '30',
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 60,
  },
  progressBar: {
    height: '100%',
    backgroundColor: Colors.gold,
    borderRadius: 4,
  },
  controlsContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
  },
  controlButton: {
    backgroundColor: Colors.accent,
    borderRadius: 50,
    padding: 20,
    alignItems: 'center',
  },
  controlText: {
    color: Colors.white,
    fontSize: 16,
    fontWeight: '600',
    marginTop: 5,
  },
  motivationContainer: {
    padding: 20,
    alignItems: 'center',
  },
  motivationText: {
    fontSize: 18,
    color: Colors.white,
    textAlign: 'center',
  },
});