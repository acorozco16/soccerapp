import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';

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

export default function TimerSelectionScreen({ route, navigation }) {
  const { drillType, drillName } = route.params;
  const [selectedMinutes, setSelectedMinutes] = useState(1);
  const [selectedSeconds, setSelectedSeconds] = useState(0);

  const handleStart = () => {
    const totalSeconds = (selectedMinutes * 60) + selectedSeconds;
    
    if (totalSeconds === 0) {
      alert('Please select a duration');
      return;
    }
    
    navigation.navigate('Timer', {
      drillType,
      drillName,
      duration: totalSeconds,
    });
  };

  const quickSelectOptions = [
    { label: '30 sec', minutes: 0, seconds: 30 },
    { label: '1 min', minutes: 1, seconds: 0 },
    { label: '2 min', minutes: 2, seconds: 0 },
    { label: '3 min', minutes: 3, seconds: 0 },
    { label: '5 min', minutes: 5, seconds: 0 },
  ];

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity 
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <MaterialIcons name="arrow-back" size={24} color={Colors.primary} />
        </TouchableOpacity>
        <Text style={styles.title}>Set Practice Duration</Text>
        <View style={styles.headerRight} />
      </View>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        <View style={styles.drillInfo}>
          <Text style={styles.drillName}>{drillName}</Text>
          <Text style={styles.subtitle}>How long do you want to practice?</Text>
        </View>

        {/* Quick Select Options */}
        <View style={styles.quickSelectContainer}>
          <Text style={styles.sectionTitle}>Quick Select</Text>
          <View style={styles.quickSelectGrid}>
            {quickSelectOptions.map((option) => (
              <TouchableOpacity
                key={option.label}
                style={[
                  styles.quickSelectButton,
                  selectedMinutes === option.minutes && 
                  selectedSeconds === option.seconds && 
                  styles.quickSelectButtonActive
                ]}
                onPress={() => {
                  setSelectedMinutes(option.minutes);
                  setSelectedSeconds(option.seconds);
                }}
              >
                <Text style={[
                  styles.quickSelectText,
                  selectedMinutes === option.minutes && 
                  selectedSeconds === option.seconds && 
                  styles.quickSelectTextActive
                ]}>
                  {option.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {/* Custom Duration Picker */}
        <View style={styles.customPickerContainer}>
          <Text style={styles.sectionTitle}>Custom Duration</Text>
          
          <View style={styles.pickerRow}>
            {/* Minutes Picker */}
            <View style={styles.pickerSection}>
              <Text style={styles.pickerLabel}>Minutes</Text>
              <View style={styles.pickerControls}>
                <TouchableOpacity
                  style={styles.pickerButton}
                  onPress={() => setSelectedMinutes(Math.max(0, selectedMinutes - 1))}
                >
                  <MaterialIcons name="remove" size={24} color={Colors.primary} />
                </TouchableOpacity>
                <Text style={styles.pickerValue}>{selectedMinutes}</Text>
                <TouchableOpacity
                  style={styles.pickerButton}
                  onPress={() => setSelectedMinutes(Math.min(5, selectedMinutes + 1))}
                >
                  <MaterialIcons name="add" size={24} color={Colors.primary} />
                </TouchableOpacity>
              </View>
            </View>

            <Text style={styles.colonText}>:</Text>

            {/* Seconds Picker */}
            <View style={styles.pickerSection}>
              <Text style={styles.pickerLabel}>Seconds</Text>
              <View style={styles.pickerControls}>
                <TouchableOpacity
                  style={styles.pickerButton}
                  onPress={() => setSelectedSeconds(selectedSeconds === 0 ? 45 : Math.max(0, selectedSeconds - 15))}
                >
                  <MaterialIcons name="remove" size={24} color={Colors.primary} />
                </TouchableOpacity>
                <Text style={styles.pickerValue}>{selectedSeconds.toString().padStart(2, '0')}</Text>
                <TouchableOpacity
                  style={styles.pickerButton}
                  onPress={() => setSelectedSeconds((selectedSeconds + 15) % 60)}
                >
                  <MaterialIcons name="add" size={24} color={Colors.primary} />
                </TouchableOpacity>
              </View>
            </View>
          </View>

          <Text style={styles.durationDisplay}>
            Total: {selectedMinutes}:{selectedSeconds.toString().padStart(2, '0')}
          </Text>
        </View>

        {/* Tips */}
        <View style={styles.tipsContainer}>
          <Text style={styles.tipsTitle}>Pro Tips:</Text>
          <Text style={styles.tipsText}>• Start with shorter sessions and build up</Text>
          <Text style={styles.tipsText}>• Focus on quality touches over quantity</Text>
          <Text style={styles.tipsText}>• The timer will beep in the last 3 seconds</Text>
        </View>
      </ScrollView>

      {/* Start Button */}
      <View style={styles.bottomContainer}>
        <TouchableOpacity
          style={styles.startButton}
          onPress={handleStart}
        >
          <MaterialIcons name="play-arrow" size={28} color={Colors.white} />
          <Text style={styles.startButtonText}>Start Timer</Text>
        </TouchableOpacity>
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
    padding: 20,
    backgroundColor: Colors.white,
    borderBottomWidth: 1,
    borderBottomColor: Colors.lightGray,
  },
  backButton: {
    padding: 5,
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    color: Colors.navy,
    flex: 1,
    textAlign: 'center',
  },
  headerRight: {
    width: 34,
  },
  content: {
    flex: 1,
    padding: 20,
  },
  drillInfo: {
    alignItems: 'center',
    marginBottom: 30,
  },
  drillName: {
    fontSize: 24,
    fontWeight: 'bold',
    color: Colors.navy,
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: Colors.darkGray,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: Colors.navy,
    marginBottom: 15,
  },
  quickSelectContainer: {
    marginBottom: 30,
  },
  quickSelectGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  quickSelectButton: {
    backgroundColor: Colors.white,
    paddingVertical: 15,
    paddingHorizontal: 20,
    borderRadius: 12,
    marginBottom: 10,
    width: '48%',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: Colors.white,
  },
  quickSelectButtonActive: {
    borderColor: Colors.primary,
    backgroundColor: Colors.primary + '10',
  },
  quickSelectText: {
    fontSize: 16,
    fontWeight: '600',
    color: Colors.navy,
  },
  quickSelectTextActive: {
    color: Colors.primary,
  },
  customPickerContainer: {
    backgroundColor: Colors.white,
    padding: 20,
    borderRadius: 12,
    marginBottom: 20,
  },
  pickerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginVertical: 20,
  },
  pickerSection: {
    alignItems: 'center',
  },
  pickerLabel: {
    fontSize: 14,
    color: Colors.darkGray,
    marginBottom: 10,
  },
  pickerControls: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  pickerButton: {
    backgroundColor: Colors.lightGray,
    borderRadius: 20,
    padding: 8,
  },
  pickerValue: {
    fontSize: 36,
    fontWeight: 'bold',
    color: Colors.navy,
    marginHorizontal: 20,
    minWidth: 60,
    textAlign: 'center',
  },
  colonText: {
    fontSize: 36,
    fontWeight: 'bold',
    color: Colors.navy,
    marginHorizontal: 10,
  },
  durationDisplay: {
    textAlign: 'center',
    fontSize: 18,
    color: Colors.primary,
    fontWeight: '600',
    marginTop: 10,
  },
  tipsContainer: {
    backgroundColor: Colors.primary + '10',
    padding: 15,
    borderRadius: 12,
    marginTop: 10,
  },
  tipsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: Colors.primary,
    marginBottom: 8,
  },
  tipsText: {
    fontSize: 13,
    color: Colors.navy,
    marginBottom: 4,
  },
  bottomContainer: {
    padding: 20,
    backgroundColor: Colors.white,
    borderTopWidth: 1,
    borderTopColor: Colors.lightGray,
  },
  startButton: {
    backgroundColor: Colors.accent,
    paddingVertical: 18,
    borderRadius: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  startButtonText: {
    color: Colors.white,
    fontSize: 18,
    fontWeight: '600',
    marginLeft: 10,
  },
});