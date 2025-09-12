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

// Real Madrid Color Palette
const Colors = {
  gold: '#FCBF00',         // Real Madrid Gold
  blue: '#004996',         // Real Madrid Blue
  white: '#FFFFFF',        // White
  red: '#E62644',          // Real Madrid Red
  lightGray: '#F8F9FA',    // Light Gray
  darkGray: '#6C757D',     // Dark Gray
  textPrimary: '#333',     // Primary text color
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


  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity 
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <MaterialIcons name="arrow-back" size={24} color={Colors.blue} />
        </TouchableOpacity>
        <Text style={styles.title}>Set Practice Duration</Text>
        <View style={styles.headerRight} />
      </View>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        <View style={styles.drillInfo}>
          <Text style={styles.drillName}>{drillName}</Text>
          <Text style={styles.subtitle}>Set timer, train, record your progress</Text>
        </View>


        {/* Duration Picker */}
        <View style={styles.customPickerContainer}>
          <Text style={styles.sectionTitle}>Select Duration</Text>
          
          <View style={styles.pickerRow}>
            {/* Minutes Picker */}
            <View style={styles.pickerSection}>
              <Text style={styles.pickerLabel}>Minutes</Text>
              <View style={styles.pickerControls}>
                <TouchableOpacity
                  style={styles.pickerButton}
                  onPress={() => setSelectedMinutes(Math.max(0, selectedMinutes - 1))}
                >
                  <MaterialIcons name="remove" size={24} color={Colors.blue} />
                </TouchableOpacity>
                <Text style={styles.pickerValue}>{selectedMinutes}</Text>
                <TouchableOpacity
                  style={styles.pickerButton}
                  onPress={() => setSelectedMinutes(Math.min(5, selectedMinutes + 1))}
                >
                  <MaterialIcons name="add" size={24} color={Colors.blue} />
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
                  <MaterialIcons name="remove" size={24} color={Colors.blue} />
                </TouchableOpacity>
                <Text style={styles.pickerValue}>{selectedSeconds.toString().padStart(2, '0')}</Text>
                <TouchableOpacity
                  style={styles.pickerButton}
                  onPress={() => setSelectedSeconds((selectedSeconds + 15) % 60)}
                >
                  <MaterialIcons name="add" size={24} color={Colors.blue} />
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
    color: Colors.textPrimary,
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
    color: Colors.textPrimary,
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: Colors.darkGray,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: Colors.textPrimary,
    marginBottom: 15,
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
    color: Colors.textPrimary,
    marginHorizontal: 20,
    minWidth: 60,
    textAlign: 'center',
  },
  colonText: {
    fontSize: 36,
    fontWeight: 'bold',
    color: Colors.textPrimary,
    marginHorizontal: 10,
  },
  durationDisplay: {
    textAlign: 'center',
    fontSize: 18,
    color: Colors.blue,
    fontWeight: '600',
    marginTop: 10,
  },
  tipsContainer: {
    backgroundColor: Colors.blue + '10',
    padding: 15,
    borderRadius: 12,
    marginTop: 10,
  },
  tipsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: Colors.blue,
    marginBottom: 8,
  },
  tipsText: {
    fontSize: 13,
    color: Colors.textPrimary,
    marginBottom: 4,
  },
  bottomContainer: {
    padding: 20,
    backgroundColor: Colors.white,
    borderTopWidth: 1,
    borderTopColor: Colors.lightGray,
  },
  startButton: {
    backgroundColor: Colors.red,
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