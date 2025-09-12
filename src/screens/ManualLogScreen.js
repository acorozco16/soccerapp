import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Alert,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
// Note: Using a simple TouchableOpacity-based selector instead of Slider for now
import drillService from '../services/drills';

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

export default function ManualLogScreen({ route, navigation }) {
  const { drillType = 'juggling', drillName = 'Juggling', duration: passedDuration } = route.params || {};
  
  const [touches, setTouches] = useState('');
  const [selectedMinutes, setSelectedMinutes] = useState(passedDuration ? Math.floor(passedDuration / 60) : 1);
  const [selectedSeconds, setSelectedSeconds] = useState(passedDuration ? passedDuration % 60 : 0);
  const [juggleType, setJuggleType] = useState('both_feet');
  const [notes, setNotes] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showTypeSelector, setShowTypeSelector] = useState(false);

  const juggleTypes = [
    { value: 'both_feet', label: 'Both Feet', icon: '👟' },
    { value: 'right_foot', label: 'Right Foot Only', icon: '➡️' },
    { value: 'left_foot', label: 'Left Foot Only', icon: '⬅️' },
  ];

  const handleSubmit = async () => {
    if (!touches || parseInt(touches) === 0) {
      Alert.alert('Missing Information', 'Please enter how many touches you completed');
      return;
    }

    setIsSubmitting(true);

    try {
      // Create a manual log entry
      const totalMinutes = selectedMinutes + (selectedSeconds / 60);
      const totalSeconds = (selectedMinutes * 60) + selectedSeconds;
      
      const logData = {
        drill_type: drillType,
        count_detected: parseInt(touches),
        duration: totalSeconds,
        manual_entry: true,
        notes: notes.trim(),
        confidence: 1.0, // Manual entries have 100% confidence
        touches_per_minute: (parseInt(touches) / totalMinutes).toFixed(1),
      };

      // Only add juggle_type for juggling drills
      if (drillType === 'juggling') {
        logData.juggle_type = juggleType;
      }

      console.log('Submitting manual log:', logData);
      
      // Call the service to save the manual entry
      const result = await drillService.logManualPractice(logData);
      
      if (result.success) {
        const durationText = selectedMinutes > 0 && selectedSeconds > 0 
          ? `${selectedMinutes}:${selectedSeconds.toString().padStart(2, '0')}`
          : selectedMinutes > 0 
          ? `${selectedMinutes} minute${selectedMinutes !== 1 ? 's' : ''}`
          : `${selectedSeconds} second${selectedSeconds !== 1 ? 's' : ''}`;
        
        Alert.alert(
          'Practice Logged! 🎉',
          `Great job! You completed ${touches} touches in ${durationText}.`,
          [
            {
              text: 'Home',
              onPress: () => navigation.navigate('MainTabs', { screen: 'Train' }),
            },
            {
              text: 'Log Another',
              onPress: () => {
                setTouches('');
                setNotes('');
                setSelectedMinutes(1);
                setSelectedSeconds(0);
                setJuggleType('both_feet');
              },
            },
          ]
        );
      } else {
        throw new Error(result.error);
      }
    } catch (error) {
      console.error('Failed to log practice:', error);
      Alert.alert('Error', 'Failed to save your practice. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const getSelectedType = () => {
    return juggleTypes.find(type => type.value === juggleType);
  };

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView 
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <View style={styles.header}>
          <TouchableOpacity 
            style={styles.backButton}
            onPress={() => navigation.goBack()}
          >
            <MaterialIcons name="arrow-back" size={24} color={Colors.white} />
          </TouchableOpacity>
          <Text style={styles.title}>Log Practice</Text>
          <View style={styles.headerRight} />
        </View>

        <ScrollView 
          style={styles.content}
          contentContainerStyle={styles.contentContainer}
          showsVerticalScrollIndicator={false}
        >
          <View style={styles.drillInfo}>
            <Text style={styles.drillName}>{drillName}</Text>
            <Text style={styles.drillSubtitle}>
              {passedDuration ? `${Math.floor(passedDuration / 60)} minute practice completed!` : 'Quick manual entry'}
            </Text>
          </View>

          {/* Touch Count Input */}
          <View style={styles.inputSection}>
            <Text style={styles.label}>
              {passedDuration ? 'Great job! How many touches did you get?' : 'How many touches?'}
            </Text>
            <TextInput
              style={styles.touchInput}
              value={touches}
              onChangeText={setTouches}
              keyboardType="number-pad"
              placeholder="0"
              placeholderTextColor={Colors.darkGray}
              maxLength={4}
            />
            <Text style={styles.touchLabel}>consecutive touches</Text>
          </View>

          {/* Juggle Type Selector - Only for juggling drill */}
          {drillType === 'juggling' && (
            <View style={styles.inputSection}>
              <Text style={styles.label}>Type of juggling</Text>
              <TouchableOpacity 
                style={styles.typeSelector}
                onPress={() => setShowTypeSelector(!showTypeSelector)}
              >
                <Text style={styles.typeSelectorText}>
                  {getSelectedType().icon} {getSelectedType().label}
                </Text>
                <MaterialIcons 
                  name={showTypeSelector ? "expand-less" : "expand-more"} 
                  size={24} 
                  color={Colors.darkGray} 
                />
              </TouchableOpacity>
              
              {showTypeSelector && (
                <View style={styles.typeOptions}>
                  {juggleTypes.map((type) => (
                    <TouchableOpacity
                      key={type.value}
                      style={[
                        styles.typeOption,
                        juggleType === type.value && styles.selectedTypeOption
                      ]}
                      onPress={() => {
                        setJuggleType(type.value);
                        setShowTypeSelector(false);
                      }}
                    >
                      <Text style={[
                        styles.typeOptionText,
                        juggleType === type.value && styles.selectedTypeOptionText
                      ]}>
                        {type.icon} {type.label}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
              )}
            </View>
          )}

          {/* Duration Selector - Only show if duration wasn't passed from timer */}
          {!passedDuration && (
            <View style={styles.inputSection}>
              <Text style={styles.label}>How long did you practice?</Text>
              <View style={styles.durationContainer}>
                <View style={styles.pickerRow}>
                  {/* Minutes Picker */}
                  <View style={styles.pickerSection}>
                    <Text style={styles.pickerLabel}>Minutes</Text>
                    <View style={styles.pickerControls}>
                      <TouchableOpacity
                        style={styles.pickerButton}
                        onPress={() => setSelectedMinutes(Math.max(0, selectedMinutes - 1))}
                      >
                        <MaterialIcons name="remove" size={24} color={Colors.navy} />
                      </TouchableOpacity>
                      <Text style={styles.pickerValue}>{selectedMinutes}</Text>
                      <TouchableOpacity
                        style={styles.pickerButton}
                        onPress={() => setSelectedMinutes(Math.min(30, selectedMinutes + 1))}
                      >
                        <MaterialIcons name="add" size={24} color={Colors.navy} />
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
                        <MaterialIcons name="remove" size={24} color={Colors.navy} />
                      </TouchableOpacity>
                      <Text style={styles.pickerValue}>{selectedSeconds.toString().padStart(2, '0')}</Text>
                      <TouchableOpacity
                        style={styles.pickerButton}
                        onPress={() => setSelectedSeconds((selectedSeconds + 15) % 60)}
                      >
                        <MaterialIcons name="add" size={24} color={Colors.navy} />
                      </TouchableOpacity>
                    </View>
                  </View>
                </View>

                <Text style={styles.durationDisplay}>
                  Total: {selectedMinutes}:{selectedSeconds.toString().padStart(2, '0')}
                </Text>
              </View>
            </View>
          )}

          {/* Notes (Optional) */}
          <View style={styles.inputSection}>
            <Text style={styles.label}>Notes (optional)</Text>
            <TextInput
              style={styles.notesInput}
              value={notes}
              onChangeText={setNotes}
              placeholder="e.g., Practiced with weak foot, used size 3 ball"
              placeholderTextColor={Colors.darkGray}
              multiline
              maxLength={200}
            />
          </View>

          {/* Quick Tips */}
          <View style={styles.tipsContainer}>
            <Text style={styles.tipsTitle}>Logging Tips:</Text>
            <Text style={styles.tipsText}>• Be honest - improvement comes from consistent practice</Text>
            <Text style={styles.tipsText}>• Track different types to see where you need work</Text>
            <Text style={styles.tipsText}>• Even 5 minutes of practice counts!</Text>
          </View>
        </ScrollView>

        {/* Submit Button */}
        <View style={styles.submitContainer}>
          <TouchableOpacity
            style={[styles.submitButton, isSubmitting && styles.submitButtonDisabled]}
            onPress={handleSubmit}
            disabled={isSubmitting}
          >
            {isSubmitting ? (
              <ActivityIndicator color={Colors.white} />
            ) : (
              <>
                <MaterialIcons name="check-circle" size={24} color={Colors.white} />
                <Text style={styles.submitButtonText}>Log Practice</Text>
              </>
            )}
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.lightGray,
  },
  keyboardView: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 20,
    backgroundColor: Colors.primary,
  },
  backButton: {
    padding: 5,
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    color: Colors.white,
    flex: 1,
    textAlign: 'center',
    marginHorizontal: 10,
  },
  headerRight: {
    width: 34,
  },
  content: {
    flex: 1,
  },
  contentContainer: {
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
    marginBottom: 5,
  },
  drillSubtitle: {
    fontSize: 16,
    color: Colors.darkGray,
  },
  inputSection: {
    marginBottom: 25,
  },
  label: {
    fontSize: 16,
    fontWeight: '600',
    color: Colors.navy,
    marginBottom: 10,
  },
  touchInput: {
    backgroundColor: Colors.white,
    borderRadius: 12,
    padding: 20,
    fontSize: 48,
    fontWeight: 'bold',
    textAlign: 'center',
    color: Colors.primary,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  touchLabel: {
    textAlign: 'center',
    color: Colors.darkGray,
    marginTop: 5,
    fontSize: 14,
  },
  typeSelector: {
    backgroundColor: Colors.white,
    borderRadius: 12,
    padding: 15,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  typeSelectorText: {
    fontSize: 16,
    color: Colors.navy,
    fontWeight: '500',
  },
  typeOptions: {
    marginTop: 10,
    backgroundColor: Colors.white,
    borderRadius: 12,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  typeOption: {
    padding: 15,
    borderBottomWidth: 1,
    borderBottomColor: Colors.lightGray,
  },
  selectedTypeOption: {
    backgroundColor: Colors.primary + '10',
  },
  typeOptionText: {
    fontSize: 16,
    color: Colors.navy,
  },
  selectedTypeOptionText: {
    color: Colors.primary,
    fontWeight: '600',
  },
  durationContainer: {
    backgroundColor: Colors.white,
    borderRadius: 12,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
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
    color: Colors.navy,
    fontWeight: '600',
    marginTop: 10,
  },
  notesInput: {
    backgroundColor: Colors.white,
    borderRadius: 12,
    padding: 15,
    fontSize: 16,
    color: Colors.navy,
    minHeight: 80,
    textAlignVertical: 'top',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  tipsContainer: {
    backgroundColor: Colors.primary + '10',
    borderRadius: 12,
    padding: 15,
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
  submitContainer: {
    padding: 20,
    backgroundColor: Colors.white,
    borderTopWidth: 1,
    borderTopColor: Colors.lightGray,
  },
  submitButton: {
    backgroundColor: Colors.accent,
    borderRadius: 12,
    padding: 18,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  submitButtonDisabled: {
    opacity: 0.7,
  },
  submitButtonText: {
    color: Colors.white,
    fontSize: 18,
    fontWeight: '600',
    marginLeft: 10,
  },
});