import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Alert,
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
  textPrimary: '#333',     // Primary text color
};

export default function DrillSelectionScreen({ navigation }) {
  const [drills, setDrills] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedDrill, setSelectedDrill] = useState(null);

  useEffect(() => {
    loadDrills();
  }, []);

  const loadDrills = async () => {
    setLoading(true);
    console.log('🎯 DrillSelection: Loading drills...');
    
    const result = await drillService.getAvailableDrills();
    console.log('🎯 DrillSelection: Result:', result);
    
    if (result.success) {
      console.log('🎯 DrillSelection: Loaded', result.drills.length, 'drills');
      setDrills(result.drills);
    } else {
      console.error('❌ DrillSelection: Failed to load drills:', result.error);
      Alert.alert(
        'Connection Error', 
        `Unable to load drills. Please check your internet connection.\n\nError: ${result.error}`,
        [
          { text: 'Retry', onPress: loadDrills },
          { text: 'Cancel', style: 'cancel' }
        ]
      );
    }
    
    setLoading(false);
  };

  const handleRecord = (drill) => {
    navigation.navigate('VideoRecording', { 
      drillType: drill.type,
      drillName: drill.name 
    });
  };

  const handleTimer = (drill) => {
    navigation.navigate('TimerSelection', { 
      drillType: drill.type,
      drillName: drill.name 
    });
  };

  const handleLog = (drill) => {
    navigation.navigate('ManualLog', { 
      drillType: drill.type,
      drillName: drill.name 
    });
  };

  const renderDrillCard = (drill) => (
    <View
      key={drill.type}
      style={[
        styles.drillCard,
        selectedDrill === drill.type && styles.selectedCard
      ]}
    >
      <View style={styles.cardHeader}>
        <Text style={styles.drillName}>{drill.name}</Text>
        <View style={styles.difficultyBadge}>
          <Text style={styles.difficultyText}>⚽</Text>
        </View>
      </View>
      
      <Text style={styles.drillDescription}>{drill.description}</Text>
      
      <View style={styles.buttonContainer}>
        <TouchableOpacity 
          style={[styles.actionButton, styles.recordButton]}
          onPress={() => handleRecord(drill)}
        >
          <MaterialIcons name="videocam" size={20} color={Colors.white} style={{ marginRight: 6 }} />
          <Text style={styles.recordButtonText}>Record</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.actionButton, styles.timerButton]}
          onPress={() => handleTimer(drill)}
        >
          <MaterialIcons name="timer" size={20} color={Colors.blue} style={{ marginRight: 6 }} />
          <Text style={styles.timerButtonText}>Timer</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.actionButton, styles.logButton]}
          onPress={() => handleLog(drill)}
        >
          <MaterialIcons name="edit" size={20} color={Colors.gold} style={{ marginRight: 6 }} />
          <Text style={styles.logButtonText}>Log</Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={Colors.blue} />
          <Text style={styles.loadingText}>Loading drills...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Select Practice</Text>
      </View>

      <ScrollView 
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        <Text style={styles.subtitle}>
          Select a drill to practice and analyze your technique
        </Text>

        {drills.length > 0 ? (
          drills.map(renderDrillCard)
        ) : (
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyText}>No drills available</Text>
            <TouchableOpacity style={styles.retryButton} onPress={loadDrills}>
              <Text style={styles.retryText}>Retry</Text>
            </TouchableOpacity>
          </View>
        )}
      </ScrollView>
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
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: Colors.textPrimary,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
  },
  subtitle: {
    fontSize: 16,
    color: Colors.darkGray,
    marginBottom: 20,
    textAlign: 'center',
  },
  drillCard: {
    backgroundColor: Colors.white,
    borderRadius: 12,
    padding: 20,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  selectedCard: {
    borderColor: Colors.blue,
    backgroundColor: Colors.blue + '10',
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  drillName: {
    fontSize: 20,
    fontWeight: '600',
    color: Colors.textPrimary,
    flex: 1,
  },
  difficultyBadge: {
    backgroundColor: Colors.gold,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  difficultyText: {
    color: Colors.textPrimary,
    fontSize: 12,
    fontWeight: '500',
  },
  drillDescription: {
    fontSize: 16,
    color: Colors.darkGray,
    marginBottom: 15,
    lineHeight: 22,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 15,
  },
  actionButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    borderRadius: 8,
    marginHorizontal: 3,
  },
  recordButton: {
    backgroundColor: Colors.red,
  },
  recordButtonText: {
    color: Colors.white,
    fontSize: 16,
    fontWeight: '600',
  },
  timerButton: {
    backgroundColor: Colors.white,
    borderWidth: 2,
    borderColor: Colors.blue,
  },
  timerButtonText: {
    color: Colors.blue,
    fontSize: 16,
    fontWeight: '600',
  },
  logButton: {
    backgroundColor: Colors.white,
    borderWidth: 2,
    borderColor: Colors.gold,
  },
  logButtonText: {
    color: Colors.gold,
    fontSize: 16,
    fontWeight: '600',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: Colors.darkGray,
  },
  emptyContainer: {
    alignItems: 'center',
    marginTop: 50,
  },
  emptyText: {
    fontSize: 18,
    color: Colors.darkGray,
    marginBottom: 20,
  },
  retryButton: {
    backgroundColor: Colors.blue,
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 8,
  },
  retryText: {
    color: Colors.white,
    fontSize: 16,
    fontWeight: '500',
  },
});