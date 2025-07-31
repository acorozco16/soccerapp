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
  primary: '#663399',      // Royal Purple
  gold: '#FFD700',         // Gold
  navy: '#001F3F',         // Navy Blue
  white: '#FFFFFF',        // White
  lightGray: '#F8F9FA',    // Light Gray
  darkGray: '#6C757D',     // Dark Gray
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
    const result = await drillService.getAvailableDrills();
    
    if (result.success) {
      setDrills(result.drills);
    } else {
      Alert.alert('Error', result.error);
    }
    
    setLoading(false);
  };

  const handleDrillSelect = (drill) => {
    setSelectedDrill(drill.type);
    // Show drill details before proceeding
    Alert.alert(
      drill.name,
      `${drill.description}\n\nSuccess Criteria: ${drill.success_criteria}`,
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Start Recording', 
          onPress: () => navigation.navigate('VideoRecording', { 
            drillType: drill.type,
            drillName: drill.name 
          })
        }
      ]
    );
  };

  const renderDrillCard = (drill) => (
    <TouchableOpacity
      key={drill.type}
      style={[
        styles.drillCard,
        selectedDrill === drill.type && styles.selectedCard
      ]}
      onPress={() => handleDrillSelect(drill)}
    >
      <View style={styles.cardHeader}>
        <Text style={styles.drillName}>{drill.name}</Text>
        <View style={styles.difficultyBadge}>
          <Text style={styles.difficultyText}>⚽</Text>
        </View>
      </View>
      
      <Text style={styles.drillDescription}>{drill.description}</Text>
      
      <View style={styles.criteriaContainer}>
        <Text style={styles.criteriaLabel}>Success Goal:</Text>
        <Text style={styles.criteriaText}>{drill.success_criteria}</Text>
      </View>
      
      <View style={styles.cardFooter}>
        <Text style={styles.timeText}>
          {drill.time_window ? `${drill.time_window}s` : 'Flexible duration'}
        </Text>
        <Text style={styles.selectText}>Tap to Select</Text>
      </View>
    </TouchableOpacity>
  );

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={Colors.primary} />
          <Text style={styles.loadingText}>Loading drills...</Text>
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
          <MaterialIcons name="arrow-back" size={24} color={Colors.primary} />
        </TouchableOpacity>
        <Text style={styles.title}>Choose Your Drill</Text>
        <View style={styles.headerRight} />
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
  backButton: {
    padding: 5,
  },
  headerRight: {
    width: 34,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: Colors.navy,
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
    borderColor: Colors.primary,
    backgroundColor: '#f8f6ff',
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
    color: Colors.navy,
    flex: 1,
  },
  difficultyBadge: {
    backgroundColor: Colors.gold,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  difficultyText: {
    color: Colors.navy,
    fontSize: 12,
    fontWeight: '500',
  },
  drillDescription: {
    fontSize: 16,
    color: Colors.darkGray,
    marginBottom: 15,
    lineHeight: 22,
  },
  criteriaContainer: {
    backgroundColor: '#f8f6ff',
    padding: 12,
    borderRadius: 8,
    marginBottom: 15,
  },
  criteriaLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: Colors.primary,
    marginBottom: 4,
  },
  criteriaText: {
    fontSize: 16,
    color: Colors.navy,
    fontWeight: '500',
  },
  cardFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  timeText: {
    fontSize: 14,
    color: Colors.darkGray,
  },
  selectText: {
    fontSize: 14,
    color: Colors.primary,
    fontWeight: '500',
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
    backgroundColor: Colors.primary,
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