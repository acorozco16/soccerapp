import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import authService from '../services/auth';

// Real Madrid Color Palette
const Colors = {
  gold: '#FCBF00',         // Real Madrid Gold
  blue: '#004996',         // Real Madrid Blue
  white: '#FFFFFF',        // White
  red: '#E62644',          // Real Madrid Red
  lightGray: '#F8F9FA',    // Light Gray
  darkGray: '#6C757D',     // Dark Gray
};

export default function HomeScreen({ navigation }) {
  const [drillStats, setDrillStats] = useState([]);

  useEffect(() => {
    loadDrillStats();
  }, []);

  const loadDrillStats = () => {
    // Mock data for now - replace with actual data from storage/API
    const mockStats = [
      { name: 'Juggling', type: 'juggling', bestTouches: 0, lastSession: null, progress: 0 },
      { name: 'Bell Touches', type: 'bell_touches', bestTouches: 0, lastSession: null, progress: 0 },
      { name: 'Inside-Outside', type: 'inside_outside', bestTouches: 0, lastSession: null, progress: 0 },
      { name: 'Sole Rolls', type: 'sole_rolls', bestTouches: 0, lastSession: null, progress: 0 },
      { name: 'Outside Foot Push', type: 'outside_foot_push', bestTouches: 0, lastSession: null, progress: 0 },
      { name: 'V Cuts', type: 'v_cuts', bestTouches: 0, lastSession: null, progress: 0 },
      { name: 'Croquetas', type: 'croquetas', bestTouches: 0, lastSession: null, progress: 0 },
      { name: 'Triangles', type: 'triangles', bestTouches: 0, lastSession: null, progress: 0 },
    ];
    setDrillStats(mockStats);
  };

  const handleLogout = async () => {
    await authService.logout();
    navigation.replace('Login');
  };

  const handleDrillSelect = (drill) => {
    navigation.navigate('DrillProgress', { 
      drillType: drill.type,
      drillName: drill.name 
    });
  };

  const handleStartPractice = () => {
    navigation.navigate('DrillSelection');
  };

  const renderDrillCard = (drill) => (
    <TouchableOpacity
      key={drill.type}
      style={styles.drillCard}
      onPress={() => handleDrillSelect(drill)}
    >
      <View style={styles.drillHeader}>
        <MaterialIcons name="sports-soccer" size={24} color={Colors.blue} />
        <Text style={styles.drillName}>{drill.name}</Text>
      </View>
      
      <View style={styles.drillStats}>
        <Text style={styles.bestLabel}>Best: {drill.bestTouches}</Text>
        <Text style={styles.progressText}>+{drill.progress}%</Text>
      </View>
      
      <Text style={styles.sessionText}>
        {drill.lastSession ? 'Last session' : 'Never'}
      </Text>
      
      <View style={styles.progressDots}>
        {[1, 2, 3, 4, 5].map((dot, index) => (
          <View
            key={index}
            style={[
              styles.progressDot,
              { backgroundColor: index === 4 ? Colors.gold : Colors.blue }
            ]}
          />
        ))}
      </View>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <Text style={styles.title}>Welcome Andrew! ⚽</Text>
          <TouchableOpacity onPress={handleLogout}>
            <MaterialIcons name="logout" size={24} color={Colors.darkGray} />
          </TouchableOpacity>
        </View>

        <Text style={styles.subtitle}>
          Ready to start your soccer training journey?
        </Text>
        
        <Text style={styles.description}>
          Select a drill below to begin tracking your touches
        </Text>

        <View style={styles.drillsSection}>
          <Text style={styles.sectionTitle}>Your Drills</Text>
          
          <View style={styles.drillsGrid}>
            {drillStats.map(renderDrillCard)}
          </View>
        </View>

        <TouchableOpacity style={styles.startButton} onPress={handleStartPractice}>
          <Text style={styles.startButtonText}>Start Practice</Text>
        </TouchableOpacity>

      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.white,
  },
  content: {
    padding: 20,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
  },
  subtitle: {
    fontSize: 20,
    fontWeight: '600',
    color: Colors.blue,
    textAlign: 'center',
    marginBottom: 10,
  },
  description: {
    fontSize: 16,
    color: Colors.darkGray,
    textAlign: 'center',
    marginBottom: 30,
  },
  drillsSection: {
    marginBottom: 30,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 20,
  },
  drillsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  drillCard: {
    backgroundColor: Colors.white,
    borderRadius: 12,
    padding: 15,
    width: '48%',
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  drillHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  drillName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginLeft: 8,
    flex: 1,
  },
  drillStats: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  bestLabel: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  progressText: {
    fontSize: 14,
    fontWeight: '600',
    color: Colors.gold,
  },
  sessionText: {
    fontSize: 14,
    color: Colors.darkGray,
    marginBottom: 10,
  },
  progressDots: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 6,
  },
  progressDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  startButton: {
    backgroundColor: Colors.blue,
    padding: 15,
    borderRadius: 25,
    alignItems: 'center',
    marginBottom: 20,
  },
  startButtonText: {
    color: Colors.white,
    fontSize: 18,
    fontWeight: '600',
  },
});