import React from 'react';
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
};

export default function AnalysisResultsScreen({ route, navigation }) {
  const { results, drillName, drillType } = route.params;

  const handleTryAgain = () => {
    navigation.navigate('VideoRecording', {
      drillType: drillType,
      drillName: drillName
    });
  };

  const handleBackToHome = () => {
    navigation.navigate('MainTabs', { screen: 'Train' });
  };

  // Extract actual metrics from results (fixed field names)
  const touchCount = results?.results?.count_detected || 0;
  const duration = results?.results?.duration || 0;
  const touchesPerMinute = results?.per_foot_counts?.touches_per_minute || 0;
  const confidence = results?.results?.confidence || 0;
  const averageInterval = duration > 0 && touchCount > 0 ? (duration / touchCount).toFixed(1) : 0;

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity 
            style={styles.backButton}
            onPress={handleBackToHome}
          >
            <MaterialIcons name="close" size={24} color={Colors.darkGray} />
          </TouchableOpacity>
        </View>

        {/* Session Complete Header */}
        <View style={styles.headerSection}>
          <View style={styles.checkmarkContainer}>
            <MaterialIcons name="check" size={20} color={Colors.gold} />
            <Text style={styles.sessionCompleteText}>SESSION COMPLETE</Text>
          </View>
          <Text style={styles.greatSessionText}>GREAT SESSION!</Text>
          <Text style={styles.drillNameText}>{drillName}</Text>
        </View>

        {/* Main Touch Count Display */}
        <View style={styles.mainMetricContainer}>
          <View style={styles.touchCountCard}>
            <Text style={styles.touchCountNumber}>{touchCount}</Text>
            <Text style={styles.touchCountLabel}>TOUCHES</Text>
          </View>
        </View>

        {/* Metrics Cards */}
        <View style={styles.metricsContainer}>
          <View style={styles.metricCard}>
            <Text style={styles.metricLabel}>DURATION</Text>
            <Text style={styles.metricValue}>{duration.toFixed(1)} seconds</Text>
          </View>

          <View style={styles.metricCard}>
            <Text style={styles.metricLabel}>TOUCHES PER MINUTE</Text>
            <Text style={styles.metricValueGold}>{touchesPerMinute.toFixed(1)}</Text>
          </View>

          <View style={styles.metricCard}>
            <Text style={styles.metricLabel}>AVERAGE INTERVAL</Text>
            <Text style={styles.metricValue}>{averageInterval} sec</Text>
          </View>
        </View>
      </ScrollView>

      {/* Action Buttons */}
      <View style={styles.buttonContainer}>
        <TouchableOpacity style={styles.secondaryButton} onPress={handleBackToHome}>
          <Text style={styles.secondaryButtonText}>BACK TO HOME</Text>
        </TouchableOpacity>
        
        <TouchableOpacity style={styles.primaryButton} onPress={handleTryAgain}>
          <Text style={styles.primaryButtonText}>TRY AGAIN</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.white,
  },
  content: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    padding: 20,
    paddingTop: 10,
  },
  backButton: {
    padding: 8,
  },
  headerSection: {
    alignItems: 'center',
    paddingHorizontal: 20,
    marginBottom: 40,
  },
  checkmarkContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  sessionCompleteText: {
    color: Colors.gold,
    fontSize: 14,
    fontWeight: '600',
    letterSpacing: 1,
    marginLeft: 8,
  },
  greatSessionText: {
    color: Colors.gold,
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: 15,
    letterSpacing: 1,
  },
  drillNameText: {
    color: Colors.darkGray,
    fontSize: 24,
    fontWeight: '500',
  },
  mainMetricContainer: {
    paddingHorizontal: 20,
    marginBottom: 40,
  },
  touchCountCard: {
    backgroundColor: Colors.white,
    borderRadius: 20,
    borderWidth: 3,
    borderColor: Colors.gold,
    paddingVertical: 40,
    paddingHorizontal: 20,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 8,
  },
  touchCountNumber: {
    color: Colors.blue,
    fontSize: 80,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  touchCountLabel: {
    color: Colors.gold,
    fontSize: 18,
    fontWeight: '600',
    letterSpacing: 2,
  },
  metricsContainer: {
    paddingHorizontal: 20,
    marginBottom: 40,
  },
  metricCard: {
    backgroundColor: Colors.lightGray,
    borderRadius: 15,
    paddingVertical: 20,
    paddingHorizontal: 25,
    marginBottom: 15,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  metricLabel: {
    color: Colors.darkGray,
    fontSize: 14,
    fontWeight: '600',
    letterSpacing: 1,
  },
  metricValue: {
    color: Colors.blue,
    fontSize: 18,
    fontWeight: 'bold',
  },
  metricValueGold: {
    color: Colors.gold,
    fontSize: 18,
    fontWeight: 'bold',
  },
  buttonContainer: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    paddingBottom: 30,
    gap: 15,
  },
  primaryButton: {
    backgroundColor: Colors.blue,
    borderRadius: 25,
    paddingVertical: 15,
    alignItems: 'center',
    flex: 1,
  },
  secondaryButton: {
    backgroundColor: Colors.white,
    borderWidth: 2,
    borderColor: Colors.blue,
    borderRadius: 25,
    paddingVertical: 15,
    alignItems: 'center',
    flex: 1,
  },
  primaryButtonText: {
    color: Colors.white,
    fontSize: 16,
    fontWeight: 'bold',
    letterSpacing: 1,
  },
  secondaryButtonText: {
    color: Colors.blue,
    fontSize: 16,
    fontWeight: 'bold',
    letterSpacing: 1,
  },
});