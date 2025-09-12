import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  ActivityIndicator,
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
  green: '#28a745',        // Success Green
};

export default function DrillProgressScreen({ route, navigation }) {
  const { drillType, drillName } = route.params;
  const [loading, setLoading] = useState(true);
  const [progressData, setProgressData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadProgressData();
  }, []);

  const loadProgressData = async () => {
    try {
      setLoading(true);
      console.log('🔍 DrillProgressScreen: Loading progress for drill type:', drillType);
      const result = await drillService.getDrillProgress(drillType);
      console.log('📊 DrillProgressScreen: API result:', result);
      if (result.success) {
        console.log('✅ DrillProgressScreen: Progress data loaded:', result.data);
        setProgressData(result.data);
      } else {
        console.log('❌ DrillProgressScreen: Failed to load:', result.error);
        setError('Failed to load progress data');
      }
    } catch (err) {
      console.error('💥 DrillProgressScreen: Exception loading progress:', err);
      setError('Connection error');
    } finally {
      setLoading(false);
    }
  };

  const calculateWeekAverage = (last7Days) => {
    if (!last7Days || last7Days.length === 0) return 0;
    // Only count days where you actually practiced (touches > 0)
    const practiceDays = last7Days.filter(day => (day.touches || 0) > 0);
    if (practiceDays.length === 0) return 0;
    const total = practiceDays.reduce((sum, day) => sum + day.touches, 0);
    return Math.round(total / practiceDays.length);
  };

  const calculateAverageMetrics = (attempts) => {
    if (!attempts || attempts.length === 0) return { duration: 0, touchesPerMinute: 0, interval: 0 };
    
    let totalDuration = 0;
    let totalTouchesPerMinute = 0;
    let totalInterval = 0;
    let validAttempts = 0;
    
    attempts.forEach(attempt => {
      const results = attempt.results || {};
      const duration = results.duration || 0;
      const touches = results.count_detected || 0;
      
      if (duration > 0 && touches > 0) {
        totalDuration += duration;
        totalTouchesPerMinute += (touches / duration) * 60; // Convert to per minute
        totalInterval += duration / touches; // Average time between touches
        validAttempts++;
      }
    });
    
    if (validAttempts === 0) return { duration: 0, touchesPerMinute: 0, interval: 0 };
    
    return {
      duration: Math.round(totalDuration / validAttempts),
      touchesPerMinute: Math.round(totalTouchesPerMinute / validAttempts),
      interval: (totalInterval / validAttempts).toFixed(1)
    };
  };

  const getTodayTouches = (last7Days) => {
    if (!last7Days || last7Days.length === 0) return 0;
    const today = new Date().toISOString().split('T')[0];
    const todayData = last7Days.find(day => day.date === today);
    return todayData ? todayData.touches : 0;
  };

  const getYesterdayTouches = (last7Days) => {
    if (!last7Days || last7Days.length < 2) return 0;
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    const yesterdayStr = yesterday.toISOString().split('T')[0];
    const yesterdayData = last7Days.find(day => day.date === yesterdayStr);
    return yesterdayData ? yesterdayData.touches : 0;
  };

  const formatDate = (dateString) => {
    try {
      if (!dateString || dateString === "Never" || dateString === "Unknown date") {
        return "Unknown Date";
      }
      
      const date = new Date(dateString);
      if (isNaN(date.getTime())) return "Invalid Date";
      
      return date.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric',
        year: 'numeric'
      });
    } catch (error) {
      console.error('Error formatting date:', dateString, error);
      return "Invalid Date";
    }
  };

  const getTimeAgo = (dateString, includeTime = false) => {
    try {
      if (!dateString || dateString === "Unknown date") return "Unknown";
      
      const date = new Date(dateString);
      if (isNaN(date.getTime())) return "Unknown";
      
      const now = new Date();
      const diffTime = now - date;
      const diffDays = Math.floor(Math.abs(diffTime) / (1000 * 60 * 60 * 24));
      const diffHours = Math.floor(Math.abs(diffTime) / (1000 * 60 * 60));
      const diffMinutes = Math.floor(Math.abs(diffTime) / (1000 * 60));
      
      // For recent sessions, include more precise timing
      if (includeTime) {
        if (diffMinutes < 1) {
          return 'Just now';
        } else if (diffMinutes < 60) {
          return `${diffMinutes}m ago`;
        } else if (diffHours < 24) {
          return `${diffHours}h ago`;
        } else if (diffDays === 1) {
          return `Yesterday at ${date.toLocaleTimeString('en-US', { 
            hour: 'numeric', 
            minute: '2-digit',
            hour12: true 
          })}`;
        } else if (diffDays < 7) {
          return `${diffDays}d ago at ${date.toLocaleTimeString('en-US', { 
            hour: 'numeric', 
            minute: '2-digit',
            hour12: true 
          })}`;
        } else {
          return date.toLocaleDateString('en-US', { 
            month: 'short', 
            day: 'numeric',
            hour: 'numeric',
            minute: '2-digit',
            hour12: true
          });
        }
      }
      
      // For other uses (like personal best), keep simple format
      if (diffDays === 0) return 'Today';
      if (diffDays === 1) return 'Yesterday';
      if (diffDays < 30) return `${diffDays} days ago`;
      
      return date.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric',
        year: 'numeric'
      });
    } catch (error) {
      console.error('Error parsing date:', dateString, error);
      return "Unknown";
    }
  };


  const renderChart = (last7Days) => {
    if (!last7Days || last7Days.length === 0) {
      return (
        <View style={styles.chartContainer}>
          <Text style={styles.noDataText}>No data available</Text>
        </View>
      );
    }

    const maxTouches = Math.max(...last7Days.map(day => day.touches || 0));
    
    // Generate dynamic day labels based on actual dates
    const dayLabels = last7Days.map(day => {
      const date = new Date(day.date);
      return date.toLocaleDateString('en-US', { weekday: 'short' });
    });

    return (
      <View>
        <View style={styles.chartContainer}>
          {last7Days.map((day, index) => {
            const height = maxTouches > 0 ? (day.touches / maxTouches) * 100 : 0;
            return (
              <View key={index} style={styles.chartBarContainer}>
                <Text style={styles.chartValue}>{day.touches || 0}</Text>
                <View 
                  style={[
                    styles.chartBar, 
                    { height: Math.max(height, 5) + '%' }
                  ]} 
                />
              </View>
            );
          })}
        </View>
        <View style={styles.chartLabels}>
          {dayLabels.map((label, index) => (
            <Text key={index} style={styles.chartLabel}>{label}</Text>
          ))}
        </View>
      </View>
    );
  };

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.header}>
          <TouchableOpacity onPress={() => navigation.goBack()}>
            <MaterialIcons name="arrow-back" size={24} color={Colors.darkGray} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>{drillName} Progress</Text>
          <View style={styles.placeholder} />
        </View>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={Colors.blue} />
        </View>
      </SafeAreaView>
    );
  }

  if (error) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.header}>
          <TouchableOpacity onPress={() => navigation.goBack()}>
            <MaterialIcons name="arrow-back" size={24} color={Colors.darkGray} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>{drillName} Progress</Text>
          <View style={styles.placeholder} />
        </View>
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>{error}</Text>
          <TouchableOpacity style={styles.retryButton} onPress={loadProgressData}>
            <Text style={styles.retryButtonText}>Try Again</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  const todayTouches = getTodayTouches(progressData?.last_7_days);
  const yesterdayTouches = getYesterdayTouches(progressData?.last_7_days);
  const weekAverage = calculateWeekAverage(progressData?.last_7_days);
  const todayChange = todayTouches - yesterdayTouches;
  const averageMetrics = calculateAverageMetrics(progressData?.recent_attempts);

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <MaterialIcons name="arrow-back" size={24} color={Colors.darkGray} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>{drillName} Progress</Text>
        <View style={styles.placeholder} />
      </View>

      <ScrollView style={styles.content}>
        {/* Hero Stats - 6 metrics in 2x3 grid */}
        <View style={styles.heroStats}>
          <View style={styles.statsGrid}>
            <View style={styles.statItem}>
              <Text style={styles.statLabel}>TODAY</Text>
              <Text style={styles.statValue}>{todayTouches}</Text>
              <Text style={styles.statChange}>touches</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statLabel}>WEEK AVG</Text>
              <Text style={styles.statValue}>{weekAverage}</Text>
              <Text style={styles.statChange}>touches</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statLabel}>BEST</Text>
              <Text style={[styles.statValue, styles.personalBestValue]}>
                {(() => {
                  const personalBest = progressData?.personal_best;
                  return typeof personalBest === 'object' ? personalBest?.touches || 0 : personalBest || 0;
                })()}
              </Text>
              <Text style={styles.statChange}>touches</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statLabel}>AVG DURATION</Text>
              <Text style={styles.statValue}>{averageMetrics.duration}</Text>
              <Text style={styles.statChange}>seconds</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statLabel}>CONSISTENCY</Text>
              <Text style={styles.statValue}>{averageMetrics.interval}</Text>
              <Text style={styles.statChange}>sec/touch</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statLabel}>TREND</Text>
              <Text style={[
                styles.statValue, 
                (progressData?.trend || 0) >= 0 ? styles.positive : styles.negative
              ]}>
                {(progressData?.trend || 0) >= 0 ? '↑' : '↓'}{Math.abs(progressData?.trend || 0)}
              </Text>
              <Text style={styles.statChange}>recent change</Text>
            </View>
          </View>
        </View>


        {/* 7 Day Chart */}
        <View style={styles.chartSection}>
          <Text style={styles.chartTitle}>Last 7 Days</Text>
          {renderChart(progressData?.last_7_days)}
        </View>

        {/* Recent Sessions */}
        <View style={styles.recentSessions}>
          <Text style={styles.sessionsTitle}>Recent Sessions</Text>
          {progressData?.recent_sessions && progressData.recent_sessions.length > 0 ? (
            progressData.recent_sessions.map((session, index) => {
              // Handle both old string format and new object format
              let touches, date;
              if (typeof session === 'string') {
                // Old format: "19 touches (2025-01-03)"
                const match = session.match(/(\d+) touches \((.+)\)/);
                touches = match ? parseInt(match[1]) : 0;
                date = match ? match[2] : '';
              } else {
                // New format: {touches: 19, date: "2025-01-03T..."}
                touches = session.touches || 0;
                date = session.date || '';
              }
              
              return (
                <View key={index} style={styles.sessionItem}>
                  <Text style={styles.sessionDate}>{getTimeAgo(date, true)}</Text>
                  <Text style={styles.sessionResult}>{touches} touches</Text>
                </View>
              );
            })
          ) : (
            <View style={styles.sessionItem}>
              <Text style={styles.sessionDate}>No sessions yet</Text>
              <Text style={styles.sessionResult}>-- touches</Text>
            </View>
          )}
        </View>

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
    justifyContent: 'space-between',
    padding: 20,
    backgroundColor: Colors.white,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: Colors.blue,
  },
  placeholder: {
    width: 24,
  },
  content: {
    padding: 20,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  errorText: {
    fontSize: 16,
    color: Colors.darkGray,
    textAlign: 'center',
    marginBottom: 20,
  },
  retryButton: {
    backgroundColor: Colors.blue,
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
  },
  retryButtonText: {
    color: Colors.white,
    fontSize: 16,
    fontWeight: '600',
  },
  heroStats: {
    backgroundColor: Colors.white,
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 5,
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statItem: {
    alignItems: 'center',
    width: '32%',
    marginBottom: 20,
  },
  statLabel: {
    fontSize: 14,
    color: Colors.darkGray,
    marginBottom: 8,
  },
  statValue: {
    fontSize: 32,
    fontWeight: 'bold',
    color: Colors.blue,
    marginBottom: 5,
  },
  statChange: {
    fontSize: 14,
    fontWeight: '600',
    textAlign: 'center',
  },
  positive: {
    color: Colors.green,
  },
  negative: {
    color: Colors.red,
  },
  personalBestValue: {
    color: Colors.gold,
  },
  chartSection: {
    backgroundColor: Colors.white,
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 5,
  },
  chartTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: Colors.blue,
    marginBottom: 20,
  },
  chartContainer: {
    height: 120,
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'space-between',
    marginBottom: 15,
  },
  chartBarContainer: {
    alignItems: 'center',
    flex: 1,
  },
  chartValue: {
    fontSize: 11,
    fontWeight: '600',
    color: Colors.blue,
    marginBottom: 5,
  },
  chartBar: {
    backgroundColor: Colors.blue,
    borderRadius: 4,
    width: 35,
    minHeight: 5,
  },
  chartLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  chartLabel: {
    fontSize: 12,
    color: Colors.darkGray,
    textAlign: 'center',
    flex: 1,
  },
  noDataText: {
    fontSize: 16,
    color: Colors.darkGray,
    textAlign: 'center',
    flex: 1,
  },
  recentSessions: {
    backgroundColor: Colors.white,
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 5,
  },
  sessionsTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: Colors.blue,
    marginBottom: 15,
  },
  sessionItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  sessionDate: {
    fontSize: 14,
    color: Colors.darkGray,
  },
  sessionResult: {
    fontSize: 16,
    fontWeight: '600',
    color: Colors.blue,
  },
});