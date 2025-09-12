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
      const result = await drillService.getDrillProgress(drillType);
      if (result.success) {
        setProgressData(result.data);
      } else {
        setError('Failed to load progress data');
      }
    } catch (err) {
      console.error('Error loading progress:', err);
      setError('Connection error');
    } finally {
      setLoading(false);
    }
  };

  const calculateWeekAverage = (last7Days) => {
    if (!last7Days || last7Days.length === 0) return 0;
    const total = last7Days.reduce((sum, day) => sum + (day.touches || 0), 0);
    return Math.round(total / last7Days.length);
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
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric' 
    });
  };

  const getTimeAgo = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    return `${diffDays} days ago`;
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
    const dayLabels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

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
        {/* Hero Stats */}
        <View style={styles.heroStats}>
          <View style={styles.statsRow}>
            <View style={styles.statItem}>
              <Text style={styles.statLabel}>TODAY</Text>
              <Text style={styles.statValue}>{todayTouches}</Text>
              <Text style={[
                styles.statChange, 
                todayChange >= 0 ? styles.positive : styles.negative
              ]}>
                {todayChange >= 0 ? '↑' : '↓'} {Math.abs(todayChange)} from yesterday
              </Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statLabel}>WEEK AVG</Text>
              <Text style={styles.statValue}>{weekAverage}</Text>
              <Text style={styles.statChange}>
                Last 7 days average
              </Text>
            </View>
          </View>
        </View>

        {/* Personal Best */}
        {progressData?.personal_best && (
          <View style={styles.personalBest}>
            <MaterialIcons name="emoji-events" size={24} color={Colors.white} />
            <Text style={styles.bestLabel}>PERSONAL BEST</Text>
            <Text style={styles.bestValue}>{progressData.personal_best.touches} touches</Text>
            <Text style={styles.bestDate}>
              Set on {formatDate(progressData.personal_best.date)}
            </Text>
          </View>
        )}

        {/* 7 Day Chart */}
        <View style={styles.chartSection}>
          <Text style={styles.chartTitle}>Last 7 Days</Text>
          {renderChart(progressData?.last_7_days)}
        </View>

        {/* Recent Sessions */}
        <View style={styles.recentSessions}>
          <Text style={styles.sessionsTitle}>Recent Sessions</Text>
          {progressData?.recent_sessions && progressData.recent_sessions.length > 0 ? (
            progressData.recent_sessions.map((session, index) => (
              <View key={index} style={styles.sessionItem}>
                <Text style={styles.sessionDate}>{getTimeAgo(session.date)}</Text>
                <Text style={styles.sessionResult}>{session.touches} touches</Text>
              </View>
            ))
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
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statItem: {
    alignItems: 'center',
    flex: 1,
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
  personalBest: {
    backgroundColor: Colors.gold,
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 5,
  },
  bestLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: Colors.white,
    marginTop: 10,
    marginBottom: 5,
    opacity: 0.9,
  },
  bestValue: {
    fontSize: 28,
    fontWeight: 'bold',
    color: Colors.white,
    marginBottom: 5,
  },
  bestDate: {
    fontSize: 14,
    color: Colors.white,
    opacity: 0.8,
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