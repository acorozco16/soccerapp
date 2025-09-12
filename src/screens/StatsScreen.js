import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';
import { MaterialIcons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import authService from '../services/auth';
import drillService from '../services/drills';

// Real Madrid Color Palette
const Colors = {
  gold: '#FCBF00',         // Real Madrid Gold
  blue: '#004996',         // Real Madrid Blue
  white: '#FFFFFF',        // White
  red: '#E62644',          // Real Madrid Red
  lightGray: '#F8F9FA',    // Light Gray
  darkGray: '#6C757D',     // Dark Gray
};

const DRILL_ICONS = {
  juggling: 'sports-soccer',
  bell_touches: 'sports-soccer',
  inside_outside: 'sports-soccer',
  sole_rolls: 'sports-soccer',
  outside_foot_push: 'sports-soccer',
  v_cuts: 'sports-soccer',
  croquetas: 'sports-soccer',
  triangles: 'sports-soccer',
};

// Helper functions for calculating streaks and weekly data
const calculateStreakAndWeekly = (sessions) => {
  console.log('📊 Calculating streak from sessions:', sessions);
  console.log('📅 Current local time:', new Date().toString());
  console.log('📅 Today date string:', `${new Date().getFullYear()}-${String(new Date().getMonth() + 1).padStart(2, '0')}-${String(new Date().getDate()).padStart(2, '0')}`);
  
  // Sort sessions by date (most recent first)
  const sortedSessions = sessions.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
  
  // Get unique dates (remove duplicates from same day) using local time
  const uniqueDates = [...new Set(sortedSessions.map(session => {
    const date = new Date(session.createdAt);
    // Use local date to avoid timezone issues
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  }))];
  
  console.log('📊 Unique session dates:', uniqueDates);
  
  // Calculate current streak
  let currentStreak = 0;
  let bestStreak = 0;
  let tempStreak = 0;
  
  const today = new Date();
  const todayDateString = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
  
  // Check if most recent session is today or yesterday
  if (uniqueDates.length > 0) {
    const mostRecentDate = new Date(uniqueDates[0] + 'T12:00:00'); // Use noon to avoid timezone issues
    const todayDate = new Date(todayDateString + 'T12:00:00');
    const daysDiff = Math.floor((todayDate - mostRecentDate) / (1000 * 60 * 60 * 24));
    
    if (daysDiff <= 1) { // Today or yesterday
      currentStreak = 1;
      
      // Count consecutive days backwards
      for (let i = 1; i < uniqueDates.length; i++) {
        const currentDate = new Date(uniqueDates[i] + 'T00:00:00');
        const prevDate = new Date(uniqueDates[i - 1] + 'T00:00:00');
        const diff = Math.floor((prevDate - currentDate) / (1000 * 60 * 60 * 24));
        
        if (diff === 1) {
          currentStreak++;
        } else {
          break;
        }
      }
    }
  }
  
  // Calculate best streak (scan all dates)
  for (let i = 0; i < uniqueDates.length; i++) {
    tempStreak = 1;
    
    for (let j = i + 1; j < uniqueDates.length; j++) {
      const currentDate = new Date(uniqueDates[j] + 'T00:00:00');
      const prevDate = new Date(uniqueDates[j - 1] + 'T00:00:00');
      const diff = Math.floor((prevDate - currentDate) / (1000 * 60 * 60 * 24));
      
      if (diff === 1) {
        tempStreak++;
      } else {
        break;
      }
    }
    
    bestStreak = Math.max(bestStreak, tempStreak);
  }
  
  // Generate weekly activity with unique drill counts (last 7 days)
  const weeklyActivity = [];
  const daysOfWeek = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'];
  
  // Group sessions by date and count unique drills per day
  const dailyDrillCounts = {};
  sessions.forEach(session => {
    const sessionDateTime = new Date(session.createdAt);
    // Use local date to avoid timezone issues
    const year = sessionDateTime.getFullYear();
    const month = String(sessionDateTime.getMonth() + 1).padStart(2, '0');
    const day = String(sessionDateTime.getDate()).padStart(2, '0');
    const sessionDate = `${year}-${month}-${day}`;
    
    if (!dailyDrillCounts[sessionDate]) {
      dailyDrillCounts[sessionDate] = new Set();
    }
    // Sessions don't have drill_type - they contain multiple drills
    if (session.drills && session.drills.length > 0) {
      session.drills.forEach(drill => {
        dailyDrillCounts[sessionDate].add(drill.drill_type);
      });
    }
  });
  
  const todayLocal = new Date();
  for (let i = 6; i >= 0; i--) {
    const date = new Date(todayLocal);
    date.setDate(date.getDate() - i);
    // Use local date formatting
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const dateString = `${year}-${month}-${day}`;
    
    const uniqueDrillCount = dailyDrillCounts[dateString] ? dailyDrillCounts[dateString].size : 0;
    const hasSession = uniqueDrillCount > 0;
    const isToday = i === 0;
    
    const dayOfWeekIndex = date.getDay();
    const dayLabel = daysOfWeek[dayOfWeekIndex === 0 ? 6 : dayOfWeekIndex - 1];
    
    weeklyActivity.push({
      day: dayLabel,
      date: dateString,
      hasSession,
      isToday,
      uniqueDrillCount
    });
  }
  
  console.log('📊 Calculated streak:', { current: currentStreak, best: bestStreak });
  console.log('📊 Weekly activity:', weeklyActivity);
  console.log('📅 Daily drill counts:', dailyDrillCounts);
  
  return {
    streak: { current: currentStreak, best: bestStreak },
    weeklyActivity
  };
};

const calculateFallbackStreakData = (drillProgress) => {
  console.log('📊 Calculating fallback streak from drill progress');
  
  // If we have recent drill activity, assume a modest streak
  const hasRecentActivity = drillProgress.some(drill => 
    drill.lastPracticed === 'Today' || drill.lastPracticed.includes('ago')
  );
  
  const todayActivity = drillProgress.some(drill => drill.lastPracticed === 'Today');
  const yesterdayActivity = drillProgress.some(drill => drill.lastPracticed === 'Yesterday' || drill.lastPracticed === '1d ago');
  
  let currentStreak = 0;
  if (todayActivity && yesterdayActivity) currentStreak = 2;
  else if (todayActivity || yesterdayActivity) currentStreak = 1;
  
  const totalSessions = drillProgress.reduce((sum, drill) => sum + (drill.totalSessions || 0), 0);
  const estimatedBestStreak = Math.min(Math.max(Math.floor(totalSessions / 3), currentStreak), 15);
  
  // Generate realistic weekly pattern
  const weeklyActivity = generateMockWeeklyData(todayActivity);
  
  return {
    streak: { current: currentStreak, best: estimatedBestStreak },
    weeklyActivity
  };
};

const generateMockWeeklyData = (todayActive = true) => {
  const today = new Date();
  const daysOfWeek = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'];
  const weeklyActivity = [];
  
  for (let i = 6; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    const dayIndex = date.getDay() === 0 ? 6 : date.getDay() - 1;
    
    const isToday = i === 0;
    const hasSession = isToday ? todayActive : Math.random() > 0.6; // 40% chance for past days
    
    weeklyActivity.push({
      day: daysOfWeek[dayIndex],
      date: date.toDateString(),
      hasSession,
      isToday,
      isFuture: false
    });
  }
  
  return weeklyActivity;
};

const generateEmptyWeeklyData = () => {
  const today = new Date();
  const daysOfWeek = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'];
  const weeklyActivity = [];
  
  for (let i = 6; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    const dayIndex = date.getDay() === 0 ? 6 : date.getDay() - 1;
    
    const isToday = i === 0;
    
    weeklyActivity.push({
      day: daysOfWeek[dayIndex],
      date: date.toDateString(),
      hasSession: false, // No sessions
      isToday,
      isFuture: false
    });
  }
  
  return weeklyActivity;
};

// Helper function to format last practiced date
const formatLastPracticed = (lastPracticedStr) => {
  if (!lastPracticedStr || lastPracticedStr === 'Never') {
    return 'Never';
  }
  
  // If it's already a relative time like "Today", "2d ago", return as-is
  if (lastPracticedStr.includes('ago') || lastPracticedStr === 'Today' || lastPracticedStr === 'Yesterday') {
    return lastPracticedStr;
  }
  
  try {
    // Parse the ISO date string
    const date = new Date(lastPracticedStr);
    const now = new Date();
    
    // Calculate difference in days
    const diffTime = Math.abs(now - date);
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) {
      return 'Today';
    } else if (diffDays === 1) {
      return 'Yesterday';
    } else if (diffDays < 7) {
      return `${diffDays}d ago`;
    } else if (diffDays < 30) {
      const weeks = Math.floor(diffDays / 7);
      return `${weeks}w ago`;
    } else {
      // Format as "Sep 4, 2025"
      const options = { month: 'short', day: 'numeric', year: 'numeric' };
      return date.toLocaleDateString('en-US', options);
    }
  } catch (error) {
    console.error('Error formatting date:', error);
    return lastPracticedStr;
  }
};

export default function StatsScreen({ navigation }) {
  const [loading, setLoading] = useState(true);
  const [userStats, setUserStats] = useState({
    sessionsThisWeek: 0,
    currentStreak: 0,
    monthlyImprovement: 0,
    totalSessions: 0,
    userName: 'there'
  });
  const [drillProgress, setDrillProgress] = useState([]);
  const [recentAchievement, setRecentAchievement] = useState(null);
  const [challengeSuggestion, setChallengeSuggestion] = useState(null);
  const [streakData, setStreakData] = useState({ current: 0, best: 0 });
  const [weeklyData, setWeeklyData] = useState([]);

  useEffect(() => {
    loadProgressData();
  }, []);

  // Refresh data when user navigates to this screen
  useFocusEffect(
    useCallback(() => {
      loadProgressData();
    }, [])
  );

  const loadProgressData = async () => {
    setLoading(true);
    try {
      console.log('🔍 StatsScreen: Loading real user progress data...');
      
      // Try to load real data first
      const [userStatsResult, drillProgressResult, sessionHistoryResult] = await Promise.all([
        drillService.getUserStats(),
        drillService.getUserDrillProgress(),
        drillService.getUserSessionHistory()
      ]);
      
      console.log('🔍 StatsScreen: Raw API responses:');
      console.log('  - userStatsResult:', userStatsResult);
      console.log('  - drillProgressResult:', drillProgressResult);
      console.log('  - sessionHistoryResult:', sessionHistoryResult);

      // Calculate streak and weekly data from session history
      if (sessionHistoryResult.success && sessionHistoryResult.sessions) {
        const { streak, weeklyActivity } = calculateStreakAndWeekly(sessionHistoryResult.sessions);
        setStreakData(streak);
        setWeeklyData(weeklyActivity);
      } else {
        console.log('⚠️ StatsScreen: No session history, using fallback calculations');
        // Try to extract session data from drill progress data
        if (drillProgressResult.success && drillProgressResult.progress) {
          // Extract all recent_attempts from drill progress data
          const allSessions = [];
          console.log('🔍 Extracting sessions from drill progress data...');
          drillProgressResult.progress.forEach(drill => {
            console.log(`🔍 Checking drill ${drill.type}:`, {
              hasAttempts: !!(drill.recent_attempts && drill.recent_attempts.length > 0),
              attemptCount: drill.recent_attempts?.length || 0
            });
            if (drill.recent_attempts && drill.recent_attempts.length > 0) {
              drill.recent_attempts.forEach(attempt => {
                console.log('📝 Adding session:', {
                  created_at: attempt.created_at,
                  drill_type: attempt.drill_type || drill.type
                });
                allSessions.push({
                  created_at: attempt.created_at,
                  drill_type: attempt.drill_type || drill.type
                });
              });
            }
          });
          console.log('🎯 Total sessions extracted:', allSessions.length);
          
          if (allSessions.length > 0) {
            console.log('📊 Calculating fallback streak from drill progress sessions:', allSessions.length);
            const { streak, weeklyActivity } = calculateStreakAndWeekly(allSessions);
            setStreakData(streak);
            setWeeklyData(weeklyActivity);
          } else {
            const fallbackData = calculateFallbackStreakData(drillProgressResult.progress);
            setStreakData(fallbackData.streak);
            setWeeklyData(fallbackData.weeklyActivity);
          }
        } else {
          // No session data - show zeros
          setStreakData({ current: 0, best: 0 });
          setWeeklyData(generateEmptyWeeklyData());
        }
      }
      
      // Update user stats (no fallback - show real data or zeros)
      if (userStatsResult.success && userStatsResult.stats) {
        console.log('✅ StatsScreen: User stats loaded:', userStatsResult.stats);
        setUserStats(userStatsResult.stats);
      } else {
        console.log('⚠️ StatsScreen: No user stats available');
        const userStr = await AsyncStorage.getItem('user');
        const user = userStr ? JSON.parse(userStr) : null;
        const userName = user?.full_name?.split(' ')[0] || 'there';
        
        setUserStats({
          sessionsThisWeek: 0,
          currentStreak: 0,
          monthlyImprovement: 0,
          totalSessions: 0,
          userName: userName
        });
      }
      
      // Update drill progress (no fake data - show real data or empty)
      if (drillProgressResult.success && drillProgressResult.progress && drillProgressResult.progress.length > 0) {
        console.log('✅ StatsScreen: Drill progress loaded:', drillProgressResult.progress.length, 'drills');
        setDrillProgress(drillProgressResult.progress);
      } else {
        console.log('⚠️ StatsScreen: No drill progress data - showing empty state');
        setDrillProgress([]);
      }
      
    } catch (error) {
      console.error('💥 StatsScreen: Failed to load progress data:', error);
      
      // Show empty state on error
      console.log('🔧 StatsScreen: Showing empty state due to error');
      const userStr = await AsyncStorage.getItem('user');
      const user = userStr ? JSON.parse(userStr) : null;
      const userName = user?.full_name?.split(' ')[0] || 'there';
      
      setUserStats({
        sessionsThisWeek: 0,
        currentStreak: 0,
        monthlyImprovement: 0,
        totalSessions: 0,
        userName: userName
      });
      
      setDrillProgress([]);
      setStreakData({ current: 0, best: 0 });
      setWeeklyData(generateEmptyWeeklyData());
    } finally {
      console.log('🏁 StatsScreen: Loading complete');
      setLoading(false);
    }
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

  const renderStatsHeader = () => (
    <View style={styles.statsHeader}>
      <View style={styles.headerRow}>
        <Text style={styles.greeting}>Hey {userStats.userName}!</Text>
        <TouchableOpacity onPress={handleLogout} style={styles.logoutButton}>
          <MaterialIcons name="logout" size={20} color={Colors.darkGray} />
        </TouchableOpacity>
      </View>
    </View>
  );

  const renderDrillCard = (drill) => (
    <TouchableOpacity 
      key={drill.type} 
      style={styles.drillCard}
      onPress={() => handleDrillSelect(drill)}
    >
      <View style={styles.drillHeader}>
        <MaterialIcons 
          name={DRILL_ICONS[drill.type] || 'sports-soccer'} 
          size={24} 
          color={Colors.blue} 
        />
        <Text style={styles.drillName}>{drill.name}</Text>
      </View>
      
      <Text style={styles.personalBest}>Best: {drill.personalBest?.touches || drill.personalBest || 0}</Text>
      <Text style={styles.lastPracticed}>{formatLastPracticed(drill.lastPracticed)}</Text>
      
      <View style={styles.sessionCount}>
        <Text style={styles.sessionCountText}>{drill.totalSessions} sessions</Text>
      </View>
    </TouchableOpacity>
  );

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={Colors.blue} />
          <Text style={styles.loadingText}>Loading your progress...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView 
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {renderStatsHeader()}
        
        {/* Streak Card */}
        <View style={styles.streakSection}>
          <View style={styles.streakCard}>
            <Text style={styles.streakNumber}>{streakData.current}</Text>
            <View style={styles.streakInfo}>
              <Text style={styles.streakLabel}>Day Streak</Text>
              <Text style={styles.streakBest}>Best: {streakData.best} days</Text>
            </View>
            <Text style={styles.fireEmoji}>🔥</Text>
          </View>
        </View>
        
        {/* Weekly Calendar */}
        <View style={styles.calendarSection}>
          <View style={styles.calendarHeader}>
            <Text style={styles.calendarTitle}>This Week</Text>
          </View>
          <View style={styles.weekDays}>
            {weeklyData.map((dayData, index) => (
              <View key={`${dayData.day}-${index}`} style={styles.dayColumn}>
                <Text style={styles.dayLabel}>{dayData.day}</Text>
                <View style={[
                  styles.dayCircle,
                  dayData.hasSession && !dayData.isToday && styles.dayCompleted,
                  dayData.isToday && dayData.hasSession && styles.dayTodayCompleted,
                  dayData.isToday && !dayData.hasSession && styles.dayToday,
                ]}>
                  {dayData.hasSession && !dayData.isToday && <Text style={styles.checkMark}>{dayData.uniqueDrillCount}</Text>}
                  {dayData.isToday && dayData.hasSession && <Text style={styles.todayNumber}>{dayData.uniqueDrillCount}</Text>}
                  {dayData.isToday && !dayData.hasSession && <Text style={styles.todayNumber}>{new Date().getDate()}</Text>}
                  {!dayData.hasSession && !dayData.isToday && <Text style={styles.futureMark}>-</Text>}
                </View>
              </View>
            ))}
          </View>
        </View>
        
        <View style={styles.drillsSection}>
          <Text style={styles.sectionTitle}>Your Drills</Text>
          {drillProgress.length > 0 ? (
            <View style={styles.drillsGrid}>
              {drillProgress.map(renderDrillCard)}
            </View>
          ) : (
            <View style={styles.emptyDrillsState}>
              <MaterialIcons name="sports-soccer" size={48} color={Colors.darkGray} />
              <Text style={styles.emptyDrillsTitle}>No Drill Progress</Text>
              <Text style={styles.emptyDrillsSubtitle}>
                Start training to see your progress here
              </Text>
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
    backgroundColor: Colors.white,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: 20,
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
  
  // Stats Header
  statsHeader: {
    backgroundColor: Colors.white,
    padding: 20,
    marginBottom: 20,
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  greeting: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  logoutButton: {
    padding: 5,
  },
  
  // Streak Section
  streakSection: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  streakCard: {
    backgroundColor: '#FF8C42', // Orange streak card
    borderRadius: 16,
    padding: 20,
    flexDirection: 'row',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  streakNumber: {
    fontSize: 48,
    fontWeight: 'bold',
    color: Colors.white,
    marginRight: 15,
  },
  streakInfo: {
    flex: 1,
  },
  streakLabel: {
    fontSize: 18,
    fontWeight: '600',
    color: Colors.white,
    marginBottom: 4,
  },
  streakBest: {
    fontSize: 14,
    color: Colors.white,
    opacity: 0.9,
  },
  fireEmoji: {
    fontSize: 32,
  },
  
  // Calendar Section
  calendarSection: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  calendarHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  calendarTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#333',
  },
  viewCalendar: {
    fontSize: 16,
    color: Colors.blue,
    fontWeight: '500',
  },
  weekDays: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    backgroundColor: Colors.white,
    borderRadius: 12,
    paddingVertical: 20,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  dayColumn: {
    alignItems: 'center',
  },
  dayLabel: {
    fontSize: 12,
    color: Colors.darkGray,
    marginBottom: 8,
  },
  dayCircle: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: Colors.lightGray,
    alignItems: 'center',
    justifyContent: 'center',
  },
  dayCompleted: {
    backgroundColor: '#4CAF50', // Green for completed days
  },
  dayToday: {
    borderWidth: 2,
    borderColor: Colors.blue,
    backgroundColor: Colors.white,
  },
  dayTodayCompleted: {
    borderWidth: 2,
    borderColor: Colors.gold,
    backgroundColor: Colors.gold,
  },
  checkMark: {
    color: Colors.white,
    fontSize: 16,
    fontWeight: 'bold',
  },
  todayNumber: {
    color: Colors.blue,
    fontSize: 14,
    fontWeight: 'bold',
  },
  futureMark: {
    color: Colors.darkGray,
    fontSize: 12,
  },
  
  // Drills Section
  drillsSection: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#333',
    marginBottom: 15,
  },
  drillsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  drillCard: {
    backgroundColor: Colors.white,
    width: '48%',
    padding: 15,
    borderRadius: 12,
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
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginLeft: 8,
    flex: 1,
  },
  personalBest: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  lastPracticed: {
    fontSize: 12,
    color: Colors.darkGray,
    marginBottom: 15,
  },
  sessionCount: {
    position: 'absolute',
    bottom: 15,
    right: 15,
  },
  sessionCountText: {
    fontSize: 12,
    color: Colors.darkGray,
    fontWeight: '500',
  },
  
  // Empty Drills State
  emptyDrillsState: {
    alignItems: 'center',
    paddingVertical: 40,
    paddingHorizontal: 20,
  },
  emptyDrillsTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginTop: 15,
    marginBottom: 8,
  },
  emptyDrillsSubtitle: {
    fontSize: 14,
    color: Colors.darkGray,
    textAlign: 'center',
    lineHeight: 20,
  },
});