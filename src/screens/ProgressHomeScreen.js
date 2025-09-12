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

export default function ProgressHomeScreen({ navigation }) {
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

  useEffect(() => {
    loadProgressData();
  }, []);

  const loadProgressData = async () => {
    setLoading(true);
    try {
      console.log('Loading real user progress data...');
      
      // Load user stats and drill progress in parallel
      const [userStatsResult, drillProgressResult] = await Promise.all([
        drillService.getUserStats(),
        drillService.getUserDrillProgress()
      ]);
      
      // Update user stats
      if (userStatsResult.success) {
        console.log('User stats loaded:', userStatsResult.stats);
        setUserStats(userStatsResult.stats);
      }
      
      // Update drill progress
      if (drillProgressResult.success) {
        console.log('Drill progress loaded:', drillProgressResult.progress.length, 'drills');
        setDrillProgress(drillProgressResult.progress);
        
        // Check for drills that haven't been practiced in a while for challenge suggestions
        const unpracticedDrills = drillProgressResult.progress.filter(drill => 
          drill.lastPracticed === 'Never' || drill.totalSessions === 0
        );
        
        if (unpracticedDrills.length > 0) {
          setChallengeSuggestion(unpracticedDrills[0]);
        }
      } else {
        console.error('Failed to load drill progress:', drillProgressResult.error);
        // Fallback to basic drill list
        const basicDrillsResult = await drillService.getAvailableDrills();
        if (basicDrillsResult.success) {
          const basicProgress = basicDrillsResult.drills.map(drill => ({
            ...drill,
            personalBest: 0,
            trend: 0,
            lastPracticed: 'Never',
            improvementPercentage: 0,
            totalSessions: 0
          }));
          setDrillProgress(basicProgress);
        }
      }
      
    } catch (error) {
      console.error('Failed to load progress data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getRandomLastPracticed = (index) => {
    const options = ['Today', 'Yesterday', '2 days ago', '3 days ago'];
    return `Last Practice: ${options[index % options.length]}`;
  };

  const handleLogout = async () => {
    await authService.logout();
    navigation.replace('Login');
  };

  const handleStartPractice = () => {
    navigation.navigate('DrillSelection');
  };

  const handleStartChallenge = () => {
    if (challengeSuggestion) {
      navigation.navigate('VideoRecording', {
        drillType: challengeSuggestion.type,
        drillName: challengeSuggestion.name,
      });
    }
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

  const renderChallenge = () => {
    if (!challengeSuggestion) return null;
    
    return (
      <View style={styles.challengeSection}>
        <Text style={styles.challengeTitle}>Ready for a challenge?</Text>
        <View style={styles.challengeCard}>
          <View style={styles.challengeContent}>
            <Text style={styles.challengeText}>
              Try {challengeSuggestion.name} again - you haven't practiced in 3 days!
            </Text>
          </View>
          <TouchableOpacity 
            style={styles.challengeButton}
            onPress={handleStartChallenge}
          >
            <Text style={styles.challengeButtonText}>Start Challenge</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  };

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
      
      <View style={styles.drillStats}>
        <Text style={styles.personalBest}>Best: {drill.personalBest}</Text>
        <View style={styles.trendContainer}>
          <MaterialIcons name="trending-up" size={16} color={Colors.gold} />
          <Text style={styles.trendText}>+{drill.improvementPercentage}%</Text>
        </View>
      </View>
      
      <Text style={styles.lastPracticed}>{drill.lastPracticed}</Text>
      
      {/* Mini trend line placeholder */}
      <View style={styles.trendLine}>
        <View style={[styles.trendPoint, { left: '10%' }]} />
        <View style={[styles.trendPoint, { left: '30%' }]} />
        <View style={[styles.trendPoint, { left: '50%' }]} />
        <View style={[styles.trendPoint, { left: '70%' }]} />
        <View style={[styles.trendPoint, { left: '90%', backgroundColor: Colors.gold }]} />
      </View>
    </TouchableOpacity>
  );

  const renderAchievement = () => {
    if (!recentAchievement) return null;
    
    return (
      <View style={styles.achievementCard}>
        <MaterialIcons name="emoji-events" size={24} color={Colors.gold} />
        <View style={styles.achievementContent}>
          <Text style={styles.achievementTitle}>{recentAchievement.achievement}</Text>
          <Text style={styles.achievementDetails}>{recentAchievement.details}</Text>
        </View>
      </View>
    );
  };

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
        
        <View style={styles.drillsSection}>
          <Text style={styles.sectionTitle}>Your Drills</Text>
          <View style={styles.drillsGrid}>
            {drillProgress.map(renderDrillCard)}
          </View>
        </View>
      </ScrollView>
      
      <View style={styles.bottomCTA}>
        <TouchableOpacity 
          style={styles.startPracticeButton}
          onPress={handleStartPractice}
        >
          <Text style={styles.startPracticeText}>Start Practice</Text>
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
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: 100, // Space for bottom CTA
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
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  sessionsText: {
    fontSize: 16,
    color: '#333',
    fontWeight: '500',
  },
  streakText: {
    fontSize: 16,
    color: Colors.gold,
    fontWeight: '600',
  },
  improvementCard: {
    backgroundColor: Colors.blue,
    padding: 15,
    borderRadius: 12,
    flexDirection: 'row',
    alignItems: 'center',
  },
  improvementText: {
    fontSize: 16,
    color: Colors.white,
    fontWeight: '600',
    marginLeft: 10,
  },
  
  // Challenge Section
  challengeSection: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  challengeTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#333',
    marginBottom: 10,
  },
  challengeCard: {
    backgroundColor: Colors.white,
    padding: 15,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: Colors.gold,
  },
  challengeContent: {
    marginBottom: 10,
  },
  challengeText: {
    fontSize: 16,
    color: '#333',
    lineHeight: 22,
  },
  challengeButton: {
    backgroundColor: Colors.blue,
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 8,
    alignSelf: 'flex-start',
  },
  challengeButtonText: {
    color: Colors.white,
    fontSize: 14,
    fontWeight: '600',
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
  drillStats: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  personalBest: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  trendContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  trendText: {
    fontSize: 12,
    color: Colors.gold,
    fontWeight: '600',
    marginLeft: 4,
  },
  lastPracticed: {
    fontSize: 12,
    color: Colors.darkGray,
    marginBottom: 10,
  },
  trendLine: {
    height: 2,
    backgroundColor: Colors.lightGray,
    borderRadius: 1,
    position: 'relative',
  },
  trendPoint: {
    position: 'absolute',
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: Colors.blue,
    top: -2,
  },
  
  // Achievement
  achievementCard: {
    backgroundColor: Colors.white,
    marginHorizontal: 20,
    padding: 15,
    borderRadius: 12,
    flexDirection: 'row',
    alignItems: 'center',
    borderLeftWidth: 4,
    borderLeftColor: Colors.gold,
    marginBottom: 20,
  },
  achievementContent: {
    flex: 1,
    marginLeft: 15,
  },
  achievementTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
  },
  achievementDetails: {
    fontSize: 14,
    color: Colors.darkGray,
    lineHeight: 20,
  },
  
  // Bottom CTA
  bottomCTA: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: Colors.white,
    paddingHorizontal: 20,
    paddingVertical: 15,
    paddingBottom: 30,
    borderTopWidth: 1,
    borderTopColor: Colors.lightGray,
  },
  startPracticeButton: {
    backgroundColor: Colors.blue,
    paddingVertical: 15,
    borderRadius: 12,
    alignItems: 'center',
  },
  startPracticeText: {
    color: Colors.white,
    fontSize: 18,
    fontWeight: '600',
  },
});