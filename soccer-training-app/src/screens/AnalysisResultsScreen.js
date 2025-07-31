import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';

// Real Madrid Color Palette
const Colors = {
  primary: '#663399',      // Royal Purple
  gold: '#FFD700',         // Gold
  navy: '#001F3F',         // Navy Blue
  white: '#FFFFFF',        // White
  lightGray: '#F8F9FA',    // Light Gray
  darkGray: '#6C757D',     // Dark Gray
};

export default function AnalysisResultsScreen({ route, navigation }) {
  const { results, drillName, drillType } = route.params;
  const [activeTab, setActiveTab] = useState('overview');

  const handleNewRecording = () => {
    navigation.navigate('VideoRecording', {
      drillType: drillType,
      drillName: drillName
    });
  };

  const handleBackToHome = () => {
    navigation.navigate('Home');
  };

  const handleTryAnotherDrill = () => {
    navigation.navigate('DrillSelection');
  };

  const renderScoreCard = (title, score, maxScore, color = Colors.primary) => (
    <View style={styles.scoreCard}>
      <Text style={styles.scoreTitle}>{title}</Text>
      <View style={styles.scoreContainer}>
        <Text style={[styles.scoreValue, { color }]}>
          {score}
        </Text>
        <Text style={styles.scoreMax}>/ {maxScore}</Text>
      </View>
      <View style={styles.scoreBarContainer}>
        <View style={styles.scoreBarBackground}>
          <View 
            style={[
              styles.scoreBarFill,
              { 
                width: `${(score / maxScore) * 100}%`,
                backgroundColor: color
              }
            ]}
          />
        </View>
      </View>
    </View>
  );

  const renderFeedbackItem = (item, index) => (
    <View key={index} style={styles.feedbackItem}>
      <MaterialIcons 
        name={item.type === 'positive' ? 'check-circle' : 'info'} 
        size={20} 
        color={item.type === 'positive' ? Colors.primary : Colors.gold} 
      />
      <Text style={styles.feedbackText}>{item.message}</Text>
    </View>
  );

  const getOverallGrade = (score) => {
    if (score >= 90) return { grade: 'A+', color: Colors.primary };
    if (score >= 80) return { grade: 'A', color: Colors.primary };
    if (score >= 70) return { grade: 'B+', color: Colors.gold };
    if (score >= 60) return { grade: 'B', color: Colors.gold };
    if (score >= 50) return { grade: 'C', color: Colors.gold };
    return { grade: 'D', color: Colors.darkGray };
  };

  const overallScore = results?.overall_score || 0;
  const gradeInfo = getOverallGrade(overallScore);

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity 
          style={styles.backButton}
          onPress={handleBackToHome}
        >
          <MaterialIcons name="home" size={24} color={Colors.darkGray} />
        </TouchableOpacity>
        <Text style={styles.title}>Analysis Results</Text>
        <TouchableOpacity 
          style={styles.shareButton}
          onPress={() => Alert.alert('Share', 'Share functionality coming soon!')}
        >
          <MaterialIcons name="share" size={24} color={Colors.darkGray} />
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {/* Overall Score */}
        <View style={styles.overallContainer}>
          <Text style={styles.drillName}>{drillName}</Text>
          <View style={styles.gradeContainer}>
            <Text style={[styles.gradeText, { color: gradeInfo.color }]}>
              {gradeInfo.grade}
            </Text>
            <Text style={styles.scoreText}>{overallScore}/100</Text>
          </View>
          <Text style={styles.completionText}>
            Analysis completed • {new Date().toLocaleDateString()}
          </Text>
        </View>

        {/* Tab Navigation */}
        <View style={styles.tabContainer}>
          <TouchableOpacity
            style={[styles.tab, activeTab === 'overview' && styles.activeTab]}
            onPress={() => setActiveTab('overview')}
          >
            <Text style={[styles.tabText, activeTab === 'overview' && styles.activeTabText]}>
              Overview
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.tab, activeTab === 'detailed' && styles.activeTab]}
            onPress={() => setActiveTab('detailed')}
          >
            <Text style={[styles.tabText, activeTab === 'detailed' && styles.activeTabText]}>
              Detailed
            </Text>
          </TouchableOpacity>
        </View>

        {activeTab === 'overview' && (
          <View style={styles.tabContent}>
            {/* Key Metrics */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Key Metrics</Text>
              <View style={styles.scoresGrid}>
                {renderScoreCard(
                  'Technique', 
                  results?.technique_score || 0, 
                  100,
                  '#2196F3'
                )}
                {renderScoreCard(
                  'Consistency', 
                  results?.consistency_score || 0, 
                  100,
                  '#FF9800'
                )}
                {renderScoreCard(
                  'Speed', 
                  results?.speed_score || 0, 
                  100,
                  '#9C27B0'
                )}
                {renderScoreCard(
                  'Accuracy', 
                  results?.accuracy_score || 0, 
                  100,
                  Colors.primary
                )}
              </View>
            </View>

            {/* Quick Insights */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Quick Insights</Text>
              <View style={styles.insightCard}>
                <MaterialIcons name="trending-up" size={24} color={Colors.primary} />
                <View style={styles.insightContent}>
                  <Text style={styles.insightTitle}>Strengths</Text>
                  <Text style={styles.insightText}>
                    {results?.strengths || 'Good ball control and consistent touches'}
                  </Text>
                </View>
              </View>
              
              <View style={styles.insightCard}>
                <MaterialIcons name="build" size={24} color={Colors.gold} />
                <View style={styles.insightContent}>
                  <Text style={styles.insightTitle}>Areas to Improve</Text>
                  <Text style={styles.insightText}>
                    {results?.improvements || 'Focus on maintaining rhythm and body positioning'}
                  </Text>
                </View>
              </View>
            </View>
          </View>
        )}

        {activeTab === 'detailed' && (
          <View style={styles.tabContent}>
            {/* Detailed Analysis */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Performance Breakdown</Text>
              
              <View style={styles.detailCard}>
                <Text style={styles.detailTitle}>Technique Analysis</Text>
                <Text style={styles.detailText}>
                  {results?.detailed_feedback?.technique || 
                   'Your ball control shows good fundamentals. Focus on keeping your touches closer to your body for better control.'}
                </Text>
              </View>

              <View style={styles.detailCard}>
                <Text style={styles.detailTitle}>Timing & Rhythm</Text>
                <Text style={styles.detailText}>
                  {results?.detailed_feedback?.timing || 
                   'Maintain consistent timing between touches. Try counting or using a metronome to improve rhythm.'}
                </Text>
              </View>

              <View style={styles.detailCard}>
                <Text style={styles.detailTitle}>Movement Pattern</Text>
                <Text style={styles.detailText}>
                  {results?.detailed_feedback?.movement || 
                   'Good overall movement. Keep your head up more to improve spatial awareness while controlling the ball.'}
                </Text>
              </View>
            </View>

            {/* Action Items */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Recommended Actions</Text>
              {(results?.action_items || [
                { type: 'positive', message: 'Continue practicing daily for consistency' },
                { type: 'improvement', message: 'Focus on weaker foot development' },
                { type: 'improvement', message: 'Practice at varying speeds' }
              ]).map(renderFeedbackItem)}
            </View>
          </View>
        )}
      </ScrollView>

      {/* Action Buttons */}
      <View style={styles.actionContainer}>
        <TouchableOpacity 
          style={[styles.actionButton, styles.secondaryButton]}
          onPress={handleTryAnotherDrill}
        >
          <Text style={styles.secondaryButtonText}>Try Another Drill</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.actionButton, styles.primaryButton]}
          onPress={handleNewRecording}
        >
          <MaterialIcons name="videocam" size={20} color={Colors.white} />
          <Text style={styles.primaryButtonText}>Record Again</Text>
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
    justifyContent: 'space-between',
    padding: 20,
    backgroundColor: Colors.white,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  backButton: {
    padding: 5,
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    color: Colors.navy,
  },
  shareButton: {
    padding: 5,
  },
  content: {
    flex: 1,
  },
  overallContainer: {
    backgroundColor: Colors.white,
    padding: 30,
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  drillName: {
    fontSize: 18,
    color: Colors.darkGray,
    marginBottom: 15,
  },
  gradeContainer: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginBottom: 10,
  },
  gradeText: {
    fontSize: 48,
    fontWeight: 'bold',
    marginRight: 10,
  },
  scoreText: {
    fontSize: 24,
    color: Colors.darkGray,
    fontWeight: '500',
  },
  completionText: {
    fontSize: 14,
    color: '#999',
  },
  tabContainer: {
    flexDirection: 'row',
    backgroundColor: Colors.white,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  tab: {
    flex: 1,
    paddingVertical: 15,
    alignItems: 'center',
  },
  activeTab: {
    borderBottomWidth: 2,
    borderBottomColor: Colors.primary,
  },
  tabText: {
    fontSize: 16,
    color: Colors.darkGray,
    fontWeight: '500',
  },
  activeTabText: {
    color: Colors.primary,
    fontWeight: '600',
  },
  tabContent: {
    flex: 1,
  },
  section: {
    padding: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: Colors.navy,
    marginBottom: 15,
  },
  scoresGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  scoreCard: {
    backgroundColor: Colors.white,
    padding: 15,
    borderRadius: 12,
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
  scoreTitle: {
    fontSize: 14,
    color: Colors.darkGray,
    marginBottom: 10,
  },
  scoreContainer: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginBottom: 10,
  },
  scoreValue: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  scoreMax: {
    fontSize: 16,
    color: '#999',
    marginLeft: 4,
  },
  scoreBarContainer: {
    height: 4,
  },
  scoreBarBackground: {
    height: '100%',
    backgroundColor: '#e0e0e0',
    borderRadius: 2,
    overflow: 'hidden',
  },
  scoreBarFill: {
    height: '100%',
    borderRadius: 2,
  },
  insightCard: {
    backgroundColor: Colors.white,
    padding: 15,
    borderRadius: 12,
    flexDirection: 'row',
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  insightContent: {
    flex: 1,
    marginLeft: 15,
  },
  insightTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: Colors.navy,
    marginBottom: 5,
  },
  insightText: {
    fontSize: 14,
    color: Colors.darkGray,
    lineHeight: 20,
  },
  detailCard: {
    backgroundColor: Colors.white,
    padding: 20,
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
  detailTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: Colors.navy,
    marginBottom: 10,
  },
  detailText: {
    fontSize: 14,
    color: Colors.darkGray,
    lineHeight: 22,
  },
  feedbackItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: Colors.white,
    padding: 15,
    borderRadius: 8,
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 1,
    },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  feedbackText: {
    flex: 1,
    fontSize: 14,
    color: Colors.darkGray,
    marginLeft: 12,
    lineHeight: 20,
  },
  actionContainer: {
    flexDirection: 'row',
    padding: 20,
    backgroundColor: Colors.white,
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  actionButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 15,
    borderRadius: 8,
    marginHorizontal: 5,
  },
  primaryButton: {
    backgroundColor: Colors.primary,
  },
  primaryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
  secondaryButton: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#ddd',
  },
  secondaryButtonText: {
    color: Colors.darkGray,
    fontSize: 16,
    fontWeight: '500',
  },
});