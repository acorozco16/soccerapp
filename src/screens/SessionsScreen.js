import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  ActivityIndicator,
  TouchableOpacity,
  RefreshControl,
  Modal,
  TextInput,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import drillService from '../services/drills';
import invitationService from '../services/invitations';

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

export default function SessionsScreen({ navigation }) {
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [error, setError] = useState(null);
  const [inviteModalVisible, setInviteModalVisible] = useState(false);
  const [inviteEmail, setInviteEmail] = useState('');
  const [sendingInvite, setSendingInvite] = useState(false);

  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Load real sessions from API
      const result = await drillService.getUserSessionHistory(20, 0);
      
      if (result.success) {
        setSessions(result.sessions);
      } else {
        setError(result.error || 'Failed to load sessions');
      }
      
    } catch (error) {
      console.error('Error loading sessions:', error);
      setError('Failed to load training sessions');
    } finally {
      setLoading(false);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadSessions();
    setRefreshing(false);
  };

  const handleClap = async (sessionId) => {
    try {
      const result = await drillService.clapForSession(sessionId);
      if (result.success) {
        // Update the sessions state with new clap count
        setSessions(prevSessions =>
          prevSessions.map(session =>
            session.id === sessionId
              ? {
                  ...session,
                  clapCount: result.data.clap_count,
                  userHasClapped: result.data.user_has_clapped
                }
              : session
          )
        );
      } else {
        console.error('Failed to clap:', result.error);
      }
    } catch (error) {
      console.error('Error clapping for session:', error);
    }
  };

  const handleSendInvite = async () => {
    if (!inviteEmail || !inviteEmail.includes('@')) {
      Alert.alert('Invalid Email', 'Please enter a valid email address');
      return;
    }

    try {
      setSendingInvite(true);
      
      const result = await invitationService.sendInvitation(inviteEmail);
      
      if (result.success) {
        Alert.alert(
          'Invitation Sent!',
          `Share this code with your friend:\n\n${result.data.invitation_code}\n\nThey can use it during registration to join your training network.`,
          [
            {
              text: 'Copy Code',
              onPress: () => {
                // In a real app, you'd copy to clipboard
                console.log('Code to copy:', result.data.invitation_code);
              }
            },
            { text: 'Done', style: 'default' }
          ]
        );

        setInviteModalVisible(false);
        setInviteEmail('');
      } else {
        Alert.alert('Error', result.error || 'Failed to send invitation');
      }
      
    } catch (error) {
      console.error('Error sending invite:', error);
      Alert.alert('Error', 'Failed to send invitation. Please try again.');
    } finally {
      setSendingInvite(false);
    }
  };

  const renderSessionCard = (session) => (
    <View key={session.id} style={styles.sessionCard}>
      <View style={styles.sessionHeader}>
        <View style={styles.userInfo}>
          <View style={[styles.avatar, session.isYou && styles.yourAvatar]}>
            <Text style={styles.avatarText}>{session.userInitials}</Text>
          </View>
          <View style={styles.userDetails}>
            <View style={styles.userNameRow}>
              <Text style={styles.userName}>{session.userName}</Text>
              <Text style={styles.sessionType}> • {session.title}</Text>
            </View>
            <Text style={styles.timeAgo}>{session.timeAgo}</Text>
          </View>
        </View>
      </View>

      <View style={styles.sessionSummary}>
        <View style={styles.summaryItem}>
          <MaterialIcons name="timer" size={16} color={Colors.darkGray} />
          <Text style={styles.summaryText}>{session.totalDuration}</Text>
        </View>
        <View style={styles.summaryItem}>
          <MaterialIcons name="sports-soccer" size={16} color={Colors.darkGray} />
          <Text style={styles.summaryText}>{session.totalTouches} total touches</Text>
        </View>
      </View>

      {/* Drill breakdown */}
      <View style={styles.drillsContainer}>
        {session.drills && session.drills.map((drill, index) => (
          <View key={index} style={styles.drillItem}>
            <View style={styles.drillInfo}>
              <Text style={styles.drillName}>{drill.drill_type}</Text>
              <View style={styles.drillStats}>
                <Text style={styles.drillDuration}>{Math.floor(drill.duration / 60)}:{(drill.duration % 60).toString().padStart(2, '0')}</Text>
                <Text style={styles.drillDivider}> • </Text>
                <Text style={styles.drillTouches}>{drill.touches} touches</Text>
                {drill.is_personal_best && (
                  <View style={styles.pbBadge}>
                    <Text style={styles.pbText}>🏆 PB</Text>
                  </View>
                )}
              </View>
            </View>
          </View>
        ))}
      </View>

      <TouchableOpacity 
        style={styles.sessionActions}
        onPress={() => handleClap(session.id)}
      >
        <View style={[
          styles.clapButton,
          session.userHasClapped && styles.clapButtonActive
        ]}>
          <Text style={styles.clapIcon}>👏</Text>
          <Text style={[
            styles.clapText,
            session.userHasClapped && styles.clapTextActive
          ]}>
            {session.clapCount > 0 ? `${session.clapCount}` : 'Clap'}
          </Text>
        </View>
      </TouchableOpacity>
    </View>
  );

  const renderEmptyState = () => (
    <View style={styles.emptyState}>
      <MaterialIcons name="people-outline" size={64} color={Colors.darkGray} />
      <Text style={styles.emptyTitle}>No Sessions Yet</Text>
      <Text style={styles.emptySubtitle}>
        Connect with friends to see their training sessions here
      </Text>
      <TouchableOpacity 
        style={styles.inviteButton}
        onPress={() => setInviteModalVisible(true)}
      >
        <MaterialIcons name="person-add" size={20} color={Colors.white} />
        <Text style={styles.inviteButtonText}>Invite Friends</Text>
      </TouchableOpacity>
    </View>
  );

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={Colors.blue} />
          <Text style={styles.loadingText}>Loading sessions...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Sessions</Text>
        <View style={styles.headerButtons}>
          <TouchableOpacity 
            style={styles.headerButton}
            onPress={() => navigation.navigate('Friends')}
          >
            <MaterialIcons name="group" size={24} color={Colors.blue} />
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.headerButton, styles.headerInviteButton]}
            onPress={() => setInviteModalVisible(true)}
          >
            <MaterialIcons name="person-add" size={24} color={Colors.blue} />
          </TouchableOpacity>
        </View>
      </View>

      <ScrollView 
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            colors={[Colors.blue]}
            tintColor={Colors.blue}
          />
        }
      >
        {error ? (
          <View style={styles.errorContainer}>
            <Text style={styles.errorText}>{error}</Text>
            <TouchableOpacity onPress={loadSessions} style={styles.retryButton}>
              <Text style={styles.retryButtonText}>Retry</Text>
            </TouchableOpacity>
          </View>
        ) : sessions.length > 0 ? (
          sessions.map(renderSessionCard)
        ) : !loading ? (
          renderEmptyState()
        ) : null}
      </ScrollView>

      {/* Invite Modal */}
      <Modal
        animationType="slide"
        transparent={true}
        visible={inviteModalVisible}
        onRequestClose={() => setInviteModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Invite Friends</Text>
              <TouchableOpacity
                onPress={() => setInviteModalVisible(false)}
                style={styles.modalCloseButton}
              >
                <MaterialIcons name="close" size={24} color={Colors.darkGray} />
              </TouchableOpacity>
            </View>

            <Text style={styles.modalSubtitle}>
              Invite friends to join your soccer training network
            </Text>

            <TextInput
              style={styles.emailInput}
              placeholder="Enter friend's email address"
              placeholderTextColor={Colors.darkGray}
              value={inviteEmail}
              onChangeText={setInviteEmail}
              keyboardType="email-address"
              autoCapitalize="none"
              autoCorrect={false}
            />

            <TouchableOpacity
              style={[styles.sendInviteButton, sendingInvite && styles.sendInviteButtonDisabled]}
              onPress={handleSendInvite}
              disabled={sendingInvite}
            >
              {sendingInvite ? (
                <ActivityIndicator size="small" color={Colors.white} />
              ) : (
                <>
                  <MaterialIcons name="send" size={20} color={Colors.white} />
                  <Text style={styles.sendInviteButtonText}>Send Invitation</Text>
                </>
              )}
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.lightGray,
  },
  header: {
    backgroundColor: Colors.white,
    paddingHorizontal: 20,
    paddingTop: 10,
    paddingBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: Colors.lightGray,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: Colors.textPrimary,
  },
  headerButtons: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  headerButton: {
    padding: 8,
    borderRadius: 20,
    backgroundColor: Colors.lightGray,
    marginLeft: 10,
  },
  headerInviteButton: {
    marginLeft: 8,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 15,
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
  
  // Session Card
  sessionCard: {
    backgroundColor: Colors.white,
    borderRadius: 12,
    padding: 15,
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
  sessionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  userInfo: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  avatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: Colors.blue,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 10,
  },
  yourAvatar: {
    backgroundColor: Colors.gold,
  },
  avatarText: {
    color: Colors.white,
    fontSize: 14,
    fontWeight: 'bold',
  },
  userDetails: {
    flex: 1,
  },
  userNameRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  userName: {
    fontSize: 16,
    fontWeight: '600',
    color: Colors.textPrimary,
  },
  sessionType: {
    fontSize: 16,
    color: Colors.darkGray,
    fontWeight: '400',
  },
  youBadge: {
    backgroundColor: Colors.blue,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    marginLeft: 6,
  },
  youBadgeText: {
    color: Colors.white,
    fontSize: 10,
    fontWeight: 'bold',
  },
  timeAgo: {
    fontSize: 12,
    color: Colors.darkGray,
  },
  drillTypeContainer: {
    alignSelf: 'flex-start',
  },
  drillType: {
    fontSize: 14,
    color: Colors.darkGray,
    fontWeight: '500',
    textAlign: 'right',
  },
  sessionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: Colors.textPrimary,
    marginBottom: 15,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 15,
  },
  stat: {
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 20,
    fontWeight: 'bold',
    color: Colors.textPrimary,
  },
  statLabel: {
    fontSize: 12,
    color: Colors.darkGray,
  },
  improvementBadge: {
    backgroundColor: Colors.blue,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 6,
    alignSelf: 'flex-start',
    marginBottom: 10,
  },
  improvementText: {
    color: Colors.white,
    fontSize: 12,
    fontWeight: '600',
    marginLeft: 4,
  },
  // Session Summary
  sessionSummary: {
    flexDirection: 'row',
    marginBottom: 12,
  },
  summaryItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 20,
  },
  summaryText: {
    fontSize: 14,
    color: Colors.darkGray,
    marginLeft: 4,
  },
  
  // Drills Container
  drillsContainer: {
    backgroundColor: Colors.lightGray,
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
  },
  drillItem: {
    marginBottom: 8,
  },
  drillInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  drillName: {
    fontSize: 15,
    fontWeight: '500',
    color: Colors.textPrimary,
    textTransform: 'capitalize',
  },
  drillStats: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  drillDuration: {
    fontSize: 14,
    color: Colors.darkGray,
  },
  drillDivider: {
    fontSize: 14,
    color: Colors.darkGray,
  },
  drillTouches: {
    fontSize: 14,
    color: Colors.darkGray,
  },
  pbBadge: {
    marginLeft: 8,
  },
  pbText: {
    fontSize: 14,
    color: Colors.gold,
  },
  
  // Clap Button
  sessionActions: {
    paddingTop: 8,
  },
  clapButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.lightGray,
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
  },
  clapButtonActive: {
    backgroundColor: Colors.gold,
  },
  clapIcon: {
    fontSize: 18,
    marginRight: 6,
  },
  clapText: {
    fontSize: 14,
    fontWeight: '600',
    color: Colors.darkGray,
  },
  clapTextActive: {
    color: Colors.white,
  },
  
  // Error State
  errorContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 100,
  },
  errorText: {
    fontSize: 16,
    color: Colors.red,
    marginBottom: 16,
    textAlign: 'center',
  },
  retryButton: {
    backgroundColor: Colors.blue,
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  retryButtonText: {
    color: Colors.white,
    fontSize: 16,
    fontWeight: '600',
  },
  
  // Empty State
  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 100,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: Colors.textPrimary,
    marginTop: 15,
    marginBottom: 8,
  },
  emptySubtitle: {
    fontSize: 16,
    color: Colors.darkGray,
    textAlign: 'center',
    lineHeight: 22,
    marginBottom: 20,
  },
  inviteButton: {
    backgroundColor: Colors.blue,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
    marginTop: 10,
  },
  inviteButtonText: {
    color: Colors.white,
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },

  // Modal Styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: Colors.white,
    borderRadius: 12,
    padding: 24,
    margin: 20,
    width: '90%',
    maxWidth: 400,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: Colors.textPrimary,
  },
  modalCloseButton: {
    padding: 4,
  },
  modalSubtitle: {
    fontSize: 16,
    color: Colors.darkGray,
    marginBottom: 20,
    textAlign: 'center',
  },
  emailInput: {
    borderWidth: 1,
    borderColor: Colors.lightGray,
    borderRadius: 8,
    paddingHorizontal: 16,
    paddingVertical: 12,
    fontSize: 16,
    marginBottom: 20,
    backgroundColor: Colors.white,
  },
  sendInviteButton: {
    backgroundColor: Colors.blue,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    borderRadius: 8,
  },
  sendInviteButtonDisabled: {
    backgroundColor: Colors.darkGray,
  },
  sendInviteButtonText: {
    color: Colors.white,
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
});