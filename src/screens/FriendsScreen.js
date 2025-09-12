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
  SectionList,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
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

export default function FriendsScreen({ navigation }) {
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [friends, setFriends] = useState([]);
  const [connectionRequests, setConnectionRequests] = useState({ received: [], sent: [] });
  const [error, setError] = useState(null);
  const [searchModalVisible, setSearchModalVisible] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searching, setSearching] = useState(false);

  useEffect(() => {
    loadFriendsData();
  }, []);

  const loadFriendsData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Load friends and connection requests in parallel
      const [friendsResult, requestsResult] = await Promise.all([
        invitationService.getFriends(),
        invitationService.getConnectionRequests()
      ]);

      if (friendsResult.success) {
        setFriends(friendsResult.data.friends || []);
      }

      if (requestsResult.success) {
        setConnectionRequests(requestsResult.data);
      }

      if (!friendsResult.success || !requestsResult.success) {
        setError('Failed to load some friend data');
      }
      
    } catch (error) {
      console.error('Error loading friends data:', error);
      setError('Failed to load friends data');
    } finally {
      setLoading(false);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadFriendsData();
    setRefreshing(false);
  };

  const handleAcceptRequest = async (connectionId) => {
    try {
      const result = await invitationService.acceptConnectionRequest(connectionId);
      
      if (result.success) {
        Alert.alert('Success', 'Friend request accepted!');
        await loadFriendsData(); // Refresh data
      } else {
        Alert.alert('Error', result.error || 'Failed to accept request');
      }
    } catch (error) {
      console.error('Error accepting request:', error);
      Alert.alert('Error', 'Failed to accept friend request');
    }
  };

  const handleRemoveFriend = (friend) => {
    Alert.alert(
      'Remove Friend',
      `Remove ${friend.full_name} from your friends?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Remove',
          style: 'destructive',
          onPress: async () => {
            try {
              const result = await invitationService.removeConnection(friend.connection_id);
              if (result.success) {
                Alert.alert('Success', 'Friend removed');
                await loadFriendsData();
              } else {
                Alert.alert('Error', result.error || 'Failed to remove friend');
              }
            } catch (error) {
              console.error('Error removing friend:', error);
              Alert.alert('Error', 'Failed to remove friend');
            }
          }
        }
      ]
    );
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      Alert.alert('Empty Search', 'Please enter an email or name to search');
      return;
    }

    try {
      setSearching(true);
      const result = await invitationService.searchUsers(searchQuery.trim());
      
      if (result.success) {
        setSearchResults(result.data.users || []);
        
        if (result.data.users.length === 0) {
          Alert.alert('No Results', 'No users found matching your search');
        }
      } else {
        Alert.alert('Search Error', result.error || 'Failed to search users');
        setSearchResults([]);
      }
    } catch (error) {
      console.error('Error searching:', error);
      Alert.alert('Search Error', 'Failed to search users');
      setSearchResults([]);
    } finally {
      setSearching(false);
    }
  };

  const handleSendFriendRequest = async (userId) => {
    try {
      const result = await invitationService.sendConnectionRequest(userId);
      
      if (result.success) {
        Alert.alert('Success', 'Friend request sent!');
        setSearchModalVisible(false);
        setSearchQuery('');
        setSearchResults([]);
        await loadFriendsData(); // Refresh data
      } else {
        Alert.alert('Error', result.error || 'Failed to send friend request');
      }
    } catch (error) {
      console.error('Error sending friend request:', error);
      Alert.alert('Error', 'Failed to send friend request');
    }
  };

  const renderFriendCard = (friend) => (
    <View key={friend.user_id} style={styles.friendCard}>
      <View style={styles.friendInfo}>
        <View style={styles.avatar}>
          <Text style={styles.avatarText}>
            {friend.full_name ? friend.full_name.charAt(0).toUpperCase() : 'F'}
          </Text>
        </View>
        <View style={styles.friendDetails}>
          <Text style={styles.friendName}>{friend.full_name || 'Unknown'}</Text>
          <Text style={styles.friendEmail}>{friend.email}</Text>
          <Text style={styles.connectedDate}>
            Connected {new Date(friend.connected_at).toLocaleDateString()}
          </Text>
        </View>
      </View>
      <TouchableOpacity
        style={styles.removeButton}
        onPress={() => handleRemoveFriend(friend)}
      >
        <MaterialIcons name="more-vert" size={20} color={Colors.darkGray} />
      </TouchableOpacity>
    </View>
  );

  const renderRequestCard = (request, type) => (
    <View key={request.id} style={styles.requestCard}>
      <View style={styles.friendInfo}>
        <View style={styles.avatar}>
          <Text style={styles.avatarText}>
            {type === 'received' 
              ? (request.requester?.full_name?.charAt(0).toUpperCase() || 'R')
              : (request.addressee?.full_name?.charAt(0).toUpperCase() || 'A')
            }
          </Text>
        </View>
        <View style={styles.friendDetails}>
          <Text style={styles.friendName}>
            {type === 'received' 
              ? (request.requester?.full_name || 'Unknown')
              : (request.addressee?.full_name || 'Unknown')
            }
          </Text>
          <Text style={styles.friendEmail}>
            {type === 'received' 
              ? request.requester?.email 
              : request.addressee?.email
            }
          </Text>
          <Text style={styles.requestDate}>
            Sent {new Date(request.created_at).toLocaleDateString()}
          </Text>
        </View>
      </View>
      {type === 'received' && (
        <TouchableOpacity
          style={styles.acceptButton}
          onPress={() => handleAcceptRequest(request.id)}
        >
          <MaterialIcons name="check" size={20} color={Colors.white} />
        </TouchableOpacity>
      )}
    </View>
  );

  const renderEmptyState = () => (
    <View style={styles.emptyState}>
      <MaterialIcons name="people-outline" size={64} color={Colors.darkGray} />
      <Text style={styles.emptyTitle}>No Friends Yet</Text>
      <Text style={styles.emptySubtitle}>
        Invite friends to join your soccer training network
      </Text>
      <TouchableOpacity 
        style={styles.inviteButton}
        onPress={() => navigation.goBack()}
      >
        <MaterialIcons name="person-add" size={20} color={Colors.white} />
        <Text style={styles.inviteButtonText}>Send Invitations</Text>
      </TouchableOpacity>
    </View>
  );

  // Prepare sections for SectionList
  const sections = [];
  
  if (connectionRequests.received.length > 0) {
    sections.push({
      title: `Friend Requests (${connectionRequests.received.length})`,
      data: connectionRequests.received,
      type: 'received'
    });
  }
  
  if (connectionRequests.sent.length > 0) {
    sections.push({
      title: `Pending Requests (${connectionRequests.sent.length})`,
      data: connectionRequests.sent,
      type: 'sent'
    });
  }
  
  if (friends.length > 0) {
    sections.push({
      title: `My Friends (${friends.length})`,
      data: friends,
      type: 'friends'
    });
  }

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={Colors.blue} />
          <Text style={styles.loadingText}>Loading friends...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Friends</Text>
        <TouchableOpacity 
          style={styles.headerButton}
          onPress={() => setSearchModalVisible(true)}
        >
          <MaterialIcons name="person-add" size={24} color={Colors.blue} />
        </TouchableOpacity>
      </View>

      {error ? (
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>{error}</Text>
          <TouchableOpacity onPress={loadFriendsData} style={styles.retryButton}>
            <Text style={styles.retryButtonText}>Retry</Text>
          </TouchableOpacity>
        </View>
      ) : sections.length > 0 ? (
        <SectionList
          sections={sections}
          keyExtractor={(item, index) => `${item.id || item.user_id}-${index}`}
          renderItem={({ item, section }) => {
            if (section.type === 'friends') {
              return renderFriendCard(item);
            } else {
              return renderRequestCard(item, section.type);
            }
          }}
          renderSectionHeader={({ section: { title } }) => (
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>{title}</Text>
            </View>
          )}
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
        />
      ) : (
        renderEmptyState()
      )}

      {/* Search Modal */}
      <Modal
        animationType="slide"
        transparent={true}
        visible={searchModalVisible}
        onRequestClose={() => setSearchModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Find Friends</Text>
              <TouchableOpacity
                onPress={() => setSearchModalVisible(false)}
                style={styles.modalCloseButton}
              >
                <MaterialIcons name="close" size={24} color={Colors.darkGray} />
              </TouchableOpacity>
            </View>

            <Text style={styles.modalSubtitle}>
              Search for friends by email or username
            </Text>

            <TextInput
              style={styles.searchInput}
              placeholder="Enter email or username"
              placeholderTextColor={Colors.darkGray}
              value={searchQuery}
              onChangeText={setSearchQuery}
              keyboardType="email-address"
              autoCapitalize="none"
              autoCorrect={false}
            />

            <TouchableOpacity
              style={[styles.searchButton, searching && styles.searchButtonDisabled]}
              onPress={handleSearch}
              disabled={searching}
            >
              {searching ? (
                <ActivityIndicator size="small" color={Colors.white} />
              ) : (
                <>
                  <MaterialIcons name="search" size={20} color={Colors.white} />
                  <Text style={styles.searchButtonText}>Search</Text>
                </>
              )}
            </TouchableOpacity>

            {/* Search Results */}
            {searchResults.length > 0 && (
              <View style={styles.searchResults}>
                <Text style={styles.resultsTitle}>Search Results</Text>
                <ScrollView style={styles.resultsContainer} nestedScrollEnabled={true}>
                  {searchResults.map((user) => (
                    <View key={user.id} style={styles.searchResultCard}>
                      <View style={styles.friendInfo}>
                        <View style={styles.avatar}>
                          <Text style={styles.avatarText}>
                            {user.full_name ? user.full_name.charAt(0).toUpperCase() : 'U'}
                          </Text>
                        </View>
                        <View style={styles.friendDetails}>
                          <Text style={styles.friendName}>{user.full_name || 'Unknown'}</Text>
                          <Text style={styles.friendEmail}>{user.email}</Text>
                        </View>
                      </View>
                      <TouchableOpacity
                        style={styles.addButton}
                        onPress={() => handleSendFriendRequest(user.id)}
                      >
                        <MaterialIcons name="person-add" size={18} color={Colors.white} />
                      </TouchableOpacity>
                    </View>
                  ))}
                </ScrollView>
              </View>
            )}
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
  headerButton: {
    padding: 8,
    borderRadius: 20,
    backgroundColor: Colors.lightGray,
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
  
  // Section Headers
  sectionHeader: {
    backgroundColor: Colors.white,
    paddingHorizontal: 15,
    paddingVertical: 10,
    borderRadius: 8,
    marginBottom: 10,
    marginTop: 10,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: Colors.blue,
  },

  // Friend Cards
  friendCard: {
    backgroundColor: Colors.white,
    borderRadius: 12,
    padding: 15,
    marginBottom: 10,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  requestCard: {
    backgroundColor: Colors.white,
    borderRadius: 12,
    padding: 15,
    marginBottom: 10,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderLeftWidth: 3,
    borderLeftColor: Colors.gold,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  friendInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  avatar: {
    width: 45,
    height: 45,
    borderRadius: 22.5,
    backgroundColor: Colors.blue,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  avatarText: {
    color: Colors.white,
    fontSize: 16,
    fontWeight: 'bold',
  },
  friendDetails: {
    flex: 1,
  },
  friendName: {
    fontSize: 16,
    fontWeight: '600',
    color: Colors.textPrimary,
    marginBottom: 2,
  },
  friendEmail: {
    fontSize: 14,
    color: Colors.darkGray,
    marginBottom: 2,
  },
  connectedDate: {
    fontSize: 12,
    color: Colors.darkGray,
  },
  requestDate: {
    fontSize: 12,
    color: Colors.gold,
  },
  
  // Action buttons
  removeButton: {
    padding: 8,
  },
  acceptButton: {
    backgroundColor: Colors.blue,
    padding: 10,
    borderRadius: 20,
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
  searchInput: {
    borderWidth: 1,
    borderColor: Colors.lightGray,
    borderRadius: 8,
    paddingHorizontal: 16,
    paddingVertical: 12,
    fontSize: 16,
    marginBottom: 20,
    backgroundColor: Colors.white,
  },
  searchButton: {
    backgroundColor: Colors.blue,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    borderRadius: 8,
  },
  searchButtonDisabled: {
    backgroundColor: Colors.darkGray,
  },
  searchButtonText: {
    color: Colors.white,
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },

  // Search Results
  searchResults: {
    marginTop: 20,
  },
  resultsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: Colors.textPrimary,
    marginBottom: 12,
  },
  resultsContainer: {
    maxHeight: 200,
  },
  searchResultCard: {
    backgroundColor: Colors.lightGray,
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  addButton: {
    backgroundColor: Colors.blue,
    padding: 8,
    borderRadius: 16,
  },
});