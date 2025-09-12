import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
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

export default function TrainScreen({ navigation }) {
  const handleStartTraining = () => {
    navigation.navigate('DrillSelection');
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <View style={styles.header}>
          <MaterialIcons name="sports-soccer" size={64} color={Colors.blue} />
          <Text style={styles.title}>Ready to Train?</Text>
          <Text style={styles.subtitle}>
            Select your drill and start improving your skills
          </Text>
        </View>

        <TouchableOpacity 
          style={styles.startButton}
          onPress={handleStartTraining}
        >
          <MaterialIcons name="play-arrow" size={24} color={Colors.white} />
          <Text style={styles.startButtonText}>Start Training</Text>
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
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  header: {
    alignItems: 'center',
    marginBottom: 60,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
    marginTop: 20,
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 16,
    color: Colors.darkGray,
    textAlign: 'center',
    lineHeight: 22,
  },
  startButton: {
    backgroundColor: Colors.blue,
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 25,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  startButtonText: {
    color: Colors.white,
    fontSize: 18,
    fontWeight: '600',
    marginLeft: 8,
  },
});