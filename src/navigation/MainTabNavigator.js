import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { MaterialIcons } from '@expo/vector-icons';

// Tab Screen Navigators
import StatsNavigator from './StatsNavigator';
import TrainNavigator from './TrainNavigator';
import SessionsNavigator from './SessionsNavigator';

const Tab = createBottomTabNavigator();

// Real Madrid Color Palette
const Colors = {
  gold: '#FCBF00',         // Real Madrid Gold
  blue: '#004996',         // Real Madrid Blue
  white: '#FFFFFF',        // White
  red: '#E62644',          // Real Madrid Red
  lightGray: '#F8F9FA',    // Light Gray
  darkGray: '#6C757D',     // Dark Gray
};

export default function MainTabNavigator() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName;

          if (route.name === 'Stats') {
            iconName = 'analytics';
          } else if (route.name === 'Train') {
            iconName = 'sports-soccer';
          } else if (route.name === 'Sessions') {
            iconName = 'people';
          }

          return <MaterialIcons name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: Colors.blue,
        tabBarInactiveTintColor: Colors.darkGray,
        tabBarStyle: {
          backgroundColor: Colors.white,
          borderTopColor: Colors.lightGray,
        },
        headerShown: false,
      })}
    >
      <Tab.Screen 
        name="Stats" 
        component={StatsNavigator}
        options={{
          tabBarLabel: 'Stats',
        }}
      />
      <Tab.Screen 
        name="Train" 
        component={TrainNavigator}
        options={{
          tabBarLabel: 'Train',
        }}
      />
      <Tab.Screen 
        name="Sessions" 
        component={SessionsNavigator}
        options={{
          tabBarLabel: 'Sessions',
        }}
      />
    </Tab.Navigator>
  );
}