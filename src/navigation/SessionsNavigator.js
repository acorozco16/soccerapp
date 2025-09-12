import React from 'react';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

// Sessions screens
import SessionsScreen from '../screens/SessionsScreen';
import FriendsScreen from '../screens/FriendsScreen';

const Stack = createNativeStackNavigator();

export default function SessionsNavigator() {
  return (
    <Stack.Navigator
      screenOptions={{
        headerShown: false,
      }}
    >
      <Stack.Screen 
        name="SessionsHome" 
        component={SessionsScreen} 
      />
      <Stack.Screen 
        name="Friends" 
        component={FriendsScreen} 
      />
    </Stack.Navigator>
  );
}