import React from 'react';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

// Stats screens
import StatsScreen from '../screens/StatsScreen';
import DrillProgressScreen from '../screens/DrillProgressScreen';

const Stack = createNativeStackNavigator();

export default function StatsNavigator() {
  return (
    <Stack.Navigator
      screenOptions={{
        headerShown: false,
      }}
    >
      <Stack.Screen 
        name="StatsHome" 
        component={StatsScreen} 
      />
      <Stack.Screen 
        name="DrillProgress" 
        component={DrillProgressScreen} 
      />
    </Stack.Navigator>
  );
}