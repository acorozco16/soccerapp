import React from 'react';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

// Training screens
import TrainScreen from '../screens/TrainScreen';
import DrillSelectionScreen from '../screens/DrillSelectionScreen';
import VideoRecordingScreen from '../screens/VideoRecordingScreen';
import VideoProcessingScreen from '../screens/VideoProcessingScreen';
import AnalysisResultsScreen from '../screens/AnalysisResultsScreen';
import ManualLogScreen from '../screens/ManualLogScreen';
import TimerSelectionScreen from '../screens/TimerSelectionScreen';
import TimerScreen from '../screens/TimerScreen';

const Stack = createNativeStackNavigator();

export default function TrainNavigator() {
  return (
    <Stack.Navigator
      screenOptions={{
        headerShown: false,
      }}
    >
      <Stack.Screen 
        name="TrainHome" 
        component={DrillSelectionScreen} 
      />
      <Stack.Screen 
        name="VideoRecording" 
        component={VideoRecordingScreen} 
      />
      <Stack.Screen 
        name="VideoProcessing" 
        component={VideoProcessingScreen} 
      />
      <Stack.Screen 
        name="AnalysisResults" 
        component={AnalysisResultsScreen} 
      />
      <Stack.Screen 
        name="ManualLog" 
        component={ManualLogScreen} 
      />
      <Stack.Screen 
        name="TimerSelection" 
        component={TimerSelectionScreen} 
      />
      <Stack.Screen 
        name="Timer" 
        component={TimerScreen} 
      />
    </Stack.Navigator>
  );
}