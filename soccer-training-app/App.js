import 'expo-dev-client';
import React, { useEffect, useState } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { StatusBar } from 'expo-status-bar';
import authService from './src/services/auth';

// Screens
import LoginScreen from './src/screens/LoginScreen';
import RegisterScreen from './src/screens/RegisterScreen';
import ProgressHomeScreen from './src/screens/ProgressHomeScreen';
import DrillSelectionScreen from './src/screens/DrillSelectionScreen';
import VideoRecordingScreen from './src/screens/VideoRecordingScreen';
import AnalysisProgressScreen from './src/screens/AnalysisProgressScreen';
import AnalysisResultsScreen from './src/screens/AnalysisResultsScreen';

const Stack = createNativeStackNavigator();

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    checkAuthStatus();
  }, []);

  const checkAuthStatus = async () => {
    const authenticated = await authService.isAuthenticated();
    setIsAuthenticated(authenticated);
    setIsLoading(false);
  };

  if (isLoading) {
    // TODO: Add a proper splash screen
    return null;
  }

  return (
    <NavigationContainer>
      <StatusBar style="auto" />
      <Stack.Navigator
        initialRouteName={isAuthenticated ? "Home" : "Login"}
        screenOptions={{
          headerShown: false,
        }}
      >
        <Stack.Screen name="Login" component={LoginScreen} />
        <Stack.Screen name="Register" component={RegisterScreen} />
        <Stack.Screen name="Home" component={ProgressHomeScreen} />
        <Stack.Screen name="DrillSelection" component={DrillSelectionScreen} />
        <Stack.Screen name="VideoRecording" component={VideoRecordingScreen} />
        <Stack.Screen name="AnalysisProgress" component={AnalysisProgressScreen} />
        <Stack.Screen name="AnalysisResults" component={AnalysisResultsScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}