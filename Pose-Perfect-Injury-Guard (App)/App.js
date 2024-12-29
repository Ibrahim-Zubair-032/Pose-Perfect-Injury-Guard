import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import HomeScreen from './screens/HomeScreen'; // Make sure to import your home screen
import PlankScreen from './screens/PlankScreen'; // Create this file next
import BicepCurl from './screens/BicepCurl';
import LungesScreen from './screens/LungesScreen';
import SquatScreen from './screens/SquatScreen';

import { View, Text } from 'react-native';

// Import the Ionicons component for icons
import { Ionicons } from '@expo/vector-icons'; 

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Home">
        <Stack.Screen 
          name="Home" 
          component={HomeScreen} 
          options={{
            headerTitle: () => (
              <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                <Ionicons name="home" size={24} color="#000" style={{ marginRight: 8 }} />
                <Text style={{ fontSize: 20, fontWeight: 'bold' }}>Home</Text>
              </View>
            ),
          }} 
        />
        <Stack.Screen name="Plank" component={PlankScreen} />
        <Stack.Screen name="BicepCurl" component={BicepCurl} />
        <Stack.Screen name="Lunges" component={LungesScreen} />
        <Stack.Screen name="Squats" component={SquatScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
