import React from 'react';
import { View, Text, StyleSheet, ImageBackground, TouchableOpacity, ScrollView, SafeAreaView } from 'react-native';

// Import the local images
import fitnessMan from '../assets/plank1.png';
import squatsImage from '../assets/Squatss.png';
import lungesImage from '../assets/Lungess.png';
import bicepCurlImage from '../assets/bicep_curl.jpg';

export default function HomeScreen({navigation}) {
    return (
        <SafeAreaView style={styles.safeArea}>
            <ScrollView contentContainerStyle={styles.container}>
                {/* Header */}
                {/* <View style={styles.headerContainer}>
                    <Text style={styles.headerText}>HOME WORKOUT</Text>
                </View> */}

                {/* Exercise Card for Plank */}
                <TouchableOpacity style={styles.cardContainer} onPress={() => navigation.navigate('Plank')}>
                    <ImageBackground
                        source={fitnessMan}
                        style={styles.cardImageBackground}
                        imageStyle={styles.cardImage}>

                        {/* Semi-transparent overlay */}
                        <View style={styles.overlay} />

                        {/* Text aligned to the left */}
                        <View style={styles.leftText}>
                            <Text style={styles.cardTitle}> Plank</Text>
                            <Text style={styles.cardDetails}>Form Analysis • Real-Time Feedback</Text>
                        </View>
                    </ImageBackground>
                </TouchableOpacity>

                {/* Exercise Card for Bicep Curl */}
                <TouchableOpacity style={styles.cardContainer} onPress={() => navigation.navigate('BicepCurl')}>
                    <ImageBackground
                        source={bicepCurlImage}
                        style={styles.cardImageBackground}
                        imageStyle={styles.cardImage}>

                        {/* Semi-transparent overlay */}
                        <View style={styles.overlay} />

                        {/* Text aligned to the left */}
                        <View style={styles.leftText}>
                            <Text style={styles.cardTitle}>Bicep Curl</Text>
                            <Text style={styles.cardDetails}>Form Analysis • Real-Time Feedback</Text>
                        </View>
                    </ImageBackground>
                </TouchableOpacity>

                {/* Exercise Card for Lunges */}
                <TouchableOpacity style={styles.cardContainer} onPress={() => navigation.navigate('Lunges')}>
                    <ImageBackground
                        source={lungesImage}
                        style={styles.cardImageBackground}
                        imageStyle={styles.cardImage}>

                        {/* Semi-transparent overlay */}
                        <View style={styles.overlay} />

                        {/* Text aligned to the left */}
                        <View style={styles.leftText}>
                            <Text style={styles.cardTitle}>Lunges</Text>
                            <Text style={styles.cardDetails}>Form Analysis • Real-Time Feedback</Text>
                        </View>
                    </ImageBackground>
                </TouchableOpacity>

                {/* Exercise Card for Squats */}
                <TouchableOpacity style={styles.cardContainer} onPress={() => navigation.navigate('Squats')}>
                    <ImageBackground
                        source={squatsImage}
                        style={styles.cardImageBackground}
                        imageStyle={styles.cardImage}>

                        {/* Semi-transparent overlay */}
                        <View style={styles.overlay} />

                        {/* Text aligned to the left */}
                        <View style={styles.leftText}>
                            <Text style={styles.cardTitle}>Squats</Text>
                            <Text style={styles.cardDetails}>Form Analysis • Real-Time Feedback</Text>
                        </View>
                    </ImageBackground>
                </TouchableOpacity>
            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    safeArea: {
        flex: 1,
        backgroundColor: '#F8F9FA',
    },
    container: {
        padding: 20,
    },
    headerContainer: {
        marginTop: 10,
        marginBottom: 10,
        backgroundColor: '#f8f9fa', // Light background
        paddingVertical: 10,
        borderRadius: 10,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 5,
        elevation: 2,
    },
    headerText: {
        fontSize: 28,
        fontWeight: 'bold',
        color: '#333', // Darker for professionalism
        textAlign: 'center',
    },
    cardContainer: {
        borderRadius: 15,
        overflow: 'hidden',
        marginVertical: 10,
        borderWidth: 1, // Subtle border
        borderColor: '#ddd', // Light grey
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.2,
        shadowRadius: 3,
        elevation: 2,
    },
    cardImageBackground: {
        width: '100%',
        height: 145, // Adjust height as needed to match the proportions
        justifyContent: 'center', // Center the content vertically
    },
    cardImage: {
        borderRadius: 15,
    },
    overlay: {
        ...StyleSheet.absoluteFillObject, // Fills the entire ImageBackground
        backgroundColor: 'rgba(0, 0, 0, 0.3)', // Adjust opacity and color as needed
        borderRadius: 15, // To ensure rounded corners match
    },
    leftText: {
        paddingLeft: 15, // Push text to the left
        justifyContent: 'center', // Keep text vertically centered
    },
    cardTitle: {
        fontSize: 22,
        fontWeight: 'bold',
        color: '#fff', // White text for contrast on the image
        textAlign: 'left', // Align text to the left
    },
    cardDetails: {
        fontSize: 16,
        color: '#fff',
        textAlign: 'left', // Align text to the left
        marginTop: 5,
    },
    
});
