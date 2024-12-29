import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert, ActivityIndicator, ScrollView, Image, Modal, ScrollView as RNScrollView } from 'react-native';
import { Video } from 'expo-av';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';
import { Ionicons } from '@expo/vector-icons';


const squatDemoVideo = require('../assets/Squat_Demo.mp4'); // Path to demo video

export default function LungesScreen() {
    const [video, setVideo] = useState(null);
    const [videoName, setVideoName] = useState('');
    const [processing, setProcessing] = useState(false); // To handle processing state
    const [serverResponse, setServerResponse] = useState(null); // To store the server's response
    const [reportContent, setReportContent] = useState(''); // Store the fetched report content
    const [isModalVisible, setModalVisible] = useState(false); // Modal visibility state

    // Function to pick a video from the gallery
    const pickVideo = async () => {
        try {
            let result = await ImagePicker.launchImageLibraryAsync({
                mediaTypes: ImagePicker.MediaTypeOptions.Videos,
                allowsEditing: true,
                quality: 1,
            });

            if (!result.canceled) {
                setVideo(result.assets[0].uri); // Set video URI from selected media
                setVideoName(result.assets[0].fileName || 'Uploaded Video'); // Set file name if available
                Alert.alert('Video uploaded!', 'Video successfully selected and ready for processing.');
            } else {
                console.log('Video selection canceled.');
            }
        } catch (error) {
            console.log('Error picking video:', error);
            Alert.alert('Error', 'There was an issue selecting the video.');
        }
    };

    // Function to record a video using the camera
    const recordVideo = async () => {
        try {
            let result = await ImagePicker.launchCameraAsync({
                mediaTypes: ImagePicker.MediaTypeOptions.Videos,
                allowsEditing: true,
                quality: 1,
            });

            if (!result.canceled) {
                setVideo(result.assets[0].uri); // Set video URI from recorded media
                setVideoName(result.assets[0].fileName || 'Recorded Video'); // Set file name if available
                Alert.alert('Video recorded!', 'Video successfully recorded and ready for processing.');
            } else {
                console.log('Video recording canceled.');
            }
        } catch (error) {
            console.log('Error recording video:', error);
            Alert.alert('Error', 'There was an issue recording the video.');
        }
    };

    // Function to process the selected video by sending it to the server
    const processVideo = async () => {
        if (!video) {
            Alert.alert('No video selected', 'Please upload or record a video first.');
            return;
        }

        setProcessing(true); // Start loading spinner
        const formData = new FormData();
        formData.append('video', {
            uri: video,
            type: 'video/mp4',
            name: videoName || 'exercise_video.mp4',
        });

        try {
            console.log('Sending video to server...');
            const response = await axios.post('http://192.168.1.51:5000/plank-analyze', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });

            // Debugging: Log the full response
            console.log('Server response:', response);

            if (response.data) {
                console.log('Response Data:', response.data);  // Log the response data to debug
                setServerResponse(response.data); // Store the server response
            } else {
                console.error('No response data from the server');
                Alert.alert('Error', 'No response from the server.');
            }

            Alert.alert('Success', 'Video processed successfully.');
        } catch (error) {
            console.error('Error processing video:', error); // More detailed error logging
            Alert.alert('Error', `Error processing video: ${error.message}`);
        } finally {
            setProcessing(false); // Ensure spinner stops in both success and error cases
        }
    };

    // Function to fetch the report from the server
    const fetchReport = async () => {
        if (!serverResponse) {
            Alert.alert('Process Video First', 'Please process the video before fetching the report.');
            return;
        }

        try {
            const response = await axios.get('http://192.168.1.51:5000/plank-get-report');
            if (response.data) {
                setReportContent(response.data); // Store the report content
                setModalVisible(true); // Show the modal with report content
            } else {
                Alert.alert('Error', 'Failed to retrieve the report.');
            }
        } catch (error) {
            console.error('Error fetching report:', error);
            Alert.alert('Error', `Failed to fetch the report: ${error.message}`);
        }
    };


    return (
        <ScrollView contentContainerStyle={styles.scrollViewContent}>
            <View style={styles.container}>
                {/* Demo Video Heading */}
                <Text style={styles.demoVideoHeading}>Demo Video</Text>

                {/* Demo Video */}
                <Video
                    source={squatDemoVideo} // Display the local demo video
                    style={styles.video}
                    useNativeControls
                    resizeMode="contain"
                    isLooping
                />

                {/* Instructions to perform exercise */}
                <View style={styles.instructionsContainer}>
                    <Text style={styles.instructionsHeading}>INSTRUCTIONS</Text>
                    <Text style={styles.instructionsText}>
                        Stand with your feet shoulder-width apart and your arms stretched forward,
                        then lower your body until your thighs are parallel with the floor.
                    </Text>
                    <Text style={styles.instructionsText}>
                        Your knees should be extended in the same direction as your toes.
                        Return to the start position and do the next rep.
                    </Text>
                    <Text style={styles.instructionsText}>
                        This works the thighs, hips, buttocks, quads, hamstrings, and lower body.
                    </Text>
                </View>
                {/* Common Mistakes Section */}
                <View style={styles.instructionsContainer}>
                    <Text style={styles.commonMistakesHeading}>COMMON MISTAKES</Text>

                    <View style={styles.mistakeItem}>
                        <Text style={styles.mistakeNumber}>1</Text>
                        <View style={styles.mistakeTextContainer}>
                            <Text style={styles.mistakeTitle}>Allowing your knees to cave in</Text>
                            <Text style={styles.mistakeDescription}>
                                This can put unnecessary stress on your knee joints and increase the risk of injury.
                                Keep your knees in line with your toes. Also, do not let your knees go past your toes.
                            </Text>
                        </View>
                    </View>

                    <View style={styles.mistakeItem}>
                        <Text style={styles.mistakeNumber}>2</Text>
                        <View style={styles.mistakeTextContainer}>
                            <Text style={styles.mistakeTitle}>Hunching the back</Text>
                            <Text style={styles.mistakeDescription}>
                                Hunching your back can put undue stress on your spine and increase the risk of injury.
                                Keep your core engaged and your back straight.
                            </Text>
                        </View>
                    </View>

                    <View style={styles.mistakeItem}>
                        <Text style={styles.mistakeNumber}>3</Text>
                        <View style={styles.mistakeTextContainer}>
                            <Text style={styles.mistakeTitle}>Not going low enough</Text>
                            <Text style={styles.mistakeDescription}>
                                If you don’t squat low enough, you’re not engaging your glutes and hamstrings as effectively.
                                Aim to squat until your thighs are parallel to the ground.
                            </Text>
                        </View>
                    </View>

                    <View style={styles.mistakeItem}>
                        <Text style={styles.mistakeNumber}>4</Text>
                        <View style={styles.mistakeTextContainer}>
                            <Text style={styles.mistakeTitle}>Letting your heels come off the ground</Text>
                            <Text style={styles.mistakeDescription}>
                                Lifting your heels can shift your weight forward and put more stress on your knees.
                                Keep your feet flat on the ground.
                            </Text>
                        </View>
                    </View>
                </View>
                {/* Instructions to Upload Video */}
                <Text style={styles.instructions}>
                    Record or upload a video of yourself performing a Squat.
                    Once uploaded, click "Process Video" to receive feedback on your form.
                </Text>

                {/* Buttons for recording or uploading video */}
                <View style={styles.buttonContainer}>
                    {/* Upload Video Button */}
                    <TouchableOpacity style={[styles.button, styles.uploadButton]} onPress={pickVideo}>
                        <Ionicons name="cloud-upload-outline" size={20} color="#fff" style={styles.buttonIcon} />
                        <Text style={styles.buttonText}>Upload Video</Text>
                    </TouchableOpacity>

                    {/* Record Video Button */}
                    <TouchableOpacity style={[styles.button, styles.recordButton]} onPress={recordVideo}>
                        <Ionicons name="camera-outline" size={20} color="#fff" style={styles.buttonIcon} />
                        <Text style={styles.buttonText}>Record Video</Text>
                    </TouchableOpacity>
                </View>

                {/* Process Video Button */}
                <TouchableOpacity style={[styles.button, styles.processButton]} onPress={processVideo} disabled={processing}>
                    {processing ? (
                        <ActivityIndicator size="small" color="#fff" />
                    ) : (
                        <>
                            <Ionicons name="checkmark-circle-outline" size={20} color="#fff" style={styles.buttonIcon} />
                            <Text style={styles.buttonText}>Process Video</Text>
                        </>
                    )}
                </TouchableOpacity>

                {/* Display selected video */}
                {video && (
                    <View style={styles.videoContainer}>
                        <Text style={styles.videoName}>{videoName}</Text>
                        <Video
                            source={{ uri: video }}
                            style={styles.selectedVideo}
                            useNativeControls
                            resizeMode="contain"
                        />
                    </View>
                )}

                {/* Display server response (labeled frames) if available */}
                {serverResponse && (
                    <View>
                        <Text style={styles.responseTitle}>Processed Video:</Text>
                        <Video
                            source={{ uri: serverResponse.processed_video_url }}
                            style={styles.selectedVideo}
                            useNativeControls
                            resizeMode="contain"
                        />
                    </View>
                )}

                {/* Show Report Button */}
                {serverResponse && (
                    <TouchableOpacity style={[styles.button, styles.reportButton]} onPress={fetchReport}>
                        <Ionicons name="document-text-outline" size={20} color="#fff" style={styles.buttonIcon} />
                        <Text style={styles.buttonText}>Show Report</Text>
                    </TouchableOpacity>
                )}

                {/* Modal for displaying the report */}
                <Modal visible={isModalVisible} animationType="slide">
                    <RNScrollView contentContainerStyle={{ padding: 20 }}>
                        <Text style={{ fontSize: 16, fontWeight: 'bold' }}>Plank Feedback Report</Text>
                        <Text style={{ marginTop: 20 }}>{reportContent}</Text>
                        <TouchableOpacity
                            style={[styles.button, { marginTop: 20 }]}
                            onPress={() => setModalVisible(false)}>
                            <Text style={styles.buttonText}>Close</Text>
                        </TouchableOpacity>
                    </RNScrollView>
                </Modal>
            </View>
        </ScrollView>
    );
}

const styles = StyleSheet.create({
    scrollViewContent: {
        flexGrow: 1,
        justifyContent: 'center',
    },
    container: {
        flex: 1,
        backgroundColor: '#fff',
        padding: 20,
    },
    demoVideoHeading: {
        fontSize: 20,
        fontWeight: 'bold',
        textAlign: 'left',
        marginBottom: 15,
        color: '#007AFF',
    },
    video: {
        width: '100%',
        height: 200,
        marginBottom: 20,
        backgroundColor: '#000',
        borderRadius: 5,
    },
    commonMistakesHeading: {
        fontSize: 18,
        fontWeight: 'bold',
        color: '#007AFF',
        marginBottom: 15,
        textAlign: 'left',
    },
    mistakeItem: {
        flexDirection: 'row',
        alignItems: 'flex-start',
        marginBottom: 15,
    },
    mistakeNumber: {
        fontSize: 16,
        fontWeight: 'bold',
        color: '#007AFF',
        width: 20,
        textAlign: 'center',
        marginRight: 10,
    },
    mistakeTextContainer: {
        flex: 1,
    },
    mistakeTitle: {
        fontSize: 16,
        fontWeight: 'bold',
        color: '#333',
        marginBottom: 5,
    },
    mistakeDescription: {
        fontSize: 14,
        color: '#555',
        lineHeight: 20,
    },
    instructions: {
        fontSize: 16,
        marginTop: 15,
        marginBottom: 20,
        textAlign: 'center',
        lineHeight: 24,
        color: '#555',
        backgroundColor: '#f9f9f9',
        padding: 15,
        borderRadius: 10,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 3,
        elevation: 1,
    },

    instructionsContainer: {
        marginTop: 20,
        marginHorizontal: 15,
        padding: 10,
        backgroundColor: '#fff',
        borderRadius: 10,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 4,
        elevation: 2,
    },
    instructionsHeading: {
        fontSize: 18,
        fontWeight: 'bold',
        color: '#007AFF', // Blue heading
        marginBottom: 10,
    },
    instructionsText: {
        fontSize: 16,
        color: '#333',
        lineHeight: 24,
        marginBottom: 10,
    },

    buttonContainer: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 20,
    },
    button: {
        flexDirection: 'row', // To align icon and text in one line
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#000', // Black background
        paddingVertical: 15,
        borderRadius: 10,
        flex: 1,
        marginHorizontal: 5,
    },
    buttonText: {
        color: '#fff', // White text
        fontWeight: 'bold',
        marginLeft: 5, // Space between icon and text
        fontSize: 16,
    },
    uploadButton: {
        marginRight: 10, // Space for alignment
    },
    recordButton: {
        marginLeft: 10, // Space for alignment
    },
    processButton: {
        marginTop: 5,
        alignSelf: 'center', // Center the button
        width: '60%', // Adjust width for central alignment
    },
    reportButton: {
        // Black background color
        padding: 15,
        marginVertical: 10,  // Add vertical margin for spacing between the processed video and the report button
    },
    buttonIcon: {
        marginRight: 5, // Space between icon and text
    },
    videoContainer: {
        marginTop: 20,
    },
    videoName: {
        fontSize: 16,
        textAlign: 'center',
        marginBottom: 10,
    },
    selectedVideo: {
        width: '100%',
        height: 200,
        backgroundColor: '#000',
        borderRadius: 5,
    },
    responseTitle: {
        fontSize: 18,
        fontWeight: 'bold',
        marginTop: 20,
        marginBottom: 10,
        textAlign: 'center',
    },
});