import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Initialize MediaPipe Pose and Drawing modules
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Function to export landmarks to CSV
def export_landmark_to_csv(dataset_path, results, label, right_knee_angle, left_knee_angle):
    # Create a list to hold the data for all specified landmarks
    data = []

    # Define the landmark names to extract
    landmark_names = [
        "NOSE",
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_KNEE",
        "RIGHT_KNEE",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
        "LEFT_HEEL",
        "RIGHT_HEEL",
        "LEFT_FOOT_INDEX",
        "RIGHT_FOOT_INDEX"
    ]

    # Extract landmarks if available
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        row = {'label': label}

        for name in landmark_names:
            # Get the index of the landmark
            index = mp_pose.PoseLandmark[name].value
            
            # Get the coordinates and visibility
            x = landmarks[index].x
            y = landmarks[index].y
            z = landmarks[index].z
            visibility = landmarks[index].visibility
            
            # Add the data to the row dictionary
            row[f"{name.lower()}_x"] = x
            row[f"{name.lower()}_y"] = y
            row[f"{name.lower()}_z"] = z
            row[f"{name.lower()}_v"] = visibility
        
        # Add angles to the row
        row['left_knee_angle'] = left_knee_angle
        row['right_knee_angle'] = right_knee_angle

        # Append the row data to the list
        data.append(row)

    # Convert to DataFrame and export to CSV
    df = pd.DataFrame(data)
    print("Exporting landmarks to CSV...")
    df.to_csv(dataset_path, mode='a', header=not pd.io.common.file_exists(dataset_path), index=False)
    print("Landmarks exported.")

# Function to rescale the frame
def rescale_frame(frame, percent=100):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

DATASET_PATH = "stage.train.csv"

cap = cv2.VideoCapture("D:\\Desktop\\Lunges FYP Model\\6525491-hd_1920_1080_25fps.mp4")
save_counts = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.8) as pose:
    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break

        # Reduce size of a frame
        image = rescale_frame(image, 50)
        image = cv2.flip(image, 1)

        video_dimensions = [image.shape[1], image.shape[0]]

        # Recolor image from BGR to RGB for MediaPipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        if not results.pose_landmarks: 
            continue

        landmarks = results.pose_landmarks.landmark

        # Calculate right knee angle
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

        # Calculate left knee angle
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

        # Recolor image from RGB back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                   mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=4), 
                                   mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Display the saved count
        cv2.putText(image, f"Saved: {save_counts}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)

        # Visualize angles
        cv2.putText(image, str(int(right_knee_angle)), tuple(np.multiply(right_knee, video_dimensions).astype(int)), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, str(int(left_knee_angle)), tuple(np.multiply(left_knee, video_dimensions).astype(int)), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("CV2", image)

        # Wait for a key press to proceed to the next frame
        k = cv2.waitKey(0) & 0xFF

        # * Press I to save as INIT stage
        if k == ord('i'): 
            export_landmark_to_csv(DATASET_PATH, results, "I", right_knee_angle, left_knee_angle)
            save_counts += 1
        # * Press M to save as MID stage
        elif k == ord("m"):
            export_landmark_to_csv(DATASET_PATH, results, "M", right_knee_angle, left_knee_angle)
            save_counts += 1
        # * Press D to save as DOWN stage
        elif k == ord("d"):
            export_landmark_to_csv(DATASET_PATH, results, "D", right_knee_angle, left_knee_angle)
            save_counts += 1
        # * Press S to skip the frame without saving
        elif k == ord("s"):
            print("Frame skipped")
        # Press q to stop
        elif k == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # (Optional) Fix bugs cannot close windows in MacOS
    for i in range(1, 5):
        cv2.waitKey(1)
