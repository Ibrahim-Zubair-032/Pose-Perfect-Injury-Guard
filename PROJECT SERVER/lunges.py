import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify
import traceback
import os
from flask import Flask, request, jsonify, send_from_directory, send_file  
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from pathlib import Path
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings(
    "ignore", 
    category=UserWarning, 
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated\."
)
warnings.filterwarnings("ignore", category=UserWarning)

source = Path(__file__).resolve().parent

model_path = source / "Model/Lunges"

# Load models
with open(model_path / "stage_SVC_model.pkl", "rb") as f:
    stage_sklearn_model = pickle.load(f)

with open(model_path / "err_LR_model.pkl", "rb") as f:
    err_sklearn_model = pickle.load(f)

# Load input scaler
with open(model_path / "input_scaler.pkl", "rb") as f:
    input_scaler = pickle.load(f)

knee_over_toe=0
left_good_counter = 0
left_not_low_enough_counter = 0
right_good_counter = 0
right_not_low_enough_counter = 0
updated_previous_knee_angle=0
previous_left_knee_angle = None
previous_right_knee_angle = None
current_stage = " "
# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Define constants
ANGLE_THRESHOLDS = [60, 135]
# Define the processed videos folder path
REPORT_FOLDER = str(source/'Report')
PROCESSED_VIDEOS_FOLDER = str(source/'processed_videos')
# Ensure that processed videos folder exists
if not os.path.exists(PROCESSED_VIDEOS_FOLDER):
    os.makedirs(PROCESSED_VIDEOS_FOLDER)

# IMPORTANT_LMS definition
IMPORTANT_LMS = [
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
    "RIGHT_FOOT_INDEX",
]

HEADERS = ["label"]
for lm in IMPORTANT_LMS:
    HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]


def reset_globals():
    global right_not_low_enough_counter  # Declare the variable as global
    global right_good_counter
    global left_good_counter
    global left_not_low_enough_counter
    global knee_over_toe
    global current_stage
    global previous_left_knee_angle
    global previous_right_knee_angle

    # Set all counters and angles to 0, and current_stage to an empty string
    right_not_low_enough_counter = 0
    right_good_counter = 0
    left_good_counter = 0
    left_not_low_enough_counter = 0
    knee_over_toe = 0
    current_stage = ""  # Empty string for current_stage
    previous_left_knee_angle = 0
    previous_right_knee_angle = 0

def extract_important_keypoints(results):
    ''' Extract important keypoints from mediapipe pose detection '''
    landmarks = results.pose_landmarks.landmark
    data = []
    for lm in IMPORTANT_LMS:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    
    return np.array(data).flatten().tolist()

def rescale_frame(frame, percent=50):
    ''' Rescale a frame to a certain percentage compared to its original frame '''
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def calculate_angle(point1, point2, point3):
    ''' Calculate the angle between 3 points in degrees '''
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)

    angleInRad = np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) - np.arctan2(point1[1] - point2[1], point1[0] - point2[0])
    angleInDeg = np.abs(angleInRad * 180.0 / np.pi)
    angleInDeg = angleInDeg if angleInDeg <= 180 else 360 - angleInDeg
    return angleInDeg
def update_right_knee_feedback(knee_coords, hip_coords, ankle_coords, angle_thresholds, current_stage):
    global previous_right_knee_angle

    # Calculate the angle
    knee_angle = calculate_angle(hip_coords, knee_coords, ankle_coords)
    # Initialize previous_knee_angle if it's the first call
    #print("Right Knee Angle : ",knee_angle) 
    #print("Current Stage : ",current_stage)
    if previous_right_knee_angle is None:
        previous_right_knee_angle = knee_angle  # Set it to the current knee angle for the first call
    # Default feedback in case no conditions match
    feedback = "UNKNOWN"
    
    # Determine feedback and increment the corresponding counters
    if current_stage == "start":
        # No feedback or counter increment in stand stage
        feedback = "STAND"
        return feedback,  0, 0

    elif current_stage == "mid":
    # Define the angle range for the halfway-down stage (adjust as needed)
      #print("Inner Knee Angle : ",knee_angle)
      if 100 < knee_angle <= 150:

        #print("MID STAGE EXECUTED")
        if previous_right_knee_angle <= 135 and knee_angle > 135:
            # previous_knee_angle = float(previous_knee_angle)
            feedback = "NOT LOW ENOUGH"
            # Update the previous knee angle for future checks
            previous_right_knee_angle = knee_angle
            # Return feedback along with counters (if needed)
            return feedback,  0, 1  # Return updated feedback
    # Update the previous knee angle to continue tracking in subsequent frames
      previous_right_knee_angle = knee_angle  # Keep tracking the previous knee angle

    elif current_stage == "down":

        if 75 <= knee_angle <= 100:  # Check for "GOOD" condition
            feedback = "GOOD"
            return feedback,  1, 0  # Increment "GOOD" by 1

    # Return default "UNKNOWN" feedback if no conditions match
    return feedback,  0, 0
def update_left_knee_feedback(knee_coords, hip_coords, ankle_coords, angle_thresholds, current_stage):
    global previous_left_knee_angle

    # Calculate the angle
    knee_angle = calculate_angle(hip_coords, knee_coords, ankle_coords)
    
    # Initialize previous_knee_angle if it's the first call 
    if previous_left_knee_angle is None:
        previous_left_knee_angle = knee_angle  # Set it to the current knee angle for the first call
    # Default feedback in case no conditions match
    feedback = "UNKNOWN"
    
    # Determine feedback and increment the corresponding counters
    if current_stage == "start":
        # No feedback or counter increment in stand stage
        feedback = "STAND"
        return feedback,  0, 0

    elif current_stage == "mid":
    # Define the angle range for the halfway-down stage (adjust as needed)
      if 100 < knee_angle <= 150:
        # Feedback when the person is not going low enough (not achieving the full lunge position)
        if previous_left_knee_angle <= 135 and knee_angle > 135:
            # previous_knee_angle = float(previous_knee_angle)
            feedback = "NOT LOW ENOUGH"
            # Update the previous knee angle for future checks
            previous_left_knee_angle = knee_angle
            # Return feedback along with counters (if needed)
            return feedback,  0, 1  # Return updated feedback
    # Update the previous knee angle to continue tracking in subsequent frames
      previous_left_knee_angle = knee_angle  # Keep tracking the previous knee angle

    elif current_stage == "down":
        if 75 <= knee_angle <= 100:  # Check for "GOOD" condition
            feedback = "GOOD"
            return feedback, 1, 0  # Increment "GOOD" by 1

    # Return default "UNKNOWN" feedback if no conditions match
    return feedback, 0, 0
    

def generate_lunge_injury_estimation_report(knee_over_toe):
    # Round the counter values for lunges
    left_good_counter_rounded = round(left_good_counter)
    left_not_low_enough_counter_rounded = round(left_not_low_enough_counter)
    right_good_counter_rounded = round(right_good_counter)
    right_not_low_enough_counter_rounded = round(right_not_low_enough_counter)

    # Calculate total evaluations
    total_left_knee_evaluations = left_good_counter_rounded + left_not_low_enough_counter_rounded
    total_right_knee_evaluations = right_good_counter_rounded + right_not_low_enough_counter_rounded

    # Calculate percentages for left and right knee
    left_good_percentage = (left_good_counter_rounded / total_left_knee_evaluations) * 100 if total_left_knee_evaluations > 0 else 0
    left_not_low_enough_percentage = (left_not_low_enough_counter_rounded / total_left_knee_evaluations) * 100 if total_left_knee_evaluations > 0 else 0

    right_good_percentage = (right_good_counter_rounded / total_right_knee_evaluations) * 100 if total_right_knee_evaluations > 0 else 0
    right_not_low_enough_percentage = (right_not_low_enough_counter_rounded / total_right_knee_evaluations) * 100 if total_right_knee_evaluations > 0 else 0

    # Knee Over Toe percentage
    knee_over_toe_percentage = (knee_over_toe / (left_good_counter_rounded + right_good_counter_rounded)) * 100 if (left_good_counter_rounded + right_good_counter_rounded) > 0 else 0

    # Start building the report content
    report_content = "Lunge Injury Estimation Report\n"
    report_content += "=" * 50 + "\n\n"

    # Left Knee Feedback
    report_content += "LEFT KNEE FEEDBACK\n"
    report_content += "-" * 50 + "\n"
    if left_good_percentage > 0:
        report_content += f"✔ GOOD FORM: {round(left_good_percentage)}%\n"
        report_content += "  Excellent alignment and stability\n"
        report_content += "  Maintain proper knee alignment.\n\n"
    if left_not_low_enough_percentage > 0:
        report_content += f"❌ NOT LOW ENOUGH: {round(left_not_low_enough_percentage)}%\n"
        report_content += "  - Limited activation of glutes and hamstrings\n"
        report_content += "  - Less effective workout for lower body\n"
        report_content += "  ✔ Lower your back knee just above the ground.\n\n"

    # Right Knee Feedback
    report_content += "RIGHT KNEE FEEDBACK\n"
    report_content += "-" * 50 + "\n"
    if right_good_percentage > 0:
        report_content += f"✔ GOOD FORM: {round(right_good_percentage)}%\n"
        report_content += "  Great alignment and balance\n"
        report_content += "  Keep your knee aligned with your toes.\n\n"
    if right_not_low_enough_percentage > 0:
        report_content += f"❌ NOT LOW ENOUGH: {round(right_not_low_enough_percentage)}%\n"
        report_content += "  - Poor engagement of quads and glutes\n"
        report_content += "  - Limited range of motion\n"
        report_content += "  ✔ Lower your knee until both knees form 90-degree angles.\n\n"

    # Knee Over Toe Feedback
    report_content += "KNEE OVER TOE FEEDBACK\n"
    report_content += "-" * 50 + "\n"
    if knee_over_toe_percentage > 0:
        report_content += f"❌ KNEE OVER TOE: {round(knee_over_toe_percentage)}%\n"
        report_content += "  - Risk of knee injuries due to excessive forward movement\n"
        report_content += "  - Increased stress on the knee joint\n"
        report_content += "  ✔ Keep your knees behind your toes.\n\n"

    # Ensure the report folder exists
    if not os.path.exists(REPORT_FOLDER):
        os.makedirs(REPORT_FOLDER)

    report_filename = "lunge_knee_feedback_report.txt"
    report_path = os.path.join(REPORT_FOLDER, report_filename)

    # Save the report to the specified folder
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report_content)

    print(f"Lunge Injury Feedback Report generated successfully at: {report_path}")

def analyze_knee_angle(
    mp_results, stage: str, angle_thresholds: list
):
    global left_good_counter, left_not_low_enough_counter
    global right_good_counter, right_not_low_enough_counter
    global current_stage 
    current_stage = stage
    results = {
        "error": None,
        "not_low_enough": False,
        "right": {"error": None, "angle": None},
        "left": {"error": None, "angle": None},
    }
    landmarks = mp_results.pose_landmarks.landmark

    # Calculate right knee angle
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    results["right"]["angle"] = calculate_angle(right_hip, right_knee, right_ankle)

    # Calculate left knee angle
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    results["left"]["angle"] = calculate_angle(left_hip, left_knee, left_ankle)

    # Get feedback and update counters for the right knee
    right_feedback, right_good_increment, right_not_low_enough_increment = update_right_knee_feedback(
        right_knee, right_hip, right_ankle, angle_thresholds, current_stage
    )

    right_good_counter += right_good_increment
    right_not_low_enough_counter += right_not_low_enough_increment
    #print("Right GOOD : ",right_good_counter)
    #print("Right NOT LOW ENOUGH : ",right_not_low_enough_counter)
    # Get feedback and update counters for the left knee
    left_feedback, left_good_increment, left_not_low_enough_increment = update_left_knee_feedback(
        left_knee, left_hip, left_ankle, angle_thresholds, current_stage
    )
    #print("Left",left_not_low_enough_counter)
    left_good_counter += left_good_increment
    left_not_low_enough_counter += left_not_low_enough_increment
    #print("Left Good: ",left_good_counter)
    #print("Left NOT LOW ENOUGH : ",left_not_low_enough_counter)
    return results



def analyze_lunge():
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video part"}), 400
        
        video_file = request.files['video']
        
        # Create a temporary file to store the uploaded video
        temp_video_path = 'temp_video.mp4'
        video_file.save(temp_video_path)

        # Generate a filename
        processed_video_name = "processed_lunges_video.mp4"
        processed_video_path = os.path.join(PROCESSED_VIDEOS_FOLDER, processed_video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Initialize video capture
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            return jsonify({"error": "Error opening video file"}), 500

        # Get the width and height of the video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(processed_video_path, fourcc, 30.0, (width, height))

        current_stage = ""
        counter = 0
        prediction_probability_threshold = 0.8

        results_list = []
        frame_count = 0
        knee_over_toe = False
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                    break

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)

                if not results.pose_landmarks:
                    continue

                image_rgb.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                row = extract_important_keypoints(results)
                X = pd.DataFrame([row], columns=HEADERS[1:])
                X = pd.DataFrame(input_scaler.transform(X))

                stage_predicted_class = stage_sklearn_model.predict(X)[0]
                stage_prediction_probabilities = stage_sklearn_model.predict_proba(X)[0]
                stage_prediction_probability = round(stage_prediction_probabilities.max(), 2)

                if stage_predicted_class == "I" and stage_prediction_probability >= prediction_probability_threshold:
                    current_stage = "start"
                elif stage_predicted_class == "M" and stage_prediction_probability >= prediction_probability_threshold:
                    current_stage = "mid"
                elif stage_predicted_class == "D" and stage_prediction_probability >= prediction_probability_threshold:
                    if (current_stage == "mid" or current_stage == "start"):
                        current_stage = "down"
                        counter += 1

                analyze_knee_angle(mp_results=results, stage=current_stage, angle_thresholds=ANGLE_THRESHOLDS)
                 # Knee over toe 
                err_predicted_class = err_prediction_probabilities = err_prediction_probability = None
                if current_stage == "down":
                    # Make prediction
                    err_predicted_class = err_sklearn_model.predict(X)[0]
                    err_prediction_probabilities = err_sklearn_model.predict_proba(X)[0]
                    err_prediction_probability = round(err_prediction_probabilities[err_prediction_probabilities.argmax()], 2)
                    # Check predicted class
                    if err_predicted_class == 0:
                        knee_over_toe += 1
              
                cv2.rectangle(image, (0, 0), (800, 60), (245, 117, 16), -1)
                cv2.putText(image, "STAGE", (460, 12), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, current_stage, (470, 30), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, "COUNTER", (360, 12), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), (380, 30), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, "KNEE FEEDBACK COUNTERS", (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                
                # Right knee counters
                cv2.putText(image, "Right Knee:", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, "GOOD: " + str(right_good_counter), (40, 90), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(image, "NOT LOW ENOUGH: " + str(right_not_low_enough_counter), (40, 110), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)
                
                # Left knee counters
                cv2.putText(image, "Left Knee:", (20, 140), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, "GOOD: " + str(left_good_counter), (40, 180), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(image, "NOT LOW ENOUGH: " + str(left_not_low_enough_counter), (40, 200), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)
                cv2.putText(image, "Knee Over Toe", (20, 240), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, str(knee_over_toe), (40, 270), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
          
                out.write(image)
                frame_count += 1

        cap.release()
        
        out.release()
        generate_lunge_injury_estimation_report(knee_over_toe)

        reset_globals()
        # Return URL of processed video
        return jsonify({"processed_video_url": f"http://{request.host}/processed_videos/{processed_video_name}"}), 200


    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def get_report():
    # Find the most recent report file
    report_files = [f for f in os.listdir(REPORT_FOLDER) if f == 'lunge_knee_feedback_report.txt']
    if not report_files:
        return jsonify({'error': 'No reports found'}), 404


    report_file = report_files[0]
    report_path = os.path.join(REPORT_FOLDER, report_file)

    # Log the report file path
    print(f"Latest report found: {report_path}")
    # Return the report file
    return send_file(report_path, as_attachment=False, mimetype='text/plain')

