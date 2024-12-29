from flask import Flask, request, jsonify, send_from_directory, send_file
import cv2
import os
import mediapipe as mp
import pandas as pd
import traceback
import math
from datetime import datetime
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="X has feature names, but LogisticRegression was fitted without feature names")
source = Path(__file__).resolve().parent
# Initialize counters for foot placement evaluations
correct_foot_placement_count = 0
too_tight_foot_placement_count = 0
too_wide_foot_placement_count = 0

# Initialize counters for knee placement evaluations
correct_knee_placement_count = 0
too_tight_knee_placement_count = 0
too_wide_knee_placement_count = 0

# Mediapipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

REPORT_FOLDER = str(source/'Report')
PROCESSED_VIDEOS_FOLDER = str(source/'processed_videos')

# Global variables for squat analysis
PREDICTION_PROB_THRESHOLD = 0.6
VISIBILITY_THRESHOLD = 0.6
FOOT_SHOULDER_RATIO_THRESHOLDS = [1.2, 2.8]
KNEE_FOOT_RATIO_THRESHOLDS = {
    "up": [0.5, 1.0],
    "middle": [0.7, 1.0],
    "down": [0.7, 1.1],
}

# Load the logistic regression model for squat classification
with open(str(source/'Model/Squat/logistic_regression_model.pkl') ,"rb") as f:
    count_model = pickle.load(f)
# Load input scaler
with open(str(source/'Model/Squat/input_scaler.pkl'), "rb") as f:
    input_scaler = pickle.load(f)


# List of important landmarks (using the official MediaPipe PoseLandmark enum names)
IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE"
]

# Create headers for the CSV or DataFrame (x, y, z, visibility for each landmark)
headers = ["label"]  # Label column
for lm in IMPORTANT_LMS:
    headers += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

# Function to extract keypoints from mediapipe results
def extract_important_keypoints(results) -> list:
    landmarks = results.pose_landmarks.landmark  # Get the list of landmarks
    keypoints = []
    for lm in IMPORTANT_LMS:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        keypoints.extend([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    return keypoints  # Return the flattened list of x, y, z, v for each landmark

# Function to rescale a frame to a percentage of its original size
def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent / 100)  # Rescale width
    height = int(frame.shape[0] * percent / 100)  # Rescale height
    dim = (width, height)  # New dimensions
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)  # Resize the frame

def calculate_distance(pointX, pointY) -> float:
    x1, y1 = pointX
    x2, y2 = pointY
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def generate_injury_estimation_report():
    # Round the counter values before calculating percentages
    correct_foot_placement_count_rounded = round(correct_foot_placement_count)
    too_tight_foot_placement_count_rounded = round(too_tight_foot_placement_count)
    too_wide_foot_placement_count_rounded = round(too_wide_foot_placement_count)
    
    correct_knee_placement_count_rounded = round(correct_knee_placement_count)
    too_tight_knee_placement_count_rounded = round(too_tight_knee_placement_count)
    too_wide_knee_placement_count_rounded = round(too_wide_knee_placement_count)

    # Foot Placement percentages
    total_foot_evaluations = correct_foot_placement_count_rounded + too_tight_foot_placement_count_rounded + too_wide_foot_placement_count_rounded
    total_knee_evaluations = correct_knee_placement_count_rounded + too_tight_knee_placement_count_rounded + too_wide_knee_placement_count_rounded

    # Calculate percentages for foot placement
    correct_foot_percentage = (correct_foot_placement_count_rounded / total_foot_evaluations) * 100 if total_foot_evaluations > 0 else 0
    too_tight_foot_percentage = (too_tight_foot_placement_count_rounded / total_foot_evaluations) * 100 if total_foot_evaluations > 0 else 0
    too_wide_foot_percentage = (too_wide_foot_placement_count_rounded / total_foot_evaluations) * 100 if total_foot_evaluations > 0 else 0
    
    # Calculate percentages for knee placement
    correct_knee_percentage = (correct_knee_placement_count_rounded / total_knee_evaluations) * 100 if total_knee_evaluations > 0 else 0
    too_tight_knee_percentage = (too_tight_knee_placement_count_rounded / total_knee_evaluations) * 100 if total_knee_evaluations > 0 else 0
    too_wide_knee_percentage = (too_wide_knee_placement_count_rounded / total_knee_evaluations) * 100 if total_knee_evaluations > 0 else 0

    # Start building the report content
    report_content = "Squat Injury Estimation Report\n"
    report_content += "=" * 50 + "\n"

    # FOOT PLACEMENT FEEDBACK
    report_content += "FOOT PLACEMENT FEEDBACK\n"
    report_content += "-" * 50 + "\n"
    
    if correct_foot_percentage > 0:
        report_content += f"CORRECT: {round(correct_foot_percentage)}%\n"
        report_content += "✔ Excellent stability and balance\n"
        report_content += "✔ Efficient push-off for squats\n\n"
    
    if too_tight_foot_percentage > 0:
        report_content += f"TOO TIGHT: {round(too_tight_foot_percentage)}%\n"
        report_content += "❗ Risk of ankle strain due to tight stance\n"
        report_content += "❗ Limited stability and balance\n"
        report_content += "✔ Widen your stance slightly\n\n"
    
    if too_wide_foot_percentage > 0:
        report_content += f"TOO WIDE: {round(too_wide_foot_percentage)}%\n"
        report_content += "❗ Potential hip discomfort due to wide stance\n"
        report_content += "❗ Less efficient push-off for squats\n"
        report_content += "✔ Narrow your stance\n\n"

    # KNEE PLACEMENT FEEDBACK
    report_content += "KNEE PLACEMENT FEEDBACK\n"
    report_content += "-" * 50 + "\n"
    
    if correct_knee_percentage > 0:
        report_content += f"CORRECT: {round(correct_knee_percentage)}%\n"
        report_content += "✔ Excellent knee alignment\n"
        report_content += "✔ Reduced injury risk\n"
        report_content += "✔ Maintain proper knee tracking\n\n"
    
    if too_tight_knee_percentage > 0:
        report_content += f"TOO TIGHT: {round(too_tight_knee_percentage)}%\n"
        report_content += "❗ Increased risk of knee strain\n"
        report_content += "❗ Poor knee tracking\n"
        report_content += "✔ Focus on pushing your knees outward\n\n"
    
    if too_wide_knee_percentage > 0:
        report_content += f"TOO WIDE: {round(too_wide_knee_percentage)}%\n"
        report_content += "❗ Potential knee valgus (knees collapsing inward)\n"
        report_content += "❗ Possible ligament strain\n"
        report_content += "✔ Keep knees aligned with toes\n\n"

        # Ensure the report folder exists
    if not os.path.exists(REPORT_FOLDER):
        os.makedirs(REPORT_FOLDER)

    report_filename = "squats_feedback_report.txt"
    report_path = os.path.join(REPORT_FOLDER, report_filename)

    # Save the report to the specified folder
    with open(report_path, "w", encoding="utf-8") as report_file:
      report_file.write(report_content)

    print("Squat Depth Feedback Report generated successfully!")
def analyze_foot_knee_placement(results, stage: str, foot_shoulder_ratio_thresholds: list, knee_foot_ratio_thresholds: dict, visibility_threshold: int) -> dict:
    '''
    Calculate the ratio between the foot and shoulder for FOOT PLACEMENT analysis
    
    Calculate the ratio between the knee and foot for KNEE PLACEMENT analysis

    Return result explanation:
        -1: Unknown result due to poor visibility
        0: Correct knee placement
        1: Placement too tight
        2: Placement too wide
    '''
    analyzed_results = {
        "foot_placement": -1,
        "knee_placement": -1,
    }

    landmarks = results.pose_landmarks.landmark

    # * Visibility check of important landmarks for foot placement analysis
    left_foot_index_vis = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].visibility
    right_foot_index_vis = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].visibility

    left_knee_vis = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
    right_knee_vis = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility

    # If visibility of any keypoints is low cancel the analysis
    if (left_foot_index_vis < visibility_threshold or right_foot_index_vis < visibility_threshold or left_knee_vis < visibility_threshold or right_knee_vis < visibility_threshold):
        return analyzed_results
    
    # * Calculate shoulder width
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    shoulder_width = calculate_distance(left_shoulder, right_shoulder)

    # * Calculate 2-foot width
    left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
    right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
    foot_width = calculate_distance(left_foot_index, right_foot_index)

    # * Calculate foot and shoulder ratio
    foot_shoulder_ratio = round(foot_width / shoulder_width, 1)

    # * Analyze FOOT PLACEMENT
    min_ratio_foot_shoulder, max_ratio_foot_shoulder = foot_shoulder_ratio_thresholds
    if min_ratio_foot_shoulder <= foot_shoulder_ratio <= max_ratio_foot_shoulder:
        analyzed_results["foot_placement"] = 0
    elif foot_shoulder_ratio < min_ratio_foot_shoulder:
        analyzed_results["foot_placement"] = 1
    elif foot_shoulder_ratio > max_ratio_foot_shoulder:
        analyzed_results["foot_placement"] = 2
    
    # * Visibility check of important landmarks for knee placement analysis
    left_knee_vis = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
    right_knee_vis = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility

    # If visibility of any keypoints is low cancel the analysis
    if (left_knee_vis < visibility_threshold or right_knee_vis < visibility_threshold):
        print("Cannot see foot")
        return analyzed_results

    # * Calculate 2 knee width
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    knee_width = calculate_distance(left_knee, right_knee)

    # * Calculate foot and shoulder ratio
    knee_foot_ratio = round(knee_width / foot_width, 1)

    # * Analyze KNEE placement
    up_min_ratio_knee_foot, up_max_ratio_knee_foot = knee_foot_ratio_thresholds.get("up")
    middle_min_ratio_knee_foot, middle_max_ratio_knee_foot = knee_foot_ratio_thresholds.get("middle")
    down_min_ratio_knee_foot, down_max_ratio_knee_foot = knee_foot_ratio_thresholds.get("down")
    if stage == "up":
        if up_min_ratio_knee_foot <= knee_foot_ratio <= up_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 0
        elif knee_foot_ratio < up_min_ratio_knee_foot:
            analyzed_results["knee_placement"] = 1
        elif knee_foot_ratio > up_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 2
    elif stage == "middle":
        if middle_min_ratio_knee_foot <= knee_foot_ratio <= middle_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 0
        elif knee_foot_ratio < middle_min_ratio_knee_foot:
            analyzed_results["knee_placement"] = 1
        elif knee_foot_ratio > middle_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 2
    elif stage == "down":
        if down_min_ratio_knee_foot <= knee_foot_ratio <= down_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 0
        elif knee_foot_ratio < down_min_ratio_knee_foot:
            analyzed_results["knee_placement"] = 1
        elif knee_foot_ratio > down_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 2
    

    # print("Function Executed")
    
    return analyzed_results

def analyze_squat():
    global correct_foot_placement_count, too_tight_foot_placement_count, too_wide_foot_placement_count
    global correct_knee_placement_count, too_tight_knee_placement_count, too_wide_knee_placement_count
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video part"}), 400
        
        video_file = request.files['video']
        temp_video_path = 'temp_video.mp4'
        video_file.save(temp_video_path)

        PROCESSED_VIDEOS_FOLDER = 'processed_videos'
        os.makedirs(PROCESSED_VIDEOS_FOLDER, exist_ok=True)


        processed_video_name = "processed_squat.mp4"
        processed_video_path = os.path.join(PROCESSED_VIDEOS_FOLDER, processed_video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        cap = cv2.VideoCapture(temp_video_path)

        #cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            return jsonify({"error": "Error opening video file"}), 500

        # Get the width and height of the video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

        counter = 0
        current_stage = ""

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)

                if not results.pose_landmarks:
                    continue

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))

                try:
                    row = extract_important_keypoints(results)
                    X = pd.DataFrame([row], columns=headers[1:])
                    predicted_class = count_model.predict(X)[0]
                    predicted_class = "down" if predicted_class == 0 else "up"
                    prediction_probability = round(count_model.predict_proba(X)[0].max(), 2)

                    if predicted_class == "down" and prediction_probability >= PREDICTION_PROB_THRESHOLD:
                        current_stage = "down"
                    elif current_stage == "down" and predicted_class == "up" and prediction_probability >= PREDICTION_PROB_THRESHOLD:
                        current_stage = "up"
                        counter += 1

                    # Analyze squat pose
                    analyzed_results = analyze_foot_knee_placement(results=results, stage=current_stage, foot_shoulder_ratio_thresholds=FOOT_SHOULDER_RATIO_THRESHOLDS, knee_foot_ratio_thresholds=KNEE_FOOT_RATIO_THRESHOLDS, visibility_threshold=VISIBILITY_THRESHOLD)

                    foot_placement_evaluation = analyzed_results["foot_placement"]
                    knee_placement_evaluation = analyzed_results["knee_placement"]

                    # * Evaluate FOOT PLACEMENT error
                    if foot_placement_evaluation == -1:
                        foot_placement = " "
                    elif foot_placement_evaluation == 0:
                        foot_placement = "Correct"
                        correct_foot_placement_count += 1
                    elif foot_placement_evaluation == 1:
                        foot_placement = "Too tight"
                        too_tight_foot_placement_count += 1
                    else:  # foot_placement_evaluation == 2
                        foot_placement = "Too wide"
                        too_wide_foot_placement_count += 1
            
                    # * Evaluate KNEE PLACEMENT error
                    if knee_placement_evaluation == -1:
                        knee_placement = " "
                    elif knee_placement_evaluation == 0:
                        knee_placement = "Correct"
                        correct_knee_placement_count += 1
                    elif knee_placement_evaluation == 1:
                        knee_placement = "Too tight"
                        too_tight_knee_placement_count += 1
                    else:  # knee_placement_evaluation == 2
                        knee_placement = "Too wide"
                        too_wide_knee_placement_count += 1


                    # Visualization
                    # Status box
                    cv2.rectangle(image, (0, 0), (500, 60), (245, 117, 16), -1)

                    # Display class
                    cv2.putText(image, "COUNT", (10, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, f'{str(counter)}' , (25, 40), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, "STAGE", (105, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, f'{str(predicted_class)}' , (110, 40), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255), 2, cv2.LINE_AA)
                    # Display Foot and Shoulder width ratio
                    # Display foot placement and counters
                    cv2.putText(image, "FOOT", (210, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, foot_placement, (195, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            
                    # Display knee placement and counters
                    cv2.putText(image, "KNEE", (350, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, knee_placement, (335, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                    out.write(image)

                except Exception as e:
                    print(f"Error: {e}")

        cap.release()
        out.release()

        # After processing, compress the video using FFmpeg
        
        generate_injury_estimation_report()
        return jsonify({"processed_video_url": f"http://{request.host}/processed_videos/{processed_video_name}"}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



def get_report():
    # Find the most recent report file
    # Find the most recent report file with the app name 'squat_feedback_report.txt'
    report_files = [f for f in os.listdir(REPORT_FOLDER) if f == 'squats_feedback_report.txt']
    if not report_files:
        return jsonify({'error': 'No reports found'}), 404


    report_file = report_files[0]
    report_path = os.path.join(REPORT_FOLDER, report_file)

    # Log the report file path
    print(f"Latest report found: {report_path}")
    # Return the report file
    return send_file(report_path, as_attachment=False, mimetype='text/plain')

