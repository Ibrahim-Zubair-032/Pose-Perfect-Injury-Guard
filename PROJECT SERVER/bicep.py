import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import os
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, send_file  
import warnings
from sklearn.exceptions import InconsistentVersionWarning
# Ignore specific warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)  # Ignore SyntaxWarnings
warnings.filterwarnings("ignore", category=UserWarning)    # Ignore UserWarnings (includes InconsistentVersionWarnings)
source = Path(__file__).resolve().parent

REPORT_FOLDER = str(source/'Report')
PROCESSED_VIDEOS_FOLDER = str(source/'processed_videos')
if not os.path.exists(PROCESSED_VIDEOS_FOLDER):
    os.makedirs(PROCESSED_VIDEOS_FOLDER)

right_counter = 0
left_counter = 0
right_weak_contraction = 0
left_weak_contraction = 0
right_loose_arm = 0
left_loose_arm=0
lean_back = 0
correct = 0

with open(str(source/'Model/BicepCurl/input_scaler.pkl'),"rb") as f:
    input_scaler = pickle.load(f)

# Load model
with open(str(source/'Model/BicepCurl/KNN_model.pkl'), "rb") as f2:
    sklearn_model = pickle.load(f2)

IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "LEFT_ELBOW",
    "RIGHT_WRIST",
    "LEFT_WRIST",       
    "LEFT_HIP",
    "RIGHT_HIP",
]

# Generate all columns of the DataFrame
HEADERS = ["label"]  # Label column
for lm in IMPORTANT_LMS:
    HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to rescale frame
def rescale_frame(frame, percent=100):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

# Function to extract important keypoints
def extract_important_keypoints(results, important_lms):
    row = []
    for lm in important_lms:
        landmark = results.pose_landmarks.landmark[getattr(mp_pose.PoseLandmark, lm)]
        row += [landmark.x, landmark.y, landmark.z, landmark.visibility]
    return row

def generate_bicep_curl_analysis_report():
    global right_counter, left_counter
    global right_weak_contraction, left_weak_contraction
    global right_loose_arm, left_loose_arm
    global lean_back, correct

    # Determine total repetitions based on the larger count (right or left)
    total_reps = max(right_counter, left_counter)

    # Calculate percentages for weak contraction and loose arm
    right_weak_contraction_percentage = (right_weak_contraction / total_reps * 100) if total_reps > 0 else 0
    left_weak_contraction_percentage = (left_weak_contraction / total_reps * 100) if total_reps > 0 else 0

    right_loose_arm_percentage = (right_loose_arm / total_reps * 100) if total_reps > 0 else 0
    left_loose_arm_percentage = (left_loose_arm / total_reps * 100) if total_reps > 0 else 0

    # Calculate percentages for correct and lean back combined
    total_evaluations = right_counter + left_counter + correct + lean_back
    correct_leanback_percentage = (correct + lean_back) / total_evaluations * 100 if total_evaluations > 0 else 0

    # Start building the report content
    report_content = "💪 Bicep Curl Analysis Report\n"
    report_content += "═" * 60 + "\n"
    report_content += "🏋️  Movement          | 🔢 Percentage   | 🔍 Impact                                   | 📝 Advice\n"
    report_content += "─" * 60 + "\n"

    # Correct Feedback
    if correct_leanback_percentage > 0:
        report_content += f"✔️  CORRECT            | {round(correct_leanback_percentage, 2):>5}%        | ✅ Optimal form.                          | Keep elbows close to your torso.\n"

    # Lean Back Feedback
    if correct_leanback_percentage < 100:
        report_content += f"❌ LEAN BACK          | {round(100 - correct_leanback_percentage, 2):>5}%        | ⚠️  Lower back strain risk.             | Keep your back straight throughout.\n"
        report_content += "                     |                  | ⬇️  Limited range of motion.              | Avoid leaning back during curls.\n"

    # Right Arm: Weak Contraction Feedback
    if right_weak_contraction_percentage > 0:
        report_content += f"🔴 RIGHT WEAK ARM     | {round(right_weak_contraction_percentage, 2):>5}%        | 🔻 Weak contraction.                     | Focus on full range of motion.\n"

    # Left Arm: Weak Contraction Feedback
    if left_weak_contraction_percentage > 0:
        report_content += f"🟡 LEFT WEAK ARM      | {round(left_weak_contraction_percentage, 2):>5}%        | 🔻 Weak contraction.                     | Tension biceps for maximum activation.\n"

    # Right Arm: Loose Arm Feedback
    if right_loose_arm_percentage > 0:
        report_content += f"🔵 RIGHT LOOSE ARM    | {round(right_loose_arm_percentage, 2):>5}%        | ⚠️  Loose upper arm.                     | Keep upper arm stationary.\n"

    # Left Arm: Loose Arm Feedback
    if left_loose_arm_percentage > 0:
        report_content += f"🟣 LEFT LOOSE ARM     | {round(left_loose_arm_percentage, 2):>5}%        | ⚠️  Loose upper arm.                     | Keep elbows tucked in.\n"

    # Ensure the report folder exists
    if not os.path.exists(REPORT_FOLDER):
        os.makedirs(REPORT_FOLDER)

    # Save the report
    report_filename = "bicep_curl_feedback_report.txt"
    report_path = os.path.join(REPORT_FOLDER, report_filename)

    with open(report_path, "w",encoding = "utf-8") as report_file:
        report_file.write(report_content)

    print(f"💪 Bicep Curl Analysis Report generated successfully at: {report_path}")


def calculate_angle(point1: list, point2: list, point3: list) -> float:
    '''
    Calculate the angle between 3 points
    Unit of the angle will be in Degree
    '''
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)

    # Calculate algo
    angleInRad = np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) - np.arctan2(point1[1] - point2[1], point1[0] - point2[0])
    angleInDeg = np.abs(angleInRad * 180.0 / np.pi)

    angleInDeg = angleInDeg if angleInDeg <= 180 else 360 - angleInDeg
    return angleInDeg
# Parameters for analysis
VISIBILITY_THRESHOLD = 0.65
STAGE_UP_THRESHOLD = 90
STAGE_DOWN_THRESHOLD = 120
PEAK_CONTRACTION_THRESHOLD = 60
LOOSE_UPPER_ARM_ANGLE_THRESHOLD = 40
POSTURE_ERROR_THRESHOLD = 0.7
posture = "Correct"
# Initialize analysis classes
class BicepPoseAnalysis:
    def __init__(self, side: str, stage_down_threshold: float, stage_up_threshold: float, peak_contraction_threshold: float, loose_upper_arm_angle_threshold: float, visibility_threshold: float):
        # Initialize thresholds
        self.stage_down_threshold = stage_down_threshold
        self.stage_up_threshold = stage_up_threshold
        self.peak_contraction_threshold = peak_contraction_threshold
        self.loose_upper_arm_angle_threshold = loose_upper_arm_angle_threshold
        self.visibility_threshold = visibility_threshold

        self.side = side
        self.counter = 0
        self.stage = "down"
        self.is_visible = True
        self.detected_errors = {
            "LOOSE_UPPER_ARM": 0,
            "PEAK_CONTRACTION": 0,
        }

        # Params for loose upper arm error detection
        self.loose_upper_arm = False

        # Params for peak contraction error detection
        self.peak_contraction_angle = 1000
        self.peak_contraction_frame = None
    
    def get_joints(self, landmarks) -> bool:
        '''
        Check for joints' visibility then get joints coordinate
        '''
        side = self.side.upper()

        # Check visibility
        joints_visibility = [ landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].visibility, landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].visibility, landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].visibility ]

        is_visible = all([ vis > self.visibility_threshold for vis in joints_visibility ])
        self.is_visible = is_visible

        if not is_visible:
            return self.is_visible
        
        # Get joints' coordinates
        self.shoulder = [ landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].y ]
        self.elbow = [ landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].y ]
        self.wrist = [ landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].y ]

        return self.is_visible
    
    def analyze_pose(self, landmarks, frame):
        '''
        - Bicep Counter
        - Errors Detection
        '''
        self.get_joints(landmarks)

        # Cancel calculation if visibility is poor
        if not self.is_visible:
            return (None, None)

        # * Calculate curl angle for counter
        bicep_curl_angle = int(calculate_angle(self.shoulder, self.elbow, self.wrist))
        if bicep_curl_angle > self.stage_down_threshold:
            self.stage = "down"
        elif bicep_curl_angle < self.stage_up_threshold and self.stage == "down":
            self.stage = "up"
            self.counter += 1
        
        # * Calculate the angle between the upper arm (shoulder & joint) and the Y axis
        shoulder_projection = [ self.shoulder[0], 1 ] # Represent the projection of the shoulder to the X axis
        ground_upper_arm_angle = int(calculate_angle(self.elbow, self.shoulder, shoulder_projection))

        # * Evaluation for LOOSE UPPER ARM error
        if ground_upper_arm_angle > self.loose_upper_arm_angle_threshold:
            # Limit the saved frame
            if not self.loose_upper_arm:
                self.loose_upper_arm = True
                # save_frame_as_image(frame, f"Loose upper arm: {ground_upper_arm_angle}")
                self.detected_errors["LOOSE_UPPER_ARM"] += 1
        else:
            self.loose_upper_arm = False
        
        # * Evaluate PEAK CONTRACTION error
        if self.stage == "up" and bicep_curl_angle < self.peak_contraction_angle:
            # Save peaked contraction every rep
            self.peak_contraction_angle = bicep_curl_angle
            self.peak_contraction_frame = frame
            
        elif self.stage == "down":
            # * Evaluate if the peak is higher than the threshold if True, marked as an error then saved that frame
            if self.peak_contraction_angle != 1000 and self.peak_contraction_angle >= self.peak_contraction_threshold:
                # save_frame_as_image(self.peak_contraction_frame, f"{self.side} - Peak Contraction: {self.peak_contraction_angle}")
                self.detected_errors["PEAK_CONTRACTION"] += 1
            
            # Reset params
            self.peak_contraction_angle = 1000
            self.peak_contraction_frame = None
        
        return (bicep_curl_angle, ground_upper_arm_angle)



def analyze_bicepCurl():

    global correct, lean_back
    global right_counter
    global left_counter
    global left_weak_contraction
    global right_weak_contraction
    global left_loose_arm
    global right_loose_arm


    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video provided"}), 400

        video_file = request.files['video']
        temp_video_path = 'temp_bicep_curl_video.mp4'
        video_file.save(temp_video_path)

        processed_video_name = "processed_bicep_curl_video.mp4"
        processed_video_path = os.path.join(PROCESSED_VIDEOS_FOLDER, processed_video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            return jsonify({"error": "Error opening video file"}), 500

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(processed_video_path, fourcc, 30.0, (width, height))

        left_arm_analysis = BicepPoseAnalysis(
            side="left",
            stage_down_threshold=STAGE_DOWN_THRESHOLD,
            stage_up_threshold=STAGE_UP_THRESHOLD,
            peak_contraction_threshold=PEAK_CONTRACTION_THRESHOLD,
            loose_upper_arm_angle_threshold=LOOSE_UPPER_ARM_ANGLE_THRESHOLD,
            visibility_threshold=VISIBILITY_THRESHOLD
        )

        right_arm_analysis = BicepPoseAnalysis(
            side="right",
            stage_down_threshold=STAGE_DOWN_THRESHOLD,
            stage_up_threshold=STAGE_UP_THRESHOLD,
            peak_contraction_threshold=PEAK_CONTRACTION_THRESHOLD,
            loose_upper_arm_angle_threshold=LOOSE_UPPER_ARM_ANGLE_THRESHOLD,
            visibility_threshold=VISIBILITY_THRESHOLD
        )

        with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
            while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                    break

                # image = rescale_frame(image, 65)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                image_rgb.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
                    )
                    try:
                        landmarks = results.pose_landmarks.landmark
            
                        # Analyze pose for bicep curl for both arms
                        right_arm_analysis.analyze_pose(landmarks=landmarks, frame=image)
                        left_arm_analysis.analyze_pose(landmarks=landmarks, frame=image)

                        # Extract keypoints from frame for the input
                        row = extract_important_keypoints(results, IMPORTANT_LMS)
                        X = pd.DataFrame([row], columns=HEADERS[1:])
                        X = pd.DataFrame(input_scaler.transform(X))

                        # Make prediction and its probability
                        predicted_class = sklearn_model.predict(X)[0]
                        prediction_probabilities = sklearn_model.predict_proba(X)[0]
                        class_prediction_probability = round(prediction_probabilities[np.argmax(prediction_probabilities)], 2)

                        if class_prediction_probability >= POSTURE_ERROR_THRESHOLD:
                            posture = predicted_class
            
                        # Display posture status and predictions
                        cv2.rectangle(image, (0, 0), (550, 120), (245, 117, 16), -1)
                        # cv2.putText(image, "Posture: " + str(posture), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                        # Display counters and errors
                        cv2.putText(image, "RIGHT ----> COUNTER", (5, 12), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(right_arm_analysis.counter), (125, 30), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(image, "LEFT ----> COUNTER", (15, 60), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(left_arm_analysis.counter), (125, 80), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA)

                        # # Display errors for both arms
            
                        cv2.putText(image, "LOOSE ARM", (350, 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(left_arm_analysis.detected_errors["LOOSE_UPPER_ARM"]), (380, 80), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                        # Check for arm analysis errors and increment counters accordingly
         
                        # Now display these counters on the image
            
                        # Display errors for the right arm
                        cv2.putText(image, "WEAK CONTRACTION", (190, 12), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(right_arm_analysis.detected_errors["PEAK_CONTRACTION"]), (225, 30), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA)
            
                        cv2.putText(image, "LOOSE ARM", (350, 12), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(right_arm_analysis.detected_errors["LOOSE_UPPER_ARM"]), (380, 30), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA)
            
                        # Display errors for the left arm
                        cv2.putText(image, "WEAK CONTRACTION", (190, 60), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(left_arm_analysis.detected_errors["PEAK_CONTRACTION"]), (225, 80), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA)
            
                        cv2.putText(image, "LOOSE ARM", (350, 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(left_arm_analysis.detected_errors["LOOSE_UPPER_ARM"]), (380, 80), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
                        # Display posture error (lean back)
                        cv2.putText(image, "Posture", (210, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
                        # Determine the label based on predicted class
                        if predicted_class == 'C':
                            label = "Correct"
                            correct+=1
                        elif predicted_class == 'L':
                            label = "Lean Back"
                            lean_back+=1
                        else:
                            label = " "  # Default in case of any other class
            
                        # Display the posture label on the image
                        cv2.putText(image, label, (210, 120), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    except Exception as e:
                        print(f"Error processing frame: {e}")

                out.write(image)

        cap.release()
        out.release()

        right_counter = right_arm_analysis.counter
        left_counter = left_arm_analysis.counter
        left_weak_contraction = left_arm_analysis.detected_errors['PEAK_CONTRACTION']
        right_weak_contraction = right_arm_analysis.detected_errors['PEAK_CONTRACTION']
        left_loose_arm = left_arm_analysis.detected_errors['LOOSE_UPPER_ARM']
        right_loose_arm = right_arm_analysis.detected_errors['LOOSE_UPPER_ARM']
        generate_bicep_curl_analysis_report()
        return jsonify({"processed_video_url": f"http://{request.host}/processed_videos/{processed_video_name}"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



def get_report():
    # Find the most recent report file
    report_files = [f for f in os.listdir(REPORT_FOLDER) if f == 'bicep_curl_feedback_report.txt']
    if not report_files:
        return jsonify({'error': 'No reports found'}), 404

    # Since there's only one file (overwritten each time), just pick the first one
    report_file = report_files[0]
    report_path = os.path.join(REPORT_FOLDER, report_file)

    # Log the report file path
    print(f"Report found: {report_path}")
    # Return the report file
    return send_file(report_path, as_attachment=False, mimetype='text/plain')

