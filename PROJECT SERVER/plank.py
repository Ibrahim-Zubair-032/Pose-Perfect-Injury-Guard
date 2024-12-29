
from flask import request, jsonify, send_from_directory, send_file  
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')


source = Path(__file__).resolve().parent
# Setup for MediaPipe and other constants
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
correct = 0
low_back = 0
high_back = 0
prediction_probability_threshold = 0.6
VIDEO_PROCESS_STATUS = {}  # Dictionary to store video processing status
PROCESSED_VIDEOS_FOLDER = str(source/'processed_videos')
REPORT_FOLDER = str(source/'Report')
IMPORTANT_LMS = [
    "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]

HEADERS = ["label"]  # Label column
for lm in IMPORTANT_LMS:
    HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

# Load model and scaler
with open(str(source/'Model/Plank/LR_model.pkl'), "rb") as f:
    sklearn_model = pickle.load(f)

with open(str(source/'Model/Plank/input_scaler.pkl'), "rb") as f2:
    input_scaler = pickle.load(f2)

def extract_important_keypoints(results) -> list:
    landmarks = results.pose_landmarks.landmark
    data = []
    for lm in IMPORTANT_LMS:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    return np.array(data).flatten().tolist()


def generate_plank_injury_estimation_report(correct_counter, low_back_counter, high_back_counter):
    # Rounding counters
    correct_counter_rounded = round(correct_counter)
    low_back_counter_rounded = round(low_back_counter)
    high_back_counter_rounded = round(high_back_counter)
    
    # Total evaluations
    total_evaluations = correct_counter_rounded + low_back_counter_rounded + high_back_counter_rounded
    
    # Calculating percentages
    correct_percentage = (correct_counter_rounded / total_evaluations) * 100 if total_evaluations > 0 else 0
    low_back_percentage = (low_back_counter_rounded / total_evaluations) * 100 if total_evaluations > 0 else 0
    high_back_percentage = (high_back_counter_rounded / total_evaluations) * 100 if total_evaluations > 0 else 0

    # Rounding percentages
    correct_percentage = round(correct_percentage)
    low_back_percentage = round(low_back_percentage)
    high_back_percentage = round(high_back_percentage)

    # Creating the report content
    report_content = "Plank Injury Estimation Report\n" + "=" * 50 + "\n"
    if correct_counter_rounded > 0:
        report_content += f"CORRECT: {correct_percentage}%\n✔ Excellent core engagement\n✔ Reduced injury risk\n✔ Maintain a straight line from head to heels\n\n"
    if low_back_counter_rounded > 0:
        report_content += f"LOW BACK: {low_back_percentage}%\n❗ Risk of lower back injury (compression)\n❗ Limited core activation\n✔ Lift your hips and align your back\n\n"
    if high_back_counter_rounded > 0:
        report_content += f"HIGH BACK: {high_back_percentage}%\n❗ Increased strain on the lower back\n❗ Risk of back strain\n✔ Engage your core and avoid excessive arching\n\n"


    report_name = "plank_injury_report.txt"
    # Ensuring the REPORT_FOLDER exists
    # Creating the full path for the report
    report_path = os.path.join(REPORT_FOLDER, report_name)

    # Saving the report to the specified path
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report_content)
    print("Plank Report Generated Successfully")


def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def get_class(prediction: float) -> str:
    return {0: "C", 1: "H", 2: "L"}.get(prediction)


# @app.route('/', methods=['GET'])
# def index():
#     return "Welcome to the Plank Analysis API. Please use the /analyze endpoint to submit your video."
# @app.route('/analyze', methods=['POST'])
def process_video():

    if 'video' not in request.files:
            return jsonify({"error": "No video part"}), 400
        
    video_file = request.files['video']
        
        # Create a temporary file to store the uploaded video
    temp_video_path = 'temp_video.mp4'
    video_file.save(temp_video_path)

        # Define folder for processed videos and create if necessary
    PROCESSED_VIDEOS_FOLDER = 'processed_videos'
    if not os.path.exists(PROCESSED_VIDEOS_FOLDER):
        os.makedirs(PROCESSED_VIDEOS_FOLDER)

    processed_video_name = "processed_plank.mp4"
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
    global correct 
    global low_back 
    global high_back 
    global current_stage
    current_stage = ''

    
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
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            try:
                row = extract_important_keypoints(results)
                X = pd.DataFrame([row], columns=HEADERS[1:])
                X = pd.DataFrame(input_scaler.transform(X))

                predicted_class = sklearn_model.predict(X)[0]
                predicted_class = get_class(predicted_class)
                prediction_probability = sklearn_model.predict_proba(X)[0]

                if predicted_class == "C" and prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold:
                    current_stage = "Correct"
                    correct += 1
                elif predicted_class == "L" and prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold:
                    current_stage = "Low back"
                    low_back += 1
                elif predicted_class == "H" and prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold:
                    current_stage = "High back"
                    high_back += 1
                else:
                    current_stage = " "

                # Visualization
                # Status box
                cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
                # Display class
                cv2.putText(image, "Stage", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, current_stage, (90, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


            except Exception as e:
                print(f"Error: {e}")
            
            out.write(image)
            

        cap.release()
        out.release()

    # Generate the report and save it with a fixed name
    
    generate_plank_injury_estimation_report(correct, low_back, high_back)
    
    return jsonify({
    "processed_video_url": f"http://{request.host}/processed_videos/{processed_video_name}"
}), 200


def get_report():
    # Find the most recent report file
    report_files = [f for f in os.listdir(REPORT_FOLDER) if f == 'plank_injury_report.txt']
    if not report_files:
        return jsonify({'error': 'No reports found'}), 404

    # Since there's only one file (overwritten each time), just pick the first one
    report_file = report_files[0]
    report_path = os.path.join(REPORT_FOLDER, report_file)

    # Log the report file path
    print(f"Report found: {report_path}")
    # Return the report file
    return send_file(report_path, as_attachment=False, mimetype='text/plain')

