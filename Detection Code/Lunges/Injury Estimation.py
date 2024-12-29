import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import traceback
import pickle

import warnings
warnings.filterwarnings('ignore')
# Load models
with open("D:\Desktop\Lunges FYP Model\Trained Models\stage_SVC_model.pkl", "rb") as f:
    stage_sklearn_model = pickle.load(f)

with open("D:\Desktop\Lunges FYP Model\Trained Models\err_LR_model.pkl", "rb") as f:
    err_sklearn_model = pickle.load(f)

# Load input scaler
with open("D:\Desktop\Lunges FYP Model\Trained Models\input_scaler.pkl", "rb") as f:
    input_scaler = pickle.load(f)

# Define constants
    
# Global counters
#left_too_low_counter = 0
knee_over_toe=0
left_good_counter = 0
left_not_low_enough_counter = 0
#right_too_low_counter = 0
right_good_counter = 0
right_not_low_enough_counter = 0
updated_previous_knee_angle=0
previous_left_knee_angle = None
previous_right_knee_angle = None
#VIDEO_PATH = r"D:\Desktop\Lunges FYP Model\5025761-hd_1080_1920_25fps.mp4"
#VIDEO_PATH = r"D:\Desktop\Lunges FYP Model\6525491-hd_1920_1080_25fps.mp4"
VIDEO_PATH = r"D:\Desktop\Lunges FYP Model\lunge_demo (1).mp4"
#VIDEO_PATH = r"D:\Desktop\Lunges FYP Model\5469836-uhd_3840_2160_30fps.mp4"
#VIDEO_PATH = r"D:\Desktop\Lunges FYP Model\lunges.webm"
#VIDEO_PATH = r"D:\Desktop\Lunges FYP Model\lunges2.webm"
#VIDEO_PATH = r"D:\Desktop\Lunges FYP Model\lunges3.webm"
#VIDEO_PATH = r"D:\Desktop\Lunges FYP Model\lunges4.webm"
#VIDEO_PATH = r"D:\Desktop\Lunges FYP Model\lunges5.mp4"
#VIDEO_PATH = r"D:\Desktop\Lunges FYP Model\lunges6.mp4"
# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# Determine important landmarks for lunge
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

# Generate all columns of the data frame

HEADERS = ["label"] # Label column
#left_too_low_counter = 0
left_good_counter = 0
left_not_low_enough_counter = 0

#right_too_low_counter = 0
right_good_counter = 0
right_not_low_enough_counter = 0
for lm in IMPORTANT_LMS:
    HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]
def extract_important_keypoints(results) -> list:
    '''
    Extract important keypoints from mediapipe pose detection
    '''
    landmarks = results.pose_landmarks.landmark

    data = []
    for lm in IMPORTANT_LMS:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    
    return np.array(data).flatten().tolist()


def rescale_frame(frame, percent=50):
    '''
    Rescale a frame to a certain percentage compare to its original frame
    '''
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


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

def update_right_knee_feedback(knee_coords, hip_coords, ankle_coords, angle_thresholds, current_stage):
    global previous_right_knee_angle

    # Calculate the angle
    knee_angle = calculate_angle(hip_coords, knee_coords, ankle_coords)
    
    # Initialize previous_knee_angle if it's the first call 
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
      if 100 < knee_angle <= 150:
        # Check if the knee angle is above 90°, but less than or equal to 120° (halfway down position)
        # Feedback when the person is not going low enough (not achieving the full lunge position)
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


def generate_report():
    report_content = (
        "Final Knee Feedback Report:\n\n"
        "Left Knee Feedback:\n"
        f" - GOOD: {left_good_counter} times\n"
        f" - NOT LOW ENOUGH: {left_not_low_enough_counter} times\n\n"
        "Right Knee Feedback:\n"
        f" - GOOD: {right_good_counter} times\n"
        f" - NOT LOW ENOUGH: {right_not_low_enough_counter} times\n"
        "Knee Over Toe:\n"
        f" - {knee_over_toe} times\n"
    )
    
def generate_lunge_injury_estimation_report():
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
    report_content = "Lunge Knee Feedback Report\n"
    report_content += "=" * 60 + "\n"
    report_content += "Knee Placement         | Percentage   | Impact                                   | Advice                                             \n"
    report_content += "-" * 60 + "\n"
    
    # Left Knee Feedback
    if left_good_percentage > 0:
        report_content += f"LEFT GOOD             | {round(left_good_percentage, 2)}%        | Excellent                               | Maintain proper knee alignment.\n"
    
    if left_not_low_enough_percentage > 0:
        report_content += f"LEFT NOT LOW ENOUGH   | {round(left_not_low_enough_percentage, 2)}%        | - Limited activation of glutes and hamstrings. | Lower your back knee just above the ground.\n"
        report_content += "                        |              | - Less effective workout for lower body.   |                                                    \n"
    
    # Right Knee Feedback
    if right_good_percentage > 0:
        report_content += f"RIGHT GOOD            | {round(right_good_percentage, 2)}%        | Excellent                               | Keep knee aligned with the toes.\n"
    
    if right_not_low_enough_percentage > 0:
        report_content += f"RIGHT NOT LOW ENOUGH  | {round(right_not_low_enough_percentage, 2)}%        | - Poor engagement of quads and glutes.      | Lower your knee until both knees form 90-degree angles.\n"
        report_content += "                        |              | - Limited range of motion.               |                                                    \n"
    
    # Knee Over Toe Feedback
    if knee_over_toe_percentage > 0:
        report_content += f"KNEE OVER TOE         | {round(knee_over_toe_percentage, 2)}%        | - Risk of knee injuries (ACL, MCL) due to excessive forward movement. | Keep your knees behind your toes.\n"
        report_content += "                        |              | - Increased stress on the knee joint.     |                                                    \n"

    # Save the report to a text file
    with open("lunge_knee_feedback_report.txt", "w") as report_file:
        report_file.write(report_content)

    print("Lunge Knee Feedback Report generated successfully!")
    
def analyze_knee_angle(
    mp_results, stage: str, angle_thresholds: list, draw_to_image: tuple = None
):
    global  left_good_counter, left_not_low_enough_counter
    global  right_good_counter, right_not_low_enough_counter

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
    right_feedback,  right_good_increment, right_not_low_enough_increment= update_right_knee_feedback(
        right_knee, right_hip, right_ankle, angle_thresholds,current_stage
    )

    right_good_counter += right_good_increment
    right_not_low_enough_counter += right_not_low_enough_increment

    # # # Get feedback and update counters for the left knee
    left_feedback,  left_good_increment, left_not_low_enough_increment= update_left_knee_feedback(
        left_knee, left_hip, left_ankle, angle_thresholds,current_stage 
    )
 
    left_good_counter += left_good_increment
    left_not_low_enough_counter += left_not_low_enough_increment

    # Draw angles and feedback on image (optional)
    if draw_to_image is not None:
        (image, video_dimensions) = draw_to_image

        # Visualize angles
        cv2.putText(image, str(int(results["right"]["angle"])), tuple(np.multiply(right_knee, video_dimensions).astype(int)),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, str(int(results["left"]["angle"])), tuple(np.multiply(left_knee, video_dimensions).astype(int)),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Display feedback
        cv2.putText(image, f"Right Knee: {right_feedback}", (230, 30), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f"Left Knee: {left_feedback}", (230, 50), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return results



cap = cv2.VideoCapture(VIDEO_PATH)
current_stage = ""
counter = 0

prediction_probability_threshold = 0.8
ANGLE_THRESHOLDS = [60, 135]

knee_over_toe = False

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break

        # Reduce size of a frame
        image = rescale_frame(image,65)
        video_dimensions = [image.shape[1], image.shape[0]]

        # Recolor image from BGR to RGB for mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        if not results.pose_landmarks:
            print("No human found")
            continue

        # Recolor image from BGR to RGB for mediapipe
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))

        # Make detection
        try:
            # Extract keypoints from frame for the input
            row = extract_important_keypoints(results)
            X = pd.DataFrame([row], columns=HEADERS[1:])
            X = pd.DataFrame(input_scaler.transform(X))

            # Make prediction and its probability
            stage_predicted_class = stage_sklearn_model.predict(X)[0]
            stage_prediction_probabilities = stage_sklearn_model.predict_proba(X)[0]
            stage_prediction_probability = round(stage_prediction_probabilities[stage_prediction_probabilities.argmax()], 2)
            # Evaluate model prediction
            if stage_predicted_class == "I" and stage_prediction_probability >= prediction_probability_threshold:
                current_stage = "start"
            elif stage_predicted_class == "M" and stage_prediction_probability >= prediction_probability_threshold:
                current_stage = "mid"
            elif stage_predicted_class == "D" and stage_prediction_probability >= prediction_probability_threshold:
                # Updated Logic: Increment counter if current_stage is "mid" or "start" and predicted is "D"
                if (current_stage == "mid" or current_stage == "start") and stage_prediction_probability > 0.5:
                    counter += 1  # Increment counter here
                # Do not increment if already in "down"
                current_stage = "down"
            
            
            # Error detection
            # Knee angle
            analyze_knee_angle(mp_results=results, stage=current_stage, angle_thresholds=ANGLE_THRESHOLDS, draw_to_image=(image, video_dimensions))

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
              
                
                
            # Visualization
            # Status box
            cv2.rectangle(image, (0, 0), (800, 60), (245, 117, 16), -1)
            
            # Display stage prediction
            cv2.putText(image, "STAGE", (460, 12), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            #cv2.putText(image, str(stage_prediction_probability), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, current_stage, (470, 30), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            
           
            # Display Counter
            cv2.putText(image, "COUNTER", (360, 12), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (380, 30), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            # Display feedback counters for both knees
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
          
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            break
        
        cv2.imshow("CV2", image)
        
        # # Press Q to close cv2 window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()
    generate_report()
    generate_lunge_injury_estimation_report()
    #generate_injury_report('knee_feedback_report.txt')
    for i in range (1, 5):
        cv2.waitKey(1)
